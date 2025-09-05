#!/usr/bin/env python3
"""
Gender Classification FastAPI Service
A FastAPI service for gender classification using DSP techniques
"""

import os
import tempfile
import logging
import math
from typing import Dict, Any
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pyngrok import ngrok

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_audio_features(audio_path: str) -> Dict[str, float]:
    """Extract DSP features from audio for gender classification using manual DSP"""
    try:
        import librosa  # Only for audio loading and basic operations
        from services.manual_dsp_core import ManualDSPCore, frequency_bins
        
        # Load audio (still use librosa for file I/O convenience)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Manual silence trimming using energy-based detection
        def trim_silence(signal, top_db=20):
            # Calculate frame energy
            frame_length = 2048
            hop_length = 512
            
            energy = []
            for i in range(0, len(signal) - frame_length + 1, hop_length):
                frame = signal[i:i + frame_length]
                frame_energy = sum(x**2 for x in frame)
                energy.append(frame_energy)
            
            if not energy:
                return signal
            
            # Convert to dB
            max_energy = max(energy)
            if max_energy == 0:
                return signal
            
            energy_db = [10 * math.log10(e / max_energy) if e > 0 else -100 for e in energy]
            threshold = max(energy_db) - top_db
            
            # Find start and end of non-silent region
            start_idx = 0
            end_idx = len(energy) - 1
            
            for i, e in enumerate(energy_db):
                if e > threshold:
                    start_idx = i
                    break
            
            for i in range(len(energy_db) - 1, -1, -1):
                if energy_db[i] > threshold:
                    end_idx = i
                    break
            
            # Convert frame indices to sample indices
            start_sample = start_idx * hop_length
            end_sample = min(len(signal), (end_idx + 1) * hop_length + frame_length)
            
            return signal[start_sample:end_sample]
        
        y = trim_silence(y)
        
        if len(y) == 0:
            raise ValueError("Audio file contains no sound")
        
        features = {}
        
        # 1. Fundamental Frequency (F0) - Key gender indicator using manual autocorrelation
        try:
            # Frame-based F0 detection using autocorrelation
            frame_length = 1024
            hop_length = 512
            f0_values = []
            
            for i in range(0, len(y) - frame_length + 1, hop_length):
                frame = y[i:i + frame_length]
                
                # Apply window to reduce edge effects
                window = ManualDSPCore.hamming_window(frame_length)
                windowed_frame = ManualDSPCore.apply_window(frame.tolist(), window)
                
                # Calculate autocorrelation
                autocorr = ManualDSPCore.autocorrelation(windowed_frame, frame_length // 2)
                
                # Find F0 in expected range (50-500 Hz)
                min_period = int(sr / 500)  # Max F0 = 500Hz
                max_period = int(sr / 50)   # Min F0 = 50Hz
                
                if max_period < len(autocorr):
                    # Search for maximum in the valid range
                    search_range = autocorr[min_period:min(max_period, len(autocorr))]
                    if search_range:
                        max_idx = search_range.index(max(search_range))
                        period = max_idx + min_period
                        
                        # Validate the peak (should be significant)
                        if autocorr[period] > 0.3 * autocorr[0]:  # At least 30% of zero-lag
                            f0 = sr / period
                            if 50 <= f0 <= 500:  # Valid F0 range
                                f0_values.append(f0)
            
            if len(f0_values) > 3:  # Need at least some valid frames
                features['f0_mean'] = float(sum(f0_values) / len(f0_values))
                
                # Manual standard deviation calculation
                mean_f0 = features['f0_mean']
                variance = sum((f0 - mean_f0) ** 2 for f0 in f0_values) / len(f0_values)
                features['f0_std'] = float(math.sqrt(variance))
                
                # Manual median calculation
                sorted_f0 = sorted(f0_values)
                n = len(sorted_f0)
                if n % 2 == 0:
                    features['f0_median'] = float((sorted_f0[n//2 - 1] + sorted_f0[n//2]) / 2)
                else:
                    features['f0_median'] = float(sorted_f0[n//2])
            else:
                features['f0_mean'] = 0.0
                features['f0_std'] = 0.0
                features['f0_median'] = 0.0
                
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}")
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_median'] = 0.0
        
        # 2. Spectral Centroid - Brightness of voice using manual FFT
        try:
            frame_length = 2048
            hop_length = 512
            centroids = []
            
            for i in range(0, len(y) - frame_length + 1, hop_length):
                frame = y[i:i + frame_length]
                
                # Apply window
                window = ManualDSPCore.hamming_window(frame_length)
                windowed_frame = ManualDSPCore.apply_window(frame.tolist(), window)
                
                # Calculate DFT using course method
                X_real, X_imag = ManualDSPCore.dft_real(windowed_frame)
                
                # Calculate spectral centroid using course approach
                centroid = ManualDSPCore.spectral_centroid_from_dft(X_real, X_imag, sr)
                if centroid > 0:
                    centroids.append(centroid)
            
            if centroids:
                features['spectral_centroid_mean'] = float(sum(centroids) / len(centroids))
                
                # Manual standard deviation
                mean_centroid = features['spectral_centroid_mean']
                variance = sum((c - mean_centroid) ** 2 for c in centroids) / len(centroids)
                features['spectral_centroid_std'] = float(math.sqrt(variance))
            else:
                features['spectral_centroid_mean'] = 0.0
                features['spectral_centroid_std'] = 0.0
                
        except Exception as e:
            logger.warning(f"Spectral centroid extraction failed: {e}")
            features['spectral_centroid_mean'] = 0.0
            features['spectral_centroid_std'] = 0.0
        
        # 3. MFCCs - Vocal tract characteristics using manual implementation
        try:
            def manual_mfcc(signal, sr, n_mfcc=13, n_mels=26):
                """Manual MFCC implementation"""
                frame_length = 2048
                hop_length = 512
                
                # Create mel filter bank
                def mel_scale(f):
                    return 2595 * math.log10(1 + f / 700)
                
                def inv_mel_scale(m):
                    return 700 * (10**(m / 2595) - 1)
                
                # Mel frequency range
                mel_min = mel_scale(0)
                mel_max = mel_scale(sr / 2)
                mel_points = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
                hz_points = [inv_mel_scale(m) for m in mel_points]
                
                # Convert to FFT bin numbers
                fft_bins = [int(f * frame_length / sr) for f in hz_points]
                
                # Create mel filters
                mel_filters = []
                for i in range(1, len(fft_bins) - 1):
                    filter_bank = [0.0] * (frame_length // 2 + 1)
                    
                    # Left slope
                    for j in range(fft_bins[i-1], fft_bins[i]):
                        if fft_bins[i] != fft_bins[i-1]:
                            filter_bank[j] = (j - fft_bins[i-1]) / (fft_bins[i] - fft_bins[i-1])
                    
                    # Right slope
                    for j in range(fft_bins[i], fft_bins[i+1]):
                        if fft_bins[i+1] != fft_bins[i]:
                            filter_bank[j] = (fft_bins[i+1] - j) / (fft_bins[i+1] - fft_bins[i])
                    
                    mel_filters.append(filter_bank)
                
                # Process frames
                mfcc_frames = []
                for i in range(0, len(signal) - frame_length + 1, hop_length):
                    frame = signal[i:i + frame_length]
                    
                    # Apply window
                    window = ManualDSPCore.hamming_window(frame_length)
                    windowed_frame = ManualDSPCore.apply_window(frame.tolist(), window)
                    
                    # DFT using course method
                    X_real, X_imag = ManualDSPCore.dft_real(windowed_frame)
                    power_spectrum = ManualDSPCore.power_spectrum(X_real, X_imag)
                    
                    # Apply mel filters
                    mel_energies = []
                    for mel_filter in mel_filters:
                        energy = sum(p * f for p, f in zip(power_spectrum, mel_filter))
                        mel_energies.append(max(energy, 1e-10))  # Avoid log(0)
                    
                    # Log mel energies
                    log_mel = [math.log(e) for e in mel_energies]
                    
                    # DCT to get MFCCs (simplified version)
                    mfccs_frame = []
                    for k in range(n_mfcc):
                        mfcc_k = 0
                        for n in range(len(log_mel)):
                            mfcc_k += log_mel[n] * math.cos(math.pi * k * (n + 0.5) / len(log_mel))
                        mfccs_frame.append(mfcc_k)
                    
                    mfcc_frames.append(mfccs_frame)
                
                return mfcc_frames
            
            mfcc_frames = manual_mfcc(y.tolist(), sr, n_mfcc=13)
            
            if mfcc_frames:
                # Calculate statistics for first 5 MFCCs
                for i in range(min(5, 13)):
                    mfcc_values = [frame[i] for frame in mfcc_frames]
                    
                    if mfcc_values:
                        # Mean
                        features[f'mfcc_{i}_mean'] = float(sum(mfcc_values) / len(mfcc_values))
                        
                        # Standard deviation
                        mean_val = features[f'mfcc_{i}_mean']
                        variance = sum((val - mean_val) ** 2 for val in mfcc_values) / len(mfcc_values)
                        features[f'mfcc_{i}_std'] = float(math.sqrt(variance))
                    else:
                        features[f'mfcc_{i}_mean'] = 0.0
                        features[f'mfcc_{i}_std'] = 0.0
            else:
                for i in range(5):
                    features[f'mfcc_{i}_mean'] = 0.0
                    features[f'mfcc_{i}_std'] = 0.0
                    
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            for i in range(5):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
        
        # 4. Simplified formant estimation using manual spectral peaks
        try:
            # Use entire signal for formant estimation
            window = ManualDSPCore.hamming_window(len(y))
            windowed_signal = ManualDSPCore.apply_window(y.tolist(), window)
            
            # Get power spectrum using manual DFT
            X_real, X_imag = ManualDSPCore.dft_real(windowed_signal)
            magnitude = ManualDSPCore.magnitude_spectrum(X_real, X_imag)
            freq_bins = frequency_bins(len(y), sr)
            
            # Manual smoothing using moving average
            smoothed_magnitude = ManualDSPCore.smooth_signal(magnitude, 51)
            
            # Find peaks in formant range (200-4000 Hz)
            formant_start_idx = 0
            formant_end_idx = len(freq_bins) - 1
            
            for i, f in enumerate(freq_bins):
                if f >= 200 and formant_start_idx == 0:
                    formant_start_idx = i
                if f <= 4000:
                    formant_end_idx = i
            
            # Extract formant region
            formant_freqs = freq_bins[formant_start_idx:formant_end_idx + 1]
            formant_mags = smoothed_magnitude[formant_start_idx:formant_end_idx + 1]
            
            if formant_mags:
                # Find peaks manually
                threshold = max(formant_mags) * 0.1
                peaks = ManualDSPCore.find_peaks(formant_mags, height=threshold, distance=50)
                
                if len(peaks) >= 2:
                    # Get peak frequencies
                    peak_freqs = [formant_freqs[p] for p in peaks]
                    peak_heights = [formant_mags[p] for p in peaks]
                    
                    # Sort by height (strongest first)
                    sorted_pairs = sorted(zip(peak_freqs, peak_heights), key=lambda x: x[1], reverse=True)
                    peak_freqs_sorted = [pair[0] for pair in sorted_pairs]
                    
                    # Try to identify F1 and F2 based on typical ranges
                    f1_candidates = [f for f in peak_freqs_sorted if 200 <= f <= 1200]
                    f2_candidates = [f for f in peak_freqs_sorted if 800 <= f <= 3000]
                    
                    features['f1_approx'] = float(f1_candidates[0]) if f1_candidates else 0.0
                    features['f2_approx'] = float(f2_candidates[0]) if f2_candidates else 0.0
                    
                    # Ensure F2 > F1 (basic sanity check)
                    if features['f1_approx'] > 0 and features['f2_approx'] > 0:
                        if features['f2_approx'] <= features['f1_approx']:
                            if len(f2_candidates) > 1:
                                features['f2_approx'] = float(f2_candidates[1])
                            else:
                                features['f2_approx'] = features['f1_approx'] * 1.5  # Rough estimate
                else:
                    features['f1_approx'] = 0.0
                    features['f2_approx'] = 0.0
            else:
                features['f1_approx'] = 0.0
                features['f2_approx'] = 0.0
                
        except Exception as e:
            logger.warning(f"Formant extraction failed: {e}")
            features['f1_approx'] = 0.0
            features['f2_approx'] = 0.0
        
        # 5. Spectral rolloff using manual implementation
        try:
            frame_length = 2048
            hop_length = 512
            rolloffs = []
            
            for i in range(0, len(y) - frame_length + 1, hop_length):
                frame = y[i:i + frame_length]
                
                # Apply window
                window = ManualDSPCore.hamming_window(frame_length)
                windowed_frame = ManualDSPCore.apply_window(frame.tolist(), window)
                
                # Calculate DFT using course method
                X_real, X_imag = ManualDSPCore.dft_real(windowed_frame)
                
                # Calculate spectral rolloff using course approach
                rolloff = ManualDSPCore.spectral_rolloff_from_dft(X_real, X_imag, sr, 0.85)
                rolloffs.append(rolloff)
            
            if rolloffs:
                features['spectral_rolloff_mean'] = float(sum(rolloffs) / len(rolloffs))
            else:
                features['spectral_rolloff_mean'] = 0.0
                
        except Exception as e:
            logger.warning(f"Spectral rolloff extraction failed: {e}")
            features['spectral_rolloff_mean'] = 0.0
        
        # 6. Zero crossing rate using manual implementation
        try:
            features['zcr_mean'] = float(ManualDSPCore.zero_crossing_rate(y.tolist()))
        except Exception as e:
            logger.warning(f"ZCR extraction failed: {e}")
            features['zcr_mean'] = 0.0
        
        # 7. Harmonic-to-noise ratio approximation using manual method
        try:
            # Simple harmonic extraction using autocorrelation-based pitch detection
            frame_length = 2048
            hop_length = 512
            harmonic_energies = []
            noise_energies = []
            
            for i in range(0, len(y) - frame_length + 1, hop_length):
                frame = y[i:i + frame_length]
                
                # Estimate fundamental period using autocorrelation
                autocorr = ManualDSPCore.autocorrelation(frame.tolist(), frame_length // 2)
                
                # Find the period (skip first few lags to avoid zero-lag peak)
                min_period = int(sr / 500)  # Max F0 = 500Hz
                max_period = int(sr / 50)   # Min F0 = 50Hz
                
                if max_period < len(autocorr):
                    search_range = autocorr[min_period:min(max_period, len(autocorr))]
                    if search_range:
                        period = search_range.index(max(search_range)) + min_period
                        
                        # Extract harmonic component (simplified)
                        harmonic_signal = [0.0] * len(frame)
                        for j in range(len(frame)):
                            if j + period < len(frame):
                                harmonic_signal[j] = (frame[j] + frame[j + period]) / 2
                        
                        # Calculate energies
                        harmonic_energy = sum(h**2 for h in harmonic_signal)
                        total_energy = sum(f**2 for f in frame)
                        noise_energy = total_energy - harmonic_energy
                        
                        if noise_energy > 0:
                            harmonic_energies.append(harmonic_energy)
                            noise_energies.append(noise_energy)
            
            if harmonic_energies and noise_energies:
                avg_harmonic = sum(harmonic_energies) / len(harmonic_energies)
                avg_noise = sum(noise_energies) / len(noise_energies)
                
                if avg_noise > 0:
                    features['hnr_approx'] = float(10 * math.log10(avg_harmonic / avg_noise))
                else:
                    features['hnr_approx'] = 20.0  # High HNR if no noise
            else:
                features['hnr_approx'] = 10.0  # Default moderate value
                
        except Exception as e:
            logger.warning(f"HNR extraction failed: {e}")
            features['hnr_approx'] = 10.0
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

def classify_gender(features: Dict[str, float]) -> Dict[str, Any]:
    """Classify gender based on extracted DSP features - comprehensive multi-feature approach"""
    
    # Initialize feature scores (0 = neutral, positive = male, negative = female)
    feature_scores = []
    feature_weights = []
    feature_details = {}
    
    # 1. Fundamental Frequency (F0) - Primary indicator (weight: 0.4)
    f0_mean = features.get('f0_mean', 0)
    f0_score = 0
    f0_weight = 0.4
    
    if f0_mean > 50:  # Valid F0
        if f0_mean <= 120:
            f0_score = 0.9  # Strong male
        elif f0_mean <= 150:
            f0_score = 0.6  # Male
        elif f0_mean <= 170:
            f0_score = 0.3  # Lean male
        elif f0_mean <= 190:
            f0_score = 0.1  # Slight male
        elif f0_mean <= 210:
            f0_score = -0.1  # Slight female
        elif f0_mean <= 240:
            f0_score = -0.6  # Female
        else:
            f0_score = -0.9  # Strong female
        
        feature_scores.append(f0_score)
        feature_weights.append(f0_weight)
        feature_details['f0'] = {'value': f0_mean, 'score': f0_score, 'weight': f0_weight}
    
    # 2. Spectral Centroid - Voice brightness (weight: 0.25)
    spectral_centroid = features.get('spectral_centroid_mean', 0)
    if spectral_centroid > 1000:  # Valid spectral data
        sc_score = 0
        sc_weight = 0.25
        
        if spectral_centroid < 1600:
            sc_score = 0.6  # Dark/low = male
        elif spectral_centroid < 2000:
            sc_score = 0.3  # Somewhat dark = lean male
        elif spectral_centroid < 2400:
            sc_score = 0.0  # Neutral
        elif spectral_centroid < 2800:
            sc_score = -0.3  # Bright = lean female
        else:
            sc_score = -0.6  # Very bright = female
        
        feature_scores.append(sc_score)
        feature_weights.append(sc_weight)
        feature_details['spectral_centroid'] = {'value': spectral_centroid, 'score': sc_score, 'weight': sc_weight}
    
    # 3. Formant Analysis - Vocal tract characteristics (weight: 0.2)
    f1 = features.get('f1_approx', 0)
    f2 = features.get('f2_approx', 0)
    
    if f1 > 200 and f2 > 800:  # Valid formants
        formant_score = 0
        formant_weight = 0.2
        
        # Male typical: F1 ~300-600Hz, F2 ~900-1300Hz
        # Female typical: F1 ~400-800Hz, F2 ~1300-2100Hz
        
        if f1 < 450 and f2 < 1200:  # Low formants = male
            formant_score = 0.7
        elif f1 < 550 and f2 < 1400:  # Somewhat low = lean male
            formant_score = 0.4
        elif f1 > 650 and f2 > 1600:  # High formants = female
            formant_score = -0.7
        elif f1 > 550 and f2 > 1400:  # Somewhat high = lean female
            formant_score = -0.4
        # else neutral (0)
        
        feature_scores.append(formant_score)
        feature_weights.append(formant_weight)
        feature_details['formants'] = {'f1': f1, 'f2': f2, 'score': formant_score, 'weight': formant_weight}
    
    # 4. MFCC Analysis - Vocal tract shape (weight: 0.1)
    mfcc_1_mean = features.get('mfcc_1_mean', None)
    mfcc_2_mean = features.get('mfcc_2_mean', None)
    
    if mfcc_1_mean is not None and mfcc_2_mean is not None:
        mfcc_score = 0
        mfcc_weight = 0.1
        
        # MFCC coefficients tend to be different for male/female
        # This is a simplified heuristic based on typical patterns
        if mfcc_1_mean < -20:  # Lower MFCC1 often indicates male
            mfcc_score += 0.3
        elif mfcc_1_mean > -10:  # Higher MFCC1 often indicates female
            mfcc_score -= 0.3
        
        if mfcc_2_mean > 10:  # Higher MFCC2 variation
            mfcc_score -= 0.2  # Often female
        elif mfcc_2_mean < 0:
            mfcc_score += 0.2  # Often male
        
        feature_scores.append(mfcc_score)
        feature_weights.append(mfcc_weight)
        feature_details['mfcc'] = {'mfcc1': mfcc_1_mean, 'mfcc2': mfcc_2_mean, 'score': mfcc_score, 'weight': mfcc_weight}
    
    # 5. Harmonic-to-Noise Ratio - Voice quality (weight: 0.05)
    hnr = features.get('hnr_approx', None)
    if hnr is not None:
        hnr_score = 0
        hnr_weight = 0.05
        
        # Males often have slightly lower HNR due to vocal fold differences
        if hnr < 10:
            hnr_score = 0.3  # Lower HNR = lean male
        elif hnr > 18:
            hnr_score = -0.3  # Higher HNR = lean female
        
        feature_scores.append(hnr_score)
        feature_weights.append(hnr_weight)
        feature_details['hnr'] = {'value': hnr, 'score': hnr_score, 'weight': hnr_weight}
    
    # Calculate weighted average score
    if len(feature_scores) == 0:
        # No valid features found
        male_prob = 0.5
        female_prob = 0.5
        weighted_score = 0
        logger.warning("No valid features found for gender classification")
    else:
        # Normalize weights to sum to 1
        total_weight = sum(feature_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in feature_weights]
            weighted_score = sum(score * weight for score, weight in zip(feature_scores, normalized_weights))
        else:
            weighted_score = 0
        
        # Convert weighted score to probabilities using sigmoid-like function
        # Score ranges from -1 (strong female) to +1 (strong male)
        sigmoid_factor = 4  # Controls steepness of probability curve
        male_prob = 1 / (1 + np.exp(-sigmoid_factor * weighted_score))
        female_prob = 1.0 - male_prob
        
        # Ensure reasonable bounds
        male_prob = max(0.05, min(0.95, male_prob))
        female_prob = 1.0 - male_prob
    
    # Determine final prediction
    if abs(male_prob - female_prob) < 0.1:  # Very close
        predicted_gender = "unknown"
        confidence = 0.5
    elif male_prob > female_prob:
        predicted_gender = "male"
        confidence = male_prob
    else:
        predicted_gender = "female"
        confidence = female_prob
    
    # Enhanced logging
    logger.info(f"=== MULTI-FEATURE GENDER CLASSIFICATION ===")
    for feature_name, details in feature_details.items():
        logger.info(f"{feature_name}: {details}")
    logger.info(f"Final weighted score: {weighted_score:.3f}")
    logger.info(f"Male probability: {male_prob:.3f}, Female probability: {female_prob:.3f}")
    logger.info(f"FINAL DECISION - Gender: {predicted_gender}, Confidence: {confidence:.3f}")
    logger.info("=== END CLASSIFICATION ===")
    
    return {
        "gender": predicted_gender,
        "confidence": float(confidence),
        "probabilities": {
            "male": float(male_prob),
            "female": float(female_prob)
        },
        "features_used": {
            "f0_mean": f0_mean,
            "spectral_centroid": spectral_centroid,
            "f1_approx": f1,
            "f2_approx": f2,
            "mfcc_1_mean": mfcc_1_mean,
            "mfcc_2_mean": mfcc_2_mean,
            "hnr_approx": hnr,
        },
        "feature_analysis": feature_details,
        "classification_method": "multi-feature_weighted_scoring"
    }

app = FastAPI(
    title="Gender Classification Service",
    description="A FastAPI service for gender classification using DSP techniques (FFT, MFCC, Formants)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Gender Classification Service",
        "method": "DSP-based (F0, MFCC, Formants, Spectral Analysis)"
    }

@app.post("/predict")
async def predict_gender(audio: UploadFile = File(...)):
    """Predict gender from uploaded audio file using DSP techniques"""
    try:
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Extract DSP features
            logger.info(f"Extracting features from: {audio.filename}")
            features = extract_audio_features(temp_path)
            
            # Comprehensive debug logging
            logger.info("=== FEATURE EXTRACTION DEBUG ===")
            for key, value in features.items():
                logger.info(f"{key}: {value}")
            logger.info("=== END FEATURES ===")
            
            # Classify gender
            result = classify_gender(features)
            
            # Debug classification process
            logger.info(f"=== CLASSIFICATION DEBUG ===")
            logger.info(f"Final result: {result}")
            logger.info("=== END CLASSIFICATION ===")
            
            logger.info(f"Gender classified as: {result['gender']} (confidence: {result['confidence']:.2f})")
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "gender": "unknown",
                "confidence": 0.0
            }
        )

@app.get("/info")
async def service_info():
    """Get service information"""
    return {
        "service": "Gender Classification Service",
        "version": "1.0.0",
        "method": "DSP-based classification",
        "techniques": [
            "Fundamental Frequency (F0) analysis",
            "Spectral Centroid",
            "MFCC (Mel-frequency cepstral coefficients)",
            "Formant estimation via LPC",
            "Spectral Rolloff",
            "Harmonic-to-Noise Ratio",
            "Zero Crossing Rate"
        ],
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Gender classification (multipart/form-data with 'audio' file)",
            "/info": "GET - Service information",
            "/docs": "GET - API documentation (Swagger UI)"
        }
    }

if __name__ == '__main__':
    print("🚀 Starting Gender Classification Service...")
    print("📡 Server will be available at: http://localhost:5001")
    print("📖 API documentation: http://localhost:5001/docs")
    print("🔗 Health check: http://localhost:5001/health")
    
    public_url = ngrok.connect(5001)  
    print("🔗 Public URL:", public_url)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        log_level="info"
    )