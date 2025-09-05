#!/usr/bin/env python3
"""
Gender Classification FastAPI Service
A FastAPI service for gender classification using DSP techniques
"""

import os
import tempfile
import logging
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
    """Extract DSP features from audio for gender classification"""
    try:
        import librosa
        import scipy.signal
        from scipy.fft import fft
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Remove silence
        y = librosa.effects.trim(y, top_db=20)[0]
        
        if len(y) == 0:
            raise ValueError("Audio file contains no sound")
        
        features = {}
        
        # 1. Fundamental Frequency (F0) - Key gender indicator
        # Try multiple F0 detection methods
        try:
            # Method 1: YIN algorithm
            f0_yin = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            f0_clean = f0_yin[f0_yin > 0]
            
            if len(f0_clean) > 5:  # Need at least some frames
                features['f0_mean'] = float(np.mean(f0_clean))
                features['f0_std'] = float(np.std(f0_clean))
                features['f0_median'] = float(np.median(f0_clean))
            else:
                # Method 2: Try piptrack as fallback
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1, fmin=50, fmax=500)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if len(pitch_values) > 5:
                    features['f0_mean'] = float(np.mean(pitch_values))
                    features['f0_std'] = float(np.std(pitch_values))
                    features['f0_median'] = float(np.median(pitch_values))
                else:
                    # Last resort: estimate from autocorrelation
                    from scipy.signal import correlate
                    
                    # Use middle section of audio
                    mid_start = len(y) // 4
                    mid_end = 3 * len(y) // 4
                    y_mid = y[mid_start:mid_end]
                    
                    if len(y_mid) > 1000:
                        autocorr = correlate(y_mid, y_mid, mode='full')
                        autocorr = autocorr[autocorr.size // 2:]
                        
                        # Find peak in expected F0 range
                        min_period = int(sr / 500)  # Max F0 = 500Hz
                        max_period = int(sr / 50)   # Min F0 = 50Hz
                        
                        if max_period < len(autocorr):
                            search_range = autocorr[min_period:max_period]
                            if len(search_range) > 0:
                                peak_idx = np.argmax(search_range) + min_period
                                estimated_f0 = sr / peak_idx
                                features['f0_mean'] = float(estimated_f0)
                                features['f0_std'] = 0.0
                                features['f0_median'] = float(estimated_f0)
                            else:
                                features['f0_mean'] = 0.0
                                features['f0_std'] = 0.0
                                features['f0_median'] = 0.0
                        else:
                            features['f0_mean'] = 0.0
                            features['f0_std'] = 0.0
                            features['f0_median'] = 0.0
                    else:
                        features['f0_mean'] = 0.0
                        features['f0_std'] = 0.0
                        features['f0_median'] = 0.0
        except:
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_median'] = 0.0
        
        # 2. Spectral Centroid - Brightness of voice
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # 3. MFCCs - Vocal tract characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(min(5, mfccs.shape[0])):  # Use first 5 MFCCs
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        # 4. Simplified formant estimation using spectral peaks
        try:
            # Get power spectrum
            fft = np.fft.rfft(y)
            magnitude = np.abs(fft)
            freq_bins = np.fft.rfftfreq(len(y), 1/sr)
            
            # Smooth the spectrum to find broad peaks
            from scipy.signal import savgol_filter, find_peaks
            smoothed_magnitude = savgol_filter(magnitude, 51, 3)
            
            # Find peaks in formant range (200-4000 Hz)
            formant_mask = (freq_bins >= 200) & (freq_bins <= 4000)
            formant_freqs = freq_bins[formant_mask]
            formant_mags = smoothed_magnitude[formant_mask]
            
            # Find prominent peaks
            peaks, properties = find_peaks(formant_mags, height=np.max(formant_mags) * 0.1, distance=50)
            
            if len(peaks) >= 2:
                # Sort by frequency (formants should be ordered)
                peak_freqs = formant_freqs[peaks]
                peak_heights = formant_mags[peaks]
                
                # Get the two strongest peaks in formant range
                sorted_indices = np.argsort(peak_heights)[::-1]  # Highest first
                
                # Try to identify F1 and F2 based on typical ranges
                f1_candidates = peak_freqs[(peak_freqs >= 200) & (peak_freqs <= 1200)]
                f2_candidates = peak_freqs[(peak_freqs >= 800) & (peak_freqs <= 3000)]
                
                features['f1_approx'] = float(f1_candidates[0]) if len(f1_candidates) > 0 else 0.0
                features['f2_approx'] = float(f2_candidates[0]) if len(f2_candidates) > 0 else 0.0
                
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
                
        except Exception as e:
            features['f1_approx'] = 0.0
            features['f2_approx'] = 0.0
        
        # 5. Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        
        # 6. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        
        # 7. Harmonic-to-noise ratio approximation
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_power = np.mean(harmonic**2)
        noise_power = np.mean((y - harmonic)**2)
        if noise_power > 0:
            features['hnr_approx'] = float(10 * np.log10(harmonic_power / noise_power))
        else:
            features['hnr_approx'] = 20.0  # High HNR if no noise
        
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