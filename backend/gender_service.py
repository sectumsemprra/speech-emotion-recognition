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
    """Classify gender based on extracted DSP features - simplified and robust"""
    
    # Start with baseline probabilities
    male_prob = 0.5
    female_prob = 0.5
    
    # F0 is the most reliable - weight it heavily
    f0_mean = features.get('f0_mean', 0)
    
    logger.info(f"CLASSIFICATION DEBUG - F0 mean: {f0_mean}Hz")
    
    if f0_mean > 50:  # Valid F0
        if f0_mean <= 120:  # Very deep male voice
            male_prob = 0.95
            female_prob = 0.05
        elif f0_mean <= 150:  # Typical male
            male_prob = 0.85
            female_prob = 0.15
        elif f0_mean <= 170:  # Low male or very deep female
            male_prob = 0.70
            female_prob = 0.30
        elif f0_mean <= 190:  # Boundary region - lean male
            male_prob = 0.60
            female_prob = 0.40
        elif f0_mean <= 210:  # Boundary region - lean female
            male_prob = 0.40
            female_prob = 0.60
        elif f0_mean <= 240:  # Typical female
            male_prob = 0.15
            female_prob = 0.85
        else:  # High female voice
            male_prob = 0.05
            female_prob = 0.95
            
        logger.info(f"After F0 analysis - Male: {male_prob:.3f}, Female: {female_prob:.3f}")
    else:
        logger.info("No valid F0 found - using secondary features")
        
        # Fallback to spectral features if F0 fails
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        
        if spectral_centroid > 1000:  # Valid spectral data
            if spectral_centroid < 1800:  # Dark
                male_prob = 0.75
                female_prob = 0.25
            elif spectral_centroid > 2800:  # Bright
                male_prob = 0.25
                female_prob = 0.75
            # else keep 50/50
            
            logger.info(f"Using spectral centroid {spectral_centroid:.0f}Hz - Male: {male_prob:.3f}, Female: {female_prob:.3f}")
    
    # Minor adjustments based on other features (don't override F0 too much)
    adjustment_factor = 0.1  # Small adjustments only
    
    # Formant check
    f1 = features.get('f1_approx', 0)
    f2 = features.get('f2_approx', 0)
    
    if f1 > 200 and f2 > 800:  # Valid formants
        if f1 < 400 and f2 < 1100:  # Low formants = male
            male_prob = min(0.95, male_prob + adjustment_factor)
            female_prob = 1.0 - male_prob
        elif f1 > 700 and f2 > 1600:  # High formants = female
            female_prob = min(0.95, female_prob + adjustment_factor)
            male_prob = 1.0 - female_prob
    
    # Determine final prediction
    if abs(male_prob - female_prob) < 0.05:  # Very close
        predicted_gender = "unknown"
        confidence = 0.5
    elif male_prob > female_prob:
        predicted_gender = "male"
        confidence = male_prob
    else:
        predicted_gender = "female"
        confidence = female_prob
    
    logger.info(f"FINAL DECISION - Gender: {predicted_gender}, Confidence: {confidence:.3f}")
    
    return {
        "gender": predicted_gender,
        "confidence": float(confidence),
        "probabilities": {
            "male": float(male_prob),
            "female": float(female_prob)
        },
        "features_used": {
            "f0_mean": f0_mean,
            "spectral_centroid": features.get('spectral_centroid_mean', 0),
            "f1_approx": f1,
            "f2_approx": f2
        }
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