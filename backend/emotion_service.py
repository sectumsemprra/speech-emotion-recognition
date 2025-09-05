#!/usr/bin/env python3
"""
Local Emotion Detection FastAPI Service
A FastAPI service that runs locally for emotion detection with DSP preprocessing
"""

import os
import json
import tempfile
import logging
from typing import Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pyngrok import ngrok
# Import gender classification functions from gender_service.py
from gender_service import extract_audio_features, classify_gender as classify_gender_dsp
from services.dsp_preprocess import dsp_preprocess
from services.dsp_preprocess_library import dsp_preprocess as dsp_preprocess_library
from services.emotion_timeline import compute_emotion_timeline
from fastapi.staticfiles import StaticFiles
from services.manual_dsp_core import ManualDSPCore, frequency_bins

from services.dsp_filters import preprocess_audio_file, get_filter_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NGROK_AUTH = os.getenv("NGROK_AUTH", "31y3EqKzzZqEmFcksJJiO0jFShJ_76ywiqbQqegsSHLULmtL")

def extract_emotion_features(audio_path: str, manual: bool = False) -> Dict[str, float]:
    """Extract DSP features for emotion classification using course-based methods or FFT"""
    try:
        import librosa
        import math
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Manual silence trimming
        def trim_silence(signal, top_db=20):
            frame_length = 2048
            hop_length = 512
            
            energy = []
            for i in range(0, len(signal) - frame_length + 1, hop_length):
                frame = signal[i:i + frame_length]
                frame_energy = sum(x**2 for x in frame)
                energy.append(frame_energy)
            
            if not energy:
                return signal
            
            max_energy = max(energy)
            if max_energy == 0:
                return signal
            
            energy_db = [10 * math.log10(e / max_energy) if e > 0 else -100 for e in energy]
            threshold = max(energy_db) - top_db
            
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
            
            start_sample = start_idx * hop_length
            end_sample = min(len(signal), (end_idx + 1) * hop_length + frame_length)
            
            return signal[start_sample:end_sample]
        
        y = trim_silence(y)
        
        if len(y) == 0:
            raise ValueError("Audio file contains no sound")
        
        features = {}
        
        # 1. Energy-based features (emotional intensity indicators)
        frame_length = 2048
        hop_length = 512
        
        # Frame-based energy analysis
        frame_energies = []
        for i in range(0, len(y) - frame_length + 1, hop_length):
            frame = y[i:i + frame_length]
            energy = sum(x**2 for x in frame)
            frame_energies.append(energy)
        
        if frame_energies:
            features['energy_mean'] = float(sum(frame_energies) / len(frame_energies))
            energy_mean = features['energy_mean']
            energy_variance = sum((e - energy_mean) ** 2 for e in frame_energies) / len(frame_energies)
            features['energy_std'] = float(math.sqrt(energy_variance))
            features['energy_max'] = float(max(frame_energies))
            features['energy_min'] = float(min(frame_energies))
        else:
            features['energy_mean'] = 0.0
            features['energy_std'] = 0.0
            features['energy_max'] = 0.0
            features['energy_min'] = 0.0
        
        # 2. F0 Analysis (pitch-based emotion indicators)
        f0_values = []
        for i in range(0, len(y) - frame_length + 1, hop_length):
            frame = y[i:i + frame_length]
            
            # Apply window
            window = ManualDSPCore.hamming_window(frame_length)
            windowed_frame = ManualDSPCore.apply_window(list(frame), window)
            
            # Autocorrelation for F0
            autocorr = ManualDSPCore.autocorrelation_fast(windowed_frame, frame_length // 2, manual=manual)
            
            # Find F0 in speech range
            min_period = int(sr / 400)  # Max F0 = 400Hz
            max_period = int(sr / 80)   # Min F0 = 80Hz
            
            if max_period < len(autocorr):
                search_range = autocorr[min_period:min(max_period, len(autocorr))]
                if search_range:
                    max_idx = search_range.index(max(search_range))
                    period = max_idx + min_period
                    
                    if autocorr[period] > 0.3 * autocorr[0]:
                        f0 = sr / period
                        if 80 <= f0 <= 400:
                            f0_values.append(f0)
        
        if len(f0_values) > 3:
            features['f0_mean'] = float(sum(f0_values) / len(f0_values))
            f0_mean = features['f0_mean']
            f0_variance = sum((f0 - f0_mean) ** 2 for f0 in f0_values) / len(f0_values)
            features['f0_std'] = float(math.sqrt(f0_variance))
            features['f0_range'] = float(max(f0_values) - min(f0_values))
            
            # F0 contour slope (emotional prosody)
            if len(f0_values) > 1:
                f0_slope = (f0_values[-1] - f0_values[0]) / len(f0_values)
                features['f0_slope'] = float(f0_slope)
            else:
                features['f0_slope'] = 0.0
        else:
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_range'] = 0.0
            features['f0_slope'] = 0.0
        
        # 3. Spectral features (emotional timbre)
        spectral_centroids = []
        spectral_rolloffs = []
        spectral_spreads = []
        
        for i in range(0, len(y) - frame_length + 1, hop_length):
            frame = y[i:i + frame_length]
            
            # Apply window
            window = ManualDSPCore.hamming_window(frame_length)
            windowed_frame = ManualDSPCore.apply_window(list(frame), window)
            
            # Spectral centroid with fast/manual option
            centroid = ManualDSPCore.spectral_centroid_fast(windowed_frame, sr, manual=manual)
            if centroid > 0:
                spectral_centroids.append(centroid)
            
            # Spectral rolloff with fast/manual option
            rolloff = ManualDSPCore.spectral_rolloff_fast(windowed_frame, sr, 0.85, manual=manual)
            spectral_rolloffs.append(rolloff)
            
            # Spectral spread (bandwidth around centroid)
            if centroid > 0:
                X_real, X_imag = ManualDSPCore.dft_fast(windowed_frame, manual=manual)
                magnitude = ManualDSPCore.magnitude_spectrum(X_real, X_imag)
                freqs = frequency_bins(frame_length, sr)
                
                if len(magnitude) == len(freqs):
                    power_spectrum = [mag ** 2 for mag in magnitude[:len(freqs)]]
                    total_power = sum(power_spectrum)
                    
                    if total_power > 0:
                        spread_sum = sum((f - centroid) ** 2 * p for f, p in zip(freqs, power_spectrum))
                        spread = math.sqrt(spread_sum / total_power)
                        spectral_spreads.append(spread)
        
        # Calculate spectral statistics
        if spectral_centroids:
            features['spectral_centroid_mean'] = float(sum(spectral_centroids) / len(spectral_centroids))
            sc_mean = features['spectral_centroid_mean']
            sc_variance = sum((sc - sc_mean) ** 2 for sc in spectral_centroids) / len(spectral_centroids)
            features['spectral_centroid_std'] = float(math.sqrt(sc_variance))
        else:
            features['spectral_centroid_mean'] = 0.0
            features['spectral_centroid_std'] = 0.0
        
        if spectral_rolloffs:
            features['spectral_rolloff_mean'] = float(sum(spectral_rolloffs) / len(spectral_rolloffs))
        else:
            features['spectral_rolloff_mean'] = 0.0
        
        if spectral_spreads:
            features['spectral_spread_mean'] = float(sum(spectral_spreads) / len(spectral_spreads))
        else:
            features['spectral_spread_mean'] = 0.0
        
        # 4. Zero Crossing Rate (speech/emotion articulation)
        features['zcr'] = float(ManualDSPCore.zero_crossing_rate(list(y)))
        
        # 5. Temporal dynamics (rhythm and timing)
        features['duration'] = float(len(y) / sr)
        
        # 6. Spectral flux (rate of spectral change - emotional dynamics)
        spectral_flux = []
        prev_magnitude = None
        
        for i in range(0, len(y) - frame_length + 1, hop_length):
            frame = y[i:i + frame_length]
            window = ManualDSPCore.hamming_window(frame_length)
            windowed_frame = ManualDSPCore.apply_window(list(frame), window)
            
            X_real, X_imag = ManualDSPCore.dft_fast(windowed_frame, manual=manual)
            magnitude = ManualDSPCore.magnitude_spectrum(X_real, X_imag)
            
            if prev_magnitude is not None:
                # Calculate spectral flux
                flux = sum(max(0, mag - prev_mag) for mag, prev_mag in zip(magnitude, prev_magnitude))
                spectral_flux.append(flux)
            
            prev_magnitude = magnitude
        
        if spectral_flux:
            features['spectral_flux_mean'] = float(sum(spectral_flux) / len(spectral_flux))
        else:
            features['spectral_flux_mean'] = 0.0
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting emotion features: {e}")
        raise

def classify_emotion_dsp(features: Dict[str, float]) -> Dict[str, Any]:
    """Classify emotion based on DSP features using acoustic-phonetic rules"""
    
    # Initialize emotion scores
    emotion_scores = {
        'happy': 0.0,
        'sad': 0.0,
        'angry': 0.0,
        'fear': 0.0,
        'surprise': 0.0,
        'disgust': 0.0,
        'neutral': 0.0
    }
    
    # Feature weights for different emotions
    feature_weights = {
        'energy': 0.25,
        'f0': 0.30,
        'spectral': 0.25,
        'temporal': 0.20
    }
    
    # 1. Energy-based emotion indicators
    energy_mean = features.get('energy_mean', 0)
    energy_std = features.get('energy_std', 0)
    energy_max = features.get('energy_max', 0)
    
    # Normalize energy (typical speech energy ranges from 0.1 to 100)
    if energy_mean > 0.1:  # Threshold for significant energy
        # Use logarithmic scaling for energy since it varies widely
        import math
        energy_factor = min(math.log10(energy_mean + 1) / 2.0, 1.0)  # Log scale normalization
        
        # Energy level classification
        if energy_mean > 10.0:  # Very high energy
            if energy_std > energy_mean * 0.3:  # High variation
                emotion_scores['angry'] += 0.4 * energy_factor
                emotion_scores['surprise'] += 0.3 * energy_factor
                emotion_scores['fear'] += 0.2 * energy_factor
            else:  # Stable high energy
                emotion_scores['happy'] += 0.5 * energy_factor
                emotion_scores['surprise'] += 0.2 * energy_factor
        elif energy_mean > 1.0:  # Medium energy
            emotion_scores['happy'] += 0.3 * energy_factor
            emotion_scores['neutral'] += 0.4 * energy_factor
            if energy_std > energy_mean * 0.4:
                emotion_scores['angry'] += 0.2 * energy_factor
        else:  # Low energy
            emotion_scores['sad'] += 0.5 * (1 - energy_factor)
            emotion_scores['neutral'] += 0.4 * (1 - energy_factor)
    
    # 2. F0-based emotion indicators
    f0_mean = features.get('f0_mean', 0)
    f0_std = features.get('f0_std', 0)
    f0_range = features.get('f0_range', 0)
    f0_slope = features.get('f0_slope', 0)
    
    if f0_mean > 80:  # Valid F0
        # F0 ranges: Male 85-180Hz, Female 165-265Hz
        if f0_mean > 220:  # Very high F0
            emotion_scores['surprise'] += 0.4
            emotion_scores['fear'] += 0.3
            emotion_scores['happy'] += 0.3
        elif f0_mean > 180:  # High F0  
            emotion_scores['happy'] += 0.4
            emotion_scores['surprise'] += 0.2
            emotion_scores['neutral'] += 0.2
        elif f0_mean > 120:  # Medium F0
            emotion_scores['neutral'] += 0.4
            emotion_scores['happy'] += 0.2
        else:  # Low F0 (80-120)
            emotion_scores['sad'] += 0.3
            emotion_scores['angry'] += 0.2
            emotion_scores['neutral'] += 0.2
        
        # F0 variation indicates emotional activation
        if f0_std > 25:  # High variation
            emotion_scores['angry'] += 0.2
            emotion_scores['surprise'] += 0.2
            emotion_scores['fear'] += 0.1
            emotion_scores['neutral'] -= 0.1
        elif f0_std < 10:  # Very stable
            emotion_scores['neutral'] += 0.2
            emotion_scores['sad'] += 0.1
        
        # F0 contour
        if f0_slope > 5:  # Rising pitch = surprise/happy
            emotion_scores['surprise'] += 0.3
            emotion_scores['happy'] += 0.2
        elif f0_slope < -5:  # Falling pitch = sad
            emotion_scores['sad'] += 0.3
    
    # 3. Spectral characteristics
    spectral_centroid = features.get('spectral_centroid_mean', 0)
    spectral_spread = features.get('spectral_spread_mean', 0)
    spectral_flux = features.get('spectral_flux_mean', 0)
    
    if spectral_centroid > 1000:
        # Spectral centroid ranges: 1000-4000Hz typical for speech
        if spectral_centroid > 3000:  # Very bright
            emotion_scores['surprise'] += 0.3
            emotion_scores['fear'] += 0.2
            emotion_scores['happy'] += 0.2
        elif spectral_centroid > 2200:  # Bright
            emotion_scores['happy'] += 0.3
            emotion_scores['neutral'] += 0.2
        elif spectral_centroid > 1600:  # Medium
            emotion_scores['neutral'] += 0.3
            emotion_scores['happy'] += 0.1
        else:  # Dark (1000-1600)
            emotion_scores['sad'] += 0.2
            emotion_scores['angry'] += 0.1
        
        # Spectral spread indicates voice quality variation
        if spectral_spread > 800:  # High spread
            emotion_scores['angry'] += 0.1
            emotion_scores['fear'] += 0.1
        elif spectral_spread < 400:  # Low spread (focused spectrum)
            emotion_scores['neutral'] += 0.1
        
        # High spectral flux = dynamic emotions
        if spectral_flux > 0.2:  # Increased threshold
            emotion_scores['angry'] += 0.1  # Reduced weight
            emotion_scores['surprise'] += 0.1
            emotion_scores['fear'] += 0.1
    
    # 4. Temporal features
    zcr = features.get('zcr', 0)
    duration = features.get('duration', 0)
    
    # Zero crossing rate indicates voicing characteristics
    if zcr > 0.15:  # Very high ZCR = lots of unvoiced sounds
        emotion_scores['angry'] += 0.1  # Reduced weight
        emotion_scores['fear'] += 0.1
        emotion_scores['disgust'] += 0.1
    elif zcr < 0.05:  # Very low ZCR = mostly voiced
        emotion_scores['sad'] += 0.1
        emotion_scores['neutral'] += 0.1
    
    # Duration effects
    if duration > 0:
        if duration < 1.0:  # Short utterances = surprise
            emotion_scores['surprise'] += 0.2
        elif duration > 3.0:  # Long utterances = sad/neutral
            emotion_scores['sad'] += 0.1
            emotion_scores['neutral'] += 0.1
    
    # Add baseline scores to prevent extreme predictions
    emotion_scores['neutral'] = max(0.3, emotion_scores['neutral'])
    emotion_scores['happy'] = max(0.1, emotion_scores['happy'])
    emotion_scores['sad'] = max(0.1, emotion_scores['sad'])
    
    # Normalize scores using softmax-like approach for better distribution
    total_score = sum(emotion_scores.values())
    if total_score > 0:
        for emotion in emotion_scores:
            emotion_scores[emotion] = emotion_scores[emotion] / total_score
    
    # Apply smoothing to prevent overconfident predictions
    smoothing = 0.05
    for emotion in emotion_scores:
        emotion_scores[emotion] = (1 - smoothing) * emotion_scores[emotion] + smoothing / len(emotion_scores)
    
    # Find best emotion
    best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    predicted_emotion = best_emotion[0]
    confidence = best_emotion[1]
    
    # Create sorted emotion list
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    top_emotions = [{"emotion": emotion, "score": float(score)} for emotion, score in sorted_emotions]
    
    logger.info(f"=== DSP-BASED EMOTION CLASSIFICATION ===")
    logger.info(f"Features: Energy={energy_mean:.4f}, F0={f0_mean:.1f}Hz, Centroid={spectral_centroid:.1f}Hz, ZCR={zcr:.3f}")
    logger.info(f"Energy factor: {min(math.log10(energy_mean + 1) / 2.0, 1.0):.3f}")
    logger.info(f"Raw emotion scores: {emotion_scores}")
    logger.info(f"Predicted: {predicted_emotion} (confidence: {confidence:.3f})")
    logger.info("=== END EMOTION CLASSIFICATION ===")
    
    return {
        "emotion": predicted_emotion,
        "confidence": float(confidence),
        "emotion_scores": emotion_scores,
        "topEmotions": top_emotions,
        "features_used": {
            "energy_mean": energy_mean,
            "f0_mean": f0_mean,
            "spectral_centroid": spectral_centroid,
            "spectral_spread": spectral_spread,
            "zcr": zcr,
            "duration": duration
        },
        "classification_method": "dsp_acoustic_phonetic_rules"
    }

def detect_emotion_from_file(
    audio_file_path: str, 
    apply_filtering: bool = True,
    filter_type: str = 'bandpass',
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0,
    cutoff: Optional[float] = None,
    filter_order: int = 5,
    manual: bool = False,
    use_hardcoded_dsp: bool = True
) -> Dict[str, Any]:
    """Run DSP-based emotion detection on an audio file with optional preprocessing"""
    
    # Choose DSP preprocessing method based on parameter
    if use_hardcoded_dsp:
        dsp_report, processed_wav, artifacts = dsp_preprocess(
            audio_path=audio_file_path,
            fs_target=16000,
            apply_quantization_for_analysis=True,   # analysis plots only; model uses clean processed signal
            quant_bits=8,
            use_mu_law=True,
            preemph_alpha=0.97,
            agc_target_rms=0.1,
        )
    else:
        dsp_report, processed_wav, artifacts = dsp_preprocess_library(
            audio_path=audio_file_path,
            fs_target=16000,
            apply_quantization_for_analysis=True,   # analysis plots only; model uses clean processed signal
            quant_bits=8,
            use_mu_law=True,
            preemph_alpha=0.97,
            agc_target_rms=0.1,
        )

    timeline_report, heatmap_path = compute_emotion_timeline(model, processed_wav, frame_ms=400, hop_ms=200, fs_target=16000, plot=True)
    
    try:
        logger.info(f"Processing audio file: {processed_wav}")
        
        # Apply DSP preprocessing if requested
        processed_audio_path = processed_wav
        filter_applied = False
        filter_info = {}
        
        if apply_filtering and filter_type.lower() != 'none':
            try:
                logger.info("🔧 Applying DSP preprocessing...")
                processed_audio_path = preprocess_audio_file(
                    input_path=audio_file_path,
                    filter_type=filter_type,
                    low_cutoff=low_cutoff,
                    high_cutoff=high_cutoff,
                    cutoff=cutoff,
                    order=filter_order
                )
                filter_applied = True
                filter_info = get_filter_info(filter_type, low_cutoff, high_cutoff, cutoff, filter_order)
                logger.info(f"✅ DSP preprocessing completed: {filter_info}")
                
            except Exception as e:
                logger.warning(f"⚠️  DSP preprocessing failed, using original audio: {e}")
                processed_audio_path = audio_file_path
                filter_applied = False
        
        # Extract DSP features for emotion classification
        logger.info("🎭 Extracting emotion features using DSP methods...")
        features = extract_emotion_features(processed_audio_path, manual=manual)
        
        # Classify emotion using DSP-based rules
        logger.info("🎯 Classifying emotion using acoustic-phonetic rules...")
        emotion_result = classify_emotion_dsp(features)
        
        # Create artifact URLs
        artifact_urls = [f"/artifacts/{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" for p in artifacts]
        processed_url = f"/artifacts/{os.path.basename(os.path.dirname(processed_wav))}/{os.path.basename(processed_wav)}"

        # Clean up processed file if it's different from original
        if processed_audio_path != audio_file_path:
            try:
                os.unlink(processed_audio_path)
            except OSError:
                pass
        
        # Prepare result with DSP-based emotion detection
        result_dict = {
            "emotion": emotion_result["emotion"],
            "confidence": emotion_result["confidence"],
            "topEmotions": emotion_result["topEmotions"],
            "emotion_scores": emotion_result["emotion_scores"],
            "preprocessing": {
                "filter_applied": filter_applied,
                "filter_info": filter_info,
                "dsp_method": "hardcoded" if use_hardcoded_dsp else "library"
            },
            "dsp_report": dsp_report,
            "features_used": emotion_result["features_used"],
            "classification_method": emotion_result["classification_method"],
            "artifacts": {
                "processed_wav": processed_url,
                "plots": artifact_urls,
                "emotion_timeline": "DSP-based classification (no timeline)"
            }
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error during DSP-based emotion detection: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="DSP-Based Emotion Detection Service",
    description="A FastAPI service for speech emotion recognition using pure DSP techniques and acoustic-phonetic rules",
    version="2.0.0"
)

# Serve generated plots & processed audio
if not os.path.exists("artifacts"):
    os.makedirs("artifacts", exist_ok=True)
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DSP-Based Emotion Detection Service",
        "classification_method": "acoustic_phonetic_rules",
        "features": ["dsp_emotion_detection", "gender_classification", "dsp_preprocessing"],
        "dsp_techniques": ["DFT", "autocorrelation", "spectral_analysis", "convolution"]
    }

@app.post("/predict")
async def predict_emotion(
    audio: UploadFile = File(...),
    apply_filtering: bool = Query(True, description="Apply DSP preprocessing"),
    filter_type: str = Query('bandpass', description="Filter type: bandpass, lowpass, highpass, none"),
    low_cutoff: float = Query(300.0, description="Low cutoff frequency (Hz) for bandpass"),
    high_cutoff: float = Query(3400.0, description="High cutoff frequency (Hz) for bandpass"),
    cutoff: Optional[float] = Query(None, description="Single cutoff frequency (Hz) for lowpass/highpass"),
    filter_order: int = Query(5, description="Filter order (1-10)"),
    use_hardcoded_dsp: bool = Query(True, description="Use hardcoded DSP methods (True) or library functions (False)"),
    manual = False
    # manual: bool = Query(False, description="Use manual DSP implementations (slower but educational)")
):
    """Predict emotion from uploaded audio file with optional DSP preprocessing"""
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Validate filter parameters
        if filter_order < 1 or filter_order > 10:
            raise HTTPException(status_code=400, detail="Filter order must be between 1 and 10")
        
        if filter_type.lower() not in ['bandpass', 'lowpass', 'highpass', 'none']:
            raise HTTPException(status_code=400, detail="Invalid filter type")
        
        if filter_type.lower() == 'bandpass' and low_cutoff >= high_cutoff:
            raise HTTPException(status_code=400, detail="Low cutoff must be less than high cutoff for bandpass filter")
        
        # Check file type (optional validation)
        allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/webm']
        if audio.content_type and audio.content_type not in allowed_types:
            logger.warning(f"Unusual audio type: {audio.content_type}")
        
        # Save uploaded file temporarily with original extension
        file_ext = os.path.splitext(audio.filename)[1]
        if not file_ext:
            # If no extension, use content-type to determine appropriate extension
            if audio.content_type:
                if 'webm' in audio.content_type:
                    file_ext = '.webm'
                elif 'mp4' in audio.content_type:
                    file_ext = '.mp4'
                elif 'mpeg' in audio.content_type:
                    file_ext = '.mp3'
                else:
                    file_ext = '.wav'
            else:
                file_ext = '.wav'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Run emotion detection with DSP preprocessing
            result = detect_emotion_from_file(
                audio_file_path=temp_path,
                apply_filtering=apply_filtering,
                filter_type=filter_type,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                cutoff=cutoff,
                filter_order=filter_order,
                use_hardcoded_dsp=use_hardcoded_dsp,
                manual=manual
            )
            
            logger.info(f"Emotion detected: {result['emotion']} ({result['confidence']:.2f}) "
                       f"[Filter applied: {result['preprocessing']['filter_applied']}]")
            
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
                "emotion": "unknown",
                "confidence": 0.0,
                "topEmotions": [],
                "preprocessing": {
                    "filter_applied": False,
                    "filter_info": {},
                    "dsp_method": "hardcoded" if use_hardcoded_dsp else "library"
                }
            }
        )

@app.post("/classify-gender")
async def classify_gender_endpoint(
    audio: UploadFile = File(...),
    apply_filtering: bool = Query(True, description="Apply DSP preprocessing"),
    filter_type: str = Query('bandpass', description="Filter type: bandpass, lowpass, highpass, none"),
    low_cutoff: float = Query(300.0, description="Low cutoff frequency (Hz)"),
    high_cutoff: float = Query(3400.0, description="High cutoff frequency (Hz)"),
    filter_order: int = Query(5, description="Filter order"),
    manual = False
    # manual: bool = Query(False, description="Use manual DSP implementations (slower but educational)")
):
    """Classify gender from uploaded audio file using DSP features with optional preprocessing"""
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Check file type (optio, nal validation)
        allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/webm']
        if audio.content_type and audio.content_type not in allowed_types:
            logger.warning(f"Unusual audio type: {audio.content_type}")
        
        # Save uploaded file temporarily with original extension
        file_ext = os.path.splitext(audio.filename)[1]
        if not file_ext:
            # If no extension, use content-type to determine appropriate extension
            if audio.content_type:
                if 'webm' in audio.content_type:
                    file_ext = '.webm'
                elif 'mp4' in audio.content_type:
                    file_ext = '.mp4'
                elif 'mpeg' in audio.content_type:
                    file_ext = '.mp3'
                else:
                    file_ext = '.wav'
            else:
                file_ext = '.wav'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Apply DSP preprocessing if requested
            processed_audio_path = temp_path
            filter_applied = False
            filter_info = {}
            
            if apply_filtering and filter_type.lower() != 'none':
                try:
                    processed_audio_path = preprocess_audio_file(
                        input_path=temp_path,
                        filter_type=filter_type,
                        low_cutoff=low_cutoff,
                        high_cutoff=high_cutoff,
                        order=filter_order
                    )
                    filter_applied = True
                    filter_info = get_filter_info(filter_type, low_cutoff, high_cutoff, None, filter_order)
                    
                except Exception as e:
                    logger.warning(f"DSP preprocessing failed for gender classification: {e}")
                    processed_audio_path = temp_path
            
            # Run gender classification using gender_service.py functions
            logger.info(f"Extracting features from: {audio.filename}")
            features = extract_audio_features(processed_audio_path, manual=manual)
            
            # Comprehensive debug logging
            logger.info("=== FEATURE EXTRACTION DEBUG ===")
            for key, value in features.items():
                logger.info(f"{key}: {value}")
            logger.info("=== END FEATURES ===")
            
            # Classify gender
            result = classify_gender_dsp(features)
            
            # Add preprocessing info to result
            result['preprocessing'] = {
                "filter_applied": filter_applied,
                "filter_info": filter_info
            }
            
            # Clean up processed file if different from original
            if processed_audio_path != temp_path:
                try:
                    os.unlink(processed_audio_path)
                except OSError:
                    pass
            
            logger.info(f"Gender classified: {result['gender']} (confidence: {result.get('confidence', 0):.2f}) "
                       f"[Filter applied: {filter_applied}]")
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
        logger.error(f"Error in gender classification endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "gender": "unknown",
                "confidence": 0.0,
                "method": "error",
                "preprocessing": {
                    "filter_applied": False,
                    "filter_info": {}
                }
            }
        )

@app.get("/filter-presets")
async def get_filter_presets():
    """Get common filter presets for different use cases"""
    return {
        "presets": {
            "telephone": {
                "filter_type": "bandpass",
                "low_cutoff": 300.0,
                "high_cutoff": 3400.0,
                "description": "Telephone bandwidth (300-3400 Hz)"
            },
            "speech": {
                "filter_type": "bandpass", 
                "low_cutoff": 85.0,
                "high_cutoff": 8000.0,
                "description": "Human speech range (85-8000 Hz)"
            },
            "voice_fundamental": {
                "filter_type": "bandpass",
                "low_cutoff": 80.0,
                "high_cutoff": 1000.0, 
                "description": "Voice fundamental frequencies (80-1000 Hz)"
            },
            "noise_reduction": {
                "filter_type": "highpass",
                "cutoff": 100.0,
                "description": "Remove low-frequency noise (>100 Hz)"
            },
            "no_filter": {
                "filter_type": "none",
                "description": "No filtering applied"
            }
        }
    }

@app.get("/info")
async def service_info():
    """Get service information"""
    return {
        "service": "DSP-Based Emotion Detection & Gender Classification Service",
        "version": "2.0.0",
        "classification_methods": {
            "emotion": "Acoustic-phonetic rules using DSP features",
            "gender": "Multi-feature DSP classifier (F0, MFCC, Formants, Spectral Analysis)"
        },
        "dsp_techniques": {
            "transforms": ["DFT with sin/cos", "IDFT", "Autocorrelation"],
            "convolution": ["Linear convolution", "Circular convolution", "DFT-based convolution"],
            "spectral_analysis": ["Magnitude spectrum", "Power spectrum", "Phase spectrum", "Spectral centroid", "Spectral rolloff"],
            "feature_extraction": ["F0 via autocorrelation", "Energy analysis", "Zero crossing rate", "Spectral flux", "MFCC with mel filters"]
        },
        "emotion_features": {
            "energy": "Frame-based energy analysis for emotional intensity",
            "f0": "Pitch analysis for emotional prosody (autocorrelation-based)",
            "spectral": "Spectral centroid, rolloff, spread for timbre analysis",
            "temporal": "Duration, ZCR, spectral flux for dynamics"
        },
        "emotion_rules": {
            "happy": "High energy + high F0 + bright spectrum",
            "sad": "Low energy + low F0 + dark spectrum + falling pitch",
            "angry": "High energy variation + moderate F0 + high spectral flux",
            "surprise": "High F0 + rising pitch + short duration",
            "fear": "High F0 + high energy variation + bright spectrum",
            "neutral": "Moderate values across all features"
        },
        "dsp_filters": {
            "types": ["bandpass", "lowpass", "highpass", "none"],
            "default_bandpass": "300-3400 Hz (telephone bandwidth)",
            "filter_orders": "1-10 (default: 5)"
        },
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - DSP-based emotion detection",
            "/classify-gender": "POST - DSP-based gender classification", 
            "/filter-presets": "GET - Common filter presets",
            "/info": "GET - Service information",
            "/docs": "GET - API documentation (Swagger UI)"
        },
        "dsp_preprocessing": {
            "course_techniques": [
                "DFT/IDFT using sin/cos", "Linear/Circular convolution", "Autocorrelation for F0",
                "Spectral analysis", "Energy-based VAD", "Manual windowing", "MFCC with mel filters"
            ],
            "artifacts_served_at": "/artifacts"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize DSP-based services"""
    logger.info("🚀 Starting DSP-Based Emotion Detection & Gender Classification Service...")
    logger.info("🎭 DSP-based Emotion Detection is ready!")
    logger.info("👥 DSP-based Gender Classification is ready!")
    logger.info("🔧 Course-based DSP techniques loaded!")
    logger.info("📊 Available techniques: DFT/IDFT, Convolution, Autocorrelation, Spectral Analysis")

if __name__ == '__main__':
    print("🚀 Starting DSP-Based Emotion Detection & Gender Classification Service...")
    print("📡 Server will be available at: http://localhost:5000")
    print("📖 API documentation: http://localhost:5000/docs")
    print("🔗 Health check: http://localhost:5000/health")
    print("🎭 DSP-based emotion detection: POST /predict")
    print("👥 DSP-based gender classification: POST /classify-gender")
    print("🔧 Filter presets: GET /filter-presets")
    print("🎛️  DSP Techniques: DFT/IDFT, Convolution, Autocorrelation, Spectral Analysis")
    print("📚 Course Methods: Sin/Cos transforms, Manual feature extraction")
    ngrok.set_auth_token(NGROK_AUTH)

    public_url = ngrok.connect(5000)  
    print("🔗 Public URL:", public_url)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )