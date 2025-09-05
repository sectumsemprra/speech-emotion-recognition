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
from services.emotion_timeline import compute_emotion_timeline
from fastapi.staticfiles import StaticFiles

from services.dsp_filters import preprocess_audio_file, get_filter_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

NGROK_AUTH = os.getenv("NGROK_AUTH", "31y3EqKzzZqEmFcksJJiO0jFShJ_76ywiqbQqegsSHLULmtL")

def load_model():
    """Load the emotion detection model"""
    global model
    try:
        logger.info("Loading emotion2vec_plus_large model...")
        from funasr import AutoModel
        model = AutoModel(model="emotion2vec_plus_large")
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("Make sure to install dependencies: pip install funasr numpy")
        return False

def detect_emotion_from_file(
    audio_file_path: str, 
    apply_filtering: bool = True,
    filter_type: str = 'bandpass',
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0,
    cutoff: Optional[float] = None,
    filter_order: int = 5
) -> Dict[str, Any]:
    """Run emotion detection on an audio file with optional DSP preprocessing"""
    global model
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    dsp_report, processed_wav, artifacts = dsp_preprocess(
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
        
        # Run model inference on processed audio
        result = model.generate(processed_audio_path, granularity="utterance")
        data = result[0]
        
        # Extract emotions and scores
        emotions = None
        scores = None
        
        if 'predictions' in data and 'scores' in data:
            emotions = data['predictions']
            scores = data['scores']
        elif 'labels' in data and 'scores' in data:
            emotions = [e.split('/')[-1] for e in data['labels']]
            scores = data['scores']
        else:
            raise ValueError("No valid predictions or labels found")
        
        # Find best prediction
        best_idx = int(np.argmax(scores))
        best_emotion = emotions[best_idx]
        confidence = float(scores[best_idx])
        
        # Replace <unk> with neutral for better user experience
        if best_emotion == '<unk>':
            best_emotion = 'neutral'
        
        # Sort all emotions by score
        emotion_pairs = list(zip(emotions, scores))
        sorted_emotions = sorted(emotion_pairs, key=lambda x: x[1], reverse=True)
        # Replace <unk> with neutral in the top emotions list too
        top_emotions = [{"emotion": "neutral" if e == "<unk>" else e, "score": float(s)} for e, s in sorted_emotions]
        
        artifact_urls = [f"/artifacts/{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" for p in artifacts]
        processed_url = f"/artifacts/{os.path.basename(os.path.dirname(processed_wav))}/{os.path.basename(processed_wav)}"

        # Clean up processed file if it's different from original
        if processed_audio_path != audio_file_path:
            try:
                os.unlink(processed_audio_path)
            except OSError:
                pass
        
        # Prepare result
        result_dict = {
            "emotion": best_emotion,
            "confidence": confidence,
            "topEmotions": top_emotions,
            "preprocessing": {
                "filter_applied": filter_applied,
                "filter_info": filter_info
            },
            "dsp_report": dsp_report,
            "artifacts": {
                "processed_wav": processed_url,
                "plots": artifact_urls,
                "emotion_timeline": timeline_report
            }
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error during emotion detection: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Local Emotion Detection Service with DSP",
    description="A FastAPI service for speech emotion recognition using emotion2vec_plus_large model with DSP preprocessing",
    version="1.1.0"
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
        "service": "Local Emotion Detection Service with DSP",
        "model_loaded": model is not None,
        "features": ["emotion_detection", "gender_classification", "dsp_preprocessing"]
    }

@app.post("/predict")
async def predict_emotion(
    audio: UploadFile = File(...),
    apply_filtering: bool = Query(True, description="Apply DSP preprocessing"),
    filter_type: str = Query('bandpass', description="Filter type: bandpass, lowpass, highpass, none"),
    low_cutoff: float = Query(300.0, description="Low cutoff frequency (Hz) for bandpass"),
    high_cutoff: float = Query(3400.0, description="High cutoff frequency (Hz) for bandpass"),
    cutoff: Optional[float] = Query(None, description="Single cutoff frequency (Hz) for lowpass/highpass"),
    filter_order: int = Query(5, description="Filter order (1-10)")
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
                filter_order=filter_order
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
                    "filter_info": {}
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
    filter_order: int = Query(5, description="Filter order")
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
            features = extract_audio_features(processed_audio_path)
            
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
        "service": "Local Emotion Detection & Gender Classification Service with DSP",
        "version": "1.1.0",
        "models": {
            "emotion": "emotion2vec_plus_large",
            "gender": "DSP-based classifier (F0, MFCC, Formants, Spectral Analysis)"
        },
        "dsp_features": {
            "filters": ["bandpass", "lowpass", "highpass", "none"],
            "default_bandpass": "300-3400 Hz (telephone bandwidth)",
            "filter_orders": "1-10 (default: 5)"
        },
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Emotion detection with DSP preprocessing",
            "/classify-gender": "POST - Gender classification with DSP preprocessing", 
            "/filter-presets": "GET - Common filter presets",
            "/info": "GET - Service information",
            "/docs": "GET - API documentation (Swagger UI)"
        },
        "dsp_preprocessing": {
            "steps": [
                "silence_trim (STE)", "anti_alias + resample", "pre_emphasis", "AGC (RMS normalize)",
                "FIR LPF (convolution)", "DFT/IDFT (duality)", "STFT OLA (PR error)",
                "Hilbert envelope/inst. freq", "window leakage demo", "quantization (μ-law, SQNR)",
                "FFT vs time-domain convolution speed", "impulse/step responses", "filter freq/phase/group delay",
                "linearity test (sinusoidal fidelity)"
            ],
            "artifacts_served_at": "/artifacts"
        }

    }

@app.on_event("startup")
async def startup_event():
    """Load model when service starts"""
    logger.info("🚀 Starting Local Emotion Detection & Gender Classification Service with DSP...")
    if load_model():
        logger.info("🎭 Emotion Detection Service is ready!")
        logger.info("👥 Gender Classification Service is ready!")
        logger.info("🔧 DSP Preprocessing is ready!")
    else:
        logger.error("❌ Failed to load emotion model. Emotion detection will not work properly.")
        logger.info("👥 Gender Classification Service is still available (DSP-based).")
        logger.info("🔧 DSP Preprocessing is still available.")

if __name__ == '__main__':
    print("🚀 Starting Local Emotion Detection & Gender Classification Service with DSP...")
    print("📡 Server will be available at: http://localhost:5000")
    print("📖 API documentation: http://localhost:5000/docs")
    print("🔗 Health check: http://localhost:5000/health")
    print("🎭 Emotion detection: POST /predict")
    print("👥 Gender classification: POST /classify-gender")
    print("🔧 Filter presets: GET /filter-presets")
    print("🎛️  DSP Features: Bandpass (300-3400Hz default), Lowpass, Highpass filters")
    ngrok.set_auth_token(NGROK_AUTH)

    public_url = ngrok.connect(5000)  
    print("🔗 Public URL:", public_url)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )