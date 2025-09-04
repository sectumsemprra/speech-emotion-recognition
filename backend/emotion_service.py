#!/usr/bin/env python3
"""
Local Emotion Detection FastAPI Service
A FastAPI service that runs locally for emotion detection
"""

import os
import json
import tempfile
import logging
from typing import Dict, Any
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pyngrok import ngrok
from services.gender_classifier import classify_gender
from services.dsp_preprocess import dsp_preprocess
from fastapi.staticfiles import StaticFiles

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

def detect_emotion_from_file(audio_file_path: str) -> Dict[str, Any]:
    """Run emotion detection on an audio file"""
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
    
    try:
        
        logger.info(f"Processing audio file: {audio_file_path}")
        
        # Run model inference
        result = model.generate(processed_wav, granularity="utterance")
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
        
        # Sort all emotions by score
        emotion_pairs = list(zip(emotions, scores))
        sorted_emotions = sorted(emotion_pairs, key=lambda x: x[1], reverse=True)
        top_emotions = [{"emotion": e, "score": float(s)} for e, s in sorted_emotions]
        
        artifact_urls = [f"/artifacts/{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" for p in artifacts]
        processed_url = f"/artifacts/{os.path.basename(os.path.dirname(processed_wav))}/{os.path.basename(processed_wav)}"

        return {
            "emotion": best_emotion,
            "confidence": confidence,
            "topEmotions": top_emotions,
            "dsp_report": dsp_report,
            "artifacts": {
                "processed_wav": processed_url,
                "plots": artifact_urls
            }
        }
        
    except Exception as e:
        logger.error(f"Error during emotion detection: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Local Emotion Detection Service",
    description="A FastAPI service for speech emotion recognition using emotion2vec_plus_large model",
    version="1.0.0"
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
        "service": "Local Emotion Detection Service",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_emotion(audio: UploadFile = File(...)):
    """Predict emotion from uploaded audio file"""
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Check file type (optional validation)
        allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/webm']
        if audio.content_type and audio.content_type not in allowed_types:
            logger.warning(f"Unusual audio type: {audio.content_type}")
        
        # Save uploaded file temporarily with original extension
        file_ext = os.path.splitext(audio.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Run emotion detection
            result = detect_emotion_from_file(temp_path)
            logger.info(f"Emotion detected: {result['emotion']} ({result['confidence']:.2f})")
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
                "topEmotions": []
            }
        )

@app.post("/classify-gender")
async def classify_gender_endpoint(audio: UploadFile = File(...)):
    """Classify gender from uploaded audio file using DSP features"""
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Check file type (optional validation)
        allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/webm']
        if audio.content_type and audio.content_type not in allowed_types:
            logger.warning(f"Unusual audio type: {audio.content_type}")
        
        # Save uploaded file temporarily with original extension
        file_ext = os.path.splitext(audio.filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Run gender classification
            result = classify_gender(temp_path, method='auto')
            logger.info(f"Gender classified: {result['gender']} (confidence: {result.get('confidence', 0):.2f})")
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
                "method": "error"
            }
        )

@app.get("/info")
async def service_info():
    """Get service information"""
    return {
        "service": "Local Emotion Detection & Gender Classification Service",
        "version": "1.0.0",
        "models": {
            "emotion": "emotion2vec_plus_large",
            "gender": "DSP-based threshold classifier"
        },
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Emotion detection (multipart/form-data with 'audio' file)",
            "/classify-gender": "POST - Gender classification (multipart/form-data with 'audio' file)",
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
    logger.info("🚀 Starting Local Emotion Detection & Gender Classification Service...")
    if load_model():
        logger.info("🎭 Emotion Detection Service is ready!")
        logger.info("👥 Gender Classification Service is ready!")
    else:
        logger.error("❌ Failed to load emotion model. Emotion detection will not work properly.")
        logger.info("👥 Gender Classification Service is still available (DSP-based).")

if __name__ == '__main__':
    print("🚀 Starting Local Emotion Detection & Gender Classification Service...")
    print("📡 Server will be available at: http://localhost:5000")
    print("📖 API documentation: http://localhost:5000/docs")
    print("🔗 Health check: http://localhost:5000/health")
    print("🎭 Emotion detection: POST /predict")
    print("👥 Gender classification: POST /classify-gender")
    ngrok.set_auth_token(NGROK_AUTH)

    public_url = ngrok.connect(5000)  
    print("🔗 Public URL:", public_url)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )