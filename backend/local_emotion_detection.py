#!/usr/bin/env python3
"""
Local Emotion Detection Service
Converts the Colab-based emotion detection to work locally without ngrok
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages if not available"""
    try:
        import numpy as np
        from funasr import AutoModel
        logger.info("All required packages are already installed")
        return True
    except ImportError as e:
        logger.error(f"Required package not found: {e}")
        logger.error("Please install required packages:")
        logger.error("pip install numpy funasr")
        return False

def load_emotion_model():
    """Load the emotion detection model locally"""
    try:
        logger.info("Loading emotion2vec_plus_large model...")
        model = AutoModel(model="emotion2vec_plus_large")
        logger.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

def detect_emotion(model, audio_file_path: str) -> Dict[str, Any]:
    """
    Run emotion detection on an audio file
    
    Args:
        model: The loaded FunASR model
        audio_file_path: Path to the audio file
        
    Returns:
        Dictionary containing emotion detection results
    """
    try:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        logger.info(f"Processing audio file: {audio_file_path}")
        
        # Run model inference
        result = model.generate(audio_file_path, granularity="utterance")
        data = result[0]
        
        # Extract emotions and scores from model output
        emotions = None
        scores = None
        
        if 'predictions' in data and 'scores' in data:
            emotions = data['predictions']
            scores = data['scores']
        elif 'labels' in data and 'scores' in data:
            # Clean up label names (remove path prefixes if present)
            emotions = [e.split('/')[-1] for e in data['labels']]
            scores = data['scores']
        else:
            raise ValueError("No valid predictions or labels found in model output")
        
        # Find the best emotion prediction
        import numpy as np
        best_idx = int(np.argmax(scores))
        best_emotion = emotions[best_idx]
        confidence = float(scores[best_idx])
        
        # Create sorted list of all emotions with scores
        emotion_score_pairs = list(zip(emotions, scores))
        sorted_emotions = sorted(emotion_score_pairs, key=lambda x: x[1], reverse=True)
        top_emotions = [{"emotion": e, "score": float(s)} for e, s in sorted_emotions]
        
        result_data = {
            "emotion": best_emotion,
            "confidence": confidence,
            "topEmotions": top_emotions
        }
        
        logger.info(f"✅ Emotion detected: {best_emotion} ({confidence:.2f})")
        return result_data
        
    except Exception as e:
        logger.error(f"❌ Error during emotion detection: {e}")
        raise

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Local Speech Emotion Detection")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--output", help="Output JSON file path (optional)")
    args = parser.parse_args()
    
    # Check if required packages are installed
    if not install_requirements():
        sys.exit(1)
    
    # Load the model
    model = load_emotion_model()
    if model is None:
        sys.exit(1)
    
    try:
        # Run emotion detection
        result = detect_emotion(model, args.audio_file)
        
        # Output results
        output_json = json.dumps(result, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            logger.info(f"Results saved to {args.output}")
        else:
            print(output_json)
            
    except Exception as e:
        error_result = {
            "error": str(e),
            "emotion": "unknown",
            "confidence": 0.0,
            "topEmotions": []
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()