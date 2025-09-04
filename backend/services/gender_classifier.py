#!/usr/bin/env python3
"""
Gender Classification Module
Implements gender classification using DSP features and threshold-based classification
"""

from backend.services import dsp_preprocess
import numpy as np
from typing import Dict, Tuple, Any
import logging
from .dsp_features import extract_gender_relevant_features

logger = logging.getLogger(__name__)

class ThresholdGenderClassifier:
    """
    Simple threshold-based gender classifier using DSP features
    Based on typical acoustic differences between male and female speech
    """
    
    def __init__(self):
        # Typical thresholds based on research literature
        # These can be adjusted based on your data
        self.thresholds = {
            'f0_mean': 165,  # Hz - fundamental frequency threshold
            'formant_f1_mean': 730,  # Hz - first formant
            'formant_f2_mean': 1090,  # Hz - second formant
            'spectral_centroid': 2000,  # Hz - spectral centroid
        }
        
        # Weights for different features in final decision
        self.feature_weights = {
            'f0_mean': 0.4,  # F0 is most important for gender
            'formant_f1_mean': 0.2,
            'formant_f2_mean': 0.2,
            'spectral_centroid': 0.2
        }
    
    def classify_gender(self, audio_path: str) -> Dict[str, Any]:
        """
        Classify gender from audio file using threshold-based approach
        
        Returns:
            Dict containing gender prediction, confidence, and feature analysis
        """
        try:
            # Extract features
            features = extract_gender_relevant_features(audio_path)
            
            # Analyze each feature
            feature_votes = {}
            feature_confidences = {}
            
            # F0 (Fundamental Frequency) Analysis
            f0_mean = features.get('f0_mean', 0)
            if f0_mean > 0:
                if f0_mean > self.thresholds['f0_mean']:
                    feature_votes['f0'] = 'female'
                    # Higher F0 = more confident it's female
                    confidence = min(1.0, (f0_mean - self.thresholds['f0_mean']) / 100)
                else:
                    feature_votes['f0'] = 'male'
                    # Lower F0 = more confident it's male
                    confidence = min(1.0, (self.thresholds['f0_mean'] - f0_mean) / 100)
                feature_confidences['f0'] = max(0.1, confidence)
            else:
                feature_votes['f0'] = 'unknown'
                feature_confidences['f0'] = 0.0
            
            # Formant Analysis
            f1_mean = features.get('formant_f1_mean', 0)
            if f1_mean > 0:
                if f1_mean > self.thresholds['formant_f1_mean']:
                    feature_votes['f1'] = 'female'
                    confidence = min(1.0, (f1_mean - self.thresholds['formant_f1_mean']) / 200)
                else:
                    feature_votes['f1'] = 'male'
                    confidence = min(1.0, (self.thresholds['formant_f1_mean'] - f1_mean) / 200)
                feature_confidences['f1'] = max(0.1, confidence)
            else:
                feature_votes['f1'] = 'unknown'
                feature_confidences['f1'] = 0.0
            
            f2_mean = features.get('formant_f2_mean', 0)
            if f2_mean > 0:
                if f2_mean > self.thresholds['formant_f2_mean']:
                    feature_votes['f2'] = 'female'
                    confidence = min(1.0, (f2_mean - self.thresholds['formant_f2_mean']) / 300)
                else:
                    feature_votes['f2'] = 'male'
                    confidence = min(1.0, (self.thresholds['formant_f2_mean'] - f2_mean) / 300)
                feature_confidences['f2'] = max(0.1, confidence)
            else:
                feature_votes['f2'] = 'unknown'
                feature_confidences['f2'] = 0.0
            
            # Spectral Centroid Analysis
            spec_centroid = features.get('spectral_centroid', 0)
            if spec_centroid > 0:
                if spec_centroid > self.thresholds['spectral_centroid']:
                    feature_votes['spectral'] = 'female'
                    confidence = min(1.0, (spec_centroid - self.thresholds['spectral_centroid']) / 1000)
                else:
                    feature_votes['spectral'] = 'male'
                    confidence = min(1.0, (self.thresholds['spectral_centroid'] - spec_centroid) / 1000)
                feature_confidences['spectral'] = max(0.1, confidence)
            else:
                feature_votes['spectral'] = 'unknown'
                feature_confidences['spectral'] = 0.0
            
            # Weighted voting
            male_score = 0.0
            female_score = 0.0
            total_weight = 0.0
            
            for feature, vote in feature_votes.items():
                weight_key = {
                    'f0': 'f0_mean',
                    'f1': 'formant_f1_mean',
                    'f2': 'formant_f2_mean',
                    'spectral': 'spectral_centroid'
                }.get(feature)
                
                if weight_key and vote != 'unknown':
                    weight = self.feature_weights[weight_key]
                    confidence = feature_confidences[feature]
                    
                    if vote == 'male':
                        male_score += weight * confidence
                    else:  # female
                        female_score += weight * confidence
                    
                    total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                male_score /= total_weight
                female_score /= total_weight
            
            # Make final decision
            if male_score > female_score:
                predicted_gender = 'male'
                confidence_score = male_score
            elif female_score > male_score:
                predicted_gender = 'female'
                confidence_score = female_score
            else:
                predicted_gender = 'unknown'
                confidence_score = 0.0
            
            # Create detailed analysis
            analysis = {
                'gender': predicted_gender,
                'confidence': float(confidence_score),
                'feature_analysis': {
                    'f0_hz': float(f0_mean),
                    'f1_hz': float(f1_mean),
                    'f2_hz': float(f2_mean),
                    'spectral_centroid_hz': float(spec_centroid),
                    'feature_votes': feature_votes,
                    'feature_confidences': {k: float(v) for k, v in feature_confidences.items()}
                },
                'scores': {
                    'male_score': float(male_score),
                    'female_score': float(female_score)
                },
                'all_features': {k: float(v) for k, v in features.items()}
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in gender classification: {e}")
            return {
                'gender': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'feature_analysis': {},
                'scores': {'male_score': 0.0, 'female_score': 0.0},
                'all_features': {}
            }

class MLGenderClassifier:
    """
    Machine Learning based gender classifier
    Can be extended to use pre-trained models or train custom models
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'f0_mean', 'f0_std', 'spectral_centroid', 'formant_f1_mean',
            'formant_f2_mean', 'formant_f3_mean', 'spectral_rolloff',
            'spectral_bandwidth', 'harmonicity', 'harmonic_noise_ratio'
        ]
        
        # Try to load a pre-trained model if available
        self._try_load_pretrained_model()
    
    def _try_load_pretrained_model(self):
        """
        Try to load a pre-trained model
        For now, we'll fall back to threshold-based classification
        """
        try:
            # Here you could load a pre-trained model
            # from sklearn.externals import joblib
            # self.model = joblib.load('gender_model.pkl')
            # self.scaler = joblib.load('gender_scaler.pkl')
            pass
        except Exception as e:
            logger.info(f"No pre-trained model found, using threshold-based classification: {e}")
            self.model = None
    
    def classify_gender(self, audio_path: str) -> Dict[str, Any]:
        """
        Classify gender using ML model or fall back to threshold-based
        """
        if self.model is None:
            # Fall back to threshold-based classification
            threshold_classifier = ThresholdGenderClassifier()
            result = threshold_classifier.classify_gender(audio_path)
            result['method'] = 'threshold-based'
            return result
        
        try:
            # Extract features
            features = extract_gender_relevant_features(audio_path)
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Predict
            prediction = self.model.predict(feature_vector)[0]
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_vector)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 0.8  # Default confidence for models without probability
            
            return {
                'gender': prediction,
                'confidence': confidence,
                'method': 'machine-learning',
                'all_features': {k: float(v) for k, v in features.items()}
            }
            
        except Exception as e:
            logger.error(f"Error in ML gender classification: {e}")
            # Fall back to threshold-based
            threshold_classifier = ThresholdGenderClassifier()
            result = threshold_classifier.classify_gender(audio_path)
            result['method'] = 'threshold-based (fallback)'
            result['ml_error'] = str(e)
            return result

# Main interface function
def classify_gender(audio_path: str, method: str = 'auto') -> Dict[str, Any]:
    """
    Classify gender from audio file
    
    Args:
        audio_path: Path to audio file
        method: 'threshold', 'ml', or 'auto' (try ML first, fallback to threshold)
    
    Returns:
        Dictionary with gender classification results
    """
    dsp_report, processed_wav, artifacts = dsp_preprocess(
        audio_path=audio_path,
        fs_target=16000,
        apply_quantization_for_analysis=True,   # analysis plots only; model uses clean processed signal
        quant_bits=8,
        use_mu_law=True,    
        preemph_alpha=0.97,
        agc_target_rms=0.1,
    )

    if method == 'threshold':
        classifier = ThresholdGenderClassifier()
    elif method == 'ml':
        classifier = MLGenderClassifier()
    else:  # auto
        classifier = MLGenderClassifier()  # Will fallback to threshold if no ML model
    
    return classifier.classify_gender(audio_path=processed_wav)
