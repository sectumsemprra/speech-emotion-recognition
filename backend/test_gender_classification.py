#!/usr/bin/env python3
"""
Test script for gender classification functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.gender_classifier import classify_gender
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gender_classification():
    """Test gender classification with the sample audio file"""
    
    # Check if sample audio file exists
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    try:
        logger.info(f"Testing gender classification with {sample_file}...")
        
        # Test threshold-based classification
        result = classify_gender(sample_file, method='threshold')
        
        logger.info("=== Gender Classification Results ===")
        logger.info(f"Predicted Gender: {result['gender']}")
        logger.info(f"Confidence: {result['confidence']:.3f}")
        logger.info(f"Method: {result.get('method', 'unknown')}")
        
        if 'feature_analysis' in result:
            analysis = result['feature_analysis']
            logger.info("\n=== Feature Analysis ===")
            logger.info(f"F0 (Fundamental Frequency): {analysis.get('f0_hz', 0):.1f} Hz")
            logger.info(f"F1 (First Formant): {analysis.get('f1_hz', 0):.1f} Hz")
            logger.info(f"F2 (Second Formant): {analysis.get('f2_hz', 0):.1f} Hz")
            logger.info(f"Spectral Centroid: {analysis.get('spectral_centroid_hz', 0):.1f} Hz")
            
            if 'feature_votes' in analysis:
                logger.info("\n=== Individual Feature Votes ===")
                for feature, vote in analysis['feature_votes'].items():
                    confidence = analysis['feature_confidences'].get(feature, 0)
                    logger.info(f"{feature}: {vote} (confidence: {confidence:.3f})")
        
        if 'scores' in result:
            scores = result['scores']
            logger.info(f"\n=== Final Scores ===")
            logger.info(f"Male Score: {scores['male_score']:.3f}")
            logger.info(f"Female Score: {scores['female_score']:.3f}")
        
        logger.info("\n✅ Gender classification test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during gender classification test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dsp_features():
    """Test DSP feature extraction"""
    
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    try:
        from services.dsp_features import DSPFeatureExtractor
        
        logger.info(f"Testing DSP feature extraction with {sample_file}...")
        
        extractor = DSPFeatureExtractor()
        features = extractor.extract_all_features(sample_file)
        
        logger.info("\n=== DSP Features ===")
        logger.info(f"Duration: {features.get('duration', 0):.2f} seconds")
        logger.info(f"Sample Rate: {features.get('sample_rate', 0)} Hz")
        logger.info(f"RMS Energy: {features.get('rms_energy', 0):.6f}")
        logger.info(f"Zero Crossing Rate: {features.get('zero_crossing_rate', 0):.6f}")
        logger.info(f"F0 Mean: {features.get('f0_mean', 0):.1f} Hz")
        logger.info(f"F0 Std: {features.get('f0_std', 0):.1f} Hz")
        logger.info(f"Spectral Centroid: {features.get('spectral_centroid', 0):.1f} Hz")
        logger.info(f"Spectral Rolloff: {features.get('spectral_rolloff', 0):.1f} Hz")
        logger.info(f"Spectral Bandwidth: {features.get('spectral_bandwidth', 0):.1f} Hz")
        
        # Show formants
        for i in range(1, 4):
            f_mean = features.get(f'formant_f{i}_mean', 0)
            f_std = features.get(f'formant_f{i}_std', 0)
            logger.info(f"Formant F{i}: {f_mean:.1f} ± {f_std:.1f} Hz")
        
        # Show some MFCCs
        logger.info("\n=== MFCC Features (first 5) ===")
        for i in range(5):
            mfcc_mean = features.get(f'mfcc_{i}_mean', 0)
            mfcc_std = features.get(f'mfcc_{i}_std', 0)
            logger.info(f"MFCC {i}: {mfcc_mean:.3f} ± {mfcc_std:.3f}")
        
        logger.info(f"\n✅ DSP feature extraction test completed! Extracted {len(features)} features.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during DSP feature extraction test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 Testing Gender Classification System")
    print("=" * 50)
    
    # Test DSP feature extraction
    print("\n1. Testing DSP Feature Extraction...")
    dsp_success = test_dsp_features()
    
    # Test gender classification
    print("\n2. Testing Gender Classification...")
    gender_success = test_gender_classification()
    
    print("\n" + "=" * 50)
    if dsp_success and gender_success:
        print("🎉 All tests passed! Gender classification system is working.")
    else:
        print("❌ Some tests failed. Check the logs above.")
        sys.exit(1)
