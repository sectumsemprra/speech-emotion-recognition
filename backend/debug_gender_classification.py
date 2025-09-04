#!/usr/bin/env python3
"""
Debug script for gender classification issues
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.gender_classifier import classify_gender
from services.manual_dsp import extract_gender_relevant_features_manual
from services.dsp_features import extract_gender_relevant_features
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_feature_extraction():
    """Debug feature extraction to see actual values"""
    
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    print("🔍 DEBUGGING FEATURE EXTRACTION")
    print("=" * 50)
    
    try:
        # Test manual feature extraction
        print("📊 Manual DSP Features:")
        manual_features = extract_gender_relevant_features_manual(sample_file)
        for key, value in manual_features.items():
            print(f"  {key:20s}: {value:10.2f}")
        
        print("\n📊 Library DSP Features:")
        library_features = extract_gender_relevant_features(sample_file)
        for key, value in library_features.items():
            print(f"  {key:20s}: {value:10.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_classification_logic():
    """Debug the classification logic step by step"""
    
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    print("\n🎯 DEBUGGING CLASSIFICATION LOGIC")
    print("=" * 50)
    
    try:
        # Get classification result with manual DSP
        result = classify_gender(sample_file, method='threshold', use_manual_dsp=True)
        
        print("📋 Classification Result:")
        print(f"  Gender: {result['gender']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Method: {result.get('method', 'unknown')}")
        
        if 'feature_analysis' in result:
            analysis = result['feature_analysis']
            print(f"\n🔍 Feature Analysis:")
            print(f"  F0 (pitch): {analysis.get('f0_hz', 0):.1f} Hz")
            print(f"  F1 (formant): {analysis.get('f1_hz', 0):.1f} Hz") 
            print(f"  F2 (formant): {analysis.get('f2_hz', 0):.1f} Hz")
            print(f"  Spectral Centroid: {analysis.get('spectral_centroid_hz', 0):.1f} Hz")
            
            if 'feature_votes' in analysis:
                print(f"\n🗳️  Feature Votes:")
                for feature, vote in analysis['feature_votes'].items():
                    confidence = analysis['feature_confidences'].get(feature, 0)
                    print(f"  {feature:12s}: {vote:7s} (confidence: {confidence:.3f})")
        
        if 'scores' in result:
            print(f"\n📊 Final Scores:")
            print(f"  Male Score: {result['scores']['male_score']:.3f}")
            print(f"  Female Score: {result['scores']['female_score']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_thresholds():
    """Analyze if thresholds are appropriate"""
    
    print("\n📏 THRESHOLD ANALYSIS")
    print("=" * 50)
    
    # Current thresholds
    thresholds = {
        'f0_mean': 165,  # Hz
        'formant_f1_mean': 730,  # Hz  
        'formant_f2_mean': 1090,  # Hz
        'spectral_centroid': 2000,  # Hz
    }
    
    print("Current Thresholds:")
    for feature, threshold in thresholds.items():
        print(f"  {feature:20s}: {threshold:6.0f} Hz")
    
    print("\nTypical Ranges (from literature):")
    print("  F0 (Male):      80-180 Hz")
    print("  F0 (Female):   165-265 Hz")
    print("  F1 (Male):     300-800 Hz")
    print("  F1 (Female):   400-1000 Hz") 
    print("  F2 (Male):     800-1200 Hz")
    print("  F2 (Female):  1000-1600 Hz")
    print("  Spectral Centroid varies widely")
    
    # Suggested improved thresholds
    improved_thresholds = {
        'f0_mean': 150,  # Slightly lower threshold
        'formant_f1_mean': 650,  # Lower threshold
        'formant_f2_mean': 1200,  # Higher threshold  
        'spectral_centroid': 1500,  # Lower threshold
    }
    
    print("\nSuggested Improved Thresholds:")
    for feature, threshold in improved_thresholds.items():
        print(f"  {feature:20s}: {threshold:6.0f} Hz")

def test_with_different_thresholds():
    """Test classification with different threshold values"""
    
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    print("\n🧪 TESTING DIFFERENT THRESHOLDS")
    print("=" * 50)
    
    # Import the classifier to modify thresholds
    from services.gender_classifier import ThresholdGenderClassifier
    
    # Test different threshold sets
    threshold_sets = [
        ("Current", {'f0_mean': 165, 'formant_f1_mean': 730, 'formant_f2_mean': 1090, 'spectral_centroid': 2000}),
        ("Lower F0", {'f0_mean': 140, 'formant_f1_mean': 730, 'formant_f2_mean': 1090, 'spectral_centroid': 2000}),
        ("Adjusted", {'f0_mean': 150, 'formant_f1_mean': 650, 'formant_f2_mean': 1200, 'spectral_centroid': 1500}),
        ("Conservative", {'f0_mean': 130, 'formant_f1_mean': 600, 'formant_f2_mean': 1300, 'spectral_centroid': 1200}),
    ]
    
    for name, thresholds in threshold_sets:
        classifier = ThresholdGenderClassifier(use_manual_dsp=True)
        classifier.thresholds = thresholds
        
        result = classifier.classify_gender(sample_file)
        
        print(f"\n{name} Thresholds:")
        print(f"  Result: {result['gender']} (confidence: {result['confidence']:.3f})")
        if 'scores' in result:
            print(f"  Male: {result['scores']['male_score']:.3f}, Female: {result['scores']['female_score']:.3f}")

if __name__ == '__main__':
    print("🐛 Gender Classification Debug Tool")
    print("=" * 50)
    
    # Debug feature extraction
    debug_feature_extraction()
    
    # Debug classification logic
    debug_classification_logic()
    
    # Analyze thresholds
    analyze_thresholds()
    
    # Test different thresholds
    test_with_different_thresholds()
    
    print("\n" + "=" * 50)
    print("🔧 Debug complete! Check the output above for issues.")
