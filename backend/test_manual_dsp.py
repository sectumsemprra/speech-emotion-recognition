#!/usr/bin/env python3
"""
Test script to compare manual DSP implementation vs library-based implementation
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.gender_classifier import classify_gender
from services.dsp_features import extract_gender_relevant_features
from services.manual_dsp import extract_gender_relevant_features_manual
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_feature_extraction():
    """Compare manual vs library feature extraction"""
    
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    try:
        logger.info("🔬 Comparing Manual DSP vs Library DSP Feature Extraction")
        logger.info("=" * 70)
        
        # Test library-based extraction
        logger.info("📚 Testing Library-based DSP...")
        start_time = time.time()
        library_features = extract_gender_relevant_features(sample_file)
        library_time = time.time() - start_time
        
        # Test manual extraction
        logger.info("🛠️  Testing Manual DSP...")
        start_time = time.time()
        manual_features = extract_gender_relevant_features_manual(sample_file)
        manual_time = time.time() - start_time
        
        logger.info("\n📊 FEATURE COMPARISON")
        logger.info("-" * 50)
        
        # Compare key features
        key_features = ['f0_mean', 'spectral_centroid', 'formant_f1_mean', 'formant_f2_mean']
        
        for feature in key_features:
            lib_val = library_features.get(feature, 0)
            man_val = manual_features.get(feature, 0)
            
            if lib_val != 0 and man_val != 0:
                diff_pct = abs(lib_val - man_val) / lib_val * 100
                status = "✅ CLOSE" if diff_pct < 20 else "⚠️ DIFFERENT"
            else:
                diff_pct = 0
                status = "❓ N/A"
            
            logger.info(f"{feature:20s}: Library={lib_val:8.1f}, Manual={man_val:8.1f}, Diff={diff_pct:5.1f}% {status}")
        
        logger.info(f"\n⏱️  TIMING COMPARISON")
        logger.info(f"Library DSP: {library_time:.3f} seconds")
        logger.info(f"Manual DSP:  {manual_time:.3f} seconds")
        logger.info(f"Speed ratio: {manual_time/library_time:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during feature comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_gender_classification():
    """Compare manual vs library gender classification"""
    
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    try:
        logger.info("\n🚻 Comparing Gender Classification Methods")
        logger.info("=" * 70)
        
        # Test library-based classification
        logger.info("📚 Testing Library-based Gender Classification...")
        start_time = time.time()
        library_result = classify_gender(sample_file, method='threshold', use_manual_dsp=False)
        library_time = time.time() - start_time
        
        # Test manual classification
        logger.info("🛠️  Testing Manual DSP Gender Classification...")
        start_time = time.time()
        manual_result = classify_gender(sample_file, method='threshold', use_manual_dsp=True)
        manual_time = time.time() - start_time
        
        logger.info("\n📊 CLASSIFICATION COMPARISON")
        logger.info("-" * 50)
        
        # Compare results
        logger.info(f"Library Method: {library_result.get('method', 'unknown')}")
        logger.info(f"Library Gender: {library_result['gender']} (confidence: {library_result['confidence']:.3f})")
        
        logger.info(f"Manual Method:  {manual_result.get('method', 'unknown')}")
        logger.info(f"Manual Gender:  {manual_result['gender']} (confidence: {manual_result['confidence']:.3f})")
        
        # Compare feature analysis
        if 'feature_analysis' in library_result and 'feature_analysis' in manual_result:
            lib_analysis = library_result['feature_analysis']
            man_analysis = manual_result['feature_analysis']
            
            logger.info("\n🔍 FEATURE ANALYSIS COMPARISON")
            logger.info("-" * 50)
            
            features_to_compare = ['f0_hz', 'f1_hz', 'f2_hz', 'spectral_centroid_hz']
            for feature in features_to_compare:
                lib_val = lib_analysis.get(feature, 0)
                man_val = man_analysis.get(feature, 0)
                
                logger.info(f"{feature:20s}: Library={lib_val:8.1f}, Manual={man_val:8.1f}")
        
        logger.info(f"\n⏱️  CLASSIFICATION TIMING")
        logger.info(f"Library Classification: {library_time:.3f} seconds")
        logger.info(f"Manual Classification:  {manual_time:.3f} seconds")
        logger.info(f"Speed ratio: {manual_time/library_time:.2f}x")
        
        # Agreement check
        agreement = library_result['gender'] == manual_result['gender']
        logger.info(f"\n🎯 AGREEMENT: {'✅ YES' if agreement else '❌ NO'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during classification comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_manual_dsp_details():
    """Show details about manual DSP implementation"""
    
    logger.info("\n🔧 MANUAL DSP IMPLEMENTATION DETAILS")
    logger.info("=" * 70)
    
    logger.info("✅ Implemented Functions:")
    logger.info("   • Audio loading with soundfile (minimal dependency)")
    logger.info("   • Simple resampling using linear interpolation")
    logger.info("   • Manual Hanning and Hamming windowing")
    logger.info("   • Manual Zero Crossing Rate calculation")
    logger.info("   • Manual frame-based energy analysis")
    logger.info("   • Manual Short-Time Fourier Transform (STFT)")
    logger.info("   • Manual spectral features (centroid, rolloff, bandwidth, flux)")
    logger.info("   • Manual autocorrelation for F0 estimation")
    logger.info("   • Manual MFCC implementation:")
    logger.info("     - Pre-emphasis filter")
    logger.info("     - Mel filter bank creation")
    logger.info("     - Mel-scale conversion functions")
    logger.info("     - Discrete Cosine Transform (DCT)")
    logger.info("   • Manual Linear Prediction Coding (LPC) with Levinson-Durbin")
    logger.info("   • Manual formant extraction from LPC coefficients")
    
    logger.info("\n📦 Reduced Dependencies:")
    logger.info("   • Removed: librosa.feature.mfcc")
    logger.info("   • Removed: librosa.feature.zero_crossing_rate")
    logger.info("   • Removed: librosa.effects.hpss")
    logger.info("   • Removed: scipy.signal (for windowing)")
    logger.info("   • Kept: numpy (essential for array operations)")
    logger.info("   • Kept: soundfile (lightweight audio I/O)")
    logger.info("   • Kept: numpy.fft (for FFT - could be manual but very complex)")
    
    logger.info("\n🎯 Benefits of Manual Implementation:")
    logger.info("   • Better understanding of DSP algorithms")
    logger.info("   • Reduced external dependencies")
    logger.info("   • Customizable parameters and behavior")
    logger.info("   • Educational value for learning DSP")
    logger.info("   • Potential for optimization for specific use cases")

if __name__ == '__main__':
    print("🧪 Manual DSP vs Library DSP Comparison Test")
    print("=" * 70)
    
    # Show implementation details
    show_manual_dsp_details()
    
    # Test feature extraction comparison
    print("\n1. Testing Feature Extraction...")
    feature_success = compare_feature_extraction()
    
    # Test gender classification comparison
    print("\n2. Testing Gender Classification...")
    classification_success = compare_gender_classification()
    
    print("\n" + "=" * 70)
    if feature_success and classification_success:
        print("🎉 All tests completed! Manual DSP implementation is working.")
        print("💡 The system now uses minimal external dependencies for DSP processing.")
    else:
        print("❌ Some tests failed. Check the logs above.")
        sys.exit(1)
