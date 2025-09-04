#!/usr/bin/env python3
"""
Quick test to verify gender classification fixes
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.gender_classifier import classify_gender
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gender_classification_fix():
    """Test the improved gender classification"""
    
    sample_file = "OAF_back_happy.wav"
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample audio file {sample_file} not found!")
        return False
    
    print("🧪 Testing Improved Gender Classification")
    print("=" * 50)
    
    try:
        # Test with improved thresholds
        result = classify_gender(sample_file, method='threshold', use_manual_dsp=True)
        
        print(f"🎯 Classification Result:")
        print(f"   Gender: {result['gender']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Method: {result.get('method', 'unknown')}")
        
        if 'feature_analysis' in result:
            analysis = result['feature_analysis']
            print(f"\n🔍 Feature Analysis:")
            print(f"   F0 (Pitch): {analysis.get('f0_hz', 0):.1f} Hz")
            print(f"   F1 (Formant): {analysis.get('f1_hz', 0):.1f} Hz")
            print(f"   F2 (Formant): {analysis.get('f2_hz', 0):.1f} Hz") 
            print(f"   Spectral Centroid: {analysis.get('spectral_centroid_hz', 0):.1f} Hz")
            
            if 'feature_votes' in analysis:
                print(f"\n🗳️  Individual Feature Votes:")
                for feature, vote in analysis['feature_votes'].items():
                    confidence = analysis['feature_confidences'].get(feature, 0)
                    print(f"   {feature:12s}: {vote:7s} (confidence: {confidence:.3f})")
        
        if 'scores' in result:
            print(f"\n📊 Final Scores:")
            print(f"   Male Score: {result['scores']['male_score']:.3f}")
            print(f"   Female Score: {result['scores']['female_score']:.3f}")
            
        print(f"\n✅ Test completed! Result: {result['gender']}")
        
        # Check if it's no longer always female
        if result['gender'] != 'female':
            print("🎉 SUCCESS: Not defaulting to female anymore!")
            return True
        else:
            print("⚠️  Still classifying as female - may need further adjustment")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_gender_classification_fix()
    
    if success:
        print("\n🎉 Gender classification bias has been reduced!")
    else:
        print("\n⚠️  May need further threshold adjustments.")
        print("💡 Try testing with different audio samples or adjust thresholds further.")
