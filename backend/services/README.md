# DSP-Based Gender Classification Service

This module implements a comprehensive Digital Signal Processing (DSP) based gender classification system for speech audio. The system extracts various acoustic features from speech signals and uses them to classify gender using threshold-based classification.

## Architecture

### 1. DSP Feature Extraction (`dsp_features.py`)

The `DSPFeatureExtractor` class implements various DSP techniques to extract meaningful features from speech signals:

#### Energy-Based Features
- **RMS Energy**: Root Mean Square energy of the signal
- **Zero Crossing Rate**: Rate at which signal changes sign
- **Energy Variance**: Variability in frame-wise energy
- **Energy Mean**: Average frame energy

#### Spectral Features (using FFT/DFT)
- **Spectral Centroid**: Center of mass of the spectrum (brightness)
- **Spectral Rolloff**: Frequency below which 85% of energy is contained
- **Spectral Bandwidth**: Weighted standard deviation around spectral centroid
- **Spectral Flux**: Rate of change in spectral power

#### Pitch and Fundamental Frequency Features
- **F0 (Fundamental Frequency)**: Estimated using autocorrelation method
- **F0 Statistics**: Mean, standard deviation, min, max of F0
- **Jitter**: Pitch variation measure

#### MFCCs (Mel-Frequency Cepstral Coefficients)
- 13 MFCC coefficients with statistical measures (mean, std)
- Represents spectral envelope characteristics

#### Formant Features (using Linear Prediction Coding)
- **F1, F2, F3**: First three formant frequencies
- Extracted using LPC analysis and root finding
- Statistical measures (mean, std) for each formant

#### Harmonic Features
- **Harmonic-to-Noise Ratio (HNR)**: Measure of voice quality
- **Harmonicity**: Ratio of harmonic to total energy

### 2. Gender Classification (`gender_classifier.py`)

#### Threshold-Based Classifier

Uses acoustic differences between male and female speech:

**Key Gender Differences:**
- **Fundamental Frequency (F0)**: 
  - Males: ~85-180 Hz
  - Females: ~165-265 Hz
  - Threshold: 165 Hz

- **Formants**:
  - F1 threshold: 730 Hz
  - F2 threshold: 1090 Hz
  - Females typically have higher formants

- **Spectral Centroid**: 
  - Threshold: 2000 Hz
  - Females typically have higher spectral centroids

#### Classification Process

1. **Feature Extraction**: Extract gender-relevant features
2. **Individual Feature Voting**: Each feature votes for male/female
3. **Confidence Calculation**: Based on distance from thresholds
4. **Weighted Voting**: Combine votes with feature weights:
   - F0: 40% weight (most important)
   - F1, F2, Spectral Centroid: 20% each
5. **Final Decision**: Highest weighted score determines gender

#### ML Classifier (Extensible)

Framework for machine learning-based classification:
- Can load pre-trained models
- Falls back to threshold-based if no ML model available
- Supports feature scaling and probability estimation

## API Integration

### Gender Classification Endpoint

```http
POST /classify-gender
Content-Type: multipart/form-data

Parameters:
- audio: Audio file (WAV, MP3, MP4, WebM)
```

**Response:**
```json
{
  "gender": "male|female|unknown",
  "confidence": 0.85,
  "method": "threshold-based",
  "feature_analysis": {
    "f0_hz": 120.5,
    "f1_hz": 650.2,
    "f2_hz": 950.8,
    "spectral_centroid_hz": 1800.3,
    "feature_votes": {
      "f0": "male",
      "f1": "male",
      "f2": "male",
      "spectral": "male"
    },
    "feature_confidences": {
      "f0": 0.8,
      "f1": 0.6,
      "f2": 0.7,
      "spectral": 0.5
    }
  },
  "scores": {
    "male_score": 0.85,
    "female_score": 0.15
  },
  "all_features": {
    // Complete feature vector
  }
}
```

## DSP Techniques Implemented

### 1. Windowing
- Hanning window for frame processing
- Reduces spectral leakage in FFT analysis

### 2. FFT/DFT
- Fast Fourier Transform for frequency domain analysis
- Power spectrum calculation
- Frequency bin analysis

### 3. Autocorrelation
- F0 estimation using autocorrelation method
- Peak detection in autocorrelation function

### 4. Linear Prediction Coding (LPC)
- Formant extraction using LPC analysis
- Levinson-Durbin algorithm implementation
- Root finding for formant frequencies

### 5. Mel-Scale Processing
- MFCC extraction using mel-scale filter banks
- Perceptually relevant frequency representation

### 6. Harmonic-Percussive Separation
- Separation of harmonic and percussive components
- Harmonic-to-noise ratio calculation

## Usage Examples

### Basic Usage
```python
from services.gender_classifier import classify_gender

# Classify gender from audio file
result = classify_gender('audio.wav', method='threshold')
print(f"Gender: {result['gender']}, Confidence: {result['confidence']}")
```

### Feature Extraction Only
```python
from services.dsp_features import DSPFeatureExtractor

extractor = DSPFeatureExtractor()
features = extractor.extract_all_features('audio.wav')
print(f"F0 mean: {features['f0_mean']} Hz")
```

### Testing
```bash
cd backend
python test_gender_classification.py
```

## Technical Notes

### Limitations
- Threshold values are based on general population statistics
- May need calibration for specific populations or recording conditions
- Performance depends on audio quality and recording conditions

### Future Improvements
- Machine learning model training with labeled data
- Adaptive thresholds based on audio characteristics
- Integration with deep learning models
- Real-time processing capabilities

### Dependencies
- librosa: Audio processing and feature extraction
- numpy: Numerical computations
- scipy: Signal processing functions
- fastapi: Web API framework

## References

The implementation is based on established DSP and speech processing techniques:
- Fundamental frequency estimation using autocorrelation
- Formant analysis using Linear Prediction Coding
- Spectral feature extraction using FFT
- Gender classification based on acoustic phonetics research
