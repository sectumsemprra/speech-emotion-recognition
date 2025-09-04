# Manual DSP Implementation

This document describes the manual Digital Signal Processing (DSP) implementation that minimizes external library dependencies while maintaining functionality for gender classification from speech.

## 🎯 Goals Achieved

- **Reduced Dependencies**: Minimized reliance on external libraries like librosa and scipy
- **Educational Value**: Implemented core DSP algorithms by hand for better understanding
- **Customization**: Full control over DSP parameters and implementations
- **Maintainability**: Self-contained DSP functions that don't rely on complex external libraries

## 📦 Dependencies Comparison

### Before (Library-based)
```python
import librosa              # Heavy audio processing library
import scipy.signal         # Signal processing functions
import numpy as np          # Array operations
```

### After (Manual DSP)
```python
import soundfile as sf      # Lightweight audio I/O only
import numpy as np          # Essential array operations (kept)
# numpy.fft (kept for FFT - manual FFT is very complex)
```

## 🛠️ Manual Implementations

### 1. Audio Loading and Preprocessing
- **Manual resampling** using linear interpolation
- **Stereo to mono conversion**
- **Pre-emphasis filtering** for speech enhancement

```python
def _simple_resample(self, y, orig_sr, target_sr):
    """Simple resampling using linear interpolation"""
    ratio = target_sr / orig_sr
    new_length = int(len(y) * ratio)
    old_indices = np.arange(len(y))
    new_indices = np.linspace(0, len(y) - 1, new_length)
    return np.interp(new_indices, old_indices, y)
```

### 2. Windowing Functions
- **Hanning window**: `0.5 * (1 - cos(2πn/(N-1)))`
- **Hamming window**: `0.54 - 0.46 * cos(2πn/(N-1))`

### 3. Energy-based Features
- **RMS Energy**: Manual root-mean-square calculation
- **Zero Crossing Rate**: Manual sign change detection
- **Frame Energy**: Manual frame-wise energy computation

### 4. Spectral Analysis
- **Manual STFT**: Short-Time Fourier Transform implementation
- **Spectral Centroid**: Center of mass of spectrum
- **Spectral Rolloff**: 85% energy threshold frequency
- **Spectral Bandwidth**: Weighted standard deviation around centroid
- **Spectral Flux**: Rate of spectral change

### 5. Pitch Analysis
- **Manual Autocorrelation**: For fundamental frequency estimation
- **F0 Estimation**: Peak detection in autocorrelation function
- **Jitter Calculation**: Pitch variation measurement

### 6. MFCC Implementation (Fully Manual)

Complete implementation of Mel-Frequency Cepstral Coefficients:

#### Step 1: Pre-emphasis Filter
```python
def _pre_emphasis_manual(self, y, alpha=0.97):
    return np.append(y[0], y[1:] - alpha * y[:-1])
```

#### Step 2: Windowing and FFT
- Apply Hanning window to frames
- Compute FFT (using numpy.fft)
- Calculate power spectrum

#### Step 3: Mel Filter Bank
```python
def _hz_to_mel_manual(self, hz):
    return 2595 * np.log10(1 + hz / 700)

def _mel_to_hz_manual(self, mel):
    return 700 * (10**(mel / 2595) - 1)
```

- Create triangular filters in mel-scale
- Map FFT bins to mel-scale frequencies

#### Step 4: Logarithm
- Apply log to mel-filtered spectrum
- Add small epsilon to avoid log(0)

#### Step 5: Discrete Cosine Transform (DCT)
```python
def _dct_manual(self, x):
    """Manual DCT Type-II implementation"""
    n_mels, n_frames = x.shape
    dct_matrix = np.zeros((n_mfcc, n_mels))
    
    for k in range(n_mfcc):
        for n in range(n_mels):
            dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    
    # Apply normalization
    dct_matrix[0] *= np.sqrt(1 / n_mels)
    dct_matrix[1:] *= np.sqrt(2 / n_mels)
    
    return np.dot(dct_matrix, x)
```

### 7. Linear Prediction Coding (LPC)

Manual implementation using Levinson-Durbin algorithm:

```python
def _lpc_analysis_manual(self, signal, order):
    """Manual LPC using Levinson-Durbin algorithm"""
    R = self._autocorrelation_manual(signal)
    a = np.zeros(order + 1)
    a[0] = 1.0
    E = R[0]  # Prediction error
    
    for i in range(1, order + 1):
        # Reflection coefficient
        k = -R[i]
        for j in range(1, i):
            k -= a[j] * R[i - j]
        k /= E
        
        # Update coefficients
        a[i] = k
        for j in range(1, i):
            a[j] = a[j] + k * a[i - j]
        
        # Update error
        E *= (1 - k * k)
    
    return a
```

### 8. Formant Extraction
- **Root finding** from LPC polynomial
- **Frequency conversion** from complex roots
- **Formant filtering** for valid frequency ranges

## 🎯 Gender Classification Features

The manual DSP implementation extracts these key features for gender classification:

### Primary Features (with thresholds)
- **F0 (Fundamental Frequency)**: 165 Hz threshold
- **F1 (First Formant)**: 730 Hz threshold  
- **F2 (Second Formant)**: 1090 Hz threshold
- **Spectral Centroid**: 2000 Hz threshold

### Additional Features
- **F0 Statistics**: mean, std, min, max, jitter
- **Spectral Features**: rolloff, bandwidth, flux
- **MFCC Coefficients**: first 13 coefficients with statistics
- **Energy Features**: RMS, variance, zero crossing rate

## 📊 Performance Comparison

### Accuracy
- Manual implementation produces results within 20% of library-based implementation for key features
- Gender classification agreement between manual and library methods is typically high
- Some differences expected due to implementation details and parameter choices

### Speed
- Manual implementation may be slightly slower than optimized libraries
- Trade-off between dependency reduction and computational efficiency
- Can be optimized further for production use

### Memory Usage
- Reduced memory footprint due to fewer loaded libraries
- More predictable memory usage patterns
- Better control over temporary array allocation

## 🔧 Usage

### Enable Manual DSP (Default)
```python
from services.gender_classifier import classify_gender

# Uses manual DSP by default
result = classify_gender('audio.wav', use_manual_dsp=True)
```

### Compare Manual vs Library
```python
# Library-based
library_result = classify_gender('audio.wav', use_manual_dsp=False)

# Manual DSP
manual_result = classify_gender('audio.wav', use_manual_dsp=True)
```

### Testing
```bash
cd backend
python test_manual_dsp.py
```

## 🎓 Educational Benefits

1. **Understanding DSP Algorithms**: See how MFCCs, LPC, and spectral features actually work
2. **Signal Processing Concepts**: Learn windowing, FFT, autocorrelation, and filter banks
3. **Algorithm Implementation**: Practice implementing mathematical concepts in code
4. **Debugging and Optimization**: Control every step of the processing pipeline

## 🚀 Future Improvements

1. **Manual FFT Implementation**: Replace numpy.fft with manual FFT (Cooley-Tukey algorithm)
2. **Optimized Algorithms**: Use more efficient implementations for production
3. **Parallel Processing**: Add multi-threading for frame-based processing
4. **Custom Audio I/O**: Replace soundfile with manual WAV file reading
5. **Advanced Features**: Add more sophisticated DSP techniques

## 📝 Code Structure

```
backend/services/
├── manual_dsp.py           # Manual DSP implementations
├── dsp_features.py         # Original library-based implementations  
├── gender_classifier.py    # Updated to support both methods
└── test_manual_dsp.py     # Comparison and testing script
```

## 🎯 Key Takeaways

- **Minimal Dependencies**: Reduced from heavy audio libraries to just soundfile + numpy
- **Educational Value**: Complete understanding of DSP pipeline
- **Flexibility**: Full control over all processing parameters
- **Maintainability**: Self-contained code that doesn't rely on external library changes
- **Performance**: Comparable accuracy with reasonable computational cost

The manual DSP implementation successfully demonstrates core signal processing concepts while maintaining practical functionality for gender classification from speech signals.
