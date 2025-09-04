#!/usr/bin/env python3
"""
Manual DSP Implementation Module
Implements DSP functions manually to reduce dependency on external libraries
"""

import numpy as np
import soundfile as sf  # Only for audio loading
import librosa  # Fallback for unsupported formats
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ManualDSPProcessor:
    """
    Manual implementation of DSP functions for speech analysis
    Minimizes use of external libraries by implementing core DSP algorithms by hand
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 40
        self.n_mfcc = 13
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file manually"""
        try:
            # Try soundfile first (lighter than librosa)
            y, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            
            # Resample if needed (simple decimation/interpolation)
            if sr != self.sample_rate:
                y = self._simple_resample(y, sr, self.sample_rate)
                sr = self.sample_rate
            
            return y, sr
        except Exception as e:
            # soundfile failed, try librosa for broader format support (WebM, MP4, etc.)
            try:
                logger.warning(f"soundfile failed ({e}), trying librosa for {audio_path}")
                y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                return y, sr
            except Exception as e2:
                logger.error(f"Both soundfile and librosa failed: {e}, {e2}")
                raise RuntimeError(f"Could not load audio file with soundfile ({e}) or librosa ({e2})")
    
    def _simple_resample(self, y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation"""
        if orig_sr == target_sr:
            return y
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(y) * ratio)
        
        # Create new time indices
        old_indices = np.arange(len(y))
        new_indices = np.linspace(0, len(y) - 1, new_length)
        
        # Linear interpolation
        resampled = np.interp(new_indices, old_indices, y)
        return resampled
    
    def extract_all_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract all DSP features using manual implementations"""
        try:
            # Load audio manually
            y, sr = self.load_audio(audio_path)
            
            features = {}
            
            # Basic signal properties
            features['duration'] = len(y) / sr
            features['sample_rate'] = sr
            
            # Energy-based features (manual)
            features.update(self._extract_energy_features_manual(y))
            
            # Spectral features (manual FFT-based)
            features.update(self._extract_spectral_features_manual(y, sr))
            
            # Pitch features (manual autocorrelation)
            features.update(self._extract_pitch_features_manual(y, sr))
            
            # MFCCs (fully manual implementation)
            features.update(self._extract_mfcc_features_manual(y, sr))
            
            # Formant features (manual LPC)
            features.update(self._extract_formant_features_manual(y, sr))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _hanning_window(self, length: int) -> np.ndarray:
        """Manual Hanning window implementation"""
        n = np.arange(length)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (length - 1)))
    
    def _hamming_window(self, length: int) -> np.ndarray:
        """Manual Hamming window implementation"""
        n = np.arange(length)
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (length - 1))
    
    def _extract_energy_features_manual(self, y: np.ndarray) -> Dict[str, float]:
        """Extract energy features manually"""
        features = {}
        
        # Root Mean Square Energy
        rms = np.sqrt(np.mean(y**2))
        features['rms_energy'] = float(rms)
        
        # Zero Crossing Rate (manual implementation)
        zcr = self._zero_crossing_rate_manual(y)
        features['zero_crossing_rate'] = float(zcr)
        
        # Energy variation (frame-based)
        frame_energies = self._frame_energy_manual(y)
        features['energy_variance'] = float(np.var(frame_energies))
        features['energy_mean'] = float(np.mean(frame_energies))
        
        return features
    
    def _zero_crossing_rate_manual(self, y: np.ndarray) -> float:
        """Manual zero crossing rate calculation"""
        # Sign changes indicate zero crossings
        signs = np.sign(y)
        sign_changes = np.diff(signs) != 0
        return np.mean(sign_changes)
    
    def _frame_energy_manual(self, y: np.ndarray) -> np.ndarray:
        """Manual frame energy calculation"""
        frame_length = self.n_fft
        hop_length = self.hop_length
        
        energies = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            energy = np.sum(frame**2)
            energies.append(energy)
        
        return np.array(energies)
    
    def _extract_spectral_features_manual(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features using manual FFT"""
        features = {}
        
        # Manual STFT
        stft_matrix = self._stft_manual(y)
        magnitude_spectrum = np.abs(stft_matrix)
        power_spectrum = magnitude_spectrum**2
        
        # Frequency bins
        freqs = np.fft.fftfreq(self.n_fft, 1/sr)[:self.n_fft//2]
        
        # Average across time frames
        avg_magnitude = np.mean(magnitude_spectrum, axis=1)
        avg_power = np.mean(power_spectrum, axis=1)
        
        # Spectral Centroid (manual)
        spectral_centroid = np.sum(freqs * avg_magnitude) / np.sum(avg_magnitude)
        features['spectral_centroid'] = float(spectral_centroid)
        
        # Spectral Rolloff (manual)
        cumulative_energy = np.cumsum(avg_power)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = 0.85 * total_energy
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0][0]
        spectral_rolloff = freqs[rolloff_idx]
        features['spectral_rolloff'] = float(spectral_rolloff)
        
        # Spectral Bandwidth (manual)
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid)**2) * avg_magnitude) / 
            np.sum(avg_magnitude)
        )
        features['spectral_bandwidth'] = float(spectral_bandwidth)
        
        # Spectral Flux (manual)
        if stft_matrix.shape[1] > 1:
            spectral_flux = np.mean(np.diff(magnitude_spectrum, axis=1)**2)
            features['spectral_flux'] = float(spectral_flux)
        else:
            features['spectral_flux'] = 0.0
        
        return features
    
    def _stft_manual(self, y: np.ndarray) -> np.ndarray:
        """Manual Short-Time Fourier Transform implementation"""
        frame_length = self.n_fft
        hop_length = self.hop_length
        
        # Calculate number of frames
        n_frames = (len(y) - frame_length) // hop_length + 1
        
        # Initialize STFT matrix
        stft_matrix = np.zeros((frame_length // 2, n_frames), dtype=complex)
        
        # Hanning window
        window = self._hanning_window(frame_length)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = y[start:start + frame_length]
            
            # Apply window
            windowed_frame = frame * window
            
            # FFT
            fft_frame = np.fft.fft(windowed_frame)
            
            # Keep only positive frequencies
            stft_matrix[:, i] = fft_frame[:frame_length // 2]
        
        return stft_matrix
    
    def _extract_pitch_features_manual(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract pitch features using manual autocorrelation"""
        f0_values = self._estimate_f0_autocorr_manual(y, sr)
        
        # Filter out invalid F0 values
        valid_f0 = f0_values[f0_values > 0]
        
        if len(valid_f0) > 0:
            features = {
                'f0_mean': float(np.mean(valid_f0)),
                'f0_std': float(np.std(valid_f0)),
                'f0_min': float(np.min(valid_f0)),
                'f0_max': float(np.max(valid_f0))
            }
            
            # Jitter calculation
            if len(valid_f0) > 1:
                jitter = np.mean(np.abs(np.diff(valid_f0)) / valid_f0[:-1])
                features['jitter'] = float(jitter)
            else:
                features['jitter'] = 0.0
        else:
            features = {
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'f0_min': 0.0,
                'f0_max': 0.0,
                'jitter': 0.0
            }
        
        return features
    
    def _estimate_f0_autocorr_manual(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Manual F0 estimation using autocorrelation"""
        frame_length = self.n_fft
        hop_length = self.hop_length
        
        f0_values = []
        window = self._hanning_window(frame_length)
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            
            # Apply window
            windowed = frame * window
            
            # Manual autocorrelation
            autocorr = self._autocorrelation_manual(windowed)
            
            # Find peak in expected F0 range (80-400 Hz)
            min_period = int(sr / 400)  # 400 Hz
            max_period = int(sr / 80)   # 80 Hz
            
            if max_period < len(autocorr):
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    f0 = sr / peak_idx if peak_idx > 0 else 0
                else:
                    f0 = 0
            else:
                f0 = 0
            
            f0_values.append(f0)
        
        return np.array(f0_values)
    
    def _autocorrelation_manual(self, x: np.ndarray) -> np.ndarray:
        """Manual autocorrelation implementation"""
        n = len(x)
        autocorr = np.zeros(n)
        
        for lag in range(n):
            if lag < n:
                autocorr[lag] = np.sum(x[:-lag if lag > 0 else n] * x[lag:])
        
        return autocorr
    
    def _extract_mfcc_features_manual(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Manual MFCC implementation"""
        # Step 1: Pre-emphasis filter
        pre_emphasized = self._pre_emphasis_manual(y)
        
        # Step 2: STFT
        stft_matrix = self._stft_manual(pre_emphasized)
        power_spectrum = np.abs(stft_matrix)**2
        
        # Step 3: Mel filter bank
        mel_filters = self._create_mel_filter_bank_manual(sr)
        mel_spectrum = np.dot(mel_filters, power_spectrum)
        
        # Step 4: Log
        log_mel_spectrum = np.log(mel_spectrum + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Step 5: DCT
        mfccs = self._dct_manual(log_mel_spectrum)
        
        # Extract statistical features
        features = {}
        for i in range(min(self.n_mfcc, mfccs.shape[0])):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        return features
    
    def _pre_emphasis_manual(self, y: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Manual pre-emphasis filter"""
        return np.append(y[0], y[1:] - alpha * y[:-1])
    
    def _hz_to_mel_manual(self, hz: float) -> float:
        """Convert Hz to Mel scale manually"""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz_manual(self, mel: float) -> float:
        """Convert Mel to Hz scale manually"""
        return 700 * (10**(mel / 2595) - 1)
    
    def _create_mel_filter_bank_manual(self, sr: int) -> np.ndarray:
        """Create Mel filter bank manually"""
        n_fft = self.n_fft
        n_mels = self.n_mels
        
        # Frequency range
        low_freq_mel = 0
        high_freq_mel = self._hz_to_mel_manual(sr / 2)
        
        # Equally spaced in Mel scale
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = np.array([self._mel_to_hz_manual(m) for m in mel_points])
        
        # Convert to FFT bin numbers
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        # Create filter bank
        filter_bank = np.zeros((n_mels, n_fft // 2))
        
        for i in range(1, n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            # Left slope
            for j in range(left, center):
                if center != left:
                    filter_bank[i - 1, j] = (j - left) / (center - left)
            
            # Right slope
            for j in range(center, right):
                if right != center:
                    filter_bank[i - 1, j] = (right - j) / (right - center)
        
        return filter_bank
    
    def _dct_manual(self, x: np.ndarray) -> np.ndarray:
        """Manual Discrete Cosine Transform (Type-II)"""
        n_mels, n_frames = x.shape
        n_mfcc = self.n_mfcc
        
        # DCT matrix
        dct_matrix = np.zeros((n_mfcc, n_mels))
        
        for k in range(n_mfcc):
            for n in range(n_mels):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
        
        # Apply normalization
        dct_matrix[0] *= np.sqrt(1 / n_mels)
        dct_matrix[1:] *= np.sqrt(2 / n_mels)
        
        # Apply DCT
        mfccs = np.dot(dct_matrix, x)
        
        return mfccs
    
    def _extract_formant_features_manual(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract formant features using manual LPC"""
        frame_length = self.n_fft
        hop_length = self.hop_length
        
        formant_freqs = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            
            # Pre-emphasis
            pre_emphasized = self._pre_emphasis_manual(frame)
            
            # Window the frame
            window = self._hamming_window(len(pre_emphasized))
            windowed = pre_emphasized * window
            
            try:
                # Manual LPC analysis
                lpc_coeffs = self._lpc_analysis_manual(windowed, order=12)
                
                # Find formants from LPC coefficients
                formants = self._find_formants_from_lpc_manual(lpc_coeffs, sr)
                formant_freqs.append(formants)
            except:
                formant_freqs.append([0, 0, 0])  # F1, F2, F3
        
        formant_freqs = np.array(formant_freqs)
        
        # Extract formant statistics
        features = {}
        for i in range(3):  # F1, F2, F3
            valid_formants = formant_freqs[:, i]
            valid_formants = valid_formants[valid_formants > 0]
            if len(valid_formants) > 0:
                features[f'formant_f{i+1}_mean'] = float(np.mean(valid_formants))
                features[f'formant_f{i+1}_std'] = float(np.std(valid_formants))
            else:
                features[f'formant_f{i+1}_mean'] = 0.0
                features[f'formant_f{i+1}_std'] = 0.0
        
        return features
    
    def _lpc_analysis_manual(self, signal: np.ndarray, order: int) -> np.ndarray:
        """Manual Linear Prediction Coding analysis using Levinson-Durbin"""
        # Autocorrelation
        R = self._autocorrelation_manual(signal)
        R = R[:order + 1]  # Only need first order+1 values
        
        if len(R) <= order or R[0] == 0:
            return np.zeros(order + 1)
        
        # Levinson-Durbin algorithm
        a = np.zeros(order + 1)
        a[0] = 1.0
        
        E = R[0]  # Prediction error
        
        for i in range(1, order + 1):
            if i < len(R) and E != 0:
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
    
    def _find_formants_from_lpc_manual(self, lpc_coeffs: np.ndarray, sr: int) -> List[float]:
        """Find formant frequencies from LPC coefficients manually"""
        try:
            # Find roots of LPC polynomial
            roots = np.roots(lpc_coeffs)
            
            # Keep only roots inside unit circle with positive imaginary part
            formants = []
            
            for root in roots:
                if abs(root) < 1 and np.imag(root) > 0:
                    # Convert to frequency
                    freq = np.angle(root) * sr / (2 * np.pi)
                    if 200 < freq < 4000:  # Typical formant range
                        formants.append(freq)
            
            # Sort formants
            formants.sort()
            
            # Return first 3 formants, pad with zeros if needed
            while len(formants) < 3:
                formants.append(0)
            
            return formants[:3]
        except:
            return [0, 0, 0]

# Convenience function for gender classification
def extract_gender_relevant_features_manual(audio_path: str) -> Dict[str, float]:
    """
    Extract features most relevant for gender classification using manual DSP
    """
    processor = ManualDSPProcessor()
    all_features = processor.extract_all_features(audio_path)
    
    # Select features most relevant for gender classification
    gender_features = {
        'f0_mean': all_features.get('f0_mean', 0),
        'f0_std': all_features.get('f0_std', 0),
        'spectral_centroid': all_features.get('spectral_centroid', 0),
        'formant_f1_mean': all_features.get('formant_f1_mean', 0),
        'formant_f2_mean': all_features.get('formant_f2_mean', 0),
        'formant_f3_mean': all_features.get('formant_f3_mean', 0),
        'spectral_rolloff': all_features.get('spectral_rolloff', 0),
        'spectral_bandwidth': all_features.get('spectral_bandwidth', 0),
        # Add some MFCC features for additional discrimination
        'mfcc_0_mean': all_features.get('mfcc_0_mean', 0),
        'mfcc_1_mean': all_features.get('mfcc_1_mean', 0),
        'mfcc_2_mean': all_features.get('mfcc_2_mean', 0),
    }
    
    return gender_features
