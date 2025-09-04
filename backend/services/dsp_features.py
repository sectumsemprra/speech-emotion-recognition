#!/usr/bin/env python3
"""
DSP Feature Extraction Module
Implements various digital signal processing techniques for speech analysis
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class DSPFeatureExtractor:
    """
    Digital Signal Processing Feature Extractor for speech analysis
    Implements various DSP techniques by hand where possible
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def extract_all_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract all DSP features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            features = {}
            
            # Basic signal properties
            features['duration'] = len(y) / sr
            features['sample_rate'] = sr
            
            # Energy-based features
            features.update(self._extract_energy_features(y))
            
            # Spectral features
            features.update(self._extract_spectral_features(y, sr))
            
            # Pitch and fundamental frequency features
            features.update(self._extract_pitch_features(y, sr))
            
            # MFCCs (using librosa but understanding the process)
            features.update(self._extract_mfcc_features(y, sr))
            
            # Formant features
            features.update(self._extract_formant_features(y, sr))
            
            # Harmonic features
            features.update(self._extract_harmonic_features(y, sr))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _extract_energy_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract energy-based features"""
        features = {}
        
        # Root Mean Square Energy
        rms = np.sqrt(np.mean(y**2))
        features['rms_energy'] = float(rms)
        
        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features['zero_crossing_rate'] = float(zcr)
        
        # Energy variation (standard deviation of frame energies)
        frame_length = 2048
        hop_length = 512
        frame_energies = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            energy = np.sum(frame**2)
            frame_energies.append(energy)
        
        features['energy_variance'] = float(np.var(frame_energies))
        features['energy_mean'] = float(np.mean(frame_energies))
        
        return features
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features using FFT and DFT concepts"""
        features = {}
        
        # Compute FFT
        fft = np.fft.fft(y)
        magnitude_spectrum = np.abs(fft)
        power_spectrum = magnitude_spectrum**2
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        
        # Only use positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude_spectrum[:len(magnitude_spectrum)//2]
        positive_power = power_spectrum[:len(power_spectrum)//2]
        
        # Spectral Centroid (center of mass of spectrum)
        spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
        features['spectral_centroid'] = float(spectral_centroid)
        
        # Spectral Rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(positive_power)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = 0.85 * total_energy
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0][0]
        spectral_rolloff = positive_freqs[rolloff_idx]
        features['spectral_rolloff'] = float(spectral_rolloff)
        
        # Spectral Bandwidth (weighted standard deviation around spectral centroid)
        spectral_bandwidth = np.sqrt(
            np.sum(((positive_freqs - spectral_centroid)**2) * positive_magnitude) / 
            np.sum(positive_magnitude)
        )
        features['spectral_bandwidth'] = float(spectral_bandwidth)
        
        # Spectral Flux (measure of how quickly the power spectrum changes)
        stft = librosa.stft(y)
        spectral_flux = np.mean(np.diff(np.abs(stft), axis=1)**2)
        features['spectral_flux'] = float(spectral_flux)
        
        return features
    
    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract pitch and fundamental frequency features"""
        features = {}
        
        # Fundamental frequency using autocorrelation
        f0_autocorr = self._estimate_f0_autocorr(y, sr)
        features['f0_mean'] = float(np.mean(f0_autocorr[f0_autocorr > 0]))
        features['f0_std'] = float(np.std(f0_autocorr[f0_autocorr > 0]))
        features['f0_min'] = float(np.min(f0_autocorr[f0_autocorr > 0]))
        features['f0_max'] = float(np.max(f0_autocorr[f0_autocorr > 0]))
        
        # Pitch variation (jitter)
        valid_f0 = f0_autocorr[f0_autocorr > 0]
        if len(valid_f0) > 1:
            jitter = np.mean(np.abs(np.diff(valid_f0)) / valid_f0[:-1])
            features['jitter'] = float(jitter)
        else:
            features['jitter'] = 0.0
        
        return features
    
    def _estimate_f0_autocorr(self, y: np.ndarray, sr: int, 
                             frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Estimate F0 using autocorrelation method"""
        f0_values = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i + frame_length]
            
            # Apply window
            windowed = frame * np.hanning(len(frame))
            
            # Autocorrelation
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peak in expected F0 range (80-400 Hz)
            min_period = int(sr / 400)  # 400 Hz
            max_period = int(sr / 80)   # 80 Hz
            
            if max_period < len(autocorr):
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                f0 = sr / peak_idx if peak_idx > 0 else 0
            else:
                f0 = 0
            
            f0_values.append(f0)
        
        return np.array(f0_values)
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract MFCC features"""
        # Use librosa for MFCCs but understand the process:
        # 1. Pre-emphasis filter
        # 2. Windowing and FFT
        # 3. Mel filter bank
        # 4. Logarithm
        # 5. DCT
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        features = {}
        # Statistical measures of MFCCs
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        return features
    
    def _extract_formant_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract formant frequencies using Linear Prediction Coding (LPC)"""
        features = {}
        
        try:
            # Frame the signal
            frame_length = 2048
            hop_length = 512
            
            formant_freqs = []
            
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                
                # Pre-emphasis
                pre_emphasized = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
                
                # Window the frame
                windowed = pre_emphasized * np.hanning(len(pre_emphasized))
                
                # LPC analysis
                try:
                    # Use scipy's LPC implementation
                    lpc_coeffs = self._lpc_analysis(windowed, order=12)
                    
                    # Find formants from LPC coefficients
                    formants = self._find_formants_from_lpc(lpc_coeffs, sr)
                    formant_freqs.append(formants)
                except:
                    formant_freqs.append([0, 0, 0])  # F1, F2, F3
            
            formant_freqs = np.array(formant_freqs)
            
            # Extract formant statistics
            for i in range(3):  # F1, F2, F3
                valid_formants = formant_freqs[:, i]
                valid_formants = valid_formants[valid_formants > 0]
                if len(valid_formants) > 0:
                    features[f'formant_f{i+1}_mean'] = float(np.mean(valid_formants))
                    features[f'formant_f{i+1}_std'] = float(np.std(valid_formants))
                else:
                    features[f'formant_f{i+1}_mean'] = 0.0
                    features[f'formant_f{i+1}_std'] = 0.0
        
        except Exception as e:
            logger.warning(f"Error extracting formants: {e}")
            # Set default values
            for i in range(3):
                features[f'formant_f{i+1}_mean'] = 0.0
                features[f'formant_f{i+1}_std'] = 0.0
        
        return features
    
    def _lpc_analysis(self, signal: np.ndarray, order: int) -> np.ndarray:
        """Linear Prediction Coding analysis"""
        # Autocorrelation method
        R = np.correlate(signal, signal, mode='full')
        R = R[len(R)//2:]
        
        # Solve Yule-Walker equations using Levinson-Durbin algorithm
        if len(R) <= order:
            return np.zeros(order + 1)
        
        # Simple implementation of Levinson-Durbin
        a = np.zeros(order + 1)
        a[0] = 1.0
        
        for i in range(1, order + 1):
            if i < len(R):
                k = -R[i] / R[0] if R[0] != 0 else 0
                a[i] = k
                for j in range(1, i):
                    a[j] = a[j] + k * a[i-j]
        
        return a
    
    def _find_formants_from_lpc(self, lpc_coeffs: np.ndarray, sr: int) -> List[float]:
        """Find formant frequencies from LPC coefficients"""
        # Find roots of LPC polynomial
        roots = np.roots(lpc_coeffs)
        
        # Keep only roots inside unit circle and with positive imaginary part
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
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract harmonic-related features"""
        features = {}
        
        # Harmonic-to-noise ratio
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate HNR
        harmonic_energy = np.sum(y_harmonic**2)
        noise_energy = np.sum(y_percussive**2)
        
        if noise_energy > 0:
            hnr = 10 * np.log10(harmonic_energy / noise_energy)
        else:
            hnr = float('inf')
        
        features['harmonic_noise_ratio'] = float(hnr) if hnr != float('inf') else 50.0
        
        # Spectral harmonicity
        features['harmonicity'] = float(harmonic_energy / (harmonic_energy + noise_energy))
        
        return features

def extract_gender_relevant_features(audio_path: str) -> Dict[str, float]:
    """
    Extract features most relevant for gender classification
    """
    extractor = DSPFeatureExtractor()
    all_features = extractor.extract_all_features(audio_path)
    
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
        'harmonicity': all_features.get('harmonicity', 0),
        'harmonic_noise_ratio': all_features.get('harmonic_noise_ratio', 0)
    }
    
    return gender_features
