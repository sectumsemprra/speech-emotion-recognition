#!/usr/bin/env python3
"""
Manual DSP Core Module
Hand-written implementations using course fundamentals: DFT, IDFT, Convolution, Sin/Cos
With optional FFT acceleration for performance
"""

import math
import numpy as np
from typing import List, Tuple, Union, Optional

class ManualDSPCore:
    """
    Hand-written DSP implementations using fundamental course concepts
    """
    
    @staticmethod
    def dft_real(x: List[float]) -> Tuple[List[float], List[float]]:
        """
        Discrete Fourier Transform - Real implementation using sin/cos
        
        X_real[k] = Σ(n=0 to N-1) x[n] * cos(2πkn/N)
        X_imag[k] = -Σ(n=0 to N-1) x[n] * sin(2πkn/N)
        
        Args:
            x: Real-valued input signal
            
        Returns:
            Tuple of (real_part, imaginary_part) of DFT coefficients
        """
        N = len(x)
        X_real = []
        X_imag = []
        
        for k in range(N):
            real_sum = 0.0
            imag_sum = 0.0
            
            for n in range(N):
                angle = 2 * math.pi * k * n / N
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                
                real_sum += x[n] * cos_val
                imag_sum += x[n] * (-sin_val)  # Negative for forward DFT
            
            X_real.append(real_sum)
            X_imag.append(imag_sum)
            
        return X_real, X_imag
    
    @staticmethod
    def idft_real(X_real: List[float], X_imag: List[float]) -> List[float]:
        """
        Inverse Discrete Fourier Transform - Real implementation using sin/cos
        
        x[n] = (1/N) * Σ(k=0 to N-1) [X_real[k]*cos(2πkn/N) - X_imag[k]*sin(2πkn/N)]
        
        Args:
            X_real: Real part of DFT coefficients
            X_imag: Imaginary part of DFT coefficients
            
        Returns:
            Real-valued time domain signal
        """
        N = len(X_real)
        x = []
        
        for n in range(N):
            sum_val = 0.0
            
            for k in range(N):
                angle = 2 * math.pi * k * n / N
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                
                # IDFT formula: real*cos - imag*sin
                sum_val += X_real[k] * cos_val - X_imag[k] * sin_val
            
            x.append(sum_val / N)
            
        return x
    
    @staticmethod
    def magnitude_spectrum(X_real: List[float], X_imag: List[float]) -> List[float]:
        """
        Calculate magnitude spectrum from real and imaginary parts
        
        |X[k]| = sqrt(X_real[k]² + X_imag[k]²)
        
        Args:
            X_real: Real part of DFT
            X_imag: Imaginary part of DFT
            
        Returns:
            Magnitude spectrum
        """
        magnitude = []
        for real, imag in zip(X_real, X_imag):
            mag = math.sqrt(real**2 + imag**2)
            magnitude.append(mag)
        return magnitude
    
    @staticmethod
    def phase_spectrum(X_real: List[float], X_imag: List[float]) -> List[float]:
        """
        Calculate phase spectrum from real and imaginary parts
        
        ∠X[k] = atan2(X_imag[k], X_real[k])
        
        Args:
            X_real: Real part of DFT
            X_imag: Imaginary part of DFT
            
        Returns:
            Phase spectrum in radians
        """
        phase = []
        for real, imag in zip(X_real, X_imag):
            ph = math.atan2(imag, real)
            phase.append(ph)
        return phase
    
    @staticmethod
    def power_spectrum(X_real: List[float], X_imag: List[float]) -> List[float]:
        """
        Calculate power spectrum from real and imaginary parts
        
        |X[k]|² = X_real[k]² + X_imag[k]²
        
        Args:
            X_real: Real part of DFT
            X_imag: Imaginary part of DFT
            
        Returns:
            Power spectrum
        """
        power = []
        for real, imag in zip(X_real, X_imag):
            pwr = real**2 + imag**2
            power.append(pwr)
        return power
    
    @staticmethod
    def hamming_window(N: int) -> List[float]:
        """
        Hamming window function
        
        w[n] = 0.54 - 0.46 * cos(2πn/(N-1))
        
        Args:
            N: Window length
            
        Returns:
            Hamming window coefficients
        """
        window = []
        for n in range(N):
            w = 0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1))
            window.append(w)
        return window
    
    @staticmethod
    def hann_window(N: int) -> List[float]:
        """
        Hann (Hanning) window function
        
        w[n] = 0.5 * (1 - cos(2πn/(N-1)))
        
        Args:
            N: Window length
            
        Returns:
            Hann window coefficients
        """
        window = []
        for n in range(N):
            w = 0.5 * (1 - math.cos(2 * math.pi * n / (N - 1)))
            window.append(w)
        return window
    
    @staticmethod
    def blackman_window(N: int) -> List[float]:
        """
        Blackman window function
        
        w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
        
        Args:
            N: Window length
            
        Returns:
            Blackman window coefficients
        """
        window = []
        for n in range(N):
            w = (0.42 - 0.5 * math.cos(2 * math.pi * n / (N - 1)) + 
                 0.08 * math.cos(4 * math.pi * n / (N - 1)))
            window.append(w)
        return window
    
    @staticmethod
    def apply_window(signal: List[float], window: List[float]) -> List[float]:
        """
        Apply window function to signal
        
        Args:
            signal: Input signal
            window: Window coefficients
            
        Returns:
            Windowed signal
        """
        if len(signal) != len(window):
            raise ValueError("Signal and window must have same length")
        
        return [s * w for s, w in zip(signal, window)]
    
    @staticmethod
    def autocorrelation(x: List[float], max_lag: Optional[int] = None) -> List[float]:
        """
        Autocorrelation function using time-domain computation
        
        R[m] = Σ(n=0 to N-1-m) x[n] * x[n+m]
        
        Args:
            x: Input signal
            max_lag: Maximum lag to compute (if None, use len(x)-1)
            
        Returns:
            Autocorrelation coefficients
        """
        N = len(x)
        if max_lag is None:
            max_lag = N - 1
        
        autocorr = []
        for m in range(max_lag + 1):
            sum_val = 0
            for n in range(N - m):
                sum_val += x[n] * x[n + m]
            autocorr.append(sum_val)
        
        return autocorr
    
    @staticmethod
    def cross_correlation(x: List[float], y: List[float]) -> List[float]:
        """
        Cross-correlation between two signals
        
        Args:
            x: First signal
            y: Second signal
            
        Returns:
            Cross-correlation coefficients
        """
        N = len(x)
        M = len(y)
        max_lag = N + M - 1
        
        xcorr = []
        for lag in range(-M + 1, N):
            sum_val = 0
            for n in range(max(0, lag), min(N, M + lag)):
                if 0 <= n - lag < M:
                    sum_val += x[n] * y[n - lag]
            xcorr.append(sum_val)
        
        return xcorr
    
    @staticmethod
    def butter_coefficients(cutoff: float, fs: float, order: int, btype: str = 'lowpass') -> Tuple[List[float], List[float]]:
        """
        Generate Butterworth filter coefficients
        
        Args:
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
            btype: Filter type ('lowpass', 'highpass', 'bandpass')
            
        Returns:
            Tuple of (numerator, denominator) coefficients
        """
        # Normalize frequency
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        
        # For simplicity, implement basic 2nd order sections
        if btype == 'lowpass':
            # Simple 1st order lowpass (for demonstration)
            if order == 1:
                alpha = math.exp(-2 * math.pi * normal_cutoff)
                b = [1 - alpha]
                a = [1, -alpha]
                return b, a
        
        # For higher orders and other types, would need more complex implementation
        # This is a simplified version for demonstration
        raise NotImplementedError(f"Filter type {btype} with order {order} not implemented")
    
    @staticmethod
    def filter_signal(signal: List[float], b: List[float], a: List[float]) -> List[float]:
        """
        Apply digital filter to signal using difference equation
        
        y[n] = Σ(b[k]*x[n-k]) - Σ(a[k]*y[n-k])
        
        Args:
            signal: Input signal
            b: Numerator coefficients
            a: Denominator coefficients
            
        Returns:
            Filtered signal
        """
        N = len(signal)
        filtered = [0.0] * N
        
        for n in range(N):
            # Numerator part (feedforward)
            for k in range(len(b)):
                if n - k >= 0:
                    filtered[n] += b[k] * signal[n - k]
            
            # Denominator part (feedback) - skip a[0] which should be 1
            for k in range(1, len(a)):
                if n - k >= 0:
                    filtered[n] -= a[k] * filtered[n - k]
        
        return filtered
    
    @staticmethod
    def convolution(x: List[float], h: List[float]) -> List[float]:
        """
        Linear convolution of two signals - Course fundamental approach
        
        y[n] = Σ(k=0 to M-1) h[k] * x[n-k]  (for n = 0, 1, ..., N+M-2)
        
        This is the basic convolution formula taught in DSP courses.
        
        Args:
            x: Input signal (length N)
            h: Impulse response (length M)
            
        Returns:
            Convolved signal (length N+M-1)
        """
        N = len(x)
        M = len(h)
        y = [0.0] * (N + M - 1)
        
        # Direct implementation of convolution sum
        for n in range(len(y)):
            for k in range(M):
                if 0 <= n - k < N:
                    y[n] += h[k] * x[n - k]
        
        return y
    
    @staticmethod
    def circular_convolution(x: List[float], h: List[float]) -> List[float]:
        """
        Circular convolution - another course fundamental
        
        y[n] = Σ(k=0 to N-1) h[k] * x[(n-k) mod N]
        
        Args:
            x: Input signal
            h: Impulse response (same length as x)
            
        Returns:
            Circularly convolved signal (same length as input)
        """
        N = len(x)
        if len(h) != N:
            # Pad or truncate h to match x length
            if len(h) < N:
                h = h + [0.0] * (N - len(h))
            else:
                h = h[:N]
        
        y = [0.0] * N
        
        for n in range(N):
            for k in range(N):
                # Circular indexing: (n-k) mod N
                idx = (n - k) % N
                y[n] += h[k] * x[idx]
        
        return y
    
    @staticmethod
    def convolution_using_dft(x: List[float], h: List[float]) -> List[float]:
        """
        Convolution using DFT - demonstrates convolution theorem
        
        Convolution in time domain = Multiplication in frequency domain
        conv(x, h) = IDFT(DFT(x) * DFT(h))
        
        This shows the fundamental relationship taught in courses.
        
        Args:
            x: Input signal
            h: Impulse response
            
        Returns:
            Convolved signal
        """
        N = len(x)
        M = len(h)
        L = N + M - 1  # Length for linear convolution
        
        # Zero-pad both signals to length L
        x_padded = x + [0.0] * (L - N)
        h_padded = h + [0.0] * (L - M)
        
        # Compute DFTs
        X_real, X_imag = ManualDSPCore.dft_real(x_padded)
        H_real, H_imag = ManualDSPCore.dft_real(h_padded)
        
        # Multiply in frequency domain (complex multiplication)
        Y_real = []
        Y_imag = []
        for i in range(L):
            # (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
            real_part = X_real[i] * H_real[i] - X_imag[i] * H_imag[i]
            imag_part = X_real[i] * H_imag[i] + X_imag[i] * H_real[i]
            Y_real.append(real_part)
            Y_imag.append(imag_part)
        
        # Inverse DFT to get result
        y = ManualDSPCore.idft_real(Y_real, Y_imag)
        
        return y
    
    @staticmethod
    def dft_fast(x: List[float], manual: bool = False) -> Tuple[List[float], List[float]]:
        """
        Fast DFT implementation with manual/FFT option
        
        Args:
            x: Real-valued input signal
            manual: If True, use manual sin/cos implementation. If False, use FFT
            
        Returns:
            Tuple of (real_part, imaginary_part) of DFT coefficients
        """
        if manual:
            return ManualDSPCore.dft_real(x)
        else:
            # Use numpy FFT for speed
            X_complex = np.fft.fft(x)
            X_real = X_complex.real.tolist()
            X_imag = X_complex.imag.tolist()
            return X_real, X_imag
    
    @staticmethod
    def idft_fast(X_real: List[float], X_imag: List[float], manual: bool = False) -> List[float]:
        """
        Fast IDFT implementation with manual/FFT option
        
        Args:
            X_real: Real part of DFT coefficients
            X_imag: Imaginary part of DFT coefficients
            manual: If True, use manual sin/cos implementation. If False, use IFFT
            
        Returns:
            Real-valued time domain signal
        """
        if manual:
            return ManualDSPCore.idft_real(X_real, X_imag)
        else:
            # Use numpy IFFT for speed
            X_complex = np.array(X_real) + 1j * np.array(X_imag)
            x_complex = np.fft.ifft(X_complex)
            return x_complex.real.tolist()
    
    @staticmethod
    def spectral_centroid_fast(signal: List[float], fs: float, manual: bool = False) -> float:
        """
        Fast spectral centroid calculation with manual/FFT option
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            manual: If True, use manual DFT. If False, use FFT
            
        Returns:
            Spectral centroid in Hz
        """
        if manual:
            X_real, X_imag = ManualDSPCore.dft_real(signal)
            return ManualDSPCore.spectral_centroid_from_dft(X_real, X_imag, fs)
        else:
            # Fast FFT-based implementation
            X = np.fft.rfft(signal)
            magnitude = np.abs(X)
            freqs = np.fft.rfftfreq(len(signal), 1/fs)
            
            # Calculate spectral centroid
            power_spectrum = magnitude ** 2
            weighted_sum = np.sum(freqs * power_spectrum)
            total_power = np.sum(power_spectrum)
            
            if total_power == 0:
                return 0.0
            
            return float(weighted_sum / total_power)
    
    @staticmethod
    def spectral_rolloff_fast(signal: List[float], fs: float, rolloff_percent: float = 0.85, manual: bool = False) -> float:
        """
        Fast spectral rolloff calculation with manual/FFT option
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            rolloff_percent: Percentage of total energy (default 85%)
            manual: If True, use manual DFT. If False, use FFT
            
        Returns:
            Rolloff frequency in Hz
        """
        if manual:
            X_real, X_imag = ManualDSPCore.dft_real(signal)
            return ManualDSPCore.spectral_rolloff_from_dft(X_real, X_imag, fs, rolloff_percent)
        else:
            # Fast FFT-based implementation
            X = np.fft.rfft(signal)
            power_spectrum = np.abs(X) ** 2
            freqs = np.fft.rfftfreq(len(signal), 1/fs)
            
            total_energy = np.sum(power_spectrum)
            if total_energy == 0:
                return 0.0
            
            # Find frequency where cumulative energy reaches threshold
            cumulative_energy = np.cumsum(power_spectrum)
            threshold = rolloff_percent * total_energy
            
            rolloff_idx = np.where(cumulative_energy >= threshold)[0]
            if len(rolloff_idx) > 0:
                return float(freqs[rolloff_idx[0]])
            else:
                return float(freqs[-1])
    
    @staticmethod
    def autocorrelation_fast(x: List[float], max_lag: Optional[int] = None, manual: bool = False) -> List[float]:
        """
        Fast autocorrelation with manual/FFT option
        
        Args:
            x: Input signal
            max_lag: Maximum lag to compute
            manual: If True, use manual time-domain. If False, use FFT-based
            
        Returns:
            Autocorrelation coefficients
        """
        if manual:
            return ManualDSPCore.autocorrelation(x, max_lag)
        else:
            # Fast FFT-based autocorrelation
            x_array = np.array(x)
            N = len(x_array)
            
            # Pad with zeros to avoid circular correlation
            x_padded = np.concatenate([x_array, np.zeros(N)])
            
            # FFT-based autocorrelation
            X = np.fft.fft(x_padded)
            autocorr_full = np.fft.ifft(X * np.conj(X)).real
            
            # Take only positive lags and trim to max_lag
            autocorr = autocorr_full[:N]
            
            if max_lag is not None:
                autocorr = autocorr[:max_lag + 1]
            
            return autocorr.tolist()
    
    @staticmethod
    def convolution_fast(x: List[float], h: List[float], manual: bool = False) -> List[float]:
        """
        Fast convolution with manual/FFT option
        
        Args:
            x: Input signal
            h: Impulse response
            manual: If True, use manual time-domain. If False, use FFT-based
            
        Returns:
            Convolved signal
        """
        if manual:
            return ManualDSPCore.convolution(x, h)
        else:
            # Fast FFT-based convolution
            return np.convolve(x, h, mode='full').tolist()
    
    @staticmethod
    def spectral_centroid_from_dft(X_real: List[float], X_imag: List[float], fs: float) -> float:
        """
        Calculate spectral centroid using DFT results
        
        Centroid = Σ(f[k] * |X[k]|²) / Σ(|X[k]|²)
        
        Args:
            X_real: Real part of DFT
            X_imag: Imaginary part of DFT
            fs: Sampling frequency
            
        Returns:
            Spectral centroid in Hz
        """
        N = len(X_real)
        
        # Generate frequency bins
        freqs = [k * fs / N for k in range(N // 2 + 1)]  # Only positive frequencies
        
        # Calculate power spectrum (only positive frequencies)
        power_spectrum = []
        for k in range(N // 2 + 1):
            power = X_real[k]**2 + X_imag[k]**2
            power_spectrum.append(power)
        
        # Calculate weighted sum
        weighted_sum = sum(f * p for f, p in zip(freqs, power_spectrum))
        total_power = sum(power_spectrum)
        
        if total_power == 0:
            return 0.0
        
        return weighted_sum / total_power
    
    @staticmethod
    def spectral_rolloff_from_dft(X_real: List[float], X_imag: List[float], fs: float, rolloff_percent: float = 0.85) -> float:
        """
        Calculate spectral rolloff frequency using DFT results
        
        Args:
            X_real: Real part of DFT
            X_imag: Imaginary part of DFT
            fs: Sampling frequency
            rolloff_percent: Percentage of total energy (default 85%)
            
        Returns:
            Rolloff frequency in Hz
        """
        N = len(X_real)
        
        # Generate frequency bins (only positive frequencies)
        freqs = [k * fs / N for k in range(N // 2 + 1)]
        
        # Calculate power spectrum (only positive frequencies)
        power_spectrum = []
        for k in range(N // 2 + 1):
            power = X_real[k]**2 + X_imag[k]**2
            power_spectrum.append(power)
        
        total_energy = sum(power_spectrum)
        
        if total_energy == 0:
            return 0.0
        
        # Find frequency where cumulative energy reaches threshold
        cumulative_energy = 0
        threshold = rolloff_percent * total_energy
        
        for i, power in enumerate(power_spectrum):
            cumulative_energy += power
            if cumulative_energy >= threshold:
                return freqs[i]
        
        return freqs[-1]  # Return highest frequency if threshold not reached
    
    @staticmethod
    def find_peaks(signal: List[float], height: Optional[float] = None, distance: int = 1) -> List[int]:
        """
        Find peaks in signal
        
        Args:
            signal: Input signal
            height: Minimum peak height
            distance: Minimum distance between peaks
            
        Returns:
            List of peak indices
        """
        peaks = []
        
        for i in range(1, len(signal) - 1):
            # Check if it's a local maximum
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                # Check height threshold
                if height is None or signal[i] >= height:
                    peaks.append(i)
        
        # Apply distance constraint
        if distance > 1:
            filtered_peaks = []
            for peak in peaks:
                # Check if peak is far enough from already selected peaks
                too_close = False
                for selected_peak in filtered_peaks:
                    if abs(peak - selected_peak) < distance:
                        too_close = True
                        break
                
                if not too_close:
                    filtered_peaks.append(peak)
            
            peaks = filtered_peaks
        
        return peaks
    
    @staticmethod
    def smooth_signal(signal: List[float], window_size: int) -> List[float]:
        """
        Smooth signal using moving average
        
        Args:
            signal: Input signal
            window_size: Size of smoothing window
            
        Returns:
            Smoothed signal
        """
        if window_size < 1:
            return signal
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window + 1)
            
            window_sum = sum(signal[start:end])
            window_length = end - start
            
            smoothed.append(window_sum / window_length)
        
        return smoothed
    
    @staticmethod
    def zero_crossing_rate(signal: List[float]) -> float:
        """
        Calculate zero crossing rate
        
        Args:
            signal: Input signal
            
        Returns:
            Zero crossing rate (crossings per sample)
        """
        if len(signal) < 2:
            return 0.0
        
        crossings = 0
        for i in range(1, len(signal)):
            if (signal[i-1] >= 0) != (signal[i] >= 0):
                crossings += 1
        
        return crossings / (len(signal) - 1)
    
    @staticmethod
    def rms_energy(signal: List[float]) -> float:
        """
        Calculate RMS (Root Mean Square) energy
        
        Args:
            signal: Input signal
            
        Returns:
            RMS energy
        """
        if not signal:
            return 0.0
        
        sum_squares = sum(x ** 2 for x in signal)
        return math.sqrt(sum_squares / len(signal))

# Convenience functions for course-based DSP
def frequency_bins(N: int, fs: float) -> List[float]:
    """Generate frequency bins for DFT"""
    return [k * fs / N for k in range(N // 2 + 1)]

def db_scale(magnitude: List[float]) -> List[float]:
    """Convert magnitude to dB scale"""
    db = []
    for mag in magnitude:
        if mag > 0:
            db.append(20 * math.log10(mag))
        else:
            db.append(-100)  # Very small value for log(0)
    return db
