# services/dsp_preprocessing.py
import os, time, math
import numpy as np
import soundfile as sf
import librosa
from fractions import Fraction
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --------------------------
# Hardcoded DSP Implementations
# --------------------------

def _hardcoded_fft(x: np.ndarray) -> np.ndarray:
    """Cooley-Tukey FFT implementation"""
    N = len(x)
    if N <= 1:
        return x.astype(complex)
    
    # Pad to next power of 2 if needed
    if N & (N-1) != 0:
        next_pow2 = 1 << (N-1).bit_length()
        x_padded = np.zeros(next_pow2, dtype=complex)
        x_padded[:N] = x
        result = _hardcoded_fft(x_padded)
        return result[:N]
    
    # Recursive FFT
    even = _hardcoded_fft(x[0::2])
    odd = _hardcoded_fft(x[1::2])
    
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + T * odd, even - T * odd])

def _hardcoded_ifft(X: np.ndarray) -> np.ndarray:
    """Inverse FFT using conjugate property"""
    return np.conj(_hardcoded_fft(np.conj(X))) / len(X)

def _hardcoded_hamming_window(N: int) -> np.ndarray:
    """Hamming window implementation"""
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

def _hardcoded_kaiser_window(N: int, beta: float = 8.0) -> np.ndarray:
    """Kaiser window implementation"""
    def i0(x):
        """Modified Bessel function of first kind, order 0"""
        # Series expansion approximation
        result = 1.0
        term = 1.0
        for k in range(1, 50):  # Sufficient terms for convergence
            term *= (x / 2) ** 2 / (k ** 2)
            result += term
            if term < 1e-12:
                break
        return result
    
    n = np.arange(N)
    alpha = (N - 1) / 2
    return np.array([i0(beta * np.sqrt(1 - ((n[i] - alpha) / alpha) ** 2)) / i0(beta) for i in range(N)])

def _hardcoded_firwin(numtaps: int, cutoff: float, fs: float, window: str = "hamming") -> np.ndarray:
    """FIR filter design using windowing method"""
    if numtaps % 2 == 0:
        numtaps += 1  # Make odd for linear phase
    
    # Normalized cutoff frequency
    wc = 2 * np.pi * cutoff / fs
    
    # Design ideal lowpass filter (sinc function)
    n = np.arange(numtaps)
    center = (numtaps - 1) / 2
    h_ideal = np.sinc((n - center) * wc / np.pi) * wc / np.pi
    
    # Apply window
    if window == "hamming":
        w = _hardcoded_hamming_window(numtaps)
    elif window == "kaiser":
        w = _hardcoded_kaiser_window(numtaps)
    else:
        w = np.ones(numtaps)  # rectangular
    
    return h_ideal * w

def _hardcoded_resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Polyphase resampling with anti-aliasing"""
    # Design anti-aliasing filter
    nyquist = 0.5 * min(up, down) / max(up, down)
    h = _hardcoded_firwin(127, nyquist * 0.9, 1.0, "kaiser")
    
    # Upsample by inserting zeros
    x_up = np.zeros(len(x) * up)
    x_up[::up] = x
    
    # Apply anti-aliasing filter
    x_filt = _hardcoded_convolve(x_up, h, mode="same")
    
    # Downsample
    x_down = x_filt[::down] * up  # Compensate for upsampling gain
    
    return x_down

def _hardcoded_convolve(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    """Time-domain convolution"""
    if len(x) == 0 or len(h) == 0:
        return np.array([])
    
    # Pad signals
    N = len(x) + len(h) - 1
    x_pad = np.zeros(N)
    h_pad = np.zeros(N)
    x_pad[:len(x)] = x
    h_pad[:len(h)] = h
    
    # Convolution in time domain
    y = np.zeros(N)
    for n in range(N):
        for k in range(len(h)):
            if 0 <= n - k < len(x):
                y[n] += x[n - k] * h[k]
    
    # Return based on mode
    if mode == "same":
        start = (len(h) - 1) // 2
        return y[start:start + len(x)]
    elif mode == "valid":
        start = len(h) - 1
        end = N - len(h) + 1
        return y[start:end]
    else:  # "full"
        return y

def _hardcoded_fftconvolve(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
    """FFT-based convolution"""
    if len(x) == 0 or len(h) == 0:
        return np.array([])
    
    # Determine output length
    N = len(x) + len(h) - 1
    
    # Zero-pad both signals to same length (power of 2 for efficiency)
    fft_size = 1 << (N - 1).bit_length()
    X = _hardcoded_fft(np.concatenate([x, np.zeros(fft_size - len(x))]))
    H = _hardcoded_fft(np.concatenate([h, np.zeros(fft_size - len(h))]))
    
    # Multiply in frequency domain
    Y = X * H
    
    # Inverse FFT and take real part
    y = _hardcoded_ifft(Y).real[:N]
    
    # Return based on mode
    if mode == "same":
        start = (len(h) - 1) // 2
        return y[start:start + len(x)]
    elif mode == "valid":
        start = len(h) - 1
        end = N - len(h) + 1
        return y[start:end]
    else:  # "full"
        return y

def _hardcoded_spectrogram(x: np.ndarray, fs: int, nperseg: int = 256, noverlap: int = None, window: str = "hann") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT-based spectrogram"""
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Window function
    if window == "hann":
        win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    elif window == "hamming":
        win = _hardcoded_hamming_window(nperseg)
    else:
        win = np.ones(nperseg)
    
    # Calculate number of frames
    hop = nperseg - noverlap
    nframes = 1 + (len(x) - nperseg) // hop
    
    # Initialize spectrogram
    nfreqs = nperseg // 2 + 1
    Sxx = np.zeros((nfreqs, nframes))
    
    # Compute STFT
    for i in range(nframes):
        start = i * hop
        frame = x[start:start + nperseg] * win
        
        # Pad frame if necessary
        if len(frame) < nperseg:
            frame = np.concatenate([frame, np.zeros(nperseg - len(frame))])
        
        # FFT and power spectral density
        X = _hardcoded_fft(frame)
        Sxx[:, i] = np.abs(X[:nfreqs]) ** 2
    
    # Frequency and time axes
    f = np.linspace(0, fs / 2, nfreqs)
    t = np.arange(nframes) * hop / fs
    
    return f, t, Sxx

def _hardcoded_stft(x: np.ndarray, fs: int, nperseg: int = 256, noverlap: int = None, window: str = "hann") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-Time Fourier Transform"""
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Window function
    if window == "hann":
        win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    elif window == "hamming":
        win = _hardcoded_hamming_window(nperseg)
    else:
        win = np.ones(nperseg)
    
    # Calculate parameters
    hop = nperseg - noverlap
    nframes = 1 + (len(x) - nperseg) // hop
    nfreqs = nperseg // 2 + 1
    
    # Initialize STFT matrix
    Z = np.zeros((nfreqs, nframes), dtype=complex)
    
    # Compute STFT
    for i in range(nframes):
        start = i * hop
        frame = x[start:start + nperseg] * win
        
        if len(frame) < nperseg:
            frame = np.concatenate([frame, np.zeros(nperseg - len(frame))])
        
        X = _hardcoded_fft(frame)
        Z[:, i] = X[:nfreqs]
    
    # Frequency and time axes
    f = np.linspace(0, fs / 2, nfreqs)
    t = np.arange(nframes) * hop / fs
    
    return f, t, Z

def _hardcoded_istft(Z: np.ndarray, fs: int, nperseg: int = 256, noverlap: int = None, window: str = "hann") -> np.ndarray:
    """Inverse Short-Time Fourier Transform with overlap-add"""
    nfreqs, nframes = Z.shape
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Window function
    if window == "hann":
        win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    elif window == "hamming":
        win = _hardcoded_hamming_window(nperseg)
    else:
        win = np.ones(nperseg)
    
    # Calculate parameters
    hop = nperseg - noverlap
    x_length = (nframes - 1) * hop + nperseg
    x = np.zeros(x_length)
    norm = np.zeros(x_length)
    
    # Overlap-add reconstruction
    for i in range(nframes):
        start = i * hop
        
        # Reconstruct frame (make two-sided spectrum)
        if nperseg % 2 == 0:
            X_full = np.concatenate([Z[:, i], np.conj(Z[-2:0:-1, i])])
        else:
            X_full = np.concatenate([Z[:, i], np.conj(Z[-1:0:-1, i])])
        
        # IFFT
        frame = _hardcoded_ifft(X_full).real
        
        # Apply window and overlap-add
        windowed_frame = frame * win
        x[start:start + nperseg] += windowed_frame
        norm[start:start + nperseg] += win ** 2
    
    # Normalize
    norm[norm < 1e-10] = 1
    x = x / norm
    
    return x

def _hardcoded_hilbert(x: np.ndarray) -> np.ndarray:
    """Hilbert transform using FFT"""
    N = len(x)
    
    # FFT of input
    X = _hardcoded_fft(x)
    
    # Create Hilbert transform multiplier
    h = np.zeros(N)
    h[0] = 1  # DC component
    if N % 2 == 0:
        h[1:N//2] = 2  # Positive frequencies
        h[N//2] = 1   # Nyquist frequency
    else:
        h[1:(N+1)//2] = 2  # Positive frequencies
    
    # Apply Hilbert transform
    X_hilbert = X * h
    
    # IFFT to get analytic signal
    return _hardcoded_ifft(X_hilbert)

def _hardcoded_group_delay(h: np.ndarray, fs: float, nfft: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group delay of FIR filter"""
    # For FIR filters, group delay is approximately (N-1)/2 samples
    N = len(h)
    group_delay_samples = (N - 1) / 2
    
    # Create frequency axis
    w = np.linspace(0, fs/2, nfft//2, endpoint=False)
    
    # Group delay is constant for linear-phase FIR filters
    gd = np.full_like(w, group_delay_samples)
    
    return w, gd

# --------------------------
# Utilities (keeping original implementations where appropriate)
# --------------------------
def _mk_session_dir(src_path: str) -> str:
    base = os.path.splitext(os.path.basename(src_path))[0]
    ts = int(time.time() * 1000)
    out_dir = os.path.join("artifacts", f"{base}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def _design_fir_lpf(fc_hz: float, fs: float, numtaps: int = 129) -> np.ndarray:
    # Hamming-windowed low-pass (linear-phase FIR)
    return _hardcoded_firwin(numtaps, cutoff=fc_hz, fs=fs, window="hamming")

def _pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y

def _rms_normalize(x: np.ndarray, target_rms: float = 0.1, limiter: float = 0.99) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    g = (target_rms / (rms + 1e-12))
    y = np.clip(x * g, -limiter, limiter)
    return y

def _short_time_energy(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win: 
        return np.array([np.sum(x**2)])
    frames = 1 + (len(x) - win) // hop
    ste = np.zeros(frames)
    for i in range(frames):
        seg = x[i*hop : i*hop + win]
        ste[i] = np.sum(seg**2)
    return ste

def _trim_silence(x: np.ndarray, fs: int, win_ms=25, hop_ms=10, thresh_db=-35) -> np.ndarray:
    win = int(fs * win_ms/1000.0)
    hop = int(fs * hop_ms/1000.0)
    ste = _short_time_energy(x, win, hop)
    ste_db = 10*np.log10(ste + 1e-12)
    thr = np.median(ste_db) + thresh_db  # relative threshold
    mask = (ste_db > thr)
    if not np.any(mask): 
        return x  # nothing to trim
    start_f = np.argmax(mask)
    end_f = len(mask) - 1 - np.argmax(mask[::-1])
    start = max(0, start_f*hop)
    end = min(len(x), end_f*hop + win)
    return x[start:end]

def _uniform_quantize(x: np.ndarray, n_bits: int = 8) -> np.ndarray:
    # mid-tread uniform quantizer in [-1,1]
    x = np.clip(x, -1, 1)
    levels = 2**n_bits
    step = 2.0/(levels - 1)
    q = np.round(x/step)*step
    q = np.clip(q, -1, 1)
    return q

def _snr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    sig_p = np.mean(ref**2) + 1e-12
    err_p = np.mean((ref - approx)**2) + 1e-12
    return 10*np.log10(sig_p/err_p)

def _mu_law_compand(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    x = np.clip(x, -1, 1)
    return np.sign(x)*np.log1p(mu*np.abs(x))/np.log1p(mu)

def _mu_law_expand(y: np.ndarray, mu: float = 255.0) -> np.ndarray:
    return np.sign(y)*( (1+mu)**np.abs(y) - 1 )/mu

def _fft_freqresp(h: np.ndarray, fs: float, n: int = 4096):
    H = _hardcoded_fft(np.concatenate([h, np.zeros(n - len(h))]))
    w = np.linspace(0, fs/2, n//2, endpoint=False)
    mag = 20*np.log10(np.maximum(np.abs(H[:n//2]), 1e-12))
    phase = np.unwrap(np.angle(H[:n//2]))
    # group delay using hardcoded implementation
    w_gd, gd = _hardcoded_group_delay(h, fs, n)
    return w, mag, phase, w_gd, gd

def _stft_ola(x: np.ndarray, fs: int, win="hann", nperseg=512, noverlap=256):
    f, t, Z = _hardcoded_stft(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap)
    x_rec = _hardcoded_istft(Z, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap)
    pr_err = np.mean((x[:len(x_rec)] - x_rec)**2)
    return f, t, Z, x_rec, pr_err

def _fft_convolution_speed(x: np.ndarray, h: np.ndarray) -> Tuple[float, float]:
    import time as _t
    t0 = _t.perf_counter(); y1 = _hardcoded_convolve(x, h, mode="same"); td = _t.perf_counter() - t0
    t0 = _t.perf_counter(); y2 = _hardcoded_fftconvolve(x, h, mode="same"); tf = _t.perf_counter() - t0
    return td, tf

def _window_leakage_demo(fs: int, N: int = 1024, f0: float = 440.5) -> Dict[str, Any]:
    n = np.arange(N)
    x = np.sin(2*np.pi*f0*n/fs)
    X_rect = np.abs(_hardcoded_fft(x))[:N//2]
    X_hamm = np.abs(_hardcoded_fft(x*_hardcoded_hamming_window(N)))[:N//2]
    freqs = np.linspace(0, fs/2, N//2, endpoint=False)
    # estimate mainlobe width & sidelobe level
    main_rect = np.max(X_rect)
    side_rect = np.max(X_rect[ (freqs>f0+50) | (freqs<f0-50) ])
    main_hamm = np.max(X_hamm)
    side_hamm = np.max(X_hamm[ (freqs>f0+50) | (freqs<f0-50) ])
    sll_rect_db = 20*np.log10(side_rect/(main_rect+1e-12)+1e-12)
    sll_hamm_db = 20*np.log10(side_hamm/(main_hamm+1e-12)+1e-12)
    return {
        "freqs": freqs, "X_rect": X_rect, "X_hamm": X_hamm,
        "sidelobe_rect_dB": float(sll_rect_db), "sidelobe_hamm_dB": float(sll_hamm_db)
    }

@dataclass
class DSPReport:
    fs_in: int
    fs_out: int
    trimmed_seconds: float
    preemph_alpha: float
    agc_target_rms: float
    quant_bits: int
    used_mu_law: bool
    sqnr_db: float
    fft_vs_time_conv_ms: Dict[str, float]
    pr_error_stft: float
    idft_recon_mse: float
    linearity_gain_error_db: float
    artifacts: List[str]
    processed_wav: str
    notes: Dict[str, Any]

# --------------------------
# Main preprocessing
# --------------------------
def dsp_preprocess(
    audio_path: str,
    fs_target: int = 16000,
    apply_quantization_for_analysis: bool = True,
    quant_bits: int = 8,
    use_mu_law: bool = True,
    preemph_alpha: float = 0.97,
    agc_target_rms: float = 0.1,
) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Returns (report_dict, processed_wav_path, artifact_file_paths)
    The processed WAV is what you feed into your classifiers.
    """
    out_dir = _mk_session_dir(audio_path)
    artifacts: List[str] = []

    # 1) Load + mono - try soundfile first, fallback to librosa for unsupported formats
    try:
        x, fs_in = sf.read(audio_path)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
    except Exception as e:
        # soundfile failed, try librosa for broader format support (WebM, MP4, etc.)
        try:
            x, fs_in = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e2:
            raise RuntimeError(f"Could not load audio file with soundfile ({e}) or librosa ({e2})")

    # 2) Silence trimming (energy-based)
    x_trim = _trim_silence(x, fs_in)
    trimmed_seconds = max(0.0, (len(x)-len(x_trim))/fs_in)

    # 3) Anti-alias + rational resample to fs_target (A/D)
    if fs_in != fs_target:
        frac = Fraction(fs_target, fs_in).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        # Anti-alias filter is built into resample_poly
        x_rs = _hardcoded_resample_poly(x_trim, up, down)
        fs = fs_target
    else:
        x_rs = x_trim
        fs = fs_in

    # 4) Pre-emphasis + AGC (static linearity / sinusoidal fidelity aid)
    x_pe = _pre_emphasis(x_rs, alpha=preemph_alpha)
    x_agc = _rms_normalize(x_pe, target_rms=agc_target_rms, limiter=0.99)

    # 5) FIR low-pass demo + frequency response (systems & convolution)
    h = _design_fir_lpf(fc_hz=3400.0, fs=fs, numtaps=129)
    y_filt_time = _hardcoded_convolve(x_agc, h, mode="same")

    # FFT vs time convolution speed
    td, tf = _fft_convolution_speed(x_agc, h)

    # Frequency response, phase, group delay
    w, mag_db, phase, w_gd, gd = _fft_freqresp(h, fs, n=4096)

    # Plot: waveform, filtered, spectrogram
    plt.figure(figsize=(12, 6))
    t_axis = np.arange(len(x_agc))/fs
    plt.subplot(2,1,1); plt.plot(t_axis, x_agc); plt.title("Preprocessed waveform (pre-emphasis + AGC)")
    plt.subplot(2,1,2); plt.plot(t_axis, y_filt_time); plt.title("FIR LPF output (time-domain convolution)")
    p = os.path.join(out_dir, "waveforms.png"); _save_plot(p); artifacts.append(p)

    f_spec, t_spec, Sxx = _hardcoded_spectrogram(x_agc, fs=fs, nperseg=512, noverlap=256)
    plt.figure(figsize=(10,5))
    plt.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx+1e-12), shading="gouraud")
    plt.ylabel("Hz"); plt.xlabel("s"); plt.title("Spectrogram (preprocessed)"); 
    p = os.path.join(out_dir, "spectrogram.png"); _save_plot(p); artifacts.append(p)

    # Plot: filter response
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(w, mag_db); plt.xlabel("Hz"); plt.ylabel("Mag (dB)"); plt.title("FIR LPF |H(f)|")
    plt.subplot(1,2,2); plt.plot(w_gd, gd); plt.xlabel("Hz"); plt.ylabel("Samples"); plt.title("Group delay")
    p = os.path.join(out_dir, "filter_response.png"); _save_plot(p); artifacts.append(p)

    # 6) DFT/IDFT – duality & reconstruction error (polar notation)
    N = len(x_agc)
    X = _hardcoded_fft(x_agc)
    x_idft = _hardcoded_ifft(X).real
    idft_mse = float(np.mean((x_agc - x_idft)**2))

    # Plot: magnitude/phase (real DFT half)
    freqs = np.linspace(0, fs, N, endpoint=False)
    half = N//2
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1); plt.plot(freqs[:half], np.abs(X[:half])); plt.title("Magnitude spectrum |X(f)|")
    plt.subplot(2,1,2); plt.plot(freqs[:half], np.unwrap(np.angle(X[:half]))); plt.title("Phase spectrum ∠X(f)")
    p = os.path.join(out_dir, "dft_polar.png"); _save_plot(p); artifacts.append(p)

    # 7) STFT OLA reconstruction (applications of DFT)
    f, t, Z, x_rec, pr_err = _stft_ola(x_agc, fs, win="hann", nperseg=512, noverlap=256)
    plt.figure(figsize=(12,4))
    plt.plot(x_agc[:len(x_rec)], label="original"); plt.plot(x_rec, alpha=0.7, label="OLA recon")
    plt.legend(); plt.title(f"STFT OLA – PR MSE={pr_err:.2e}")
    p = os.path.join(out_dir, "stft_reconstruction.png"); _save_plot(p); artifacts.append(p)

    # 8) Hilbert transform (analytic signal → envelope + inst. freq)
    analytic = _hardcoded_hilbert(x_agc)
    env = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = (np.diff(inst_phase)/(2*np.pi))*fs
    plt.figure(figsize=(12,5))
    plt.subplot(2,1,1); plt.plot(env); plt.title("Hilbert envelope")
    plt.subplot(2,1,2); plt.plot(inst_freq); plt.title("Instantaneous frequency (Hz)")
    p = os.path.join(out_dir, "hilbert.png"); _save_plot(p); artifacts.append(p)

    # 9) Windowing & leakage demo (rect vs hamming)
    leak = _window_leakage_demo(fs=fs, N=1024, f0=440.5)
    plt.figure(figsize=(12,5))
    plt.plot(leak["freqs"], leak["X_rect"], label="Rect"); 
    plt.plot(leak["freqs"], leak["X_hamm"], label="Hamming", alpha=0.8)
    plt.title(f"Window leakage (Rect vs Hamming)\nSLL Rect={leak['sidelobe_rect_dB']:.1f} dB, Hamming={leak['sidelobe_hamm_dB']:.1f} dB")
    plt.legend(); plt.xlabel("Hz")
    p = os.path.join(out_dir, "window_leakage.png"); _save_plot(p); artifacts.append(p)

    # 10) Impulse/step response (delta function & convolution properties)
    delta = np.zeros(1024); delta[0] = 1.0
    imp_resp = _hardcoded_fftconvolve(delta, h)[:len(h)]
    step = np.ones(1024)
    step_resp = _hardcoded_fftconvolve(step, h)[:1024]
    plt.figure(figsize=(12,4))
    plt.plot(imp_resp); plt.title("Impulse response h[n] via δ * h"); 
    p = os.path.join(out_dir, "impulse_response.png"); _save_plot(p); artifacts.append(p)

    # 11) Linearity test (sinusoidal fidelity): scale input → output scales?
    t_lin = np.arange(0, 0.2, 1/fs)
    s1 = 0.3*np.sin(2*np.pi*1000*t_lin)  # test tone
    y1 = _hardcoded_convolve(_rms_normalize(_pre_emphasis(s1, preemph_alpha)), h, mode="same")
    s2 = 0.6*np.sin(2*np.pi*1000*t_lin)
    y2 = _hardcoded_convolve(_rms_normalize(_pre_emphasis(s2, preemph_alpha)), h, mode="same")
    # ideal linear gain between y2 and y1 should be ~2; measure error in dB
    gain_meas = (np.sqrt(np.mean(y2**2))+1e-9)/(np.sqrt(np.mean(y1**2))+1e-9)
    lin_err_db = 20*np.log10(abs(gain_meas/2.0) + 1e-12)
    plt.figure(figsize=(12,4)); 
    plt.plot(y1, label="0.3 tone → y1"); plt.plot(y2, label="0.6 tone → y2", alpha=0.8)
    plt.legend(); plt.title(f"Static linearity test: gain error={lin_err_db:.2f} dB")
    p = os.path.join(out_dir, "linearity.png"); _save_plot(p); artifacts.append(p)

    # 12) Quantization analysis (ADC)
    if apply_quantization_for_analysis:
        x_q_in = x_agc.copy()
        if use_mu_law:
            y_c = _mu_law_compand(x_q_in)
            y_q = _uniform_quantize(y_c, n_bits=quant_bits)
            x_q = _mu_law_expand(y_q)
        else:
            x_q = _uniform_quantize(x_q_in, n_bits=quant_bits)
        sqnr = _snr_db(x_q_in, x_q)
        plt.figure(figsize=(10,4)); 
        plt.step(np.arange(100), x_q[:100], where="mid", label="quantized")
        plt.plot(x_q_in[:100], alpha=0.7, label="original")
        plt.legend(); plt.title(f"Quantization demo (n_bits={quant_bits}, μ-law={use_mu_law}) – SQNR={sqnr:.1f} dB")
        p = os.path.join(out_dir, "quantization.png"); _save_plot(p); artifacts.append(p)
    else:
        sqnr = float("nan")

    # 13) Save the processed signal (feed this to your models)
    processed_wav = os.path.join(out_dir, "processed.wav")
    sf.write(processed_wav, x_agc, fs)

    # Summary/report
    report = DSPReport(
        fs_in=fs_in,
        fs_out=fs,
        trimmed_seconds=float(trimmed_seconds),
        preemph_alpha=preemph_alpha,
        agc_target_rms=agc_target_rms,
        quant_bits=int(quant_bits),
        used_mu_law=bool(use_mu_law and apply_quantization_for_analysis),
        sqnr_db=float(sqnr),
        fft_vs_time_conv_ms={"time_ms": float(1000*td), "fft_ms": float(1000*tf)},
        pr_error_stft=float(pr_err),
        idft_recon_mse=float(idft_mse),
        linearity_gain_error_db=float(lin_err_db),
        artifacts=artifacts,
        processed_wav=processed_wav,
        notes={
            "window_leakage_sidelobes_dB": {
                "rect": float(leak["sidelobe_rect_dB"]),
                "hamming": float(leak["sidelobe_hamm_dB"])
            },
            "filter_taps": len(h)
        }
    )

    # Convert dataclass to plain dict for JSON response
    report_dict = {
        **report.__dict__
    }
    return report_dict, processed_wav, artifacts