# services/dsp_preprocessing.py
import os, time, math
import numpy as np
import soundfile as sf
from fractions import Fraction
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.linalg import solve_toeplitz

# --------------------------
# Utilities
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
    return signal.firwin(numtaps, cutoff=fc_hz, fs=fs, window="hamming")

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
    H = fft(h, n)
    w = np.linspace(0, fs/2, n//2, endpoint=False)
    mag = 20*np.log10(np.maximum(np.abs(H[:n//2]), 1e-12))
    phase = np.unwrap(np.angle(H[:n//2]))
    # group delay (approx derivative of phase wrt rad/s -> samples)
    # Use scipy.signal.group_delay for stability on FIR:
    w_gd, gd = signal.group_delay((h, 1.0), fs=fs, whole=False, w=n//2)
    return w, mag, phase, w_gd, gd

def _stft_ola(x: np.ndarray, fs: int, win="hann", nperseg=512, noverlap=256):
    f, t, Z = signal.stft(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, boundary=None)
    _, x_rec = signal.istft(Z, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
    pr_err = np.mean((x[:len(x_rec)] - x_rec)**2)
    return f, t, Z, x_rec, pr_err

def _fft_convolution_speed(x: np.ndarray, h: np.ndarray) -> Tuple[float, float]:
    import time as _t
    t0 = _t.perf_counter(); y1 = np.convolve(x, h, mode="same"); td = _t.perf_counter() - t0
    t0 = _t.perf_counter(); y2 = signal.fftconvolve(x, h, mode="same"); tf = _t.perf_counter() - t0
    return td, tf

def _window_leakage_demo(fs: int, N: int = 1024, f0: float = 440.5) -> Dict[str, Any]:
    n = np.arange(N)
    x = np.sin(2*np.pi*f0*n/fs)
    X_rect = np.abs(fft(x))[:N//2]
    X_hamm = np.abs(fft(x*signal.get_window("hamming", N)))[:N//2]
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

    # 1) Load + mono
    x, fs_in = sf.read(audio_path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    # 2) Silence trimming (energy-based)
    x_trim = _trim_silence(x, fs_in)
    trimmed_seconds = max(0.0, (len(x)-len(x_trim))/fs_in)

    # 3) Anti-alias + rational resample to fs_target (A/D)
    if fs_in != fs_target:
        frac = Fraction(fs_target, fs_in).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        # Anti-alias filter is built into resample_poly
        x_rs = signal.resample_poly(x_trim, up, down, window=('kaiser', 8.0))
        fs = fs_target
    else:
        x_rs = x_trim
        fs = fs_in

    # 4) Pre-emphasis + AGC (static linearity / sinusoidal fidelity aid)
    x_pe = _pre_emphasis(x_rs, alpha=preemph_alpha)
    x_agc = _rms_normalize(x_pe, target_rms=agc_target_rms, limiter=0.99)

    # 5) FIR low-pass demo + frequency response (systems & convolution)
    h = _design_fir_lpf(fc_hz=3400.0, fs=fs, numtaps=129)
    y_filt_time = np.convolve(x_agc, h, mode="same")

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

    f_spec, t_spec, Sxx = signal.spectrogram(x_agc, fs=fs, nperseg=512, noverlap=256)
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
    X = fft(x_agc)
    x_idft = ifft(X).real
    idft_mse = float(np.mean((x_agc - x_idft)**2))

    # Plot: magnitude/phase (real DFT half)
    freqs = fftfreq(N, 1/fs)
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
    analytic = signal.hilbert(x_agc)
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
    imp_resp = signal.fftconvolve(delta, h)[:len(h)]
    step = np.ones(1024)
    step_resp = signal.fftconvolve(step, h)[:1024]
    plt.figure(figsize=(12,4))
    plt.plot(imp_resp); plt.title("Impulse response h[n] via δ * h"); 
    p = os.path.join(out_dir, "impulse_response.png"); _save_plot(p); artifacts.append(p)

    # 11) Linearity test (sinusoidal fidelity): scale input → output scales?
    t_lin = np.arange(0, 0.2, 1/fs)
    s1 = 0.3*np.sin(2*np.pi*1000*t_lin)  # test tone
    y1 = np.convolve(_rms_normalize(_pre_emphasis(s1, preemph_alpha)), h, mode="same")
    s2 = 0.6*np.sin(2*np.pi*1000*t_lin)
    y2 = np.convolve(_rms_normalize(_pre_emphasis(s2, preemph_alpha)), h, mode="same")
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
