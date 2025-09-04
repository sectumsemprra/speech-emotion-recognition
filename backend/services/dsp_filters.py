#!/usr/bin/env python3
"""
DSP Filters Utility
Provides audio preprocessing filters using scipy for speech emotion recognition
"""

import numpy as np
import logging
from scipy import signal
from scipy.io import wavfile
import tempfile
import os
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return audio data and sample rate
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Try scipy first (supports WAV)
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Convert to float and normalize if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        return audio_data, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio with scipy: {e}")
        
        # Fallback: try with librosa if available
        try:
            import librosa
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
            return audio_data, sample_rate
        except ImportError:
            raise RuntimeError("Could not load audio. Install librosa for better format support: pip install librosa")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {e}")

def save_audio(audio_data: np.ndarray, sample_rate: int, output_path: str) -> None:
    """
    Save audio data to file
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        output_path: Output file path
    """
    # Convert float to int16 for saving
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        # Clip to prevent overflow
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
    else:
        audio_int16 = audio_data.astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio_int16)

def design_butterworth_filter(
    cutoff: Union[float, Tuple[float, float]], 
    sample_rate: int, 
    filter_type: str = 'band', 
    order: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth filter
    
    Args:
        cutoff: Cutoff frequency(ies) in Hz
                - For lowpass/highpass: single float
                - For bandpass/bandstop: tuple of (low, high)
        sample_rate: Sample rate in Hz
        filter_type: 'low', 'high', 'band', or 'bandstop'
        order: Filter order (higher = steeper rolloff)
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyquist = sample_rate / 2.0
    
    if filter_type in ['band', 'bandstop']:
        if not isinstance(cutoff, (tuple, list)) or len(cutoff) != 2:
            raise ValueError("Band filters require cutoff as tuple (low_freq, high_freq)")
        low_freq, high_freq = cutoff
        if low_freq >= high_freq:
            raise ValueError("Low frequency must be less than high frequency")
        if high_freq >= nyquist:
            raise ValueError(f"High frequency ({high_freq}) must be less than Nyquist frequency ({nyquist})")
        
        normalized_cutoff = [low_freq / nyquist, high_freq / nyquist]
    else:
        if isinstance(cutoff, (tuple, list)):
            cutoff = cutoff[0]
        if cutoff >= nyquist:
            raise ValueError(f"Cutoff frequency ({cutoff}) must be less than Nyquist frequency ({nyquist})")
        
        normalized_cutoff = cutoff / nyquist
    
    b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return b, a

def apply_butterworth_filter(
    audio_data: np.ndarray, 
    cutoff: Union[float, Tuple[float, float]], 
    sample_rate: int,
    filter_type: str = 'band', 
    order: int = 5
) -> np.ndarray:
    """
    Apply Butterworth filter to audio data
    
    Args:
        audio_data: Input audio samples
        cutoff: Cutoff frequency(ies) in Hz
        sample_rate: Sample rate in Hz
        filter_type: 'low', 'high', 'band', or 'bandstop'
        order: Filter order
        
    Returns:
        Filtered audio data
    """
    if len(audio_data) == 0:
        return audio_data
    
    try:
        b, a = design_butterworth_filter(cutoff, sample_rate, filter_type, order)
        filtered_audio = signal.filtfilt(b, a, audio_data)
        return filtered_audio
        
    except Exception as e:
        logger.error(f"Filter application failed: {e}")
        logger.warning("Returning original audio data")
        return audio_data

def lowpass_filter(audio_data: np.ndarray, cutoff: float, sample_rate: int, order: int = 5) -> np.ndarray:
    """Apply low-pass filter"""
    return apply_butterworth_filter(audio_data, cutoff, sample_rate, 'low', order)

def highpass_filter(audio_data: np.ndarray, cutoff: float, sample_rate: int, order: int = 5) -> np.ndarray:
    """Apply high-pass filter"""
    return apply_butterworth_filter(audio_data, cutoff, sample_rate, 'high', order)

def bandpass_filter(
    audio_data: np.ndarray, 
    low_cutoff: float, 
    high_cutoff: float, 
    sample_rate: int, 
    order: int = 5
) -> np.ndarray:
    """Apply band-pass filter"""
    return apply_butterworth_filter(audio_data, (low_cutoff, high_cutoff), sample_rate, 'band', order)

def bandstop_filter(
    audio_data: np.ndarray, 
    low_cutoff: float, 
    high_cutoff: float, 
    sample_rate: int, 
    order: int = 5
) -> np.ndarray:
    """Apply band-stop filter"""
    return apply_butterworth_filter(audio_data, (low_cutoff, high_cutoff), sample_rate, 'bandstop', order)

def preprocess_audio_file(
    input_path: str,
    filter_type: str = 'bandpass',
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0,
    cutoff: Optional[float] = None,
    order: int = 5,
    output_path: Optional[str] = None
) -> str:
    """
    Preprocess audio file with DSP filtering
    
    Args:
        input_path: Path to input audio file
        filter_type: 'bandpass', 'lowpass', 'highpass', or 'none'
        low_cutoff: Low cutoff frequency for bandpass (Hz)
        high_cutoff: High cutoff frequency for bandpass (Hz)
        cutoff: Single cutoff frequency for lowpass/highpass (Hz)
        order: Filter order
        output_path: Output file path (if None, creates temp file)
        
    Returns:
        Path to processed audio file
    """
    try:
        # Load audio
        audio_data, sample_rate = load_audio(input_path)
        logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
        
        # Apply filtering
        if filter_type.lower() == 'bandpass':
            logger.info(f"Applying bandpass filter: {low_cutoff}-{high_cutoff} Hz")
            filtered_audio = bandpass_filter(audio_data, low_cutoff, high_cutoff, sample_rate, order)
            
        elif filter_type.lower() == 'lowpass':
            cutoff_freq = cutoff if cutoff is not None else high_cutoff
            logger.info(f"Applying lowpass filter: {cutoff_freq} Hz")
            filtered_audio = lowpass_filter(audio_data, cutoff_freq, sample_rate, order)
            
        elif filter_type.lower() == 'highpass':
            cutoff_freq = cutoff if cutoff is not None else low_cutoff
            logger.info(f"Applying highpass filter: {cutoff_freq} Hz")
            filtered_audio = highpass_filter(audio_data, cutoff_freq, sample_rate, order)
            
        elif filter_type.lower() == 'none':
            logger.info("No filtering applied")
            filtered_audio = audio_data
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Create output path if not provided
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            output_path = temp_file.name
            temp_file.close()
        
        # Save processed audio
        save_audio(filtered_audio, sample_rate, output_path)
        logger.info(f"Processed audio saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        # Return original file path if preprocessing fails
        return input_path

def get_filter_info(
    filter_type: str = 'bandpass',
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0,
    cutoff: Optional[float] = None,
    order: int = 5
) -> dict:
    """
    Get information about the filter configuration
    
    Returns:
        Dictionary with filter information
    """
    info = {
        "filter_type": filter_type,
        "order": order
    }
    
    if filter_type.lower() == 'bandpass':
        info.update({
            "low_cutoff_hz": low_cutoff,
            "high_cutoff_hz": high_cutoff,
            "bandwidth_hz": high_cutoff - low_cutoff
        })
    elif filter_type.lower() in ['lowpass', 'highpass']:
        cutoff_freq = cutoff if cutoff is not None else (high_cutoff if filter_type.lower() == 'lowpass' else low_cutoff)
        info["cutoff_hz"] = cutoff_freq
    
    return info