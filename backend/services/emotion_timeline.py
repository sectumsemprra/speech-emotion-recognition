#!/usr/bin/env python3
"""
services/emotion_timeline.py

Computes a time-resolved emotion trajectory (frame-level emotion probabilities)
using your existing `funasr` AutoModel instance. Produces a JSON-friendly timeline
and a heatmap saved under an `artifacts/` session folder so it can be served by
your FastAPI static mount.

Usage:
    from services.emotion_timeline import compute_emotion_timeline
    timeline, heatmap_path = compute_emotion_timeline(model, processed_wav, frame_ms=400, hop_ms=200)

The function is intentionally conservative and robust to different model outputs
(`predictions`/`scores` or `labels`/`scores`). It does per-frame inference by
writing frames to temporary WAV files and calling `model.generate(...)`.

Note: per-frame inference is slower than single-shot inference. For demo/demo
reports this is fine; tune frame/hop sizes for speed/temporal resolution trade-off.
"""

import os
import tempfile
import time
from typing import Tuple, List, Dict, Any

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly


def _mk_artifact_dir(src_path: str) -> str:
    base = os.path.splitext(os.path.basename(src_path))[0]
    ts = int(time.time() * 1000)
    out_dir = os.path.join("artifacts", f"{base}_timeline_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _frame_generator(x: np.ndarray, fs: int, frame_ms: int, hop_ms: int):
    frame_len = int(fs * frame_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    if frame_len <= 0:
        raise ValueError("frame_ms too small for sampling rate")
    if hop <= 0:
        hop = frame_len
    starts = list(range(0, max(1, len(x) - frame_len + 1), hop))
    for s in starts:
        yield s, s + frame_len
    # last partial frame (if any)
    if len(x) > 0 and (starts == [] or starts[-1] + frame_len < len(x)):
        s = max(0, len(x) - frame_len)
        yield s, len(x)


def _safe_model_generate(model, wav_path: str) -> Dict[str, Any]:
    """Call model.generate and return the first result dict in a normalized format."""
    # model.generate may return different formats; handle common cases
    res = model.generate(wav_path, granularity="utterance")
    if not isinstance(res, (list, tuple)) or len(res) == 0:
        raise RuntimeError("Model.generate returned empty or unexpected format")
    data = res[0]
    # Normalize to have `emotions` list and `scores` list
    if 'predictions' in data and 'scores' in data:
        emotions = list(data['predictions'])
        scores = list(data['scores'])
    elif 'labels' in data and 'scores' in data:
        emotions = [str(e).split('/')[-1] for e in data['labels']]
        scores = list(data['scores'])
    elif 'scores' in data and isinstance(data['scores'], dict):
        # sometimes the model might return a dict of {label:score}
        emotions = list(data['scores'].keys())
        scores = list(data['scores'].values())
    else:
        # fallback: try to introspect keys
        # Build empty
        emotions = []
        scores = []
    return {"emotions": emotions, "scores": scores}


def compute_emotion_timeline(
    model,
    audio_path: str,
    frame_ms: int = 400,
    hop_ms: int = 200,
    fs_target: int = 16000,
    plot: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """
    Compute time-resolved emotion outputs across frames.

    Returns:
      timeline_report: dict with keys:
         - frames: list of {start_s, end_s, top_emotion, top_score, probs: {label:score}}
         - emotions: ordered list of all labels found across frames
         - frame_ms, hop_ms, n_frames
         - artifacts: paths (heatmap)

      heatmap_path: path to saved heatmap PNG (or empty string if not generated)
    """
    # Load audio and ensure mono
    x, fs = sf.read(audio_path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    # Resample if needed
    if fs != fs_target:
        # use resample_poly for good quality
        gcd = np.gcd(fs_target, fs)
        up = fs_target // gcd
        down = fs // gcd
        x = resample_poly(x, up, down)
        fs = fs_target

    out_dir = _mk_artifact_dir(audio_path)
    heatmap_path = ""

    # Iterate frames and run model
    frames = []
    all_labels = set()
    per_frame_label_scores: List[Dict[str, float]] = []

    for start, end in _frame_generator(x, fs, frame_ms, hop_ms):
        seg = x[start:end]
        # skip very short segments
        if len(seg) < int(0.02 * fs):
            per_frame_label_scores.append({})
            frames.append((start, end))
            continue

        # write to temp wav
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, seg, fs)
        try:
            model_out = _safe_model_generate(model, tmp_path)
            labels = model_out.get('emotions', [])
            scores = model_out.get('scores', [])
            # If scores are empty but labels present, assign uniform
            if labels and (not scores or len(scores) != len(labels)):
                scores = [1.0/len(labels)] * len(labels)

            label_score_map = {str(l): float(s) for l, s in zip(labels, scores)}
        except Exception:
            label_score_map = {}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        per_frame_label_scores.append(label_score_map)
        all_labels.update(label_score_map.keys())
        frames.append((start, end))

    # Order emotions consistently (sort by average score desc)
    emotions = sorted(list(all_labels))
    if emotions:
        # compute average score for ordering
        avg_scores = {e: 0.0 for e in emotions}
        counts = {e: 0 for e in emotions}
        for m in per_frame_label_scores:
            for e, s in m.items():
                avg_scores[e] += s
                counts[e] += 1
        for e in emotions:
            if counts[e] > 0:
                avg_scores[e] /= counts[e]
        emotions = sorted(emotions, key=lambda e: avg_scores.get(e, 0.0), reverse=True)

    n_frames = len(per_frame_label_scores)

    # Build heatmap matrix (len(emotions) x n_frames)
    if emotions:
        mat = np.zeros((len(emotions), n_frames), dtype=float)
        for j, m in enumerate(per_frame_label_scores):
            for i, e in enumerate(emotions):
                mat[i, j] = float(m.get(e, 0.0))
    else:
        mat = np.zeros((1, n_frames), dtype=float)
        emotions = ["unknown"]

    # Build timeline entries
    timeline_frames = []
    for j, (s, e) in enumerate(frames):
        probs = {lab: float(per_frame_label_scores[j].get(lab, 0.0)) for lab in emotions}
        top_lab = max(probs.items(), key=lambda kv: kv[1])[0] if probs else "unknown"
        top_score = probs.get(top_lab, 0.0)
        timeline_frames.append({
            "start_s": float(s) / fs,
            "end_s": float(e) / fs,
            "top_emotion": top_lab,
            "top_score": float(top_score),
            "probs": probs
        })

    # Plot heatmap
    if plot and n_frames > 0:
        plt.figure(figsize=(max(6, n_frames * 0.2), max(3, len(emotions) * 0.5)))
        im = plt.imshow(mat, aspect='auto', interpolation='nearest', origin='lower')
        plt.yticks(np.arange(len(emotions)), emotions)
        xticks = np.arange(n_frames)
        xtick_labels = [f"{(frames[i][0]/fs):.2f}" for i in range(n_frames)]
        plt.xticks(xticks, xtick_labels, rotation=90, fontsize=8)
        plt.xlabel('time (s) — frame start')
        plt.colorbar(im, label='score')
        plt.title('Emotion timeline heatmap')
        heatmap_path = os.path.join(out_dir, 'emotion_timeline.png')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

    report = {
        "frames": timeline_frames,
        "emotions": emotions,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "n_frames": n_frames,
        "artifacts": {
            "heatmap": heatmap_path
        }
    }

    return report, heatmap_path
