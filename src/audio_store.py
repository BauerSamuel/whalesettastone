"""
Disk-backed audio store for memory-limited hosting.

Stores decoded audio on disk; session state holds only paths. Load on demand,
use, discard. Never keeps full waveforms in RAM across reruns.
"""

import os
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import AUDIO_TMP_DIR, MAX_AUDIO_SECONDS, RESAMPLE_RATE


def _ensure_tmp_dir() -> Path:
    AUDIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_TMP_DIR


def save_audio_to_disk(y: np.ndarray, sr: int) -> str:
    """Write audio array to a temp file; return path. Caller stores path in session."""
    d = _ensure_tmp_dir()
    path = d / f"{uuid.uuid4().hex}.wav"
    sf.write(str(path), y, sr)
    return str(path)


def load_audio_from_path(path: str):
    """Load (y, sr) from disk. Caller uses and discards; do not cache."""
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def load_audio_optimized(file_or_path, use_librosa: bool = True):
    """
    Load audio with memory limits: duration cap, optional resample.
    Returns (y, sr). Use librosa when feature extraction will follow.
    """
    import librosa

    duration = MAX_AUDIO_SECONDS if MAX_AUDIO_SECONDS > 0 else None
    sr = RESAMPLE_RATE

    if hasattr(file_or_path, "read") or isinstance(file_or_path, (str, Path)):
        y, sr_out = librosa.load(file_or_path, sr=sr, duration=duration, mono=True)
        return y, sr_out
    raise TypeError("Need path or file-like")


def clear_audio_tmp_dir():
    """Remove all temp audio files (call on new batch or session reset)."""
    if not AUDIO_TMP_DIR.exists():
        return
    for p in AUDIO_TMP_DIR.glob("*.wav"):
        try:
            p.unlink()
        except OSError:
            pass
