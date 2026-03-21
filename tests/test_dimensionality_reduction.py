"""
Tests for dimensionality reduction of audio features.

Uses FeatureExtractor and FeatureReducer (single canonical implementation).
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extractor import FeatureExtractor
from src.feature_reducer import FeatureReducer
from src.config import WHALE_CODAS_DIR, SAMPLE_AUDIO_DIR, PLOTS_DIR

# Cap files so test finishes in reasonable time (t-SNE/UMAP are slow)
_MAX_FILES = 5


def _save_wav(path: Path, *, frequency_hz: float, duration_s: float = 1.0, sample_rate: int = 44100):
    """
    Create a simple sine wave WAV file.

    This keeps the test robust when no real WAVs exist in data/.
    """
    import wave
    import struct
    import numpy as np

    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave_data = np.sin(2 * np.pi * frequency_hz * t)
    normalized = np.int16(wave_data * 32767)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for value in normalized:
            wf.writeframes(struct.pack("h", int(value)))


def test_dimensionality_reduction(tmp_path: Path):
    """Run dimensionality reduction and save plots; close figures when done."""
    import matplotlib.pyplot as plt
    try:
        for search_dir in (WHALE_CODAS_DIR, SAMPLE_AUDIO_DIR):
            if not search_dir.exists():
                continue
            wav_files = list(search_dir.glob("*.wav"))[: _MAX_FILES]
            if wav_files:
                break
        else:
            pytest.skip("No WAV files in data/whale_codas or data/sample_audio.")

        extractor = FeatureExtractor()
        features = []
        for path in wav_files:
            feats = extractor.extract_features(path)
            if feats:
                features.append(feats)

        # UMAP is sensitive to small sample counts; ensure enough feature vectors.
        min_features = 8
        if len(features) < min_features:
            synth_dir = tmp_path / "synth_wavs"
            freqs = [200, 240, 280, 320, 360, 420, 500, 580, 660, 740]
            for i, f in enumerate(freqs):
                if len(features) >= min_features:
                    break
                synth_path = synth_dir / f"synth_{i}.wav"
                _save_wav(synth_path, frequency_hz=f, duration_s=1.0, sample_rate=44100)
                feats = extractor.extract_features(synth_path)
                if feats:
                    features.append(feats)

        if not features:
            pytest.skip("No features extracted.")

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        for method in ("tsne", "umap"):
            reducer = FeatureReducer(method=method)
            reduced, _ = reducer.process_features(
                features,
                output_path=str(PLOTS_DIR / f"{method}_plot.png"),
            )
            assert reduced is not None, f"{method.upper()} reduction should succeed"
    finally:
        plt.close("all")


if __name__ == "__main__":
    test_dimensionality_reduction()
