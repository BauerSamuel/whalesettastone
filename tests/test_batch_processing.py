"""
Test script for batch processing of audio files.

Uses FeatureExtractor and config paths; processes only a few files so the test stays fast.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import WHALE_CODAS_DIR, SAMPLE_AUDIO_DIR
from src.feature_extractor import FeatureExtractor


# Cap files so the test finishes in reasonable time (avoid processing 90+ WAVs)
_MAX_BATCH_FILES = 5


def test_batch_processing():
    """Extract features from a few WAVs in whale_codas (or sample_audio) and assert we get features."""
    for search_dir in (WHALE_CODAS_DIR, SAMPLE_AUDIO_DIR):
        if search_dir.exists():
            wavs = list(search_dir.glob("*.wav"))[: _MAX_BATCH_FILES]
            if wavs:
                break
    else:
        pytest.skip("No WAV files found in data/whale_codas or data/sample_audio.")

    extractor = FeatureExtractor()
    full_features = []
    for path in wavs:
        feats = extractor.extract_features(path)
        if feats:
            full_features.append(feats)

    assert len(full_features) > 0, "Expected at least one file to yield features"
    for f in full_features:
        assert "filename" in f
        assert "duration" in f
        assert "mfccs" in f


if __name__ == "__main__":
    test_batch_processing()
