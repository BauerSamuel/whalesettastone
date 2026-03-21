"""
Unit tests for the whale audio file importer.

These tests focus on validating WAV files and importing single files /
directories, including a few key error paths.
"""

import sys
from pathlib import Path
import wave

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.file_importer import FileImporter
from src.config import WHALE_CODAS_DIR


def _write_wav(
    path: Path,
    *,
    nchannels: int = 1,
    sampwidth: int = 2,
    framerate: int = 16_000,
    nframes: int = 16_000,
) -> None:
    """Helper to create a simple WAV file with the given parameters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00" * nframes * nchannels * sampwidth)


def test_validate_wav_file_valid(tmp_path: Path) -> None:
    """A WAV that matches config constraints should validate as True."""
    wav_path = tmp_path / "valid.wav"
    _write_wav(wav_path)

    importer = FileImporter()
    assert importer.validate_wav_file(wav_path) is True


def test_validate_wav_file_rejects_low_sample_rate(tmp_path: Path) -> None:
    """Sample rate below the configured minimum should be rejected."""
    wav_path = tmp_path / "low_sr.wav"
    # Enough frames so file size is above MIN_FILE_SIZE, but framerate is low.
    _write_wav(wav_path, framerate=4_000, nframes=2_000)

    importer = FileImporter()
    assert importer.validate_wav_file(wav_path) is False


def test_validate_wav_file_rejects_wrong_channels(tmp_path: Path) -> None:
    """A stereo file should be rejected when mono is required."""
    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, nchannels=2, nframes=2_000)

    importer = FileImporter()
    assert importer.validate_wav_file(wav_path) is False


def test_validate_wav_file_rejects_wrong_sample_width(tmp_path: Path) -> None:
    """A file with the wrong sample width should be rejected."""
    wav_path = tmp_path / "wrong_width.wav"
    # 1-byte samples instead of 16‑bit (2 bytes)
    _write_wav(wav_path, sampwidth=1, nframes=2_000)

    importer = FileImporter()
    assert importer.validate_wav_file(wav_path) is False


def test_validate_wav_file_rejects_too_small_file(tmp_path: Path) -> None:
    """Files smaller than MIN_FILE_SIZE should be rejected."""
    wav_path = tmp_path / "tiny.wav"
    # Very few frames so the total size stays well below MIN_FILE_SIZE.
    _write_wav(wav_path, nframes=10)

    importer = FileImporter()
    assert importer.validate_wav_file(wav_path) is False


def test_import_file_success_creates_in_whale_codas(tmp_path: Path) -> None:
    """import_file should copy a valid WAV into the whale_codas directory."""
    source = tmp_path / "to_import.wav"
    _write_wav(source)

    importer = FileImporter()
    dest = importer.import_file(source)

    assert dest is not None
    assert dest.exists()
    assert dest.parent == WHALE_CODAS_DIR
    assert dest.name == source.name


def test_import_file_nonexistent_returns_none(tmp_path: Path) -> None:
    """Nonexistent source files should return None and not crash."""
    source = tmp_path / "missing.wav"
    importer = FileImporter()
    dest = importer.import_file(source)

    assert dest is None


def test_import_directory_mixes_valid_and_invalid(tmp_path: Path) -> None:
    """
    import_directory should only return successfully imported (valid) files.
    """
    src_dir = tmp_path / "wav_dir"
    src_dir.mkdir(parents=True, exist_ok=True)

    valid_file = src_dir / "valid.wav"
    invalid_file = src_dir / "invalid.wav"

    _write_wav(valid_file)
    # Invalid: wrong number of channels
    _write_wav(invalid_file, nchannels=2, nframes=2_000)

    importer = FileImporter()
    imported = importer.import_directory(src_dir)

    # Only the valid file should end up imported.
    imported_names = {p.name for p in imported}
    assert "valid.wav" in imported_names
    assert "invalid.wav" not in imported_names
    assert all(p.parent == WHALE_CODAS_DIR for p in imported)


def test_import_directory_nonexistent_returns_empty_list(tmp_path: Path) -> None:
    """A non‑existent directory should yield an empty list, not an exception."""
    missing_dir = tmp_path / "does_not_exist"

    importer = FileImporter()
    imported = importer.import_directory(missing_dir)

    assert imported == []

