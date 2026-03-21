"""
Configuration settings for the whale audio analysis system.

This module contains all configuration parameters and settings used throughout
the system, including paths, audio parameters, and processing settings.
"""

from pathlib import Path
import os

# Directory structure
BASE_DIR = Path("data")
WHALE_CODAS_DIR = BASE_DIR / "whale_codas"
UPLOADS_DIR = BASE_DIR / "uploads"
SAMPLE_AUDIO_DIR = BASE_DIR / "sample_audio"
LOGS_DIR = BASE_DIR / "logs"
METADATA_DIR = BASE_DIR / "metadata"
PLOTS_DIR = BASE_DIR / "plots"

# Ensure all directories exist
for directory in [BASE_DIR, WHALE_CODAS_DIR, UPLOADS_DIR, SAMPLE_AUDIO_DIR, 
                 LOGS_DIR, METADATA_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File validation parameters
MIN_FILE_SIZE = 1024  # 1 KB
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MIN_SAMPLE_RATE = 8000  # 8 kHz
MAX_SAMPLE_RATE = 48000  # 48 kHz
REQUIRED_CHANNELS = 1  # Mono audio
REQUIRED_SAMPLE_WIDTH = 2  # 16-bit audio

# Feature extraction parameters
N_MFCC = 13  # Number of MFCC coefficients
N_FFT = 2048  # Length of the FFT window
HOP_LENGTH = 512  # Number of samples between successive frames

# Dimensionality reduction parameters
N_COMPONENTS = 2  # Number of dimensions to reduce to
PERPLEXITY = 30  # t-SNE perplexity parameter
N_NEIGHBORS = 15  # UMAP n_neighbors parameter

# Download parameters
DOWNLOAD_DELAY = 2  # seconds between downloads
MAX_RETRIES = 3
TIMEOUT = 30  # seconds

def _env_bool(name: str, default: bool = False) -> bool:
    """Parse truthy env vars: 1, true, yes, on (case-insensitive)."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# Upload/UI limits (can be overridden via environment variables)
MAX_UPLOAD_FILES = int(os.getenv("WHALE_MAX_UPLOAD_FILES", "20"))
MAX_UPLOAD_BYTES = int(
    os.getenv("WHALE_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))
)  # 50 MB default per file

# Show sidebar debug panel (off in production by default)
SHOW_DEBUG = _env_bool("WHALE_SHOW_DEBUG", False)

# Memory-limiting options (for free-tier hosting)
# Cap audio duration for processing (seconds); longer files are truncated. 0 = no cap.
MAX_AUDIO_SECONDS = int(os.getenv("WHALE_MAX_AUDIO_SECONDS", "60"))
# Resample to this rate during load to shrink arrays (Hz). None = keep original.
RESAMPLE_RATE = int(os.getenv("WHALE_RESAMPLE_RATE", "22050")) or None
# Temp dir for disk-backed audio (session state stores paths, not arrays)
AUDIO_TMP_DIR = BASE_DIR / "tmp_audio"