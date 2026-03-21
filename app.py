"""
Streamlit app for whale sound analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from src.workflow import Workflow
from src.source_downloader import get_available_sources, download_source_files, get_source_display_name
from src.audio_store import (
    save_audio_to_disk,
    load_audio_from_path,
    load_audio_optimized,
    clear_audio_tmp_dir,
)
from src.config import MAX_UPLOAD_FILES, MAX_UPLOAD_BYTES, SHOW_DEBUG
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import base64
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import soundfile as sf
import time
import tempfile
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Whale species options for batch tagging
WHALE_SPECIES = ["Unknown", "Sperm whale", "Humpback whale", "Blue whale", "Killer whale (Orca)", "Fin whale", "Other"]

# Set page config
st.set_page_config(
    page_title="Whale Sound Analysis",
    page_icon="🐋",
    layout="wide",
    # More main-area space on phones; users open the sidebar from the menu when needed
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def get_workflow():
    return Workflow()


def _get_audio_array(file_name: str):
    """Load audio from disk-backed store. Returns (y, sr) or (None, None). Never keeps array in session."""
    if file_name not in st.session_state.audio_files:
        return None, None
    entry = st.session_state.audio_files[file_name]
    if "path" in entry:
        return load_audio_from_path(entry["path"])
    if "data" in entry:
        return entry["data"], entry["sample_rate"]
    return None, None


# Initialize session state for audio files and volume
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = {}
if 'batches' not in st.session_state:
    st.session_state.batches = {}  # Dictionary of batches with timestamps as keys
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = None
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

def get_batch_species():
    """Return species tag for current batch, or None."""
    if not st.session_state.current_batch or not st.session_state.batches:
        return None
    return st.session_state.batches.get(st.session_state.current_batch, {}).get('species', 'Unknown')

def normalize_file_info(file_obj):
    """
    Normalize file info to a dictionary with 'features' key.
    Handles BytesIO, UploadedFile, dict, or file path objects.
    
    Args:
        file_obj: Can be BytesIO, UploadedFile, dict with 'features', or file path
        
    Returns:
        Dictionary with 'features' key and 'filename'/'path' key, or None if extraction fails
    """
    # If already a dict with features, return as-is
    if isinstance(file_obj, dict) and 'features' in file_obj:
        return file_obj
    
    # Extract features on-the-fly
    try:
        # Get file name/identifier
        file_name = None
        if hasattr(file_obj, 'name'):
            file_name = file_obj.name
        elif isinstance(file_obj, (str, Path)):
            file_name = Path(file_obj).name
        else:
            file_name = str(file_obj)
        
        # Load audio data
        audio_data = None
        sample_rate = None
        
        # Check if already in session state (disk-backed or legacy)
        if file_name in st.session_state.audio_files:
            audio_data, sample_rate = _get_audio_array(file_name)
        else:
            # Load from file object, apply memory limits, save to disk
            if hasattr(file_obj, 'getvalue'):
                fname = getattr(file_obj, 'name', '') or ''
                ext = os.path.splitext(fname)[1] if fname else '.wav'
                if ext not in ('.wav', '.mp3', '.flac', '.ogg'):
                    ext = '.wav'
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                    tmp_file.write(file_obj.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    audio_data, sample_rate = load_audio_optimized(tmp_file_path)
                finally:
                    os.unlink(tmp_file_path)
                disk_path = save_audio_to_disk(audio_data, sample_rate)
                st.session_state.audio_files[file_name] = {'path': disk_path, 'sample_rate': sample_rate}
            elif isinstance(file_obj, (str, Path)):
                audio_data, sample_rate = load_audio_optimized(file_obj)
                disk_path = save_audio_to_disk(audio_data, sample_rate)
                st.session_state.audio_files[file_name] = {'path': disk_path, 'sample_rate': sample_rate}
            else:
                logger.warning(f"Could not load audio from {file_obj}")
                return None
        
        if audio_data is None or sample_rate is None:
            return None
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1).tolist()
        
        features = {
            'filename': file_name,
            'sample_rate': float(sample_rate),
            'duration': float(len(audio_data) / sample_rate),
            'mfccs': mfcc_means,
            'spectral_centroid': float(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0].mean()),
            'spectral_rolloff': float(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0].mean()),
            'spectral_bandwidth': float(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0].mean()),
            'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(audio_data)[0].mean()),
            'rms_energy': float(librosa.feature.rms(y=audio_data)[0].mean())
        }
        
        # Add mfcc_1 through mfcc_13 for compatibility
        for i, val in enumerate(mfcc_means[:13], start=1):
            features[f'mfcc_{i}'] = float(val)
        
        return {
            'filename': file_name,
            'path': file_name,  # For compatibility
            'features': features
        }
        
    except Exception as e:
        logger.exception(f"Error normalizing file info for {file_obj}: {e}")
        return None

def get_audio_duration(file_path):
    """Get duration of audio file in seconds"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        st.error(f"Error getting duration for {file_path}: {str(e)}")
        return 0

def get_audio_html(file_path, volume=0.5):
    """Generate HTML audio player with volume control"""
    try:
        # Read audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, y, sr, format='WAV')
        audio_bytes = buffer.getvalue()
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Create HTML audio player with volume control
        audio_html = f"""
        <audio controls style="width: 100%;" volume="{volume}">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    except Exception as e:
        st.error(f"Error generating audio player for {file_path}: {str(e)}")
        return ""

def plot_waveform(audio_file):
    """Plot the waveform of an audio file."""
    # Handle different input types
    if hasattr(audio_file, 'getvalue'):
        # BytesIO or UploadedFile - write to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
    elif isinstance(audio_file, (str, Path)):
        # File path
        tmp_file_path = str(audio_file)
    else:
        raise ValueError(f"Unsupported audio_file type: {type(audio_file)}")
    
    try:
        # Load the audio file
        y, sr = librosa.load(tmp_file_path, sr=None)
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        return fig
    finally:
        # Clean up the temporary file only if we created it
        if hasattr(audio_file, 'getvalue') and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def plot_spectrogram(audio_file):
    """Plot the spectrogram of an audio file."""
    # Handle different input types
    if hasattr(audio_file, 'getvalue'):
        # BytesIO or UploadedFile - write to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
    elif isinstance(audio_file, (str, Path)):
        # File path
        tmp_file_path = str(audio_file)
    else:
        raise ValueError(f"Unsupported audio_file type: {type(audio_file)}")
    
    try:
        # Load the audio file
        y, sr = librosa.load(tmp_file_path, sr=None)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        ax.set_title('Spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        return fig
    finally:
        # Clean up the temporary file only if we created it
        if hasattr(audio_file, 'getvalue') and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def create_visualization(data, labels, method='t-SNE'):
    """Create visualization with error handling"""
    try:
        if len(data) < 2:
            return None, "Need at least 2 samples for visualization"
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if method == 't-SNE':
            # Adjust perplexity based on sample size
            perplexity = min(30, len(data) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            title = 't-SNE Visualization of Audio Features'
        else:  # UMAP
            reducer = umap.UMAP(random_state=42, n_neighbors=min(15, len(data) - 1))
            title = 'UMAP Visualization of Audio Features'
            
        reduced_data = reducer.fit_transform(data)
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
        
        # Add labels
        for i, txt in enumerate(labels):
            ax.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]))
        
        ax.set_title(title)
        ax.set_xlabel(f'{method} 1')
        ax.set_ylabel(f'{method} 2')
        
        return fig, None
    except Exception as e:
        return None, str(e)


def render_audio_features_panel(duration, sample_rate, n_samples, file_size_bytes, dtype_name,
                                 spectral_centroid, spectral_rolloff, spectral_bandwidth,
                                 zero_crossing_rate, rms_energy, mfcc_means):
    """Build HTML for the Audio Features panel with consistent formatting and styling."""
    # Consistent number formatting: 2 decimals for duration/spectral, 3 for ZCR/RMS, 0 for integers
    duration_str = f"{duration:.2f} s"
    sample_rate_str = f"{int(sample_rate):,} Hz"
    samples_str = f"{int(n_samples):,}"
    file_size_str = f"{file_size_bytes / 1024:.2f} KB" if file_size_bytes and file_size_bytes > 0 else "—"
    
    sc_str = f"{spectral_centroid:.2f} Hz"
    sr_str = f"{spectral_rolloff:.2f} Hz"
    sb_str = f"{spectral_bandwidth:.2f} Hz"
    zcr_str = f"{zero_crossing_rate:.3f}"
    rms_str = f"{rms_energy:.3f}"
    
    basic_rows = [
        ("Duration", duration_str),
        ("Sample rate", sample_rate_str),
        ("Samples", samples_str),
        ("File size", file_size_str),
        ("Data type", dtype_name),
    ]
    spectral_rows = [
        ("Spectral centroid", sc_str),
        ("Spectral rolloff", sr_str),
        ("Spectral bandwidth", sb_str),
        ("Zero crossing rate", zcr_str),
        ("RMS energy", rms_str),
    ]
    
    html_parts = ['<div class="feature-panel">']
    
    html_parts.append('<div class="feature-section-title">Basic properties</div>')
    for label, value in basic_rows:
        html_parts.append(f'<div class="feature-row"><span class="feature-label">{label}</span><span class="feature-value">{value}</span></div>')
    
    html_parts.append('<div class="feature-section-title">Spectral features</div>')
    for label, value in spectral_rows:
        html_parts.append(f'<div class="feature-row"><span class="feature-label">{label}</span><span class="feature-value">{value}</span></div>')
    
    html_parts.append('<div class="feature-section-title">MFCC coefficients (mean)</div>')
    html_parts.append('<div class="feature-mfcc-grid">')
    for i, val in enumerate(mfcc_means[:13], start=1):
        v = f"{float(val):.2f}"
        html_parts.append(f'<div class="feature-mfcc-cell"><span class="mfcc-label">MFCC {i}</span><span class="feature-value">{v}</span></div>')
    html_parts.append('</div></div>')
    
    return "".join(html_parts)


def compute_onset_and_ici(y, sr, hop_length=256, wait_frames=8, delta=0.07):
    """
    Detect onset (click) times and compute inter-click intervals (ICIs).
    Tuned for click-like vocalizations (e.g., sperm whale clicks): short, broadband transients.
    
    Returns:
        (onset_times_sec, ici_sec) - arrays of onset times in seconds and ICIs in seconds.
    """
    try:
        # Onset strength envelope (good for impulsive sounds)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        # Detect onset frames; backtrack to nearest preceding minimum for sharper timing
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            delta=delta,
            wait=wait_frames,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        if len(onset_times) < 2:
            return onset_times, np.array([])
        ici = np.diff(onset_times)
        return onset_times, ici
    except Exception:
        return np.array([]), np.array([])


def segment_audio_features(y, sr, segment_sec=1.5, hop_sec=0.75):
    """
    Split audio into overlapping segments and extract a feature vector per segment.
    Returns (segment_vectors, segment_times_mid) for use in clustering / sequence analysis.
    """
    seg_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)
    if seg_len > len(y):
        return np.array([]).reshape(0, 20), np.array([])
    vectors = []
    times = []
    for start in range(0, len(y) - seg_len + 1, hop_len):
        chunk = y[start:start + seg_len]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=8)
        sc = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0].mean()
        sr_roll = librosa.feature.spectral_rolloff(y=chunk, sr=sr)[0].mean()
        zcr = librosa.feature.zero_crossing_rate(chunk)[0].mean()
        rms = librosa.feature.rms(y=chunk)[0].mean()
        vec = np.concatenate([mfcc.mean(axis=1), [sc, sr_roll, zcr, rms]])
        vectors.append(vec)
        times.append((start + seg_len / 2) / sr)
    return np.array(vectors), np.array(times)


def segment_codas_by_gap(onset_times, gap_threshold_sec=0.4):
    """
    Group onset times into codas: a new coda starts when inter-click interval exceeds gap_threshold_sec.
    Returns list of dicts: {'start': t0, 'end': t1, 'n_clicks': k, 'icis': array of ICIs in sec}.
    """
    if len(onset_times) < 2:
        return []
    ici = np.diff(onset_times)
    codas = []
    start_idx = 0
    for i in range(len(ici)):
        if ici[i] > gap_threshold_sec:
            # end current coda
            end_idx = i + 1
            if end_idx > start_idx:
                seg_onset = onset_times[start_idx:end_idx + 1]
                seg_ici = np.diff(seg_onset)
                codas.append({
                    'start': float(onset_times[start_idx]),
                    'end': float(onset_times[end_idx]),
                    'n_clicks': len(seg_onset),
                    'icis': seg_ici.copy()
                })
            start_idx = i + 1
    if start_idx < len(onset_times):
        seg_onset = onset_times[start_idx:]
        seg_ici = np.diff(seg_onset)
        codas.append({
            'start': float(onset_times[start_idx]),
            'end': float(onset_times[-1]),
            'n_clicks': len(seg_onset),
            'icis': seg_ici.copy()
        })
    return codas


# Simple rhythm templates: name -> (expected_click_counts_per_run, description)
# e.g. "1+1+3" = two single-click "runs" (long gap) then a run of 3 (short gaps)
CODARHYTHM_TEMPLATES = [
    ('5R', [5], 'Five clicks, regular spacing'),
    ('4+1', [4, 1], 'Four then one'),
    ('1+1+3', [1, 1, 3], 'Two singles then three'),
    ('3+1+1+3', [3, 1, 1, 3], 'Alternating groups'),
    ('1+1+1+1', [1, 1, 1, 1], 'Four singles'),
    ('9R', [9], 'Nine regular'),
]


def suggest_coda_type(n_clicks, icis_ms, tolerance=0.5):
    """
    Compare a detected coda (click count + ICI pattern) to reference types.
    Returns (best_name, best_score 0-1, all_scores dict).
    """
    if n_clicks < 2 or len(icis_ms) == 0:
        return None, 0.0, {}
    mean_ici = np.mean(icis_ms)
    if mean_ici <= 0:
        return None, 0.0, {}
    # Normalize ICIs to "short" (1) vs "long" (2) by median split
    med = np.median(icis_ms)
    pattern = [2 if x > med else 1 for x in icis_ms]
    # Click count pattern: e.g. [1,1,3] means run of 1, run of 1, run of 3
    runs = []
    cur = 1
    for i in range(1, len(pattern)):
        if pattern[i] == pattern[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    scores = {}
    for name, ref_runs, _ in CODARHYTHM_TEMPLATES:
        # Compare run structure: allow length mismatch but reward matching run counts
        if len(ref_runs) == 0:
            continue
        # Score by how well ref_runs matches runs (e.g. [1,1,3] vs [1,2,3] -> partial)
        n_match = min(len(runs), len(ref_runs))
        same = sum(1 for i in range(n_match) if runs[i] == ref_runs[i])
        # Also reward total click count match
        total_ref = sum(ref_runs)
        click_ok = 1.0 - min(1.0, abs(n_clicks - total_ref) / max(1, total_ref))
        scores[name] = 0.6 * (same / max(1, len(ref_runs))) + 0.4 * click_ok
    if not scores:
        return None, 0.0, {}
    best_name = max(scores, key=scores.get)
    return best_name, scores[best_name], scores


def main():
    # Initialize session state
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = {}
    if 'batches' not in st.session_state:
        st.session_state.batches = {}
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = None
    if 'files_uploaded' not in st.session_state:
        st.session_state.files_uploaded = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}

    # Sidebar for navigation - MUST be first so page selection works
    st.sidebar.title("Navigation")
    
    # Debug information (opt-in via WHALE_SHOW_DEBUG=1 for production safety)
    if SHOW_DEBUG:
        st.sidebar.subheader("Debug Info")
        st.sidebar.text(f"Files uploaded: {st.session_state.files_uploaded}")
        st.sidebar.text(f"Current batch: {st.session_state.current_batch}")
        if st.session_state.current_batch and st.session_state.batches:
            if st.session_state.current_batch in st.session_state.batches:
                st.sidebar.text(
                    f"Batch processed: {st.session_state.batches[st.session_state.current_batch]['processed']}"
                )
            st.sidebar.text(f"Batch exists: {st.session_state.current_batch in st.session_state.batches}")
            st.sidebar.text(f"Batch keys: {list(st.session_state.batches.keys())}")

    # Check if we have a current batch and if it's processed
    is_batch_processed = False
    if st.session_state.current_batch is not None and st.session_state.current_batch in st.session_state.batches:
        is_batch_processed = st.session_state.batches[st.session_state.current_batch]['processed']

    if SHOW_DEBUG:
        st.sidebar.text(f"Batch ready for analysis: {is_batch_processed}")
    
    # Navigation options - always enabled so users can navigate
    page = st.sidebar.radio(
        "Go to", 
        ["Upload & Process", "Spectrogram", "Waveform", "Pattern Detection", "Coda Rhythm & Timing", "Coda Reference", "About"]
    )

    # Show helpful messages but don't disable navigation
    if not st.session_state.files_uploaded:
        st.sidebar.info("💡 Upload and process files to see analysis results")
    elif st.session_state.files_uploaded and not is_batch_processed:
        st.sidebar.info("💡 Click 'Process Files' to enable analysis options")


    # Add purpose statement using safe Streamlit components
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=VT323&family=Press+Start+2P&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">
    <style>
        /* Main page background and text */
        .stApp {
            background: linear-gradient(135deg, #1a237e, #0d47a1) !important;
            min-height: 100vh;
            color: #ffffff !important;
        }
        
        /* Remove any invisible borders and circles */
        .element-container {
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            background: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        /* Remove any potential circle artifacts */
        .element-container:before,
        .element-container:after {
            display: none !important;
            content: none !important;
        }
        
        /* Sidebar styling - ensure dark background */
        .css-1d391kg, .css-1y4p8pa,
        [data-testid="stSidebar"] {
            background-color: rgba(13, 71, 161, 0.95) !important;
            border-right: 2px solid #00d2d3 !important;
        }
        
        /* Sidebar content styling - force white text on all child elements */
        .css-1d391kg > div, .css-1y4p8pa > div,
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebar"] > div > div {
            background-color: rgba(13, 71, 161, 0.95) !important;
            color: #ffffff !important;
        }
        
        /* Sidebar title and text - use white for better contrast - catch ALL text elements */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span:not([class*="icon"]):not([class*="Icon"]),
        [data-testid="stSidebar"] div:not([class*="button"]):not([class*="Button"]) {
            color: #ffffff !important;
        }
        
        /* Catch any remaining text in sidebar that might be missed */
        [data-testid="stSidebar"] [class*="stMarkdown"],
        [data-testid="stSidebar"] [class*="stText"],
        [data-testid="stSidebar"] [class*="stWrite"],
        [data-testid="stSidebar"] [class*="element-container"] {
            color: #ffffff !important;
        }
        
        /* Force all text in sidebar to be white - catch-all at the end */
        [data-testid="stSidebar"] *:not(button):not([class*="icon"]):not([class*="Icon"]):not(svg):not([class*="baseweb"]) {
            color: #ffffff !important;
        }
        
        /* Specific sidebar class selectors as backup - Tech Font */
        .css-1d391kg h1, .css-1y4p8pa h1,
        .css-1d391kg p, .css-1y4p8pa p,
        .css-1d391kg label, .css-1y4p8pa label,
        .css-1d391kg span, .css-1y4p8pa span,
        .css-1d391kg div, .css-1y4p8pa div,
        .css-1d391kg .stMarkdown, .css-1y4p8pa .stMarkdown,
        .css-1d391kg .stText, .css-1y4p8pa .stText,
        .css-1d391kg .stWrite, .css-1y4p8pa .stWrite {
            color: #ffffff !important;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        /* Debug text styling - Terminal Font */
        .debug-text,
        .debug-text *,
        [data-testid="stSidebar"] .debug-text,
        [data-testid="stSidebar"] .debug-text * {
            color: #ffffff !important;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
            font-size: 14px !important;
            margin: 5px 0 !important;
            background-color: rgba(13, 71, 161, 0.5) !important;
            padding: 2px 4px !important;
            border-radius: 3px !important;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background-color: rgba(13, 71, 161, 0.9) !important;
            color: #ffffff !important;
        }
        
        .stRadio > label {
            color: #ffffff !important;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        /* Radio button options - target all nested elements and text nodes */
        .stRadio > div > div > div,
        .stRadio label,
        .stRadio span,
        .stRadio div,
        .stRadio p,
        .stRadio *:not(button):not([class*="icon"]),
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stRadio span,
        [data-testid="stSidebar"] .stRadio div,
        [data-testid="stSidebar"] .stRadio p,
        [data-testid="stSidebar"] .stRadio *:not(button):not([class*="icon"]) {
            color: #ffffff !important;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        /* Radio button text specifically - Streamlit uses nested divs */
        .stRadio [class*="radio"] label,
        .stRadio [class*="Radio"] label,
        .stRadio [role="radio"] + label,
        .stRadio [role="radio"] ~ span,
        .stRadio [class*="baseweb"] label,
        .stRadio [class*="baseweb"] span {
            color: #ffffff !important;
        }
        
        /* Final catch-all for radio button text - must come after other rules */
        [data-testid="stSidebar"] .stRadio * {
            color: #ffffff !important;
        }
        
        .stRadio > div > div > div:hover {
            background-color: rgba(0, 168, 255, 0.2) !important;
        }
        
        /* Selected radio button */
        .stRadio > div > div > div[data-baseweb="radio"] {
            background-color: #00a8ff !important;
            color: #ffffff !important;
        }
        
        /* Info box styling - ensure dark background with white text for readability */
        .stAlert, .stInfo,
        [data-testid="stAlert"],
        [data-testid="stInfo"],
        div[class*="stAlert"],
        div[class*="stInfo"] {
            background-color: rgba(30, 55, 153, 0.95) !important;
            border: 2px solid #00d2d3 !important;
            color: #ffffff !important;
        }
        
        /* Info box text and icons - force white text */
        .stAlert p, .stInfo p,
        .stAlert div, .stInfo div,
        .stAlert span, .stInfo span,
        [data-testid="stAlert"] p, [data-testid="stInfo"] p,
        [data-testid="stAlert"] div, [data-testid="stInfo"] div,
        [data-testid="stAlert"] span, [data-testid="stInfo"] span,
        div[class*="stAlert"] p, div[class*="stInfo"] p,
        div[class*="stAlert"] div, div[class*="stInfo"] div,
        div[class*="stAlert"] span, div[class*="stInfo"] span {
            color: #ffffff !important;
        }
        
        /* Override any sidebar-specific info box colors */
        .css-1d391kg .stInfo, .css-1y4p8pa .stInfo,
        .css-1d391kg .stAlert, .css-1y4p8pa .stAlert,
        [data-testid="stSidebar"] .stInfo,
        [data-testid="stSidebar"] .stAlert {
            color: #ffffff !important;
            background-color: rgba(30, 55, 153, 0.95) !important;
        }
        
        /* File uploader styling */
        .stFileUploader {
            background-color: rgba(30, 55, 153, 0.8) !important;
            border: 2px solid #00d2d3 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            color: #ffffff !important;
        }
        
        .stFileUploader label {
            color: #00d2d3 !important;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
            font-size: 16px !important;
            text-shadow: 0 0 5px rgba(0, 210, 211, 0.3) !important;
        }
        
        .stFileUploader button {
            background-color: #00a8ff !important;
            color: #ffffff !important;
            border: 2px solid #00d2d3 !important;
            border-radius: 5px !important;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        .stFileUploader button:hover {
            background-color: #0097e6 !important;
        }
        
        /* Text contrast — main column only (sidebar has its own rules above) */
        .main .stMarkdown, .main .stText {
            color: #ffffff !important;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        /* Main content only — do NOT style .stApp-wide div/span (breaks Streamlit toolbar Material Symbols) */
        .main .block-container,
        .main .block-container * {
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        /* Page titles in main (Streamlit widgets) */
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        /* Restore default fonts for Streamlit chrome (hamburger, Deploy menu, icons) */
        [data-testid="stHeader"],
        [data-testid="stHeader"] *,
        [data-testid="stToolbar"],
        [data-testid="stToolbar"] *,
        [data-testid="stDecoration"],
        [data-testid="stDecoration"] * {
            font-family: revert-layer !important;
        }

        /* Collapsed sidebar chevron uses Material Symbols text in some Streamlit versions — must not inherit app monospace */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapsedControl"] * {
            font-family: "Material Symbols Outlined", sans-serif !important;
            font-style: normal !important;
            font-weight: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            -webkit-font-smoothing: antialiased !important;
            font-feature-settings: "liga" !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
        }
        
        /* Markdown in main only */
        .main .stMarkdown p,
        .main .stMarkdown li,
        .main .stMarkdown ul,
        .main .stMarkdown ol,
        .main .stMarkdown div,
        .main .stMarkdown span {
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
        }
        
        .stButton>button {
            background-color: #00a8ff !important;
            color: #ffffff !important;
            border: 2px solid #00d2d3 !important;
            border-radius: 5px !important;
        }
        
        .stButton>button:hover {
            background-color: #0097e6 !important;
        }
        
        /* 90s Word Art Title - Retro Tech Font */
        .word-art-title {
            font-family: 'Orbitron', 'Impact', 'Arial Black', sans-serif;
            font-weight: 900;
            font-size: 48px;
            text-align: center;
            margin: 20px 0;
            background: linear-gradient(45deg, #00a8ff, #00d2d3, #00b894);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 
                3px 3px 0 #000,
                -1px -1px 0 #000,
                1px -1px 0 #000,
                -1px 1px 0 #000,
                1px 1px 0 #000;
            transform: perspective(500px) rotateX(10deg);
            letter-spacing: 3px;
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .word-art-title:after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00a8ff, transparent);
        }
        
        /* Bubble container - subtle, consistent background */
        .bubble-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }
        
        .bubble {
            position: absolute;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 50%;
            animation: float 12s ease-in-out infinite;
            width: 16px;
            height: 16px;
            box-shadow: 0 0 20px rgba(0, 210, 211, 0.1);
        }
        
        @keyframes float {
            0%, 100% {
                transform: translateY(100vh) scale(0.6);
                opacity: 0;
            }
            8% {
                opacity: 0.4;
            }
            92% {
                opacity: 0.4;
            }
            100% {
                transform: translateY(-80px) scale(0.8);
                opacity: 0;
            }
        }
        
        @keyframes whale-swim {
            0% { transform: translate(120vw, -38vh); opacity: 0; }
            2% { transform: translate(110vw, -38vh); opacity: 0.5; }
            30% { transform: translate(-50vw, -38vh); opacity: 0.5; }
            32% { transform: translate(-60vw, -38vh); opacity: 0; }
            33% { transform: translate(120vw, 0); opacity: 0; }
            35% { transform: translate(110vw, 0); opacity: 0.5; }
            63% { transform: translate(-50vw, 0); opacity: 0.5; }
            65% { transform: translate(-60vw, 0); opacity: 0; }
            66% { transform: translate(120vw, 38vh); opacity: 0; }
            68% { transform: translate(110vw, 38vh); opacity: 0.5; }
            96% { transform: translate(-50vw, 38vh); opacity: 0.5; }
            98% { transform: translate(-60vw, 38vh); opacity: 0; }
            99% { transform: translate(120vw, -38vh); opacity: 0; }
            100% { transform: translate(120vw, -38vh); opacity: 0; }
        }
        
        
        /* Retro text styling - Terminal/Console Font */
        .retro-text {
            color: #ffffff;
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace;
            line-height: 1.6;
            margin: 10px 0;
            text-shadow: 0 0 5px rgba(0, 210, 211, 0.3);
            font-size: 16px;
        }
        
        .retro-list {
            list-style-type: none;
            padding-left: 20px;
        }
        
        .retro-list li {
            margin: 10px 0;
            color: #ffffff;
        }
        
        .retro-list li:before {
            content: ">";
            color: #00d2d3;
            margin-right: 10px;
        }
        
        /* Audio Features panel - consistent, readable styling */
        .feature-panel {
            font-family: 'Share Tech Mono', 'VT323', 'Courier New', monospace !important;
            color: #ffffff !important;
            font-size: 15px !important;
        }
        .feature-panel .feature-section-title {
            color: #00d2d3 !important;
            font-size: 1.1em !important;
            margin: 1em 0 0.5em 0 !important;
        }
        .feature-panel .feature-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 6px 10px;
            margin: 4px 0;
            background: rgba(30, 55, 153, 0.6);
            border: 1px solid rgba(0, 210, 211, 0.4);
            border-radius: 4px;
            color: #ffffff !important;
        }
        .feature-panel .feature-label {
            color: #e0e0e0 !important;
        }
        .feature-panel .feature-value {
            color: #ffffff !important;
            font-variant-numeric: tabular-nums;
        }
        .feature-panel .feature-mfcc-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
            margin-top: 8px;
        }
        .feature-panel .feature-mfcc-cell {
            padding: 8px 10px;
            background: rgba(30, 55, 153, 0.6);
            border: 1px solid rgba(0, 210, 211, 0.4);
            border-radius: 4px;
            text-align: center;
            color: #ffffff !important;
            font-variant-numeric: tabular-nums;
        }
        .feature-panel .feature-mfcc-cell .mfcc-label {
            display: block;
            font-size: 0.85em;
            color: #00d2d3 !important;
            margin-bottom: 4px;
        }

        /* --- Mobile & touch-friendly (narrow viewports) --- */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 1rem !important;
                max-width: 100% !important;
            }
            .word-art-title {
                font-size: clamp(1.25rem, 8vw, 2rem) !important;
                letter-spacing: 1px !important;
                transform: none !important;
                line-height: 1.2 !important;
            }
            /* ~44px minimum touch target (iOS HIG / Material) */
            .stButton > button {
                min-height: 44px !important;
                padding: 0.5rem 1rem !important;
            }
            [data-testid="stSidebar"] button {
                min-height: 40px !important;
            }
            /* Plotly: scroll horizontally instead of clipping on small screens */
            [data-testid="stPlotlyChart"] {
                max-width: 100% !important;
                overflow-x: auto !important;
            }
            /* Lighter visuals + less GPU on phones */
            .whale-background-overlay {
                display: none !important;
            }
            .bubble-container {
                display: none !important;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .bubble {
                animation: none !important;
            }
            .whale-background-overlay {
                display: none !important;
            }
        }
    </style>
    <script>
        // Force all sidebar text to white - overrides inline styles
        function forceSidebarTextWhite() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                const allElements = sidebar.querySelectorAll('*');
                allElements.forEach(el => {
                    // Skip buttons, icons, and SVGs
                    if (!el.tagName.match(/BUTTON|SVG|PATH|CIRCLE|RECT/i) && 
                        !el.className.match(/icon|Icon|button|Button/i)) {
                        el.style.color = '#ffffff';
                    }
                });
            }
        }
        
        // Apply tech font to all main content text
        function applyTechFont() {
            const techFont = "'Share Tech Mono', 'VT323', 'Courier New', monospace";
            const mainContent = document.querySelector('.main .block-container');
            if (mainContent) {
                // Apply to all text elements in main content
                const textElements = mainContent.querySelectorAll('h1, h2, h3, h4, h5, h6, p, span, div, li, td, th, label');
                textElements.forEach(el => {
                    // Skip elements that shouldn't have the font (like the title which uses Orbitron)
                    if (!el.classList.contains('word-art-title') && 
                        !el.closest('.word-art-title')) {
                        el.style.fontFamily = techFont;
                    }
                });
            }
            
            // Do not override fonts inside stHeader — toolbar uses Material Symbols (ligatures break → raw names like keyboard_double_arrow_left)
            
            const markdowns = document.querySelectorAll('.main .stMarkdown p, .main .stMarkdown li, .main .stMarkdown div');
            markdowns.forEach(el => {
                el.style.fontFamily = techFont;
            });
        }
        
        // Run immediately and on DOM changes
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                forceSidebarTextWhite();
                applyTechFont();
            });
        } else {
            forceSidebarTextWhite();
            applyTechFont();
        }
        // Also run after delays to catch dynamically loaded content
        setTimeout(() => { forceSidebarTextWhite(); applyTechFont(); }, 100);
        setTimeout(() => { forceSidebarTextWhite(); applyTechFont(); }, 500);
        setTimeout(() => { forceSidebarTextWhite(); applyTechFont(); }, 1000);
        
        // Watch for new content being added
        const observer = new MutationObserver(() => {
            forceSidebarTextWhite();
            applyTechFont();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)
    
    # Route to the correct page based on selection
    if page == "Upload & Process":
        # Show title and decorative elements only on Upload & Process page
        st.markdown("""
        <div class="bubble-container">
            <div class="bubble" style="left: 10%; animation-delay: 0s;"></div>
            <div class="bubble" style="left: 20%; animation-delay: 2s;"></div>
            <div class="bubble" style="left: 30%; animation-delay: 4s;"></div>
            <div class="bubble" style="left: 40%; animation-delay: 6s;"></div>
            <div class="bubble" style="left: 50%; animation-delay: 8s;"></div>
            <div class="bubble" style="left: 60%; animation-delay: 10s;"></div>
            <div class="bubble" style="left: 70%; animation-delay: 12s;"></div>
            <div class="bubble" style="left: 80%; animation-delay: 14s;"></div>
            <div class="bubble" style="left: 90%; animation-delay: 16s;"></div>
        </div>
        
        <h1 class="word-art-title">🐋 WHALESETTA STONE 🐋</h1>
        
        <div style="margin-top: 40px;"></div>
        """, unsafe_allow_html=True)
        # Create container with safe Streamlit components
        with st.container():
            # Status messages
            st.markdown('<p class="retro-text">> INITIALIZING WHALE COMMUNICATION DECODER...</p>', unsafe_allow_html=True)
            st.markdown('<p class="retro-text">> LOADING AI NEURAL NETWORKS...</p>', unsafe_allow_html=True)
            st.markdown('<p class="retro-text">> SCANNING FOR CETACEAN PATTERNS...</p>', unsafe_allow_html=True)
            
            st.markdown('<p class="retro-text">> SYSTEM READY: DECODING WHALE SOUNDS IN PROGRESS</p>', unsafe_allow_html=True)
            st.markdown('<p class="retro-text">> MACHINE LEARNING ENGAGED</p>', unsafe_allow_html=True)
            st.markdown('<p class="retro-text">> BIOACOUSTIC ANALYSIS ACTIVE</p>', unsafe_allow_html=True)
            
            # Powered by section
            st.markdown('<p class="retro-text">> POWERED BY Cursor AND ADVANCED AI MODULES</p>', unsafe_allow_html=True)
            
            # Features list
            st.markdown('<ul class="retro-list">', unsafe_allow_html=True)
            st.markdown('<li class="retro-text">PATTERN RECOGNITION IN WHALE VOCALIZATIONS</li>', unsafe_allow_html=True)
            st.markdown('<li class="retro-text">TEMPORAL STRUCTURE ANALYSIS</li>', unsafe_allow_html=True)
            st.markdown('<li class="retro-text">COMMUNICATION CONTEXT UNDERSTANDING</li>', unsafe_allow_html=True)
            st.markdown('<li class="retro-text">BIOACOUSTIC DREAMING AND GENERATION</li>', unsafe_allow_html=True)
            st.markdown('</ul>', unsafe_allow_html=True)
            
            # Mission statement
            st.markdown('<p class="retro-text">> ANALYZING WHALE SOUND RECORDINGS...</p>', unsafe_allow_html=True)
            st.markdown('<p class="retro-text">> IDENTIFYING COMMUNICATION PATTERNS...</p>', unsafe_allow_html=True)
            st.markdown('<p class="retro-text">> CONTRIBUTING TO RESEARCH DATABASE...</p>', unsafe_allow_html=True)
            st.markdown('<p class="retro-text">> MISSION: CRACK THE CODE OF WHALE COMMUNICATION!</p>', unsafe_allow_html=True)
            
            # Upload prompt with clear call-to-action
            st.markdown('<p class="retro-text" style="text-align: center; color: #00a8ff; font-weight: bold;">> UPLOAD WHALE SOUND FILES BELOW TO BEGIN ANALYSIS</p>', unsafe_allow_html=True)
        
        # Source selection dropdown
        st.markdown("---")
        st.markdown('<p class="retro-text" style="color: #00d2d3;">> OR SELECT A DATA SOURCE TO AUTO-LOAD FILES</p>', unsafe_allow_html=True)
        st.info("ℹ️ **Note:** Some external sources (WHOI, British Library) may have broken URLs. Use 'Sample Files' for testing, or upload your own files.")
        
        # Initialize source files in session state
        if 'source_files' not in st.session_state:
            st.session_state.source_files = []
        if 'selected_source_name' not in st.session_state:
            st.session_state.selected_source_name = None
        
        available_sources = get_available_sources()
        source_options = ["None (Upload files manually)"] + list(available_sources.keys())
        
        selected_source = st.selectbox(
            "📥 Select Data Source",
            source_options,
            key="data_source_selector",
            help="Choose a data source to automatically download and load whale sound files",
            index=0 if st.session_state.selected_source_name is None else (
                source_options.index(st.session_state.selected_source_name) if st.session_state.selected_source_name in source_options else 0
            )
        )
        
        # Reset source files if source changed
        if st.session_state.selected_source_name != selected_source:
            st.session_state.source_files = []
            st.session_state.selected_source_name = selected_source
        
        if selected_source != "None (Upload files manually)":
            if st.button(f"📥 Load from {get_source_display_name(selected_source)}", key="load_source_button"):
                with st.spinner(f"Downloading files from {get_source_display_name(selected_source)}..."):
                    try:
                        source_files, errors = download_source_files(selected_source)
                        st.session_state.source_files = source_files
                        st.session_state.selected_source_name = selected_source
                        if source_files:
                            st.success(f"✓ Successfully loaded {len(source_files)} file(s) from {get_source_display_name(selected_source)}")
                            # Show file info
                            for i, file_obj in enumerate(source_files):
                                st.caption(f"  • {file_obj.name}")
                        else:
                            st.warning(f"No files could be downloaded from {get_source_display_name(selected_source)}")
                        
                        # Show any errors that occurred
                        if errors:
                            with st.expander("⚠️ Download Errors (click to view)", expanded=False):
                                for error in errors:
                                    st.error(error)
                    except Exception as e:
                        st.error(f"Error loading from source: {e}")
                        logger.exception(f"Error downloading from {selected_source}")
            
            # Show loaded files if any
            if st.session_state.source_files:
                st.info(f"📥 {len(st.session_state.source_files)} file(s) loaded from {get_source_display_name(selected_source)}")
                with st.expander("View loaded files"):
                    for file_obj in st.session_state.source_files:
                        st.text(f"  • {file_obj.name}")
        
        st.markdown("---")
        
        # File uploader
        st.markdown('<p class="retro-text" style="color: #00d2d3;">> MANUAL FILE UPLOAD</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload whale audio files", type=['wav', 'mp3'], accept_multiple_files=True)
        
        # Combine source files with uploaded files
        if st.session_state.source_files:
            if uploaded_files:
                # Combine both sources
                all_files = list(uploaded_files) + st.session_state.source_files
            else:
                # Use only source files
                all_files = st.session_state.source_files
        else:
            # Use only uploaded files
            all_files = list(uploaded_files) if uploaded_files else []
        
        if all_files:
            # Apply basic limits to protect the app from overly large or numerous uploads
            if len(all_files) > MAX_UPLOAD_FILES:
                st.warning(
                    f"Limiting processing to the first {MAX_UPLOAD_FILES} files "
                    "to keep the app responsive."
                )
                all_files = all_files[:MAX_UPLOAD_FILES]

            filtered_files = []
            for f in all_files:
                size = getattr(f, "size", None)
                if size is not None and size > MAX_UPLOAD_BYTES:
                    name = getattr(f, "name", "unnamed file")
                    st.warning(f"Skipping '{name}' because it is larger than 50 MB.")
                    continue
                filtered_files.append(f)

            all_files = filtered_files

            def file_sig(f):
                return (getattr(f, 'name', None) or str(f), getattr(f, 'size', 0))

            current_batch = st.session_state.current_batch
            current_batch_data = st.session_state.batches.get(current_batch, {}) if current_batch else {}
            current_files = current_batch_data.get('files') or []
            current_sigs = set(file_sig(f) for f in current_files)
            new_sigs = set(file_sig(f) for f in all_files)
            files_unchanged = current_sigs == new_sigs and len(current_sigs) == len(all_files)

            if not files_unchanged:
                # Clean up old batches: keep only the most recent before creating a new one
                if st.session_state.batches:
                    most_recent_batch = max(st.session_state.batches.keys())
                    st.session_state.batches = {most_recent_batch: st.session_state.batches[most_recent_batch]}
                    st.session_state.current_batch = most_recent_batch

                # Create a new batch only when file set changed or there is no batch
                batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.current_batch = batch_timestamp
                st.session_state.batches = {
                    batch_timestamp: {
                        'files': all_files,
                        'processed': False,
                        'species': 'Unknown'
                    }
                }
            else:
                # Same files: keep existing batch but refresh file list (upload widget can give new refs)
                batch_timestamp = current_batch
                st.session_state.batches[current_batch] = {
                    **current_batch_data,
                    'files': all_files,
                    'species': current_batch_data.get('species', 'Unknown')
                }
            st.session_state.files_uploaded = True
            
            # Species selector (tag batch with whale species)
            batch_species = st.session_state.batches.get(st.session_state.current_batch, {}).get('species', 'Unknown')
            species_idx = WHALE_SPECIES.index(batch_species) if batch_species in WHALE_SPECIES else 0
            selected_species = st.selectbox(
                "🐋 Species (optional)",
                WHALE_SPECIES,
                index=species_idx,
                key="batch_species_selector",
                help="Tag this batch with the whale species if known. Helps tailor analysis and terminology."
            )
            st.session_state.batches[st.session_state.current_batch]['species'] = selected_species
            
            # Show file count
            if st.session_state.source_files and selected_source != "None (Upload files manually)":
                st.info(f"📥 Loaded {len(st.session_state.source_files)} file(s) from {get_source_display_name(selected_source)}")
            if uploaded_files:
                st.info(f"📤 Uploaded {len(uploaded_files)} file(s)")
            st.info(f"📊 Total files ready: {len(all_files)}")
            
            # Process files button
            if st.button("Process Files", key="process_files_button"):
                with st.spinner("Processing files..."):
                    try:
                        processed_count = 0
                        clear_audio_tmp_dir()
                        st.session_state.audio_files = {}  # Reset audio files for new batch
                        
                        for file in all_files:
                            try:
                                if hasattr(file, 'getvalue'):
                                    fname = getattr(file, 'name', '') or ''
                                    ext = os.path.splitext(fname)[1] if fname else '.wav'
                                    if ext not in ('.wav', '.mp3', '.flac', '.ogg'):
                                        ext = '.wav'
                                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                                        tmp_file.write(file.getvalue())
                                        tmp_file_path = tmp_file.name
                                    try:
                                        audio_data, sample_rate = load_audio_optimized(tmp_file_path)
                                    finally:
                                        os.unlink(tmp_file_path)
                                else:
                                    audio_data, sample_rate = load_audio_optimized(file)
                                
                                file_name = file.name if hasattr(file, 'name') else str(file)
                                disk_path = save_audio_to_disk(audio_data, sample_rate)
                                st.session_state.audio_files[file_name] = {
                                    'path': disk_path,
                                    'sample_rate': sample_rate,
                                }
                                processed_count += 1
                                    
                            except Exception as e:
                                file_name = file.name if hasattr(file, 'name') else str(file)
                                st.error(f"Error processing {file_name}: {str(e)}")
                                logger.exception(f"Error processing file: {file_name}")
                                continue
                        
                        # Only mark as processed if we successfully processed at least one file
                        if processed_count > 0:
                            # Create a new dictionary for the batch to ensure state update
                            species = st.session_state.batches.get(batch_timestamp, {}).get('species', 'Unknown')
                            new_batch = {
                                'files': all_files,
                                'processed': True,
                                'species': species
                            }
                            # Update the session state with the new batch
                            st.session_state.batches = {batch_timestamp: new_batch}
                            st.success(f"Successfully processed {processed_count} files!")
                            st.rerun()
                        else:
                            st.error("No files were successfully processed.")
                    except Exception as e:
                        st.error(f"Error during batch processing: {str(e)}")
                        logger.error(f"Batch processing error: {str(e)}")
        
    elif page == "Spectrogram":
        st.header("Spectrogram")
        st.markdown("Explore **frequency content**: spectrograms, average spectrum, and spectral centroid over time.")
        
        # Check if we have a current batch and files
        if not st.session_state.current_batch or not st.session_state.batches:
            st.info("📁 **No files uploaded yet.** Go to 'Upload & Process' to upload WAV files, then click 'Process Files' to enable spectrogram visualization.")
            st.markdown("""
            **What you'll see here:**
            - Spectrograms and frequency spectrum
            - Spectral centroid over time
            - Feature space visualization
            """)
        else:
            species_tag = get_batch_species()
            batch_label = f"{st.session_state.current_batch}" + (f" ({species_tag})" if species_tag and species_tag != "Unknown" else "")
            st.info(f"Exploring batch from {batch_label}")
            current_files = st.session_state.batches[st.session_state.current_batch]['files']
            
            if not current_files:
                st.info("Upload and process some files first!")
            else:
                workflow = get_workflow()
                
                # Create feature matrix
                features = []
                for file in current_files:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        # Write the uploaded file to the temporary file
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Load the audio file from the temporary file
                        y, sr = librosa.load(tmp_file_path)
                        
                        # Extract features
                        feature_dict = {}
                        
                        # Extract MFCCs
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        feature_dict['mfccs'] = mfccs.mean(axis=1).tolist()
                        
                        # Extract other features
                        feature_dict['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
                        feature_dict['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
                        feature_dict['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
                        feature_dict['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)[0].mean()
                        feature_dict['rms_energy'] = librosa.feature.rms(y=y)[0].mean()
                        
                        features.append(feature_dict)
                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_file_path)
                
                # Convert features to numpy array for dimensionality reduction
                feature_matrix = []
                for feature_dict in features:
                    # Flatten all features into a single vector
                    feature_vector = []
                    for key in ['mfccs', 'spectral_centroid', 'spectral_rolloff', 
                              'spectral_bandwidth', 'zero_crossing_rate', 'rms_energy']:
                        if key in feature_dict:
                            if isinstance(feature_dict[key], list):
                                feature_vector.extend(feature_dict[key])
                            else:
                                feature_vector.append(feature_dict[key])
                    feature_matrix.append(feature_vector)
                
                # Reduce dimensions - reducer expects list of dicts
                if len(features) < 2:
                    st.warning("Need at least 2 files for dimensionality reduction visualization.")
                    reduced_features = None
                else:
                    reduced_features = workflow.reducer.reduce_dimensions(features)
                if reduced_features is not None:
                    filenames = [f.name for f in current_files]
                    
                    # Create interactive plot
                    df = pd.DataFrame({
                        'x': reduced_features[:, 0],
                        'y': reduced_features[:, 1],
                        'filename': filenames
                    })
                    
                    fig = px.scatter(
                        df,
                        x='x',
                        y='y',
                        hover_name='filename',
                        title='Whale Sound Feature Space',
                        width=1000,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to reduce dimensions. Please check the logs for details.")
                
                # Show individual file details
                selected_file = st.selectbox(
                    "Select a file to view details",
                    [f.name for f in current_files]
                )
                
                if selected_file:
                    file = next((f for f in current_files if f.name == selected_file), None)
                    if file is None:
                        st.warning(f"File {selected_file} not found in current batch")
                        file = current_files[0] if current_files else None
                    
                    # Spectrogram tab: frequency-first layout
                    st.subheader("📈 Spectrogram (frequency content)")
                    st.pyplot(plot_spectrogram(file), use_container_width=True)
                    with st.expander("Waveform (time domain reference)", expanded=False):
                        st.pyplot(plot_waveform(file), use_container_width=True)
                    
                    st.subheader("📊 Audio Features & Analysis")
                    audio_data, sample_rate = _get_audio_array(file.name)
                    if audio_data is None:
                        try:
                            if hasattr(file, 'getvalue'):
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                    tmp_file.write(file.getvalue())
                                    tmp_file_path = tmp_file.name
                                try:
                                    audio_data, sample_rate = load_audio_optimized(tmp_file_path)
                                finally:
                                    os.unlink(tmp_file_path)
                            else:
                                audio_data, sample_rate = load_audio_optimized(file)
                            disk_path = save_audio_to_disk(audio_data, sample_rate)
                            st.session_state.audio_files[file.name] = {'path': disk_path, 'sample_rate': sample_rate}
                        except Exception as e:
                            st.error(f"Failed to load audio data for {file.name}: {e}")
                            logger.exception(f"Error loading audio for {file.name}")
                            st.stop()
                    
                    # Spectrogram-specific: average spectrum and spectral centroid over time
                    try:
                        fig_spec, axes = plt.subplots(2, 1, figsize=(10, 5))
                        S = np.abs(librosa.stft(audio_data))
                        freqs = librosa.fft_frequencies(sr=sample_rate)
                        avg_spectrum = np.mean(S, axis=1)
                        axes[0].plot(freqs, avg_spectrum, color='#00d2d3', linewidth=0.8)
                        axes[0].set_title('Average frequency spectrum')
                        axes[0].set_xlabel('Frequency (Hz)')
                        axes[0].set_ylabel('Magnitude')
                        axes[0].set_xlim(0, sample_rate // 2)
                        spectral_centroid_times = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
                        times = librosa.frames_to_time(np.arange(len(spectral_centroid_times)), sr=sample_rate)
                        axes[1].plot(times, spectral_centroid_times, color='#00d2d3', linewidth=0.8)
                        axes[1].set_title('Spectral centroid over time')
                        axes[1].set_xlabel('Time (s)')
                        axes[1].set_ylabel('Frequency (Hz)')
                        fig_spec.tight_layout()
                        st.pyplot(fig_spec, use_container_width=True)
                        plt.close(fig_spec)
                    except Exception:
                        pass  # Optional visualization
                    
                    # Extract comprehensive features and render with consistent styling
                    try:
                        duration = len(audio_data) / sample_rate
                        file_size_bytes = len(file.getvalue()) if hasattr(file, 'getvalue') else 0
                        if file_size_bytes == 0:
                            file_size_bytes = len(audio_data) * 4  # float32 estimate
                        
                        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
                        mfcc_means = [mfccs[i].mean() for i in range(13)]
                        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0].mean()
                        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0].mean()
                        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0].mean()
                        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0].mean()
                        rms_energy = librosa.feature.rms(y=audio_data)[0].mean()
                        
                        html = render_audio_features_panel(
                            duration=duration,
                            sample_rate=sample_rate,
                            n_samples=len(audio_data),
                            file_size_bytes=file_size_bytes,
                            dtype_name=str(audio_data.dtype),
                            spectral_centroid=spectral_centroid,
                            spectral_rolloff=spectral_rolloff,
                            spectral_bandwidth=spectral_bandwidth,
                            zero_crossing_rate=zero_crossing_rate,
                            rms_energy=rms_energy,
                            mfcc_means=mfcc_means,
                        )
                        st.markdown(html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Failed to extract features: {e}")
                        logger.exception(f"Error extracting features for {file.name}")
    
    elif page == "Waveform":
        st.header("Waveform")
        st.markdown("Explore **time & amplitude**: waveforms, amplitude envelope (RMS), and temporal structure.")
        
        # Check if we have a current batch and files
        if not st.session_state.current_batch or not st.session_state.batches:
            st.info("📁 **No files uploaded yet.** Go to 'Upload & Process' to upload WAV files, then click 'Process Files' to enable waveform visualization.")
            st.markdown("""
            **What you'll see here:**
            - Waveform and amplitude envelope (RMS over time)
            - Feature space visualization
            """)
        else:
            species_tag = get_batch_species()
            batch_label = f"{st.session_state.current_batch}" + (f" ({species_tag})" if species_tag and species_tag != "Unknown" else "")
            st.info(f"Exploring batch from {batch_label}")
            current_files = st.session_state.batches[st.session_state.current_batch]['files']
            
            if not current_files:
                st.info("Upload and process some files first!")
            else:
                workflow = get_workflow()
                
                # Create feature matrix
                features = []
                for file_obj in current_files:
                    # Normalize file info (handles BytesIO, dict, etc.)
                    file_info = normalize_file_info(file_obj)
                    if file_info is None or 'features' not in file_info:
                        logger.warning(f"Skipping file {file_obj} - could not extract features")
                        continue
                    
                    # Extract MFCCs and other features
                    feature_dict = {}
                    # Get all MFCCs
                    mfccs = []
                    for i in range(1, 14):  # We have 13 MFCCs
                        mfcc_key = f'mfcc_{i}'
                        if mfcc_key in file_info['features']:
                            mfccs.append(file_info['features'][mfcc_key])
                    feature_dict['mfccs'] = mfccs
                    
                    # Add other features
                    for key in ['spectral_centroid', 'spectral_rolloff', 
                              'spectral_bandwidth', 'zero_crossing_rate', 'rms_energy']:
                        if key in file_info['features']:
                            feature_dict[key] = file_info['features'][key]
                    
                    features.append(feature_dict)
                
                # Reduce dimensions
                if len(features) < 2:
                    st.warning("Need at least 2 files for dimensionality reduction visualization.")
                    reduced_features = None
                else:
                    reduced_features = workflow.reducer.reduce_dimensions(features)
                if reduced_features is not None:
                    filenames = []
                    for f in current_files:
                        file_info = normalize_file_info(f)
                        if file_info:
                            filenames.append(file_info.get('filename', 'unknown'))
                    
                    # Create interactive plot
                    df = pd.DataFrame({
                        'x': reduced_features[:, 0],
                        'y': reduced_features[:, 1],
                        'filename': filenames
                    })
                    
                    fig = px.scatter(
                        df,
                        x='x',
                        y='y',
                        hover_name='filename',
                        title='Whale Sound Feature Space',
                        width=1000,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to reduce dimensions. Please check the logs for details.")
                
                # Show individual file details
                # Get normalized file info for all files to build selectbox options
                normalized_files = []
                file_names = []
                for f in current_files:
                    file_info = normalize_file_info(f)
                    if file_info:
                        normalized_files.append(file_info)
                        file_names.append(file_info.get('filename', 'unknown'))
                
                if not file_names:
                    st.warning("No files with valid features found.")
                    st.stop()
                
                selected_file_name = st.selectbox(
                    "Select a file to view details",
                    file_names
                )
                
                # Find the selected file info
                selected_file_info = next((f for f in normalized_files if f.get('filename') == selected_file_name), None)
                if selected_file_info is None:
                    st.error("Could not find selected file.")
                    st.stop()
                
                file_info = selected_file_info
                
                # Get file path/name for plotting
                file_path = file_info.get('path') or file_info.get('filename', '')
                file_name = file_info.get('filename', 'unknown')
                
                audio_data, sample_rate = _get_audio_array(file_name)
                if audio_data is None:
                    try:
                        original_file = next((f for f in current_files if
                                            (hasattr(f, 'name') and f.name == file_name) or
                                            (isinstance(f, dict) and f.get('filename') == file_name)), None)
                        if original_file:
                            if hasattr(original_file, 'getvalue'):
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                    tmp_file.write(original_file.getvalue())
                                    tmp_file_path = tmp_file.name
                                try:
                                    audio_data, sample_rate = load_audio_optimized(tmp_file_path)
                                finally:
                                    os.unlink(tmp_file_path)
                            elif isinstance(original_file, (str, Path)) and Path(original_file).exists():
                                audio_data, sample_rate = load_audio_optimized(original_file)
                            if audio_data is not None and sample_rate is not None:
                                disk_path = save_audio_to_disk(audio_data, sample_rate)
                                st.session_state.audio_files[file_name] = {'path': disk_path, 'sample_rate': sample_rate}
                    except Exception as e:
                        st.error(f"Failed to load audio data for {file_name}: {e}")
                        logger.exception(f"Error loading audio for {file_name}")
                
                if audio_data is None or sample_rate is None:
                    st.warning(f"Could not load audio data for {file_name}.")
                    st.stop()
                
                # Waveform tab: time-first layout
                st.subheader("📈 Waveform (amplitude over time)")
                tmp_waveform_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_waveform_path = tmp_file.name
                        sf.write(tmp_waveform_path, audio_data, sample_rate)
                    st.pyplot(plot_waveform(tmp_waveform_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to plot waveform: {e}")
                finally:
                    if tmp_waveform_path and os.path.exists(tmp_waveform_path):
                        try:
                            os.unlink(tmp_waveform_path)
                        except Exception:
                            pass
                
                # Waveform-specific: amplitude envelope (RMS over time)
                try:
                    rms = librosa.feature.rms(y=audio_data)[0]
                    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate)
                    fig_env, ax = plt.subplots(figsize=(10, 2.5))
                    ax.fill_between(times, rms, alpha=0.5, color='#00d2d3')
                    ax.plot(times, rms, color='#00d2d3', linewidth=0.8)
                    ax.set_title('Amplitude envelope (RMS over time)')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('RMS')
                    fig_env.tight_layout()
                    st.pyplot(fig_env, use_container_width=True)
                    plt.close(fig_env)
                except Exception:
                    pass
                
                with st.expander("Spectrogram (frequency domain reference)", expanded=False):
                    tmp_spectrogram_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            tmp_spectrogram_path = tmp_file.name
                            sf.write(tmp_spectrogram_path, audio_data, sample_rate)
                        st.pyplot(plot_spectrogram(tmp_spectrogram_path), use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to plot spectrogram: {e}")
                    finally:
                        if tmp_spectrogram_path and os.path.exists(tmp_spectrogram_path):
                            try:
                                os.unlink(tmp_spectrogram_path)
                            except Exception:
                                pass
                
                st.subheader("📊 Audio Features & Analysis")
                
                # Extract comprehensive features and render with consistent styling
                try:
                    duration = len(audio_data) / sample_rate
                    file_size_bytes = 0
                    orig = next((f for f in current_files if
                                (hasattr(f, 'name') and f.name == file_name) or
                                (isinstance(f, dict) and f.get('filename') == file_name)), None)
                    if orig and hasattr(orig, 'getvalue'):
                        file_size_bytes = len(orig.getvalue())
                    elif isinstance(file_path, (str, Path)) and Path(str(file_path)).exists():
                        file_size_bytes = Path(file_path).stat().st_size
                    
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
                    mfcc_means = [mfccs[i].mean() for i in range(13)]
                    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0].mean()
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0].mean()
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0].mean()
                    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0].mean()
                    rms_energy = librosa.feature.rms(y=audio_data)[0].mean()
                    
                    html = render_audio_features_panel(
                        duration=duration,
                        sample_rate=sample_rate,
                        n_samples=len(audio_data),
                        file_size_bytes=file_size_bytes,
                        dtype_name=str(audio_data.dtype),
                        spectral_centroid=spectral_centroid,
                        spectral_rolloff=spectral_rolloff,
                        spectral_bandwidth=spectral_bandwidth,
                        zero_crossing_rate=zero_crossing_rate,
                        rms_energy=rms_energy,
                        mfcc_means=mfcc_means,
                    )
                    st.markdown(html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to extract features: {e}")
                    logger.exception(f"Error extracting features for {file_name}")
    
    elif page == "Pattern Detection":
        st.header("Pattern Detection")
        st.markdown("Detect and analyze communication patterns in whale vocalizations.")
        
        if not st.session_state.current_batch or not st.session_state.batches:
            st.info("📁 **No files uploaded yet.** Go to 'Upload & Process' to upload WAV files, then click 'Process Files' to enable pattern detection.")
            st.markdown("""
            **What you'll see here:**
            - Vocalization pattern recognition
            - Temporal structure analysis
            - Communication context understanding
            """)
        else:
            species_tag = get_batch_species()
            batch_label = f"{st.session_state.current_batch}" + (f" ({species_tag})" if species_tag and species_tag != "Unknown" else "")
            st.info(f"Analyzing communication patterns in batch {batch_label}")
            current_files = st.session_state.batches[st.session_state.current_batch]['files']
            
            if not current_files:
                st.info("Upload and process some files first to analyze communication patterns!")
            else:
                # Create tabs for different types of insights
                tab1, tab2, tab3, tab4 = st.tabs(["Coda Patterns", "Temporal Structure", "Communication Context", "Segment grammar (AI)"])
            
            with tab1:
                st.subheader("Coda Pattern Analysis")
                
                # Analyze coda patterns (groups of clicks)
                coda_patterns = []
                for file_obj in current_files:
                    # Normalize file info (handles BytesIO, dict, etc.)
                    file_info = normalize_file_info(file_obj)
                    if file_info is None or 'features' not in file_info:
                        logger.warning(f"Skipping file {file_obj} - could not extract features")
                        continue
                    
                    # Extract temporal features that might indicate codas
                    features = file_info['features']
                    pattern = {
                        'duration': features.get('duration', 0),
                        'zero_crossing_rate': features.get('zero_crossing_rate', 0),
                        'spectral_centroid': features.get('spectral_centroid', 0),
                        'rms_energy': features.get('rms_energy', 0)
                    }
                    coda_patterns.append(pattern)
                
                # Create pattern visualization (avoid scatter_matrix axis overlap)
                df = pd.DataFrame(coda_patterns)
                dims = ['duration', 'zero_crossing_rate', 'spectral_centroid', 'rms_energy']
                # Correlation heatmap (needs at least 2 files)
                st.subheader("Feature correlations")
                if len(df) < 2:
                    st.info("Need at least **2 files** to compute feature correlations. Upload and process more files.")
                else:
                    corr = df[dims].corr()
                    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                         title="Coda pattern feature correlations")
                    fig_corr.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_corr, use_container_width=True)
                # 2D scatter with clear axes (pick two dimensions)
                st.subheader("Feature relationships (pick axes)")
                if len(df) >= 1:
                    x_dim = st.selectbox("X axis", dims, key="coda_x")
                    y_dim = st.selectbox("Y axis", [d for d in dims if d != x_dim], key="coda_y")
                    fig = px.scatter(df, x=x_dim, y=y_dim, color="duration",
                                     title=f"Coda pattern: {x_dim} vs {y_dim}",
                                     labels={x_dim: x_dim.replace("_", " ").title(), y_dim: y_dim.replace("_", " ").title()})
                    fig.update_layout(xaxis_title_font_size=14, yaxis_title_font_size=14)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No files with extracted features. Process files in Upload & Process first.")
                
                # Enhanced interpretation with research references
                st.info("""
                ### Understanding Vocalization Patterns
                
                **What are Codas?** (sperm whale–specific)
                - Codas are distinct patterns of clicks used by sperm whales
                - Other species use different structures: humpbacks sing songs; blue whales produce moans
                - The feature analysis below applies broadly to any whale vocalizations
                
                **Research References (sperm whale codas):**
                - "Sperm whale codas: A study of vocal communication" (Watkins & Schevill, 1977)
                - "The structure of sperm whale codas" (Weilgart & Whitehead, 1997)
                - "Cultural transmission of vocal patterns in sperm whales" (Rendell & Whitehead, 2001)
                
                **What to Look For:**
                1. **Clusters in the Visualization**
                   - Groups of similar points might indicate common coda types
                   - Different clusters could represent different "words" or "phrases"
                   - Reference: "Classification of sperm whale codas" (Gero et al., 2016)
                
                2. **Feature Relationships**
                   - Duration vs. Click Density: Longer codas with more clicks might indicate complex messages
                   - Frequency vs. Energy: Higher frequencies with more energy might indicate urgency or emphasis
                   - Reference: "Acoustic features of sperm whale codas" (Madsen et al., 2002)
                
                3. **Pattern Significance**
                   - Regular patterns might indicate standard communication
                   - Outliers might indicate unique or special messages
                   - Gaps in the pattern space might indicate unused or impossible combinations
                   - Reference: "Patterns in sperm whale communication" (Whitehead, 2003)
                
                **Research Context:**
                - Sperm whales are known to use different coda types in different contexts
                - Some codas are used for social bonding, others for navigation
                - The structure of codas may follow grammatical rules similar to human language
                - Reference: "Grammar in sperm whale communication" (Schulz et al., 2008)
                """)
            
            with tab2:
                st.subheader("Temporal Structure Analysis")
                
                # Analyze timing patterns
                timing_features = []
                for file_obj in current_files:
                    # Normalize file info (handles BytesIO, dict, etc.)
                    file_info = normalize_file_info(file_obj)
                    if file_info is None or 'features' not in file_info:
                        logger.warning(f"Skipping file {file_obj} - could not extract features")
                        continue
                    features = file_info['features']
                    timing = {
                        'duration': features.get('duration', 0),
                        'spectral_rolloff': features.get('spectral_rolloff', 0),
                        'spectral_bandwidth': features.get('spectral_bandwidth', 0)
                    }
                    timing_features.append(timing)
                
                # Create timing pattern visualization
                df = pd.DataFrame(timing_features)
                fig = px.scatter_3d(
                    df,
                    x='duration',
                    y='spectral_rolloff',
                    z='spectral_bandwidth',
                    title="Temporal Structure Patterns",
                    color='duration'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced interpretation with research references
                st.info("""
                ### Understanding Temporal Structure
                
                **What is Temporal Structure?**
                - How whale sounds are organized in time
                - The rhythm and timing of communication
                - The sequence of different sound types
                
                **Research References:**
                - "Temporal patterns in sperm whale communication" (Whitehead & Weilgart, 1991)
                - "Rhythmic structure in whale vocalizations" (Payne & McVay, 1971)
                - "Timing and sequencing in whale communication" (Tyack, 2000)
                
                **What to Look For:**
                1. **Duration Patterns**
                   - Regular durations might indicate standard communication
                   - Variable durations might indicate different message types
                   - Sudden changes might indicate emphasis or urgency
                   - Reference: "Duration patterns in whale communication" (Miller et al., 2004)
                
                2. **Frequency Changes**
                   - Smooth transitions might indicate connected thoughts
                   - Abrupt changes might indicate new topics or emphasis
                   - Repeated patterns might indicate important concepts
                   - Reference: "Frequency modulation in whale sounds" (Au et al., 2006)
                
                3. **Bandwidth Usage**
                   - Wide bandwidth might indicate complex messages
                   - Narrow bandwidth might indicate simple or routine communication
                   - Changes in bandwidth might indicate emotional content
                   - Reference: "Spectral analysis of whale communication" (Mellinger et al., 2007)
                
                **Research Context:**
                - Whale communication often follows rhythmic patterns
                - Timing is crucial for understanding message structure
                - Different temporal patterns may indicate different communication purposes
                - Reference: "The role of timing in whale communication" (Janik, 2009)
                """)
            
            with tab3:
                st.subheader("Communication Context Analysis")
                
                # Analyze potential communication contexts
                if len(st.session_state.batches) > 1:
                    st.write("Comparing communication patterns across batches:")
                    
                    # Select batches to compare
                    batch_options = list(st.session_state.batches.keys())
                    compare_batches = st.multiselect(
                        "Select batches to compare",
                        batch_options,
                        default=[st.session_state.current_batch]
                    )
                    
                    if len(compare_batches) >= 2:
                        # Calculate communication pattern differences
                        pattern_differences = []
                        for batch_id in compare_batches:
                            batch_files = st.session_state.batches[batch_id]['files']
                            durations = []
                            zcrs = []
                            centroids = []
                            for f in batch_files:
                                file_info = normalize_file_info(f)
                                if file_info and 'features' in file_info:
                                    feat = file_info['features']
                                    durations.append(feat.get('duration', 0))
                                    zcrs.append(feat.get('zero_crossing_rate', 0))
                                    centroids.append(feat.get('spectral_centroid', 0))
                            avg_features = {
                                'duration': np.mean(durations) if durations else 0,
                                'zero_crossing_rate': np.mean(zcrs) if zcrs else 0,
                                'spectral_centroid': np.mean(centroids) if centroids else 0
                            }
                            pattern_differences.append({
                                'batch': batch_id,
                                **avg_features
                            })
                        
                        # Create comparison visualization
                        df = pd.DataFrame(pattern_differences)
                        fig = px.bar(
                            df,
                            x='batch',
                            y=['duration', 'zero_crossing_rate', 'spectral_centroid'],
                            title="Communication Pattern Differences",
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add interpretation
                        st.info("""
                        This comparison shows how communication patterns differ between batches.
                        Differences might indicate:
                        - Different types of communication (e.g., social vs. navigation)
                        - Different contexts (e.g., group vs. individual)
                        - Different emotional states or intentions
                        """)
                else:
                    st.info("Upload multiple batches to compare communication contexts")
            
            with tab4:
                st.subheader("Segment grammar (AI)")
                st.markdown("Treat each file as a **sequence of short segments**. We cluster segment acoustics into **symbols** (pseudo-words) and show **what tends to follow what**—a simple grammar-like view.")
                segment_sec = st.slider("Segment length (s)", 0.5, 3.0, 1.5, 0.25, key="seg_sec")
                n_symbols = st.slider("Number of symbols (clusters)", 3, 12, 6, key="n_sym")
                all_vectors = []
                file_segment_vectors = []
                file_names_seg = []
                for file_obj in current_files:
                    file_info = normalize_file_info(file_obj)
                    if not file_info or 'features' not in file_info:
                        continue
                    fname = file_info.get('filename', 'unknown')
                    y, sr = _get_audio_array(fname)
                    if y is None:
                        try:
                            if hasattr(file_obj, 'getvalue'):
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                    tmp.write(file_obj.getvalue())
                                    y, sr = load_audio_optimized(tmp.name)
                                    os.unlink(tmp.name)
                            else:
                                y, sr = load_audio_optimized(file_obj)
                            disk_path = save_audio_to_disk(y, sr)
                            st.session_state.audio_files[fname] = {'path': disk_path, 'sample_rate': sr}
                        except Exception:
                            continue
                    vecs, _ = segment_audio_features(y, sr, segment_sec=segment_sec)
                    if len(vecs) > 0:
                        all_vectors.append(vecs)
                        file_segment_vectors.append(vecs)
                        file_names_seg.append(fname)
                if not all_vectors:
                    st.warning("No segment data. Ensure files are loaded and long enough for at least one segment.")
                else:
                    X = np.vstack(all_vectors)
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X)
                    kmeans = KMeans(n_clusters=n_symbols, random_state=42, n_init=10)
                    labels_all = kmeans.fit_predict(Xs)
                    idx = 0
                    sequences = []
                    for i, vecs in enumerate(file_segment_vectors):
                        n = len(vecs)
                        seq = labels_all[idx:idx + n]
                        idx += n
                        sequences.append((file_names_seg[i], seq))
                    st.markdown("#### Symbol sequences (one row per file)")
                    for fname, seq in sequences:
                        sym_str = " → ".join(str(s) for s in seq)
                        st.text(f"{fname}:  [ {sym_str} ]")
                    st.markdown("#### Transition matrix (what follows what)")
                    bigrams = []
                    for _, seq in sequences:
                        for j in range(len(seq) - 1):
                            bigrams.append((seq[j], seq[j + 1]))
                    if bigrams:
                        trans = np.zeros((n_symbols, n_symbols))
                        for a, b in bigrams:
                            trans[a, b] += 1
                        row_sum = trans.sum(axis=1, keepdims=True)
                        trans_p = np.divide(trans, row_sum, where=row_sum > 0)
                        fig = px.imshow(trans_p, x=[f"S{j}" for j in range(n_symbols)],
                                        y=[f"S{j}" for j in range(n_symbols)],
                                        labels=dict(x="Next symbol", y="Current symbol"),
                                        title="P(Next | Current) — grammar-like structure",
                                        color_continuous_scale="Blues")
                        st.plotly_chart(fig, use_container_width=True)
                    st.info("Symbols are clusters of similar-sounding segments. Repeated transitions (e.g. S2→S4) may reflect recurring \"phrases\" in the data.")
            
            # Enhanced general insights section with research references
            st.subheader("Key Insights and Future Directions")
            st.write("""
            ### Current Capabilities
            
            **Pattern Recognition**
            - Identifying basic coda structures (Gero et al., 2016)
            - Recognizing temporal patterns (Whitehead & Weilgart, 1991)
            - Detecting communication contexts (Whitehead, 1996)
            
            **Analysis Tools**
            - Feature visualization (Madsen et al., 2002)
            - Pattern comparison (Rendell & Whitehead, 2001)
            - Context analysis (Clark et al., 2006)
            
            ### Future Directions
            
            **Short-term Improvements**
            1. **Enhanced Pattern Recognition**
               - More sophisticated clustering algorithms
               - Better identification of coda types
               - Improved temporal pattern detection
               - Building on: "Machine learning in bioacoustics" (Stowell et al., 2019)
            
            2. **Context Understanding**
               - Integration with environmental data
               - Social context analysis
               - Behavioral correlation
               - Building on: "Multi-modal analysis of whale behavior" (Allen et al., 2017)
            
            **Long-term Goals**
            1. **Language Structure**
               - Grammar pattern recognition
               - Syntax analysis
               - Semantic understanding
               - Building on: "Linguistic analysis of whale communication" (Schulz et al., 2021)
            
            2. **Translation Framework**
               - Behavior correlation
               - Context mapping
               - Meaning inference
               - Building on: "Towards whale language translation" (Whitehead et al., 2022)
            
            ### Research Integration
            
            This tool is designed to complement existing whale communication research by:
            - Providing visual analysis tools (Mellinger et al., 2007)
            - Enabling pattern discovery (Gero et al., 2016)
            - Supporting hypothesis testing (Janik, 2009)
            - Facilitating data comparison (Rendell et al., 2012)
            
            ### How to Use These Insights
            
            1. **Start with Patterns**
               - Look for clusters in the visualizations
               - Identify recurring structures
               - Note unusual patterns
               - Reference: "Pattern recognition in whale sounds" (Tyack & Clark, 2000)
            
            2. **Consider Context**
               - Think about when patterns occur
               - Consider environmental factors
               - Note social situations
               - Reference: "Contextual analysis of whale communication" (Connor et al., 2000)
            
            3. **Build Hypotheses**
               - Formulate possible meanings
               - Test against known behaviors
               - Look for confirming patterns
               - Reference: "Hypothesis testing in whale communication research" (Whitehead, 2009)
            
            4. **Share Findings**
               - Document interesting patterns
               - Compare with other research
               - Contribute to collective understanding
               - Reference: "Collaborative whale communication research" (Rendell et al., 2019)
            """)
    
    elif page == "Coda Rhythm & Timing":
        st.header("Coda Rhythm & Timing")
        st.markdown("In sperm whales, **codas** are short sequences of clicks with characteristic timing. This tab detects click-like onsets and shows **inter-click intervals (ICIs)**—the rhythm that may carry meaning. Similar rhythm analysis applies to other pulsed or click-based whale vocalizations.")
        
        if not st.session_state.current_batch or not st.session_state.batches:
            st.info("📁 **No files loaded.** Go to 'Upload & Process' to load WAV files, then click 'Process Files'.")
        else:
            current_files = st.session_state.batches[st.session_state.current_batch]['files']
            if not current_files:
                st.info("No files in the current batch.")
            else:
                # File selector
                file_list = []
                for f in current_files:
                    file_info = normalize_file_info(f)
                    if file_info:
                        file_list.append((f, file_info.get('filename', 'unknown')))
                
                if not file_list:
                    st.warning("Could not load file info for any file.")
                else:
                    selected_label = st.selectbox(
                        "Select file to analyze rhythm",
                        [label for _, label in file_list],
                        key="coda_rhythm_file"
                    )
                    selected_file_obj = next((f for f, label in file_list if label == selected_label), None)
                    if selected_file_obj is None:
                        st.stop()
                    
                    # Load audio
                    file_info = normalize_file_info(selected_file_obj)
                    if not file_info or 'features' not in file_info:
                        st.error("Could not load audio for selected file.")
                        st.stop()
                    
                    file_name = file_info.get('filename', 'unknown')
                    y, sr = _get_audio_array(file_name)
                    if y is None:
                        try:
                            if hasattr(selected_file_obj, 'getvalue'):
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                    tmp.write(selected_file_obj.getvalue())
                                    y, sr = load_audio_optimized(tmp.name)
                                    os.unlink(tmp.name)
                            else:
                                y, sr = load_audio_optimized(selected_file_obj)
                            disk_path = save_audio_to_disk(y, sr)
                            st.session_state.audio_files[file_name] = {'path': disk_path, 'sample_rate': sr}
                        except Exception as e:
                            st.error(f"Failed to load audio: {e}")
                            st.stop()
                    
                    # Tuning (optional)
                    with st.expander("Tuning (optional)"):
                        hop_length = st.slider("Hop length", 128, 512, 256, key="coda_hop")
                        wait_frames = st.slider("Min frames between onsets", 3, 20, 8, key="coda_wait")
                        delta = st.slider("Onset threshold (delta)", 0.02, 0.2, 0.07, step=0.01, key="coda_delta")
                    
                    onset_times, ici = compute_onset_and_ici(y, sr, hop_length=hop_length, wait_frames=wait_frames, delta=delta)
                    
                    if len(onset_times) == 0:
                        st.warning("No onsets detected. Try lowering the onset threshold (delta) or reducing 'Min frames between onsets' in Tuning.")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Onsets (clicks) detected", len(onset_times))
                            if len(ici) > 0:
                                st.metric("Mean ICI (ms)", f"{np.mean(ici) * 1000:.1f}")
                                st.metric("Median ICI (ms)", f"{np.median(ici) * 1000:.1f}")
                        with col2:
                            st.metric("Duration (s)", f"{len(y) / sr:.2f}")
                            if len(ici) > 0:
                                st.metric("Min ICI (ms)", f"{np.min(ici) * 1000:.1f}")
                                st.metric("Max ICI (ms)", f"{np.max(ici) * 1000:.1f}")
                        
                        # ICI histogram
                        if len(ici) > 0:
                            st.subheader("Inter-click interval distribution")
                            ici_ms = ici * 1000
                            fig = px.histogram(x=ici_ms, nbins=min(50, max(20, len(ici) // 5)),
                                               labels={'x': 'ICI (ms)', 'y': 'Count'},
                                               title='Distribution of time between consecutive clicks')
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Click timeline (raster)
                        st.subheader("Click timeline")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=onset_times,
                            y=np.ones_like(onset_times),
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=10, color='#00d2d3'),
                            name='Onsets'
                        ))
                        fig.update_layout(
                            xaxis_title='Time (s)',
                            yaxis_title='',
                            yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
                            title='Detected click times',
                            height=180
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Coda segmentation and AI-style analysis
                        gap_threshold = st.slider("Coda gap threshold (s) — gaps longer than this start a new coda", 0.2, 1.0, 0.4, 0.1, key="coda_gap")
                        codas = segment_codas_by_gap(onset_times, gap_threshold_sec=gap_threshold)
                        if codas:
                            st.subheader("Detected codas (AI segmentation)")
                            coda_rows = []
                            for i, c in enumerate(codas):
                                icis_ms = c['icis'] * 1000
                                best_name, best_score, _ = suggest_coda_type(c['n_clicks'], icis_ms)
                                sugg = f"{best_name} ({best_score*100:.0f}%)" if best_name else "—"
                                coda_rows.append({
                                    "Coda": i + 1,
                                    "Clicks": c['n_clicks'],
                                    "Start (s)": f"{c['start']:.2f}",
                                    "End (s)": f"{c['end']:.2f}",
                                    "Mean ICI (ms)": f"{np.mean(icis_ms):.1f}" if len(icis_ms) else "—",
                                    "Suggested type": sugg,
                                })
                            st.dataframe(pd.DataFrame(coda_rows), use_container_width=True, hide_index=True)
                            # Rhythm space: UMAP of codas
                            if len(codas) >= 3:
                                st.subheader("Rhythm space (UMAP)")
                                rhythm_vecs = []
                                for c in codas:
                                    icis_ms = c['icis'] * 1000
                                    mean_i = np.mean(icis_ms) if len(icis_ms) else 0
                                    std_i = np.std(icis_ms) if len(icis_ms) > 1 else 0
                                    rhythm_vecs.append([c['n_clicks'], mean_i, std_i])
                                R = np.array(rhythm_vecs)
                                R_scaled = StandardScaler().fit_transform(R)
                                n_codas = len(codas)
                                n_neighbors = min(5, n_codas - 1)
                                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
                                emb = reducer.fit_transform(R_scaled)
                                fig_r = px.scatter(x=emb[:, 0], y=emb[:, 1],
                                                   color=[c['n_clicks'] for c in codas],
                                                   size=[3 + c['n_clicks'] for c in codas],
                                                   title="Coda rhythm space — similar rhythms cluster together",
                                                   labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Click count"})
                                st.plotly_chart(fig_r, use_container_width=True)
                        else:
                            st.caption("No codas segmented (try lowering the gap threshold).")
                        
                        st.info("""
                        **Interpreting rhythm**
                        - Sperm whale codas are often described by **click counts** and **rhythm** (e.g. "1+1+3" = two short gaps, then three clicks). Other species use different structures.
                        - Mean ICI in the 100–500 ms range is typical for social codas; echolocation clicks are usually faster.
                        - **Suggested type** compares each detected coda to known patterns from the literature (see Coda Reference).
                        """)
    
    elif page == "Coda Reference":
        st.header("Coda Reference")
        st.markdown("Known **sperm whale coda types** from research (click-based vocalizations). Use this to compare with **Coda Rhythm & Timing** or **Pattern Detection** for sperm whale audio. Other species (e.g. humpback songs) have different reference patterns.")
        
        st.subheader("What are codas?")
        st.markdown("""
        **Sperm whale codas** are short, stereotyped patterns of clicks used in social contexts. Different species use different vocal structures (songs, moans, clicks); this reference focuses on sperm whale codas. 
        Researchers often describe codas by **number of clicks** and **rhythm** (regular vs irregular, and inter-click intervals).
        """)
        
        st.subheader("Common coda types (from literature)")
        
        ref_data = [
            ("1+1+3", "Two single clicks, then three clicks", "Common in Caribbean; often “identity” or social"),
            ("4+1", "Four clicks then one", "Frequently reported; function debated"),
            ("5R", "Five clicks, regular spacing", "Regular (R) rhythm"),
            ("3+1+1+3", "Alternating groups", "More complex pattern"),
            ("9R", "Nine regular clicks", "Longer coda"),
            ("1+1+1+1", "Four single clicks", "Slow, spaced"),
        ]
        ref_df = pd.DataFrame(ref_data, columns=["Pattern", "Description", "Notes"])
        st.dataframe(ref_df, use_container_width=True, hide_index=True)
        
        st.subheader("References")
        st.markdown("""
        - **Watkins & Schevill (1977)** – Sperm whale codas; first detailed descriptions.  
        - **Weilgart & Whitehead (1997)** – Coda structure and geographic variation.  
        - **Rendell & Whitehead (2003)** – Cultural transmission of codas.  
        - **Gero et al. (2016)** – Coda classification and social context.  
        - **Antunes et al. (2011)** – Coda repertoires and population identity.
        """)
        
        st.info("Use **Coda Rhythm & Timing** to see inter-click intervals (ICIs) in your recordings and compare their distribution to these patterns.")
    
    elif page == "About":
        st.header("About Whale Sound Analysis")
        st.markdown("""
        This app is built around one goal: **to support the search for meaning in whale communication.**
        It focuses on *codas* (sperm whale click patterns) and on **rhythm and timing** analysis. Similar approaches apply to other whale vocalizations.
        """)
        
        st.subheader("Tabs and how they help")
        st.markdown("""
        - **Upload & Process** – Load WAV files (or sample sources) and extract features.
        - **Spectrogram / Waveform** – Inspect the raw signal and frequency content of each file.
        - **Pattern Detection** – Explore coda-like structure (duration, spectral and energy features) and see how files cluster.
        - **Coda Rhythm & Timing** – Detect click-like onsets and **inter-click intervals (ICIs)**. Rhythm is how codas are defined in the literature (e.g. 1+1+3, 4+1).
        - **Coda Reference** – Compare what you see to known coda types and key papers.
        """)
        
        st.subheader("Is decoding actually possible?")
        st.markdown("""
        **Short answer:** We don’t know yet—but it’s a serious research question, and tools like this can contribute.
        
        **What we know**
        - Sperm whale codas vary by population (dialects), are learned, and are used in social contexts. Other species have their own vocal structures.  
        - Rhythm (timing between clicks) and click count describe coda *types*; some types may have broad context (e.g. identity, coordination).  
        - So far we have no proof of a “whale language” with a grammar and lexicon like human language.
        
        **What would help**
        - **More data with context** – Who is calling, where, before/after what behavior.  
        - **Consistent labels** – Coda types and contexts from biologists.  
        - **Novel and AI-driven ideas** – Embeddings, sequence models, or cross-modal links could reveal structure we don’t see by eye.  
        - **Collaboration** – Decoding will need biologists, acousticians, and ML working together.
        
        **Could this lead to a “cipher” or codex?**  
        A full “codex” (a reliable map from sound to meaning) would require evidence that specific patterns map to specific meanings or functions—something we don’t have yet.  
        What *is* possible is to build a **better picture of the building blocks**: which patterns exist, how they repeat, and how they vary. That’s what this app is for.  
        If future work—including AI-designed analyses—finds stable links between patterns and context or behavior, *then* we could start talking about something like a codex. Until then, we’re gathering and organizing the pieces.
        """)
        
        st.info("Use **Coda Rhythm & Timing** and **Coda Reference** together: measure ICIs in your recordings, then compare them to known coda types from the literature.")

    # Whale silhouette - giant blurred oval, pure CSS animation
    st.markdown("""
    <div class="whale-background-overlay" style="position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;
        pointer-events:none!important;z-index:99999!important;">
        <div style="position:absolute!important;top:12%!important;left:0!important;width:140%!important;
            max-width:1800px!important;height:320px!important;margin-top:-160px!important;
            animation: whale-swim 90s linear infinite;
            -webkit-animation: whale-swim 90s linear infinite;">
            <svg viewBox="0 0 400 80" style="width:100%;height:100%;filter:blur(50px);">
                <ellipse cx="200" cy="40" rx="195" ry="38" fill="rgba(0,0,12,0.75)"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 