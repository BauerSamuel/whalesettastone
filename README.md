# Whale-lingo 🐋

A tool for analyzing and visualizing whale audio data across species (sperm whale, humpback, blue whale, and others).

## Features

- Upload and process WAV files
- Audio feature extraction (MFCCs, spectral and temporal features, optional tempo)
- Dimensionality reduction (t-SNE / UMAP) and 2D visualization
- Streamlit app: waveform, spectrogram, feature space, batch comparison
- Optional VAE-based “bioacoustic dreaming” for generative exploration

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit app

```bash
streamlit run app.py
```

Use the app to upload WAVs, view waveforms and spectrograms, run the feature pipeline, and compare batches.

### Command-line workflow

Run the full pipeline (import → extract features → reduce dimensions → save plots) on existing files in `data/whale_codas`:

```bash
python -m src.workflow
```

Or import from a directory (env or CLI), from the **repository root**:

```bash
# Use existing data/whale_codas
python scripts/import_whale_sounds.py

# Custom path (solo_recordings / group_recordings under it)
set WHALE_SOUNDS_DIR=C:\WhaleSounds\SpermWhale
python scripts/import_whale_sounds.py

# Or: python scripts/import_whale_sounds.py C:\WhaleSounds\SpermWhale
```

Optional: download sample WAVs with `python scripts/download_whale_sounds.py`. See `scripts/README.md` for other helpers.

Results and plots go to `data/analysis_results` and `data/plots`.

## Project structure

- **`app.py`** – Streamlit entrypoint (repo root)
- **`src/`** – Core logic  
  - `config.py` – Paths and parameters  
  - `file_importer.py` – WAV validation and import  
  - `feature_extractor.py` – Feature extraction (MFCCs, spectral, etc.)  
  - `feature_reducer.py` – t-SNE/UMAP and `process_features()`  
  - `workflow.py` – Full pipeline  
  - `audio_visualizer.py` – Jupyter-oriented `AudioVisualizer` + workflow’s `FeatureVisualizer`  
  - `classifier.py` – Clustering (e.g. KMeans)  
  - `coda_vae.py` / `bioacoustic_dreamer.py` – VAE and dream sequences  
  - `audio_interface.py` – Jupyter/widget interface (no WebScraper)
- **`scripts/`** – CLI helpers (download, import workflow, test audio, plot viewer); see `scripts/README.md`
- **`docs/`** – Reference docs (`WHALE_DATA_SOURCES.md`, `SRC_LAYOUT.md` for `src/` notes)
- **`tests/`** – Pytest suite
- **`data/`** – `whale_codas`, `uploads`, `sample_audio`, `plots`, `metadata`, `logs` (runtime output; see `.gitignore`)

## Tests

```bash
pytest tests/ -v
# Or run individual scripts, e.g. tests/test_workflow.py, tests/test_dimensionality_reduction.py
```

## Environment variables (optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `WHALE_SHOW_DEBUG` | (off) | Set to `1` or `true` to show sidebar debug info (keep off in production). |
| `WHALE_MAX_UPLOAD_FILES` | `20` | Max files per batch in the Streamlit uploader. |
| `WHALE_MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Max size per uploaded file (bytes). |
| `CODA_EMBEDDINGS_KEY` | — | Fernet key (base64) for encrypted embeddings; preferred over a local `.secret_key` file. |
| `WHALE_MAX_AUDIO_SECONDS` | `60` | Cap audio duration (seconds) when loading; longer files are truncated to reduce memory. `0` = no cap. |
| `WHALE_RESAMPLE_RATE` | `22050` | Resample audio to this rate (Hz) during load. Reduces array size. `0` = keep original. |

## Streamlit Community Cloud

- **Main file:** `app.py` (repository root).
- **Dependencies:** `requirements.txt` (includes `streamlit>=1.41.0` so Python **3.13** and **Altair 6** work; older Streamlit + new Altair causes `ModuleNotFoundError: altair.vegalite.v4`).
- **System libs:** `packages.txt` installs `libsndfile1` for audio I/O.
- Push the branch Cloud uses (e.g. `Main`), then **Reboot app** in **Manage app** after changing dependencies.

## Docker

Build and run locally (first build downloads dependencies and may take several minutes):

```bash
docker compose build
docker compose up
```

Open `http://localhost:8501`. Data under `./data` is mounted so uploads and outputs persist across restarts.

Run without Compose:

```bash
docker build -t whale-lingo .
docker run --rm -p 8501:8501 -v "%CD%\data:/app/data" whale-lingo
```

(On Linux/macOS, use `$(pwd)/data:/app/data`.)

**Note:** The image is large (PyTorch, notebooks, etc.). For a smaller deploy later, you can split `requirements.txt` into app-only vs dev/test dependencies.
