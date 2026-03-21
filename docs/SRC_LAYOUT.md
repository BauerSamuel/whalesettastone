# `src/` layout and technical notes

## Intended layout

| Area | Role |
|------|------|
| **Root `src/*.py`** | App-facing and shared pipeline: `config`, `workflow`, `file_importer`, `feature_*`, `classifier`, `coda_vae`, `bioacoustic_dreamer`, Streamlit helpers (`source_downloader`, `sources`), Jupyter UI (`audio_interface`, `audio_visualizer`). |
| **`src/bioacoustics/`** | Training, generative VAE, embeddings, analysis, and an alternate **BioacousticDreamer** used by tests. |

## Duplicate implementations (consolidate when you touch this area)

- **`CodaVAE`**: defined in both `src/coda_vae.py` (used by **`app.py`** and `bioacoustic_dreamer.py`) and `src/bioacoustics/generative.py` (used by training/tests and `bioacoustics/dreaming.py`). APIs differ; treat as two code paths until unified.
- **`BioacousticDreamer`**: `src/bioacoustic_dreamer.py` (Streamlit) vs `src/bioacoustics/dreaming.py` (tests / bioacoustics package). Same name, different classes.

## Modules with only tests (not used by `app.py` today)

- **`state_manager.py`** – Streamlit session helpers; the main app inlines batch state instead. Kept for tests / future refactor.

## Removed as unused (no imports)

Former stubs removed: duplicate Streamlit `Visualizer` modules, `file_handler`, `data_analyzer`, and `audio_processor` (upload flow is in the app and `FileImporter` / `FeatureExtractor`).
