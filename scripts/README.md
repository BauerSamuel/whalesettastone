# Utility scripts

Run these **from the repository root** so `data/` paths resolve correctly.

| Script | Purpose |
|--------|---------|
| `download_whale_sounds.py` | Download sample WAVs from configured sources. |
| `import_whale_sounds.py` | Run the workflow on WAV directories or existing `data/whale_codas`. |
| `create_test_audio.py` | Generate synthetic test tones under `data/sample_audio/`. |
| `view_plots.py` | Open saved `data/plots/tsne_plot.png` and `umap_plot.png` side by side. |

**Note:** `app.py` stays at the repo root (`streamlit run app.py`). These are optional CLI helpers.
