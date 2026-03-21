"""
Module for creating and managing the whale audio interface.
Uses config, FileImporter, and FeatureExtractor (no WebScraper).
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, Audio, HTML
from pathlib import Path
from typing import Dict

from .config import WHALE_CODAS_DIR, SAMPLE_AUDIO_DIR
from .file_importer import FileImporter
from .feature_extractor import FeatureExtractor
from .audio_visualizer import AudioVisualizer


class AudioInterface:
    """Manages the whale audio interface (download, upload, visualize)."""

    def __init__(self):
        self.whale_codas_dir = WHALE_CODAS_DIR
        self._importer = FileImporter()
        self._extractor = FeatureExtractor()
        self._visualizer = AudioVisualizer(Path(self.whale_codas_dir))

    def copy_sample_files(self) -> Dict[str, int]:
        """Copy WAV files from sample_audio to whale_codas. Returns counts."""
        imported = self._importer.import_directory(SAMPLE_AUDIO_DIR)
        total = len(list(SAMPLE_AUDIO_DIR.glob("*.wav"))) if SAMPLE_AUDIO_DIR.exists() else 0
        return {
            "downloaded": len(imported),
            "skipped": max(0, total - len(imported)),
            "failed": 0,
        }

    def validate_wav_file(self, file_path: Path) -> bool:
        return self._importer.validate_wav_file(file_path)

    def get_all_features(self) -> Dict[str, Dict]:
        """Extract features for all WAV files in whale_codas."""
        out = {}
        for path in self.whale_codas_dir.glob("*.wav"):
            feats = self._extractor.extract_features(path)
            if feats and "filename" in feats:
                out[feats["filename"]] = feats
        return out

    def create_download_tab(self) -> widgets.VBox:
        download_button = widgets.Button(
            description="Download Sample Files",
            icon="download",
        )
        download_output = widgets.Output()

        def on_download_click(b):
            with download_output:
                clear_output()
                print("Copying sample files...")
                results = self.copy_sample_files()
                print(f"Downloaded: {results['downloaded']}, Skipped: {results['skipped']}, Failed: {results['failed']}")

        download_button.on_click(on_download_click)
        return widgets.VBox([download_button, download_output])

    def create_upload_tab(self) -> widgets.VBox:
        upload = widgets.FileUpload(accept=".wav", multiple=True, description="Upload WAV")
        upload_button = widgets.Button(description="Process Uploads", icon="upload")
        upload_output = widgets.Output()

        def on_upload_click(b):
            with upload_output:
                clear_output()
                if not upload.value:
                    print("Please select files to upload first.")
                    return
                print("Processing uploaded files...")
                for filename, file_data in upload.value.items():
                    # Use only the final path component to avoid directory traversal
                    safe_name = Path(filename).name
                    if not safe_name:
                        continue
                    target_path = self.whale_codas_dir / safe_name
                    with open(target_path, "wb") as f:
                        f.write(file_data["content"])
                    print(f"Saved {safe_name}")
                print("Validating...")
                for filename in list(upload.value.keys()):
                    path = self.whale_codas_dir / filename
                    if self.validate_wav_file(path):
                        print(f"  ✓ {filename}")
                    else:
                        print(f"  ✗ {filename}")
                        path.unlink(missing_ok=True)
                upload.value.clear()

        upload_button.on_click(on_upload_click)
        return widgets.VBox([upload, upload_button, upload_output])

    def create_interface(self) -> widgets.Tab:
        tab = widgets.Tab()
        download_tab = self.create_download_tab()
        upload_tab = self.create_upload_tab()
        features = self.get_all_features()
        viz_tab = self._visualizer.create_visualization_interface(features)
        tab.children = [download_tab, upload_tab, viz_tab]
        tab.set_title(0, "Download")
        tab.set_title(1, "Upload")
        tab.set_title(2, "Visualize")
        return tab

    def display_welcome_message(self):
        display(HTML("<h1>Whale Audio Analysis Interface</h1>"))
        display(
            HTML(
                """
            <p>Download sample files, upload WAVs, and visualize waveforms,
            spectrograms, and feature plots.</p>
            """
            )
        )