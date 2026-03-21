"""
Workflow for whale audio analysis.

This module provides a high-level interface for the entire whale audio analysis process,
including file import, feature extraction, dimensionality reduction, and visualization.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union

from .file_importer import FileImporter
from .feature_extractor import FeatureExtractor
from .feature_reducer import FeatureReducer
from .audio_visualizer import FeatureVisualizer

class Workflow:
    """
    Manages the complete workflow for whale audio analysis.
    """
    
    def __init__(self):
        """Initialize the workflow components."""
        self.importer = FileImporter()
        self.extractor = FeatureExtractor()
        self.reducer = FeatureReducer()
        self.visualizer = FeatureVisualizer()
        
    def run_workflow(self, 
                    source_path: Optional[Union[str, Path]] = None,
                    is_directory: bool = False,
                    visualize: bool = True) -> bool:
        """
        Run the complete workflow.
        
        Args:
            source_path: Path to a file or directory to import
            is_directory: Whether source_path is a directory
            visualize: Whether to generate visualizations
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        try:
            # Step 1: Import files
            logging.info("Step 1: Importing files...")
            if source_path:
                if is_directory:
                    imported_files = self.importer.import_directory(source_path)
                else:
                    result = self.importer.import_file(source_path)
                    imported_files = [result] if result else []
            else:
                # Use existing files in whale_codas directory
                imported_files = list(Path("data/whale_codas").glob("*.wav"))
            
            if not imported_files:
                logging.error("No valid audio files found")
                return False
            
            logging.info(f"Found {len(imported_files)} valid audio files")
            
            # Step 2: Extract features
            logging.info("Step 2: Extracting features...")
            features = []
            for file_path in imported_files:
                try:
                    file_features = self.extractor.extract_features(file_path)
                    if file_features:
                        features.append(file_features)
                except Exception as e:
                    logging.error(f"Error processing {file_path.name}: {str(e)}")
            
            if not features:
                logging.error("No features could be extracted")
                return False
            
            logging.info(f"Successfully extracted features from {len(features)} files")
            
            # Step 3: Reduce dimensions
            logging.info("Step 3: Reducing dimensions...")
            reduced_features, filenames = self.reducer.process_features(features)
            
            if reduced_features is None:
                logging.error("Failed to reduce dimensions")
                return False
            
            # Step 4: Visualize results
            if visualize:
                logging.info("Step 4: Creating visualizations...")
                # Create both static and interactive visualizations
                self.visualizer.plot_features(
                    reduced_features,
                    filenames,
                    title="Whale Sound Feature Visualization",
                    filename="whale_sounds_static.png"
                )
                self.visualizer.create_interactive_plot(
                    reduced_features,
                    filenames,
                    title="Interactive Whale Sound Feature Visualization"
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Error in workflow: {str(e)}")
            return False

def main():
    """Run the complete workflow."""
    workflow = Workflow()
    workflow.run_workflow()

if __name__ == "__main__":
    main() 