"""
Feature reducer for dimensionality reduction.

This module provides functionality to reduce high-dimensional feature vectors
to lower dimensions for visualization using various techniques like t-SNE and UMAP.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

from .config import (
    N_COMPONENTS,
    PERPLEXITY,
    N_NEIGHBORS
)

class FeatureReducer:
    """
    Reduces dimensionality of feature vectors for visualization.
    """
    
    def __init__(self, method: str = 'tsne'):
        """
        Initialize the feature reducer.
        
        Args:
            method: Reduction method ('tsne' or 'umap')
        """
        self.method = method.lower()
        if self.method not in ['tsne', 'umap']:
            raise ValueError("Method must be either 'tsne' or 'umap'")
    
    def prepare_features(self, features: List[Dict]) -> np.ndarray:
        """
        Prepare features for dimensionality reduction.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Array of feature vectors
        """
        try:
            # Extract MFCC features and other numerical features
            feature_vectors = []
            for feature_dict in features:
                vector = []
                # Add MFCCs
                vector.extend(feature_dict['mfccs'])
                # Add other numerical features
                for key in ['spectral_centroid', 'spectral_rolloff', 
                          'spectral_bandwidth', 'zero_crossing_rate', 'rms_energy']:
                    if key in feature_dict:
                        vector.append(feature_dict[key])
                feature_vectors.append(vector)
            
            return np.array(feature_vectors)
            
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            return None
    
    def reduce_dimensions(self, features: List[Dict]) -> Optional[np.ndarray]:
        """
        Reduce dimensions of feature vectors.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Array of reduced dimension features
        """
        try:
            # Prepare feature vectors
            X = self.prepare_features(features)
            if X is None:
                return None
            
            # Adjust parameters based on sample size
            n_samples = X.shape[0]
            if n_samples < 2:
                logging.error(f"Need at least 2 samples for dimensionality reduction, got {n_samples}")
                return None
            perplexity = max(1, min(PERPLEXITY, n_samples - 1))
            n_neighbors = max(1, min(N_NEIGHBORS, n_samples - 1))
            
            # Perform dimensionality reduction
            if self.method == 'tsne':
                reducer = TSNE(
                    n_components=N_COMPONENTS,
                    perplexity=perplexity,
                    random_state=42
                )
            else:  # umap
                reducer = UMAP(
                    n_components=N_COMPONENTS,
                    n_neighbors=n_neighbors,
                    random_state=42
                )
            
            reduced_features = reducer.fit_transform(X)
            logging.info(f"Successfully reduced features to {N_COMPONENTS} dimensions using {self.method}")
            return reduced_features
            
        except Exception as e:
            logging.error(f"Error reducing dimensions: {str(e)}")
            return None

    def process_features(
        self, features: List[Dict], output_path: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Prepare features, reduce dimensions, and optionally save a plot.

        Args:
            features: List of feature dictionaries (must contain 'filename' and
                keys expected by prepare_features).
            output_path: Optional path to save a static scatter plot.

        Returns:
            Tuple of (reduced_features ndarray, list of filenames).
        """
        if not features:
            logging.error("No features provided to process_features")
            return None, []
        filenames = [
            f.get("filename", f"file_{i}") for i, f in enumerate(features)
        ]
        reduced = self.reduce_dimensions(features)
        if reduced is None:
            return None, filenames
        if output_path:
            self._plot_reduced(reduced, filenames, output_path)
        return reduced, filenames

    def _plot_reduced(
        self,
        reduced_features: np.ndarray,
        filenames: List[str],
        output_path: str,
    ) -> None:
        """Save a static scatter plot of reduced features."""
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        df = pd.DataFrame({
            "x": reduced_features[:, 0],
            "y": reduced_features[:, 1],
            "filename": filenames,
        })
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x="x", y="y", hue="filename", s=100)
        plt.title(f"{self.method.upper()} visualization of audio features")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Plot saved to {output_path}")