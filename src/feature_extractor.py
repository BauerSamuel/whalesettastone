"""
Feature extractor for whale audio files.

This module provides functionality to extract audio features from WAV files,
including MFCCs, spectral features, and temporal features.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import librosa

class FeatureExtractor:
    """
    Extracts audio features from WAV files.
    """
    
    def __init__(self, 
                 n_mfcc: int = 13,
                 hop_length: int = 512,
                 n_fft: int = 2048):
        """
        Initialize the feature extractor.
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract
            hop_length: Number of samples between successive frames
            n_fft: Length of the FFT window
        """
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
    
    def extract_features(self, file_path: Path) -> Optional[Dict]:
        """
        Extract features from a WAV file.
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        try:
            # Load the audio file
            y, sr = librosa.load(str(file_path), sr=None)
            
            # Extract features
            features = {
                'filename': file_path.name,
                'sample_rate': sr,
                'duration': librosa.get_duration(y=y, sr=sr),
                
                # MFCCs
                'mfccs': librosa.feature.mfcc(
                    y=y, 
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                ).mean(axis=1).tolist(),
                
                # Spectral features
                'spectral_centroid': librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                ).mean(),
                
                'spectral_rolloff': librosa.feature.spectral_rolloff(
                    y=y,
                    sr=sr,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                ).mean(),
                
                'spectral_bandwidth': librosa.feature.spectral_bandwidth(
                    y=y,
                    sr=sr,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                ).mean(),
                
                # Temporal features
                'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).mean(),
                'rms_energy': librosa.feature.rms(y=y).mean()
            }
            
            logging.info(f"Successfully extracted features from {file_path.name}")
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features from {file_path}: {str(e)}")
            return None 