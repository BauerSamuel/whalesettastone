"""
File importer for whale audio files.

This module provides functionality to import whale audio files from local directories
or individual files, with validation and processing capabilities.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Union
import wave

from .config import (
    WHALE_CODAS_DIR,
    MIN_FILE_SIZE,
    MAX_FILE_SIZE,
    MIN_SAMPLE_RATE,
    REQUIRED_CHANNELS,
    REQUIRED_SAMPLE_WIDTH
)

class FileImporter:
    """
    Handles importing of whale audio files from local sources.
    """
    
    def __init__(self):
        """Initialize the file importer."""
        # Ensure required directories exist
        WHALE_CODAS_DIR.mkdir(parents=True, exist_ok=True)
        
    def validate_wav_file(self, file_path: Path) -> bool:
        """
        Validate a WAV file meets our requirements.
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Check file size
            file_size = file_path.stat().st_size
            if not MIN_FILE_SIZE <= file_size <= MAX_FILE_SIZE:
                logging.error(f"File size {file_size} bytes outside allowed range")
                return False
            
            # Check WAV format
            with wave.open(str(file_path), 'rb') as wav_file:
                # Check sample rate
                if wav_file.getframerate() < MIN_SAMPLE_RATE:
                    logging.error(f"Sample rate {wav_file.getframerate()} below minimum {MIN_SAMPLE_RATE}")
                    return False
                
                # Check channels
                if wav_file.getnchannels() != REQUIRED_CHANNELS:
                    logging.error(f"Expected {REQUIRED_CHANNELS} channels, got {wav_file.getnchannels()}")
                    return False
                
                # Check sample width
                if wav_file.getsampwidth() != REQUIRED_SAMPLE_WIDTH:
                    logging.error(f"Expected sample width {REQUIRED_SAMPLE_WIDTH}, got {wav_file.getsampwidth()}")
                    return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating WAV file {file_path}: {str(e)}")
            return False
    
    def import_file(self, source_path: Union[str, Path]) -> Optional[Path]:
        """
        Import a single audio file.
        
        Args:
            source_path: Path to the source file
            
        Returns:
            Path to the imported file in whale_codas directory, or None if import failed
        """
        try:
            source_path = Path(source_path)
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
            
            if not source_path.exists():
                logging.error(f"Source file not found: {source_path}")
                return None
            
            # Generate output path
            output_path = WHALE_CODAS_DIR / source_path.name
            
            # Skip if file already exists
            if output_path.exists():
                logging.info(f"File already exists in whale_codas: {output_path.name}")
                return output_path
            
            # Copy and validate the file
            shutil.copy2(source_path, output_path)
            if self.validate_wav_file(output_path):
                logging.info(f"Successfully imported file: {output_path.name}")
                return output_path
            else:
                logging.error(f"File failed validation: {output_path.name}")
                output_path.unlink(missing_ok=True)
                return None
                
        except Exception as e:
            logging.error(f"Error importing file {source_path}: {str(e)}")
            return None
    
    def import_directory(self, source_dir: Union[str, Path]) -> List[Path]:
        """
        Import all WAV files from a directory.
        
        Args:
            source_dir: Path to the source directory
            
        Returns:
            List of paths to successfully imported files
        """
        try:
            source_dir = Path(source_dir)
            if not source_dir.is_absolute():
                source_dir = Path.cwd() / source_dir
            
            if not source_dir.exists() or not source_dir.is_dir():
                logging.error(f"Source directory not found: {source_dir}")
                return []
            
            imported_files = []
            for file_path in source_dir.glob("*.wav"):
                result = self.import_file(file_path)
                if result:
                    imported_files.append(result)
            
            logging.info(f"Imported {len(imported_files)} files from {source_dir}")
            return imported_files
            
        except Exception as e:
            logging.error(f"Error importing directory {source_dir}: {str(e)}")
            return [] 