"""
Tests for the coda embeddings module.
"""

import pytest
import numpy as np
from pathlib import Path
import shutil
import tempfile
from src.bioacoustics.embeddings import CodaEmbeddings
import os
import platform

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    return [
        {
            'filename': 'test1.wav',
            'mfcc_1': 0.1,
            'mfcc_2': 0.2,
            'spectral_centroid': 0.3,
            'spectral_rolloff': 0.4
        },
        {
            'filename': 'test2.wav',
            'mfcc_1': 0.5,
            'mfcc_2': 0.6,
            'spectral_centroid': 0.7,
            'spectral_rolloff': 0.8
        }
    ]

def test_prepare_embeddings(temp_dir, sample_features):
    """Test preparing embeddings from features."""
    embeddings_handler = CodaEmbeddings(embeddings_dir=temp_dir)
    embeddings, filenames = embeddings_handler.prepare_embeddings(sample_features)
    
    # Check shape
    assert embeddings.shape == (2, 4)  # 2 samples, 4 features each
    
    # Check values
    assert np.allclose(embeddings[0], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(embeddings[1], [0.5, 0.6, 0.7, 0.8])
    
    # Check filenames
    assert filenames == ['test1.wav', 'test2.wav']

def test_save_and_load_embeddings(temp_dir, sample_features):
    """Test saving and loading encrypted embeddings."""
    embeddings_handler = CodaEmbeddings(embeddings_dir=temp_dir)
    
    # Prepare and save embeddings
    embeddings, filenames = embeddings_handler.prepare_embeddings(sample_features)
    metadata = {'test': 'metadata'}
    saved_path = embeddings_handler.save_embeddings(embeddings, filenames, metadata)
    
    # Check file permissions (only on Unix-like systems)
    if platform.system() != 'Windows':
        assert oct(os.stat(saved_path).st_mode)[-3:] == '600'
    
    # Load embeddings
    loaded_embeddings, loaded_filenames, loaded_metadata = embeddings_handler.load_embeddings(saved_path)
    
    # Check loaded data matches saved data
    assert np.allclose(embeddings, loaded_embeddings)
    assert filenames == loaded_filenames
    assert metadata == loaded_metadata

def test_get_latest_embeddings(temp_dir, sample_features):
    """Test getting the latest embeddings file."""
    embeddings_handler = CodaEmbeddings(embeddings_dir=temp_dir)
    
    # Initially no embeddings
    assert embeddings_handler.get_latest_embeddings() is None
    
    # Save some embeddings
    embeddings, filenames = embeddings_handler.prepare_embeddings(sample_features)
    saved_path = embeddings_handler.save_embeddings(embeddings, filenames)
    
    # Check we can get the latest
    latest = embeddings_handler.get_latest_embeddings()
    assert latest == saved_path

def test_error_handling(temp_dir):
    """Test error handling in embeddings processing."""
    embeddings_handler = CodaEmbeddings(embeddings_dir=temp_dir)
    
    # Test with invalid features
    with pytest.raises(ValueError):
        embeddings_handler.prepare_embeddings([{'invalid': 'data'}])
    
    # Test loading non-existent file
    with pytest.raises(Exception):
        embeddings_handler.load_embeddings(Path('nonexistent.npz'))

def test_feature_validation(temp_dir):
    """Test feature validation security checks."""
    embeddings_handler = CodaEmbeddings(embeddings_dir=temp_dir)
    
    # Test missing required fields
    invalid_features = [{'filename': 'test.wav'}]  # Missing MFCCs
    with pytest.raises(ValueError):
        embeddings_handler.prepare_embeddings(invalid_features)
    
    # Test invalid filename
    invalid_features = [{
        'filename': 'test.txt',  # Wrong extension
        'mfcc_1': 0.1,
        'mfcc_2': 0.2,
        'spectral_centroid': 0.3,
        'spectral_rolloff': 0.4
    }]
    with pytest.raises(ValueError):
        embeddings_handler.prepare_embeddings(invalid_features)
    
    # Test invalid numeric values
    invalid_features = [{
        'filename': 'test.wav',
        'mfcc_1': 'not a number',  # Invalid type
        'mfcc_2': 0.2,
        'spectral_centroid': 0.3,
        'spectral_rolloff': 0.4
    }]
    with pytest.raises(ValueError):
        embeddings_handler.prepare_embeddings(invalid_features)

def test_data_integrity(temp_dir, sample_features):
    """Test data integrity checks."""
    embeddings_handler = CodaEmbeddings(embeddings_dir=temp_dir)
    
    # Save embeddings
    embeddings, filenames = embeddings_handler.prepare_embeddings(sample_features)
    saved_path = embeddings_handler.save_embeddings(embeddings, filenames)
    
    # Tamper with the file
    with open(saved_path, 'rb') as f:
        data = f.read()
    tampered_data = data[:-1]  # Remove last byte
    with open(saved_path, 'wb') as f:
        f.write(tampered_data)
    
    # Try to load tampered data
    with pytest.raises(ValueError):
        embeddings_handler.load_embeddings(saved_path)

def test_encryption(temp_dir, sample_features):
    """Test encryption functionality."""
    embeddings_handler = CodaEmbeddings(embeddings_dir=temp_dir)
    
    # Save embeddings
    embeddings, filenames = embeddings_handler.prepare_embeddings(sample_features)
    saved_path = embeddings_handler.save_embeddings(embeddings, filenames)
    
    # Try to read encrypted file directly
    with open(saved_path, 'rb') as f:
        encrypted_data = f.read()
    
    # Verify data is encrypted (should not be readable as plain text)
    assert b'embeddings' not in encrypted_data
    assert b'filenames' not in encrypted_data 