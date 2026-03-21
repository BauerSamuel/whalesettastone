"""
Tests for the generative modeling module.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from src.bioacoustics.generative import CodaDataPrep, CodaDataset
from src.bioacoustics.embeddings import CodaEmbeddings
from cryptography.fernet import Fernet
from torch.utils.data import DataLoader

@pytest.fixture
def encryption_key():
    """Create a test encryption key."""
    return Fernet.generate_key()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings with realistic dimensions."""
    # Create a larger sample size with realistic feature dimensions
    n_samples = 50  # Increased from 10 to 50
    n_features = 20  # More realistic feature dimension
    return np.random.randn(n_samples, n_features)

@pytest.fixture
def data_prep(tmp_path):
    """Create a CodaDataPrep instance with a temporary directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir()
    return CodaDataPrep(data_dir=str(data_dir))

@pytest.fixture
def saved_embeddings(data_prep, sample_embeddings):
    """Save sample embeddings and return the file path."""
    filenames = [f"sample_{i}.wav" for i in range(len(sample_embeddings))]
    embeddings_file = data_prep.embeddings_handler.save_embeddings(
        sample_embeddings,
        filenames,
        metadata={"test": True}
    )
    return embeddings_file

def test_load_and_preprocess(data_prep, saved_embeddings, sample_embeddings):
    """Test loading and preprocessing embeddings."""
    # Test with dimensionality reduction
    reduced_embeddings, filenames = data_prep.load_and_preprocess(
        reduce_dim=True,
        n_components=10
    )
    
    assert isinstance(reduced_embeddings, np.ndarray)
    assert reduced_embeddings.shape[0] == len(sample_embeddings)
    assert reduced_embeddings.shape[1] == 10
    assert len(filenames) == len(sample_embeddings)
    
    # Test without dimensionality reduction
    original_embeddings, filenames = data_prep.load_and_preprocess(
        reduce_dim=False
    )
    
    assert isinstance(original_embeddings, np.ndarray)
    assert original_embeddings.shape == sample_embeddings.shape
    assert len(filenames) == len(sample_embeddings)

def test_save_training_data(data_prep, saved_embeddings, sample_embeddings):
    """Test saving training data."""
    # First reduce dimensionality
    reduced_embeddings, filenames = data_prep.load_and_preprocess(
        reduce_dim=True,
        n_components=10
    )
    
    # Save training data
    output_file = data_prep.save_training_data(
        reduced_embeddings,
        filenames
    )
    
    assert output_file.exists()
    data = np.load(output_file)
    assert 'embeddings' in data
    assert 'filenames' in data
    assert len(data['filenames']) == len(sample_embeddings)

def test_create_data_loader(data_prep, saved_embeddings, sample_embeddings):
    """Test creating a data loader."""
    # First reduce dimensionality
    reduced_embeddings, _ = data_prep.load_and_preprocess(
        reduce_dim=True,
        n_components=10
    )
    
    # Create data loader
    batch_size = 16
    data_loader = data_prep.create_data_loader(
        reduced_embeddings,
        batch_size=batch_size
    )
    
    assert isinstance(data_loader, DataLoader)
    assert len(data_loader.dataset) == len(sample_embeddings)
    
    # Test batch iteration
    for batch in data_loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.shape[0] <= batch_size
        assert batch.shape[1] == 10

def test_coda_dataset():
    """Test CodaDataset class."""
    # Create sample data
    embeddings = np.random.normal(0, 1, (10, 5))
    
    # Create dataset
    dataset = CodaDataset(embeddings)
    
    # Test length
    assert len(dataset) == 10
    
    # Test getting item
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (5,)

def test_error_handling(temp_dir):
    """Test error handling."""
    data_prep = CodaDataPrep(data_dir=temp_dir)
    
    # Test loading non-existent embeddings
    with pytest.raises(ValueError, match="No embeddings file found"):
        data_prep.load_and_preprocess()
    
    # Test saving to invalid directory (using a path that should be invalid on all platforms)
    with pytest.raises(OSError):
        data_prep.save_training_data(
            np.array([[1, 2]]),
            ['test.wav'],
            output_dir='C:/invalid/dir/with/special/chars/*?<>|'
        ) 