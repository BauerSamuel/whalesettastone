"""Tests for VAE training module."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import shutil
import tempfile

from src.bioacoustics.generative import CodaVAE, CodaDataset
from src.bioacoustics.training import VAETrainer

@pytest.fixture
def sample_data():
    """Create sample training data."""
    n_samples = 32
    input_dim = 10
    return torch.randn(n_samples, input_dim)

@pytest.fixture
def data_loaders(sample_data):
    """Create training and validation data loaders."""
    # Split data into train and val
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(
        CodaDataset(train_data.numpy()),
        batch_size=8,
        shuffle=True
    )
    val_loader = DataLoader(
        CodaDataset(val_data.numpy()),
        batch_size=8,
        shuffle=False
    )
    
    return train_loader, val_loader

@pytest.fixture
def vae_model():
    """Create a VAE model instance."""
    return CodaVAE(
        input_dim=10,
        latent_dim=5,
        hidden_dims=[8]
    )

@pytest.fixture
def trainer(vae_model, data_loaders, tmp_path):
    """Create a trainer instance."""
    train_loader, val_loader = data_loaders
    return VAETrainer(
        model=vae_model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=str(tmp_path / "checkpoints")
    )

def test_trainer_initialization(trainer):
    """Test trainer initialization."""
    assert isinstance(trainer.model, CodaVAE)
    assert trainer.current_epoch == 0
    assert trainer.best_val_loss == float('inf')
    assert trainer.checkpoint_dir.exists()

def test_train_epoch(trainer):
    """Test training for one epoch."""
    metrics = trainer.train_epoch()
    
    # Check metrics
    expected_metrics = {
        'train_loss/reconstruction',
        'train_loss/kl_divergence',
        'train_loss/total',
        'train_reconstruction/cosine_similarity',
        'train_latent/norm',
        'train_latent/std'
    }
    assert set(metrics.keys()) == expected_metrics
    
    # Check metric values
    for metric_name, value in metrics.items():
        assert isinstance(value, float)
        if 'loss' in metric_name:
            assert value > 0

def test_validation(trainer):
    """Test validation."""
    metrics = trainer.validate()
    
    # Check metrics
    expected_metrics = {
        'val_loss/reconstruction',
        'val_loss/kl_divergence',
        'val_loss/total',
        'val_reconstruction/cosine_similarity',
        'val_latent/norm',
        'val_latent/std'
    }
    assert set(metrics.keys()) == expected_metrics

def test_checkpointing(trainer, tmp_path):
    """Test checkpoint saving and loading."""
    # Train for one epoch to get metrics
    metrics = trainer.train_epoch()
    
    # Save checkpoint
    trainer.save_checkpoint(metrics, is_best=True)
    
    # Check that files were created
    assert (trainer.checkpoint_dir / "best_model.pt").exists()
    
    # Load checkpoint
    trainer.load_checkpoint(str(trainer.checkpoint_dir / "best_model.pt"))
    assert trainer.current_epoch == 0
    
def test_full_training(trainer):
    """Test full training loop."""
    n_epochs = 2
    final_metrics = trainer.train(n_epochs=n_epochs)
    
    # Check that training progressed
    assert trainer.current_epoch == n_epochs - 1
    
    # Check final metrics
    assert isinstance(final_metrics, dict)
    assert 'train_loss/total' in final_metrics
    assert 'val_loss/total' in final_metrics 