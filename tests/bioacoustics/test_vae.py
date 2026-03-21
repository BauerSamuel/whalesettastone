import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.bioacoustics.generative import CodaVAE, CodaDataset

@pytest.fixture
def sample_batch():
    """Create a sample batch of embeddings."""
    batch_size = 16
    input_dim = 10  # Reduced dimension from UMAP
    return torch.randn(batch_size, input_dim)

@pytest.fixture
def vae_model():
    """Create a VAE model instance."""
    input_dim = 10  # Reduced dimension from UMAP
    latent_dim = 5  # Latent space dimension
    hidden_dims = [8]  # Single hidden layer
    return CodaVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)

def test_vae_initialization(vae_model):
    """Test VAE model initialization."""
    assert isinstance(vae_model, CodaVAE)
    assert vae_model.input_dim == 10
    assert vae_model.latent_dim == 5
    assert len(vae_model.hidden_dims) == 1
    assert vae_model.hidden_dims[0] == 8

def test_encoder_forward(vae_model, sample_batch):
    """Test encoder forward pass."""
    # Forward pass through encoder
    mu, log_var = vae_model.encode(sample_batch)
    
    # Check output shapes
    assert mu.shape == (sample_batch.shape[0], vae_model.latent_dim)
    assert log_var.shape == (sample_batch.shape[0], vae_model.latent_dim)
    
    # Check that log_var is negative (for numerical stability)
    assert torch.all(log_var < 0)

def test_reparameterization(vae_model, sample_batch):
    """Test reparameterization trick."""
    mu, log_var = vae_model.encode(sample_batch)
    z = vae_model.reparameterize(mu, log_var)
    
    # Check output shape
    assert z.shape == (sample_batch.shape[0], vae_model.latent_dim)
    
    # Check that z is different from mu (random sampling worked)
    assert not torch.allclose(z, mu)

def test_decoder_forward(vae_model, sample_batch):
    """Test decoder forward pass."""
    # Get latent representation
    mu, log_var = vae_model.encode(sample_batch)
    z = vae_model.reparameterize(mu, log_var)
    
    # Forward pass through decoder
    reconstructed = vae_model.decode(z)
    
    # Check output shape
    assert reconstructed.shape == sample_batch.shape

def test_forward_pass(vae_model, sample_batch):
    """Test complete forward pass."""
    # Forward pass through entire model
    reconstructed, mu, log_var = vae_model(sample_batch)
    
    # Check output shapes
    assert reconstructed.shape == sample_batch.shape
    assert mu.shape == (sample_batch.shape[0], vae_model.latent_dim)
    assert log_var.shape == (sample_batch.shape[0], vae_model.latent_dim)

def test_loss_computation(vae_model, sample_batch):
    """Test loss computation."""
    # Forward pass
    reconstructed, mu, log_var = vae_model(sample_batch)
    
    # Compute loss and metrics
    loss, metrics = vae_model.loss_function(reconstructed, sample_batch, mu, log_var)
    
    # Check that loss is a scalar
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss > 0
    
    # Check metrics
    expected_metrics = {
        'loss/reconstruction',
        'loss/kl_divergence',
        'loss/total',
        'reconstruction/cosine_similarity',
        'latent/norm',
        'latent/std'
    }
    assert set(metrics.keys()) == expected_metrics
    
    # Check metric values
    for metric_name, value in metrics.items():
        assert isinstance(value, float)
        if 'loss' in metric_name:
            assert value > 0
        if metric_name == 'reconstruction/cosine_similarity':
            assert -1 <= value <= 1

def test_validation_metrics(vae_model, sample_batch):
    """Test validation metrics computation."""
    # Create a small validation loader
    val_loader = DataLoader(
        CodaDataset(sample_batch.numpy()),
        batch_size=4,
        shuffle=False
    )
    
    # Compute validation metrics
    val_metrics = vae_model.compute_validation_metrics(val_loader)
    
    # Check metric names
    expected_metrics = {
        'val_loss/reconstruction',
        'val_loss/kl_divergence',
        'val_loss/total',
        'val_reconstruction/cosine_similarity',
        'val_latent/norm',
        'val_latent/std'
    }
    assert set(val_metrics.keys()) == expected_metrics
    
    # Check metric values
    for metric_name, value in val_metrics.items():
        assert isinstance(value, float)
        if 'loss' in metric_name:
            assert value > 0
        if 'cosine_similarity' in metric_name:
            assert -1 <= value <= 1 