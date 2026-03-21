"""
Tests for VAE visualization module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
import os

from src.bioacoustics.visualization import (
    load_metrics_from_checkpoints,
    plot_training_progress,
    plot_latent_space
)
from src.bioacoustics.generative import CodaVAE
from src.bioacoustics.data import CodaDataset

@pytest.fixture
def sample_metrics():
    """Create sample metrics for testing."""
    return {
        'train_loss/total': [1.0, 0.8, 0.6],
        'train_loss/reconstruction': [0.7, 0.5, 0.4],
        'train_loss/kl_divergence': [0.3, 0.3, 0.2],
        'train_reconstruction/cosine_similarity': [0.5, 0.7, 0.8],
        'train_latent/norm': [1.0, 1.1, 1.2],
        'train_latent/std': [0.5, 0.6, 0.7],
        'val_loss/total': [0.9, 0.7, 0.5],
        'val_reconstruction/cosine_similarity': [0.6, 0.8, 0.9]
    }

@pytest.fixture
def checkpoint_dir(sample_metrics, tmp_path):
    """Create a temporary checkpoint directory with sample metrics."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    # Create sample metrics files
    for i in range(3):
        metrics_file = checkpoint_dir / f"metrics_20240101_120000_{i}.json"
        with open(metrics_file, 'w') as f:
            json.dump({k: v[i] for k, v in sample_metrics.items()}, f)
    
    return checkpoint_dir

@pytest.fixture
def temp_dir(tmp_path):
    """Use pytest's tmp_path so output files are in a known location."""
    return str(tmp_path)

@pytest.fixture
def sample_data():
    # Create a small sample dataset
    data = np.random.randn(100, 20)  # 100 samples, 20 features
    return data

@pytest.fixture
def data_loader(sample_data):
    """Create a data loader for the sample data."""
    dataset = CodaDataset(sample_data)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False
    )

@pytest.fixture
def vae_model():
    # Create a simple VAE model
    return CodaVAE(input_dim=20, latent_dim=2, hidden_dims=[10])

def test_load_metrics_from_checkpoints(checkpoint_dir, sample_metrics):
    """Test loading metrics from checkpoint files."""
    metrics = load_metrics_from_checkpoints(str(checkpoint_dir))
    
    # Check that all metrics are loaded
    assert set(metrics.keys()) == set(sample_metrics.keys())
    
    # Check values
    for k, v in sample_metrics.items():
        assert metrics[k] == v

def test_plot_training_progress(checkpoint_dir, tmp_path):
    """Test plotting training progress."""
    output_dir = tmp_path / "plots"
    plot_training_progress(str(checkpoint_dir), str(output_dir))
    
    # Check that at least one plot was created (filename includes timestamp)
    plot_files = list(output_dir.glob("training_progress_*.png"))
    assert len(plot_files) >= 1, f"Expected at least one training_progress_*.png in {output_dir}"

def test_plot_latent_space(vae_model, data_loader, temp_dir):
    """Test plotting latent space visualization."""
    vae_model = vae_model.to("cpu")
    plot_latent_space(vae_model, data_loader, temp_dir, device="cpu")
    html_path = os.path.join(temp_dir, "latent_space.html")
    png_path = os.path.join(temp_dir, "latent_space.png")
    assert os.path.exists(html_path), f"HTML file not found at {html_path}"
    assert os.path.exists(png_path), f"PNG file not found at {png_path}" 