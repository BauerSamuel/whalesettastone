"""
Tests for the bioacoustic dreaming module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Use non-interactive backend so tests don't open windows or hang in CI
import matplotlib
matplotlib.use("Agg")

from src.bioacoustics.dreaming import BioacousticDreamer
from src.bioacoustics.generative import CodaVAE

@pytest.fixture
def vae_model():
    """Create a VAE model instance."""
    return CodaVAE(
        input_dim=10,
        latent_dim=5,
        hidden_dims=[8]
    )

@pytest.fixture
def dreamer(vae_model):
    """Create a BioacousticDreamer instance."""
    return BioacousticDreamer(vae_model)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def test_dreamer_initialization(dreamer):
    """Test dreamer initialization."""
    assert isinstance(dreamer.model, CodaVAE)
    assert dreamer.model.latent_dim == 5
    assert dreamer.model.training == False  # Should be in eval mode

def test_generate_dream_sequence(dreamer):
    """Test dream sequence generation."""
    # Test with default parameters
    sequence = dreamer.generate_dream_sequence(n_steps=50)
    assert isinstance(sequence, torch.Tensor)
    assert sequence.shape == (50, 5)  # n_steps x latent_dim
    
    # Test with custom temperature
    sequence = dreamer.generate_dream_sequence(n_steps=50, temperature=0.5)
    assert isinstance(sequence, torch.Tensor)
    assert sequence.shape == (50, 5)
    
    # Test with starting point
    start_point = torch.randn(1, 5)
    sequence = dreamer.generate_dream_sequence(n_steps=50, start_point=start_point)
    assert torch.allclose(sequence[0], start_point.squeeze())

def test_decode_dream(dreamer):
    """Test dream decoding."""
    # Generate a dream sequence
    sequence = dreamer.generate_dream_sequence(n_steps=10)
    
    # Decode the sequence
    decoded = dreamer.decode_dream(sequence)
    
    assert isinstance(decoded, torch.Tensor)
    assert decoded.shape == (10, 10)  # n_steps x input_dim

def test_analyze_dream(dreamer):
    """Test dream analysis."""
    # Generate a dream sequence
    sequence = dreamer.generate_dream_sequence(n_steps=50)
    
    # Analyze the sequence
    analysis = dreamer.analyze_dream(sequence)
    
    # Check analysis keys
    expected_keys = {'mean_norm', 'std_norm', 'smoothness', 'diversity'}
    assert set(analysis.keys()) == expected_keys
    
    # Check that values are reasonable
    for value in analysis.values():
        assert isinstance(value, float)
        assert value >= 0

def test_visualize_dream(dreamer, temp_dir):
    """Test dream visualization."""
    import matplotlib.pyplot as plt
    try:
        sequence = dreamer.generate_dream_sequence(n_steps=50)
        save_path = temp_dir / "dream_visualization.png"
        dreamer.visualize_dream(sequence, save_path=save_path)
        assert save_path.exists()
    finally:
        plt.close("all")

def test_create_dream_animation(dreamer, temp_dir):
    """Test dream animation creation."""
    import matplotlib.pyplot as plt
    try:
        sequence = dreamer.generate_dream_sequence(n_steps=50)
        save_path = temp_dir / "dream_animation.gif"
        dreamer.create_dream_animation(sequence, save_path=save_path)
        assert save_path.exists()
    finally:
        plt.close("all")

def test_error_handling(dreamer):
    """Test error handling."""
    import matplotlib.pyplot as plt
    try:
        # Test with invalid sequence shape
        with pytest.raises(ValueError):
            invalid_sequence = torch.randn(10, 3)  # Wrong latent dimension
            dreamer.decode_dream(invalid_sequence)

        # Test with invalid save path (need enough steps for t-SNE: perplexity < n_samples)
        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            invalid_path = Path("/nonexistent_parent_dir_12345/dream.png")
            sequence = dreamer.generate_dream_sequence(n_steps=50)
            dreamer.visualize_dream(sequence, save_path=invalid_path)
    finally:
        plt.close("all") 