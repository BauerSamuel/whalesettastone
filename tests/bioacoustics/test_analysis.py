"""
Tests for the coda analysis module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil
from src.bioacoustics.analysis import CodaAnalyzer

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(50, 4)

@pytest.fixture
def sample_filenames():
    """Create sample filenames for testing."""
    return [f'test_{i}.wav' for i in range(30)]

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

def test_cluster_codas(sample_embeddings):
    """Test coda clustering functionality."""
    analyzer = CodaAnalyzer()
    
    # Test kmeans clustering
    analyzer.cluster_codas(sample_embeddings, method='kmeans', n_clusters=3)
    assert len(np.unique(analyzer.cluster_labels)) == 3
    
    # Test dbscan clustering
    analyzer.cluster_codas(sample_embeddings, method='dbscan', eps=0.5, min_samples=2)
    assert len(np.unique(analyzer.cluster_labels)) >= 1

def test_analyze_clusters(sample_embeddings, sample_filenames):
    """Test cluster analysis."""
    analyzer = CodaAnalyzer()
    analyzer.cluster_codas(sample_embeddings, method='kmeans', n_clusters=3)
    
    analysis = analyzer.analyze_clusters(sample_embeddings, sample_filenames)
    
    # Check analysis results
    assert analysis['n_clusters'] == 3
    assert analysis['noise_points'] == 0
    assert 'silhouette_score' in analysis
    assert len(analysis['cluster_summary']) == 3
    
    # Check cluster summaries
    for summary in analysis['cluster_summary']:
        assert 'cluster_id' in summary
        assert 'size' in summary
        assert 'mean_features' in summary
        assert 'std_features' in summary
        assert len(summary['sample_files']) <= 5

def test_plot_cluster_distribution(sample_embeddings, temp_dir):
    """Test cluster distribution plotting."""
    analyzer = CodaAnalyzer()
    analyzer.cluster_codas(sample_embeddings, method='kmeans', n_clusters=3)
    
    # Test saving plot
    save_path = Path(temp_dir) / 'cluster_dist.png'
    analyzer.plot_cluster_distribution(sample_embeddings, save_path)
    assert save_path.exists()

def test_plot_feature_importance(sample_embeddings, temp_dir):
    """Test feature importance plotting."""
    analyzer = CodaAnalyzer()
    analyzer.cluster_codas(sample_embeddings, method='kmeans', n_clusters=3)
    
    feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
    
    # Test saving plot
    save_path = Path(temp_dir) / 'feature_importance.png'
    analyzer.plot_feature_importance(sample_embeddings, feature_names, save_path)
    assert save_path.exists()

def test_analyze_temporal_patterns(sample_embeddings):
    """Test temporal pattern analysis."""
    analyzer = CodaAnalyzer()
    analyzer.cluster_codas(sample_embeddings, method='kmeans', n_clusters=3)
    
    # Create mock timestamps
    timestamps = np.arange(len(sample_embeddings))
    patterns = analyzer.analyze_temporal_patterns(timestamps)
    
    assert isinstance(patterns, dict)
    assert 'cluster_sequences' in patterns
    assert 'transition_matrix' in patterns

def test_plot_temporal_patterns(sample_embeddings, temp_dir):
    """Test temporal pattern visualization."""
    analyzer = CodaAnalyzer()
    analyzer.cluster_codas(sample_embeddings, method='kmeans', n_clusters=3)
    
    # Create mock timestamps
    timestamps = np.arange(len(sample_embeddings))
    
    # Test saving plot
    save_path = Path(temp_dir) / 'temporal_patterns.png'
    analyzer.plot_temporal_patterns(timestamps, save_path)
    assert save_path.exists()

def test_error_handling():
    """Test error handling."""
    analyzer = CodaAnalyzer()
    
    # Test plotting before clustering
    with pytest.raises(ValueError):
        analyzer.plot_cluster_distribution(np.array([[1, 2]]))
    
    with pytest.raises(ValueError):
        analyzer.plot_feature_importance(np.array([[1, 2]]), ['f1', 'f2'])
    
    with pytest.raises(ValueError):
        analyzer.analyze_clusters(np.array([[1, 2]]), ['test.wav']) 