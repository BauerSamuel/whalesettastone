"""
Module for analyzing whale coda patterns and features.
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

class CodaAnalyzer:
    """
    Analyzes whale coda patterns using various statistical and machine learning techniques.
    """
    
    def __init__(self):
        """Initialize the coda analyzer."""
        self.scaler = StandardScaler()
        self.clusters = None
        self.cluster_labels = None
    
    def cluster_codas(self, embeddings: np.ndarray, method: str = 'dbscan',
                     eps: float = 0.5, min_samples: int = 5,
                     n_clusters: int = 5) -> np.ndarray:
        """
        Cluster coda embeddings to find patterns.
        
        Args:
            embeddings: Array of coda embeddings
            method: Clustering method ('dbscan' or 'kmeans')
            eps: Maximum distance between samples for DBSCAN
            min_samples: Minimum samples in a cluster for DBSCAN
            n_clusters: Number of clusters for KMeans
            
        Returns:
            Array of cluster labels
        """
        # Ensure input is finite
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Input contains non-finite values")
            
        # Scale the embeddings
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # Perform clustering
        if method == 'dbscan':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        
        self.cluster_labels = clusterer.fit_predict(scaled_embeddings)
        self.clusters = clusterer
        
        return self.cluster_labels
    
    def analyze_clusters(self, embeddings: np.ndarray, 
                        filenames: List[str]) -> Dict:
        """
        Analyze cluster statistics and patterns.
        
        Args:
            embeddings: Array of coda embeddings
            filenames: List of corresponding filenames
            
        Returns:
            Dictionary containing analysis results
        """
        if self.cluster_labels is None:
            raise ValueError("Must run clustering before analysis")
        
        # Basic cluster statistics
        n_clusters = len(set(self.cluster_labels[self.cluster_labels >= 0]))
        noise_points = sum(self.cluster_labels == -1)
        
        # Calculate silhouette score (excluding noise points)
        valid_mask = self.cluster_labels != -1
        if sum(valid_mask) > 1:
            silhouette = silhouette_score(
                embeddings[valid_mask], 
                self.cluster_labels[valid_mask]
            )
        else:
            silhouette = 0
        
        # Cluster sizes
        cluster_sizes = pd.Series(self.cluster_labels).value_counts()
        
        # Create cluster summary
        cluster_summary = []
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:
                continue
                
            mask = self.cluster_labels == cluster_id
            cluster_files = [f for i, f in enumerate(filenames) if mask[i]]
            
            summary = {
                'cluster_id': cluster_id,
                'size': sum(mask),
                'mean_features': embeddings[mask].mean(axis=0),
                'std_features': embeddings[mask].std(axis=0),
                'sample_files': cluster_files[:5]  # First 5 examples
            }
            cluster_summary.append(summary)
        
        return {
            'n_clusters': n_clusters,
            'noise_points': noise_points,
            'silhouette_score': silhouette,
            'cluster_sizes': cluster_sizes.to_dict(),
            'cluster_summary': cluster_summary
        }
    
    def plot_cluster_distribution(self, embeddings: np.ndarray,
                                save_path: Optional[Path] = None):
        """
        Plot the distribution of clusters.
        
        Args:
            embeddings: Array of coda embeddings
            save_path: Optional path to save the plot
        """
        if self.cluster_labels is None:
            raise ValueError("Must run clustering before plotting")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Cluster sizes
        cluster_sizes = pd.Series(self.cluster_labels).value_counts()
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, ax=ax1)
        ax1.set_title('Cluster Size Distribution')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Samples')
        
        # Plot 2: 2D visualization of first two features
        scatter = ax2.scatter(embeddings[:, 0], embeddings[:, 1], 
                            c=self.cluster_labels, cmap='viridis')
        ax2.set_title('Cluster Distribution (First 2 Features)')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax2, label='Cluster ID')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def plot_feature_importance(self, embeddings: np.ndarray,
                              feature_names: List[str],
                              save_path: Optional[Path] = None):
        """
        Plot feature importance based on cluster separation.
        
        Args:
            embeddings: Array of coda embeddings
            feature_names: List of feature names
            save_path: Optional path to save the plot
        """
        if self.cluster_labels is None:
            raise ValueError("Must run clustering before plotting")
            
        # Calculate feature importance scores
        importance_scores = []
        for i in range(embeddings.shape[1]):
            # Use the ratio of between-cluster to within-cluster variance
            cluster_means = []
            cluster_vars = []
            
            for cluster_id in set(self.cluster_labels):
                if cluster_id == -1:  # Skip noise
                    continue
                mask = self.cluster_labels == cluster_id
                cluster_means.append(embeddings[mask, i].mean())
                cluster_vars.append(embeddings[mask, i].var())
            
            between_cluster_var = np.var(cluster_means)
            within_cluster_var = np.mean(cluster_vars)
            
            if within_cluster_var == 0:
                importance_scores.append(0)
            else:
                importance_scores.append(between_cluster_var / within_cluster_var)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_names, y=importance_scores)
        plt.xticks(rotation=45)
        plt.title('Feature Importance in Cluster Separation')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def analyze_temporal_patterns(self, timestamps: np.ndarray) -> dict:
        """
        Analyze temporal patterns in coda sequences.
        
        Args:
            timestamps: Array of timestamps for each coda
            
        Returns:
            Dictionary containing:
                - cluster_sequences: List of cluster sequences
                - transition_matrix: Matrix of transition probabilities between clusters
        """
        if self.cluster_labels is None:
            raise ValueError("Must run clustering before analyzing temporal patterns")
        
        if len(timestamps) != len(self.cluster_labels):
            raise ValueError("Number of timestamps must match number of samples")
        
        # Sort clusters by timestamp
        sorted_indices = np.argsort(timestamps)
        cluster_sequence = self.cluster_labels[sorted_indices]
        
        # Calculate transition matrix
        n_clusters = len(np.unique(self.cluster_labels))
        transition_matrix = np.zeros((n_clusters, n_clusters))
        
        for i in range(len(cluster_sequence) - 1):
            current_cluster = cluster_sequence[i]
            next_cluster = cluster_sequence[i + 1]
            transition_matrix[current_cluster, next_cluster] += 1
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    where=row_sums != 0)
        
        return {
            'cluster_sequences': cluster_sequence.tolist(),
            'transition_matrix': transition_matrix.tolist()
        }

    def plot_temporal_patterns(self, timestamps: np.ndarray,
                             save_path: Optional[Path] = None):
        """
        Visualize temporal patterns in coda sequences.
        
        Args:
            timestamps: Array of timestamps for each coda
            save_path: Optional path to save the plot
        """
        if self.cluster_labels is None:
            raise ValueError("Must run clustering before plotting temporal patterns")
            
        # Get temporal patterns
        patterns = self.analyze_temporal_patterns(timestamps)
        transition_matrix = np.array(patterns['transition_matrix'])
        cluster_sequence = patterns['cluster_sequences']
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Transition Probabilities Between Coda Types',
                'Coda Type Sequence Over Time',
                'Duration Distribution by Coda Type',
                'Coda Type Statistics'
            ),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                  [{"type": "box"}, {"type": "table"}]],
            column_widths=[0.5, 0.5],
            row_heights=[0.5, 0.5]
        )
        
        # Plot 1: Transition matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix,
                x=[f"Type {i}" for i in range(transition_matrix.shape[1])],
                y=[f"Type {i}" for i in range(transition_matrix.shape[0])],
                colorscale='YlOrRd',
                text=np.round(transition_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True
            ),
            row=1, col=1
        )
        
        # Add transition statistics
        total_transitions = np.sum(transition_matrix > 0)
        most_common = np.unravel_index(np.argmax(transition_matrix), 
                                     transition_matrix.shape)
        stats_text = (f"Total unique transitions: {total_transitions}<br>"
                     f"Most common transition: {most_common[0]} → {most_common[1]}<br>"
                     f"Probability: {transition_matrix[most_common]:.2f}")
        
        # Plot 2: Cluster sequence over time
        time_points = np.arange(len(cluster_sequence))
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=cluster_sequence,
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color=cluster_sequence,
                    colorscale='Viridis',
                    showscale=True
                ),
                line=dict(color='black', width=1),
                name='Coda Type'
            ),
            row=1, col=2
        )
        
        # Add cluster statistics
        unique_clusters = np.unique(cluster_sequence)
        cluster_stats = []
        for cluster_id in unique_clusters:
            mask = cluster_sequence == cluster_id
            duration = timestamps[mask][-1] - timestamps[mask][0] if len(timestamps[mask]) > 1 else 0
            cluster_stats.append([f"Type {cluster_id}", sum(mask), f"{duration:.1f}s"])
        
        # Plot 3: Coda type duration distribution
        durations = []
        current_cluster = cluster_sequence[0]
        start_time = timestamps[0]
        
        for i in range(1, len(cluster_sequence)):
            if cluster_sequence[i] != current_cluster:
                durations.append((current_cluster, timestamps[i] - start_time))
                current_cluster = cluster_sequence[i]
                start_time = timestamps[i]
        durations.append((current_cluster, timestamps[-1] - start_time))
        
        cluster_durations = {cluster: [] for cluster in unique_clusters}
        for cluster, duration in durations:
            cluster_durations[cluster].append(duration)
        
        for cluster in unique_clusters:
            fig.add_trace(
                go.Box(
                    y=cluster_durations[cluster],
                    name=f"Type {cluster}",
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=2, col=1
            )
        
        # Plot 4: Statistics table
        total_duration = timestamps[-1] - timestamps[0]
        avg_duration = np.mean([d for _, d in durations])
        stats_data = [
            ["Total Duration", f"{total_duration:.1f}s"],
            ["Average Duration", f"{avg_duration:.1f}s"],
            ["Number of Types", str(len(unique_clusters))]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in stats_data],
                                 [row[1] for row in stats_data]])
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Temporal Analysis of Whale Coda Patterns",
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Next Coda Type", row=1, col=1)
        fig.update_yaxes(title_text="Current Coda Type", row=1, col=1)
        fig.update_xaxes(title_text="Time Step", row=1, col=2)
        fig.update_yaxes(title_text="Coda Type ID", row=1, col=2)
        fig.update_xaxes(title_text="Coda Type", row=2, col=1)
        fig.update_yaxes(title_text="Duration (seconds)", row=2, col=1)
        
        if save_path:
            fig.write_html(str(save_path))
        else:
            fig.show() 