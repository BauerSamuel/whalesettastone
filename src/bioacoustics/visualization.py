"""
Module for visualizing VAE training progress and results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import torch
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os
import pandas as pd

def load_metrics_from_checkpoints(checkpoint_dir: str) -> Dict[str, List[float]]:
    """
    Load metrics from all checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Dictionary of metric names to lists of values
    """
    checkpoint_dir = Path(checkpoint_dir)
    metrics = {}
    
    # Find all metrics files
    metrics_files = sorted(checkpoint_dir.glob("metrics_*.json"))
    
    for file in metrics_files:
        with open(file, 'r') as f:
            epoch_metrics = json.load(f)
            
            # Initialize lists for new metrics
            for k in epoch_metrics.keys():
                if k not in metrics:
                    metrics[k] = []
            
            # Append values
            for k, v in epoch_metrics.items():
                metrics[k].append(v)
    
    return metrics

def plot_training_progress(checkpoint_dir: str, output_dir: Optional[str] = None):
    """
    Plot training progress from checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_dir: Optional directory to save plots
    """
    # Load metrics
    metrics = load_metrics_from_checkpoints(checkpoint_dir)
    if not metrics:
        raise ValueError("No metrics found in checkpoint directory")
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(checkpoint_dir) / "plots"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("VAE Training Progress", fontsize=16)
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    epochs = range(1, len(metrics['train_loss/total']) + 1)
    
    ax.plot(epochs, metrics['train_loss/total'], label='Training Loss')
    if 'val_loss/total' in metrics:
        ax.plot(epochs, metrics['val_loss/total'], label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Reconstruction quality
    ax = axes[0, 1]
    ax.plot(epochs, metrics['train_reconstruction/cosine_similarity'],
            label='Training')
    if 'val_reconstruction/cosine_similarity' in metrics:
        ax.plot(epochs, metrics['val_reconstruction/cosine_similarity'],
                label='Validation')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Reconstruction Quality')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Latent space statistics
    ax = axes[1, 0]
    ax.plot(epochs, metrics['train_latent/norm'], label='Norm')
    ax.plot(epochs, metrics['train_latent/std'], label='Std')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Latent Space Statistics')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Loss components
    ax = axes[1, 1]
    ax.plot(epochs, metrics['train_loss/reconstruction'],
            label='Reconstruction Loss')
    ax.plot(epochs, metrics['train_loss/kl_divergence'],
            label='KL Divergence')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(output_dir / f"training_progress_{timestamp}.png")
    plt.close()

def plot_latent_space(
    vae_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    output_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """Plot the latent space of the VAE model.
    
    Args:
        vae_model: The trained VAE model
        data_loader: DataLoader containing the data to visualize
        output_dir: Directory to save the plot
        device: Device to run the model on
    """
    try:
        vae_model.eval()
        latents = []
        labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch
                    
                data = data.to(device)
                mu, _ = vae_model.encode(data)
                latents.append(mu.cpu().numpy())
                
                if isinstance(batch, (tuple, list)) and len(batch) > 1:
                    labels.extend(batch[1].cpu().numpy())
        
        latents = np.vstack(latents)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(latents, columns=[f'z{i+1}' for i in range(latents.shape[1])])
        if labels:
            df['label'] = labels
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save HTML plot using plotly
        try:
            fig = go.Figure()
            if labels:
                for label in np.unique(labels):
                    mask = df['label'] == label
                    fig.add_trace(go.Scatter(
                        x=df.loc[mask, 'z1'],
                        y=df.loc[mask, 'z2'],
                        mode='markers',
                        name=f'Label {label}',
                        marker=dict(
                            size=8,
                            opacity=0.7
                        )
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df['z1'],
                    y=df['z2'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        opacity=0.7
                    )
                ))
            
            fig.update_layout(
                title='Latent Space Visualization',
                xaxis_title='First Latent Dimension',
                yaxis_title='Second Latent Dimension',
                showlegend=True,
                width=1000,
                height=800,
                template='plotly_white'
            )
            
            fig.write_html(os.path.join(output_dir, 'latent_space.html'))
        except Exception as e:
            print(f"Warning: Failed to save HTML plot: {str(e)}")
        
        # Save PNG plot using matplotlib
        try:
            plt.figure(figsize=(10, 8))
            if labels:
                for label in np.unique(labels):
                    mask = df['label'] == label
                    plt.scatter(df.loc[mask, 'z1'], df.loc[mask, 'z2'], 
                              alpha=0.7, label=f'Label {label}')
                plt.legend()
            else:
                plt.scatter(df['z1'], df['z2'], alpha=0.7)
            
            plt.title('Latent Space Visualization')
            plt.xlabel('First Latent Dimension')
            plt.ylabel('Second Latent Dimension')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'latent_space.png'))
            plt.close()
        except Exception as e:
            print(f"Error saving PNG plot: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Error in plot_latent_space: {str(e)}")
        raise 