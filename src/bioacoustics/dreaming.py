"""
Module for bioacoustic dreaming analysis using VAE models.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.manifold import TSNE
from .generative import CodaVAE, CodaDataset

class BioacousticDreamer:
    """Class for generating and analyzing bioacoustic dreams."""
    
    def __init__(self, model: CodaVAE, device: Optional[str] = None):
        """
        Initialize the dreamer.
        
        Args:
            model: Trained VAE model
            device: Device to run the model on (cuda/cpu)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Log to stderr only — no on-disk log files."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_dream_sequence(self, n_steps: int = 100,
                              temperature: float = 1.0,
                              start_point: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a dream sequence in the latent space.
        
        Args:
            n_steps: Number of steps in the dream sequence
            temperature: Controls the randomness of the dream (higher = more random)
            start_point: Optional starting point in latent space
            
        Returns:
            Tensor of dream sequence in latent space
        """
        with torch.no_grad():
            # Initialize starting point
            if start_point is None:
                z = torch.randn(1, self.model.latent_dim, device=self.device)
            else:
                z = start_point.to(self.device)
            
            # Store the sequence
            dream_sequence = [z]
            
            # Generate sequence
            for _ in range(n_steps - 1):
                # Add some noise and drift
                noise = torch.randn_like(z) * temperature
                drift = torch.randn_like(z) * 0.1
                z = z + noise + drift
                
                # Keep within reasonable bounds
                z = torch.clamp(z, -3, 3)
                dream_sequence.append(z)
            
            return torch.cat(dream_sequence)
    
    def decode_dream(self, dream_sequence: torch.Tensor) -> torch.Tensor:
        """
        Decode a dream sequence from latent space to embedding space.
        
        Args:
            dream_sequence: Sequence in latent space
            
        Returns:
            Decoded sequence in embedding space
        """
        if dream_sequence.shape[-1] != self.model.latent_dim:
            raise ValueError(
                f"Dream sequence latent dim {dream_sequence.shape[-1]} does not match "
                f"model latent_dim {self.model.latent_dim}"
            )
        with torch.no_grad():
            return self.model.decode(dream_sequence)
    
    def analyze_dream(self, dream_sequence: torch.Tensor) -> Dict[str, float]:
        """
        Analyze characteristics of a dream sequence.
        
        Args:
            dream_sequence: Sequence in latent space
            
        Returns:
            Dictionary of dream characteristics
        """
        with torch.no_grad():
            # Calculate basic statistics
            mean = dream_sequence.mean(dim=0)
            std = dream_sequence.std(dim=0)
            
            # Calculate smoothness (how much the sequence changes)
            changes = torch.diff(dream_sequence, dim=0)
            smoothness = torch.mean(torch.norm(changes, dim=1))
            
            # Calculate diversity (how much the sequence explores)
            diversity = torch.mean(torch.norm(dream_sequence - mean, dim=1))
            
            return {
                'mean_norm': torch.norm(mean).item(),
                'std_norm': torch.norm(std).item(),
                'smoothness': smoothness.item(),
                'diversity': diversity.item()
            }
    
    def visualize_dream(self, dream_sequence: torch.Tensor,
                       save_path: Optional[Path] = None,
                       title: str = "Bioacoustic Dream") -> None:
        """
        Visualize a dream sequence.
        
        Args:
            dream_sequence: Sequence in latent space
            save_path: Optional path to save the visualization
            title: Title for the plot
        """
        # Convert to numpy for visualization
        dream_np = dream_sequence.cpu().numpy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot latent space trajectory
        if dream_np.shape[1] > 2:
            # Use t-SNE for dimensionality reduction if needed
            tsne = TSNE(n_components=2, random_state=42)
            dream_2d = tsne.fit_transform(dream_np)
        else:
            dream_2d = dream_np
        
        ax1.plot(dream_2d[:, 0], dream_2d[:, 1], 'b-', alpha=0.5)
        ax1.scatter(dream_2d[:, 0], dream_2d[:, 1], c=range(len(dream_2d)),
                   cmap='viridis', s=50)
        ax1.set_title(f"{title} - Latent Space Trajectory")
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")
        
        # Plot dream characteristics over time
        characteristics = self.analyze_dream(dream_sequence)
        ax2.plot(range(len(dream_np)), dream_np, alpha=0.5)
        ax2.set_title(f"{title} - Latent Dimensions")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Value")
        ax2.legend([f"Dim {i}" for i in range(dream_np.shape[1])])
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path)
                self.logger.info(f"Saved dream visualization to {save_path}")
            except Exception as e:
                plt.close()
                raise OSError(f"Failed to save dream visualization: {e}") from e
        else:
            plt.show()
        
        plt.close()
    
    def create_dream_animation(self, dream_sequence: torch.Tensor,
                             save_path: Path,
                             title: str = "Bioacoustic Dream") -> None:
        """
        Create an animation of the dream sequence.
        
        Args:
            dream_sequence: Sequence in latent space
            save_path: Path to save the animation
            title: Title for the animation
        """
        # Convert to numpy for visualization
        dream_np = dream_sequence.cpu().numpy()
        
        # Use t-SNE for dimensionality reduction if needed
        if dream_np.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            dream_2d = tsne.fit_transform(dream_np)
        else:
            dream_2d = dream_np
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(frame):
            ax.clear()
            ax.plot(dream_2d[:frame+1, 0], dream_2d[:frame+1, 1], 'b-', alpha=0.5)
            ax.scatter(dream_2d[frame, 0], dream_2d[frame, 1], c='red', s=100)
            ax.set_title(f"{title} - Step {frame+1}/{len(dream_2d)}")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.set_xlim(dream_2d[:, 0].min() - 1, dream_2d[:, 0].max() + 1)
            ax.set_ylim(dream_2d[:, 1].min() - 1, dream_2d[:, 1].max() + 1)
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(dream_2d),
                           interval=100, blit=False)
        
        # Save animation
        anim.save(save_path, writer="pillow", fps=10)
        self.logger.info(f"Saved dream animation to {save_path}")
        plt.close() 