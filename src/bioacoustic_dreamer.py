"""
Bioacoustic dreaming module for generating and analyzing whale coda dream sequences.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from .coda_vae import CodaVAE

class BioacousticDreamer:
    def __init__(
        self,
        model=None,
        model_path=None,
        input_dim=10,
        latent_dim=5,
        hidden_dims=None,
    ):
        """
        Initialize the BioacousticDreamer.

        Args:
            model: Optional CodaVAE instance to use (avoids loading from file).
            model_path: Optional path to saved model weights (used only if model is None).
            input_dim: Dimension of input features (used only if model is None).
            latent_dim: Dimension of latent space (used only if model is None).
            hidden_dims: Hidden layer dimensions (used only if model is None).
        """
        if hidden_dims is None:
            hidden_dims = [8]
        if model is not None:
            self.model = model
        else:
            self.model = CodaVAE(input_dim, latent_dim, hidden_dims)
            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
    def generate_dream_sequence(self, num_steps=100, temperature=1.0, start_point=None):
        """
        Generate a dream sequence in latent space.
        
        Args:
            num_steps (int): Number of steps in the sequence
            temperature (float): Controls randomness (higher = more random)
            start_point (torch.Tensor, optional): Starting point in latent space
            
        Returns:
            torch.Tensor: Generated dream sequence
        """
        with torch.no_grad():
            if start_point is None:
                # Start from random point in latent space
                z = torch.randn(1, self.model.fc_mu.out_features) * temperature
            else:
                z = start_point
            
            sequence = []
            for _ in range(num_steps):
                # Add small random walk
                z = z + torch.randn_like(z) * 0.1 * temperature
                # Generate sample
                sample = self.model.decode(z)
                sequence.append(sample)
            
            return torch.stack(sequence)
    
    def decode_dream(self, dream_sequence):
        """
        Decode a dream sequence from latent space to embedding space.
        
        Args:
            dream_sequence (torch.Tensor): Dream sequence in latent space
            
        Returns:
            torch.Tensor: Decoded sequence in embedding space
        """
        with torch.no_grad():
            return self.model.decode(dream_sequence)
    
    def analyze_dream(self, dream_sequence):
        """
        Analyze characteristics of a dream sequence.
        
        Args:
            dream_sequence (torch.Tensor): Dream sequence
            
        Returns:
            dict: Analysis results
        """
        sequence_np = dream_sequence.numpy()
        
        return {
            'mean': np.mean(sequence_np, axis=0),
            'std': np.std(sequence_np, axis=0),
            'smoothness': np.mean(np.diff(sequence_np, axis=0)),
            'diversity': np.mean(np.std(sequence_np, axis=1))
        }
    
    def visualize_dream(self, dream_sequence, save_path=None):
        """
        Visualize a dream sequence.
        
        Args:
            dream_sequence (torch.Tensor): Dream sequence
            save_path (str, optional): Path to save visualization
        """
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        # Plot each dimension
        sequence_np = dream_sequence.numpy()
        for i in range(sequence_np.shape[1]):
            plt.plot(sequence_np[:, i], label=f'Dimension {i+1}')
        
        plt.title('Dream Sequence Visualization')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_dream_animation(self, dream_sequence, save_path, fps=10):
        """
        Create an animation of the dream sequence.
        
        Args:
            dream_sequence (torch.Tensor): Dream sequence
            save_path (str): Path to save animation
            fps (int): Frames per second
        """
        sequence_np = dream_sequence.numpy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        def update(frame):
            ax.clear()
            for i in range(sequence_np.shape[1]):
                ax.plot(sequence_np[:frame+1, i], label=f'Dimension {i+1}')
            ax.set_title(f'Dream Sequence (Step {frame+1})')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.set_xlim(0, len(sequence_np))
            ax.set_ylim(sequence_np.min(), sequence_np.max())
        
        anim = FuncAnimation(fig, update, frames=len(sequence_np), 
                           interval=1000/fps, blit=False)
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close() 