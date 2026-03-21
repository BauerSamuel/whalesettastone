"""
Module for generative modeling of whale codas.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
import umap
from .embeddings import CodaEmbeddings

class CodaDataset(Dataset):
    """Dataset class for whale coda embeddings."""
    
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            embeddings: Array of coda embeddings
        """
        self.embeddings = torch.FloatTensor(embeddings)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]

class CodaDataPrep:
    """Prepares coda data for generative modeling."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data preparation class.
        
        Args:
            data_dir: Directory containing the data
        """
        self.data_dir = Path(data_dir)
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.reducer = None
        self.embeddings_handler = CodaEmbeddings(embeddings_dir=self.embeddings_dir)
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        from ..config import LOGS_DIR

        log_dir = LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "generative.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess(self, reduce_dim: bool = True,
                           n_components: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Load coda embeddings and preprocess them for model training.
        
        Args:
            reduce_dim: Whether to reduce dimensionality using UMAP
            n_components: Number of dimensions for UMAP reduction
            
        Returns:
            Tuple of (preprocessed embeddings array, list of filenames)
        """
        try:
            # Get latest embeddings file
            embeddings_file = self.embeddings_handler.get_latest_embeddings()
            if embeddings_file is None:
                raise ValueError("No embeddings file found")
            
            # Load embeddings
            embeddings, filenames, _ = self.embeddings_handler.load_embeddings(embeddings_file)
            
            # Check array and ensure finite values
            embeddings = check_array(embeddings, ensure_2d=True, dtype=np.float64)
            
            # Scale the embeddings
            scaled_embeddings = self.scaler.fit_transform(embeddings)
            
            # Optionally reduce dimensionality
            if reduce_dim:
                # Ensure n_components is less than the number of samples
                n_components = min(n_components, embeddings.shape[0] - 1)
                
                # Adjust UMAP parameters based on sample size
                n_neighbors = min(15, embeddings.shape[0] - 1)
                min_dist = 0.1
                
                # Initialize UMAP with adjusted parameters
                self.reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    random_state=42,
                    n_jobs=1,  # Explicitly set for reproducibility
                    low_memory=True  # Enable low memory mode for better stability
                )
                
                # Fit and transform
                reduced_embeddings = self.reducer.fit_transform(scaled_embeddings)
                self.logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {n_components} dimensions")
                return reduced_embeddings, filenames
            
            return scaled_embeddings, filenames
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def save_training_data(self, embeddings: np.ndarray,
                          filenames: List[str],
                          output_dir: Optional[str] = None) -> Path:
        """
        Save preprocessed embeddings for model training.
        
        Args:
            embeddings: Preprocessed embeddings array
            filenames: List of corresponding filenames
            output_dir: Optional output directory
            
        Returns:
            Path to saved data file
            
        Raises:
            OSError: If the output directory is invalid or cannot be created
        """
        try:
            # Set output directory
            if output_dir is None:
                output_dir = self.data_dir / "processed"
            else:
                output_dir = Path(output_dir)
                
            # Check if directory exists or can be created
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True)
                except OSError as e:
                    raise OSError(f"Could not create output directory: {str(e)}")
            
            # Save data
            output_file = output_dir / "training_data.npz"
            np.savez(
                output_file,
                embeddings=embeddings,
                filenames=filenames
            )
            
            self.logger.info(f"Saved training data to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving training data: {str(e)}")
            raise
    
    def create_data_loader(self, embeddings: np.ndarray,
                          batch_size: int = 32,
                          shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader for model training.
        
        Args:
            embeddings: Preprocessed embeddings array
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        dataset = CodaDataset(embeddings)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

class CodaVAE(nn.Module):
    """Variational Autoencoder for whale coda embeddings."""
    
    def __init__(self, input_dim: int = 10, latent_dim: int = 5,
                 hidden_dims: List[int] = None):
        """
        Initialize the VAE.
        
        Args:
            input_dim: Dimension of input embeddings
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [8]
        
        # Build encoder
        encoder_layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.ReLU()
            ])
            in_features = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Initialize weights for log variance to be negative
        nn.init.constant_(self.fc_var.weight, -0.1)
        nn.init.constant_(self.fc_var.bias, -0.1)
        
        # Build decoder
        decoder_layers = []
        in_features = latent_dim
        
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.ReLU()
            ])
            in_features = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input into latent space parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, log variance) tensors
        """
        # Pass through encoder layers
        h = self.encoder(x)
        
        # Get latent parameters
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor,
                      log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent space.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector back to input space.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed input, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss and metrics.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Tuple of (total loss, metrics dictionary)
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        # Compute additional metrics
        with torch.no_grad():
            # Cosine similarity between original and reconstructed
            cos_sim = nn.functional.cosine_similarity(
                x.view(x.size(0), -1),
                recon_x.view(recon_x.size(0), -1)
            ).mean()
            
            # Latent space statistics
            latent_norm = torch.norm(mu, dim=1).mean()
            latent_std = torch.exp(0.5 * log_var).mean()
            
            # Per-sample losses
            recon_loss_per_sample = nn.functional.mse_loss(
                recon_x, x, reduction='none'
            ).sum(dim=1).mean()
            kl_loss_per_sample = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(), dim=1
            ).mean()
        
        metrics = {
            'loss/reconstruction': recon_loss_per_sample.item(),
            'loss/kl_divergence': kl_loss_per_sample.item(),
            'loss/total': total_loss.item(),
            'reconstruction/cosine_similarity': cos_sim.item(),
            'latent/norm': latent_norm.item(),
            'latent/std': latent_std.item()
        }
        
        return total_loss, metrics
    
    def compute_validation_metrics(self, val_loader: DataLoader) -> dict:
        """
        Compute validation metrics over the entire validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.eval()
        total_metrics = {}
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass
                recon_batch, mu, log_var = self(batch)
                
                # Compute loss and metrics
                _, batch_metrics = self.loss_function(recon_batch, batch, mu, log_var)
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                n_batches += 1
        
        # Average metrics
        avg_metrics = {
            f'val_{k}': v / n_batches 
            for k, v in total_metrics.items()
        }
        
        self.train()
        return avg_metrics 