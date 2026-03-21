"""
Variational Autoencoder for whale coda analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CodaVAE(nn.Module):
    def __init__(self, input_dim=10, latent_dim=5, hidden_dims=[8]):
        """
        Initialize the CodaVAE model.
        
        Args:
            input_dim (int): Dimension of input features
            latent_dim (int): Dimension of latent space
            hidden_dims (list): List of hidden layer dimensions
        """
        super(CodaVAE, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input into latent space.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (mu, log_var) of latent space
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for sampling.
        
        Args:
            mu (torch.Tensor): Mean of latent space
            log_var (torch.Tensor): Log variance of latent space
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode from latent space.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (reconstructed_x, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var):
        """
        Calculate VAE loss.
        
        Args:
            recon_x (torch.Tensor): Reconstructed input
            x (torch.Tensor): Original input
            mu (torch.Tensor): Mean of latent space
            log_var (torch.Tensor): Log variance of latent space
            
        Returns:
            torch.Tensor: Total loss
        """
        # Reconstruction loss
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return BCE + KLD
    
    def generate(self, num_samples=1, temperature=1.0):
        """
        Generate new samples from the latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            temperature (float): Temperature for sampling (higher = more random)
            
        Returns:
            torch.Tensor: Generated samples
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.fc_mu.out_features) * temperature
            return self.decode(z) 