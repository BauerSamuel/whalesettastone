"""
Module for training VAE models.
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Optional
import json
from datetime import datetime

from .generative import CodaVAE

class VAETrainer:
    """Trainer class for VAE models."""
    
    def __init__(self, model: CodaVAE, train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-3,
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize the trainer.
        
        Args:
            model: VAE model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            learning_rate: Initial learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training components
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Setup logging
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
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save a checkpoint of the model.
        
        Args:
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save metrics
        metrics_path = self.checkpoint_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            val_loss = metrics.get('val_loss/total', metrics.get('train_loss/total'))
            self.logger.info(f"Saved best model with loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_metrics = {}
        n_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(batch)
            
            # Compute loss and metrics
            loss, batch_metrics = self.model.loss_function(recon_batch, batch, mu, log_var)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            n_batches += 1
        
        # Average metrics
        avg_metrics = {
            f'train_{k}': v / n_batches 
            for k, v in total_metrics.items()
        }
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        return self.model.compute_validation_metrics(self.val_loader)
    
    def train(self, n_epochs: int, early_stopping_patience: int = 10) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            n_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary of final metrics
        """
        self.logger.info(f"Starting training for {n_epochs} epochs")
        self.logger.info(f"Using device: {self.device}")
        
        for epoch in range(self.current_epoch, self.current_epoch + n_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.current_epoch + n_epochs} - "
                f"Train Loss: {metrics['train_loss/total']:.4f} - "
                f"Val Loss: {metrics.get('val_loss/total', 'N/A')} - "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Update learning rate
            if self.val_loader is not None:
                val_loss = metrics['val_loss/total']
                self.scheduler.step(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(metrics, is_best=True)
                else:
                    self.epochs_without_improvement += 1
                    
                # Early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch+1} epochs"
                    )
                    break
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(metrics)
        
        self.logger.info("Training completed")
        return metrics 