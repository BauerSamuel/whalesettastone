"""
Data handling module for whale coda analysis.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Union

class CodaDataset(Dataset):
    """Dataset class for whale coda data."""
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Initialize the dataset.
        
        Args:
            data: Numpy array of shape (n_samples, n_features) containing the coda embeddings
            labels: Optional numpy array of shape (n_samples,) containing labels for each coda
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Either a single tensor (data) if no labels are present,
            or a tuple of (data, label) if labels were provided
        """
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx] 