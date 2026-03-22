"""
Module for handling coda feature embeddings and preparing them for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import hashlib
import hmac
import secrets
import base64
from cryptography.fernet import Fernet
import os

class CodaEmbeddings:
    """
    Handles the preparation and management of coda feature embeddings with security features.
    """
    
    def __init__(self, embeddings_dir: str = "data/embeddings", secret_key: Optional[str] = None):
        """
        Initialize the embeddings handler.
        
        Args:
            embeddings_dir: Directory to store embeddings
            secret_key: Optional secret key for encryption. If None, a new key will be generated.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        if secret_key:
            # Explicit secret passed in (preferred for programmatic use)
            self.fernet = Fernet(secret_key.encode())
        else:
            # Prefer an environment-provided key if available (safer than on-disk)
            env_key = os.getenv("CODA_EMBEDDINGS_KEY")
            if env_key:
                self.fernet = Fernet(env_key.encode())
            else:
                # Fallback: load or generate a local key file in embeddings_dir
                key_path = self.embeddings_dir / ".secret_key"
                if key_path.exists():
                    with open(key_path, 'rb') as f:
                        key = f.read()
                    self.fernet = Fernet(key)
                else:
                    # Generate a new key and store it locally
                    key = Fernet.generate_key()
                    self.fernet = Fernet(key)
                    # Save the key securely
                    with open(key_path, 'wb') as f:
                        f.write(key)
                    try:
                        os.chmod(key_path, 0o600)  # Best-effort: owner read/write only (POSIX)
                    except PermissionError:
                        # On some platforms (e.g. Windows) chmod may be a no-op; ignore.
                        pass
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Log to stderr only — no on-disk log files (keeps repo/workspace clean)."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_features(self, features: List[Dict]) -> bool:
        """
        Validate feature data to prevent injection attacks.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            bool: True if features are valid
        """
        required_fields = {'filename', 'mfcc_1', 'mfcc_2', 'spectral_centroid', 'spectral_rolloff'}
        
        for feature in features:
            # Check required fields
            if not all(field in feature for field in required_fields):
                return False
                
            # Validate filename
            if not isinstance(feature['filename'], str) or not feature['filename'].endswith('.wav'):
                return False
                
            # Validate numeric values
            for field in ['mfcc_1', 'mfcc_2', 'spectral_centroid', 'spectral_rolloff']:
                if not isinstance(feature[field], (int, float)):
                    return False
                    
        return True
    
    def _generate_checksum(self, data: bytes) -> str:
        """Generate a secure checksum for data integrity verification."""
        return hmac.new(
            self.fernet._signing_key,
            data,
            hashlib.sha256
        ).hexdigest()
    
    def prepare_embeddings(self, features: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature embeddings from raw feature dictionaries.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Tuple of (embeddings array, filenames)
        """
        try:
            # Validate features
            if not self._validate_features(features):
                raise ValueError("Invalid feature data detected")
            
            embeddings = []
            filenames = []
            
            for feature_dict in features:
                # Extract MFCCs
                mfccs = []
                for i in range(1, 14):  # We have 13 MFCCs
                    mfcc_key = f'mfcc_{i}'
                    if mfcc_key in feature_dict:
                        mfccs.append(feature_dict[mfcc_key])
                
                # Add other features
                vector = mfccs.copy()
                for key in ['spectral_centroid', 'spectral_rolloff', 
                          'spectral_bandwidth', 'zero_crossing_rate', 'rms_energy']:
                    if key in feature_dict:
                        vector.append(feature_dict[key])
                
                embeddings.append(vector)
                filenames.append(feature_dict.get('filename', 'unknown'))
            
            return np.array(embeddings), filenames
            
        except Exception as e:
            self.logger.error(f"Error preparing embeddings: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, filenames: List[str], 
                       metadata: Optional[Dict] = None) -> Path:
        """
        Save embeddings to disk with encryption and integrity checks.
        
        Args:
            embeddings: Embeddings array
            filenames: List of filenames
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved embeddings file
        """
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            embeddings_file = self.embeddings_dir / f"embeddings_{timestamp}.npz"
            
            # Prepare data for saving
            data = {
                'embeddings': embeddings.tolist(),  # Convert to list for JSON serialization
                'filenames': filenames,
                'metadata': metadata if metadata else {},
                'timestamp': timestamp
            }
            
            # Serialize and encrypt data
            serialized_data = json.dumps(data).encode()
            encrypted_data = self.fernet.encrypt(serialized_data)
            
            # Generate checksum
            checksum = self._generate_checksum(encrypted_data)
            
            # Save encrypted data and checksum
            with open(embeddings_file, 'wb') as f:
                f.write(encrypted_data)
                f.write(b'\n')
                f.write(checksum.encode())
            
            # Set secure permissions
            os.chmod(embeddings_file, 0o600)
            
            self.logger.info(f"Saved encrypted embeddings to {embeddings_file}")
            return embeddings_file
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, embeddings_file: Path) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load embeddings from disk with integrity verification.
        
        Args:
            embeddings_file: Path to embeddings file
            
        Returns:
            Tuple of (embeddings array, filenames, metadata)
        """
        try:
            # Read encrypted data and checksum
            with open(embeddings_file, 'rb') as f:
                content = f.read()
                encrypted_data, checksum = content.split(b'\n', 1)
            
            # Verify checksum
            if self._generate_checksum(encrypted_data) != checksum.decode():
                raise ValueError("Data integrity check failed")
            
            # Decrypt data
            decrypted_data = self.fernet.decrypt(encrypted_data)
            data = json.loads(decrypted_data)
            
            # Convert numpy arrays
            embeddings = np.array(data['embeddings'])
            filenames = data['filenames']
            metadata = data['metadata']
            
            self.logger.info(f"Loaded embeddings from {embeddings_file}")
            return embeddings, filenames, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def get_latest_embeddings(self) -> Optional[Path]:
        """
        Get the path to the most recent embeddings file.
        
        Returns:
            Path to latest embeddings file or None if none exist
        """
        try:
            embedding_files = list(self.embeddings_dir.glob("embeddings_*.npz"))
            if not embedding_files:
                return None
            return max(embedding_files, key=lambda x: x.stat().st_mtime)
        except Exception as e:
            self.logger.error(f"Error getting latest embeddings: {str(e)}")
            return None

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj) 