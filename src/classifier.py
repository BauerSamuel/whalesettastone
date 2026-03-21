import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import librosa
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhaleClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.pca = PCA(n_components=2)
        
    def fit(self, features_dict):
        """Fit the classifier on the extracted features"""
        if not features_dict:
            logger.warning("Empty features dictionary provided to fit")
            return None
            
        try:
            # Convert features to array and ensure they're all the same length
            features_list = []
            for filename, features in features_dict.items():
                if isinstance(features, np.ndarray):
                    features_list.append(features)
                else:
                    features_list.append(np.array(features))
            
            # Stack features and ensure they're all the same length
            max_len = max(len(f) for f in features_list)
            padded_features = []
            for f in features_list:
                if len(f) < max_len:
                    padded = np.pad(f, (0, max_len - len(f)), 'constant')
                    padded_features.append(padded)
                else:
                    padded_features.append(f)
            
            features = np.array(padded_features)
            logger.info(f"Features shape before scaling: {features.shape}")
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            logger.info(f"Features shape after scaling: {scaled_features.shape}")
            
            # Fit KMeans
            self.kmeans.fit(scaled_features)
            
            # Fit PCA for visualization
            self.pca.fit(scaled_features)
            
            return self.kmeans.labels_
        except Exception as e:
            logger.error(f"Error in fit: {str(e)}")
            logger.error(f"Features dictionary keys: {list(features_dict.keys())}")
            return None
        
    def predict(self, features_dict):
        """Predict clusters for new features"""
        if not features_dict:
            return None
            
        # Convert features to array and ensure they're all the same length
        features_list = []
        for filename, features in features_dict.items():
            if isinstance(features, np.ndarray):
                features_list.append(features)
            else:
                features_list.append(np.array(features))
        
        # Stack features and ensure they're all the same length
        max_len = max(len(f) for f in features_list)
        padded_features = []
        for f in features_list:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), 'constant')
                padded_features.append(padded)
            else:
                padded_features.append(f)
        
        features = np.array(padded_features)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict clusters
        return self.kmeans.predict(scaled_features)
        
    def get_pca_components(self, features_dict):
        """Get PCA components for visualization"""
        if not features_dict:
            logger.warning("Empty features dictionary provided to get_pca_components")
            return None
            
        try:
            # Convert features to array and ensure they're all the same length
            features_list = []
            for filename, features in features_dict.items():
                if isinstance(features, np.ndarray):
                    features_list.append(features)
                else:
                    features_list.append(np.array(features))
            
            # Stack features and ensure they're all the same length
            max_len = max(len(f) for f in features_list)
            padded_features = []
            for f in features_list:
                if len(f) < max_len:
                    padded = np.pad(f, (0, max_len - len(f)), 'constant')
                    padded_features.append(padded)
                else:
                    padded_features.append(f)
            
            features = np.array(padded_features)
            logger.info(f"Features shape before scaling: {features.shape}")
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            logger.info(f"Features shape after scaling: {scaled_features.shape}")
            
            # Get PCA components
            pca_components = self.pca.transform(scaled_features)
            logger.info(f"PCA components shape: {pca_components.shape}")
            return pca_components
        except Exception as e:
            logger.error(f"Error in get_pca_components: {str(e)}")
            logger.error(f"Features dictionary keys: {list(features_dict.keys())}")
            return None
        
    @staticmethod
    def extract_features(audio_data, sample_rate):
        """Extract features from audio data"""
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
            mfccs_mean = np.mean(mfccs, axis=1)
            logger.info(f"MFCCs shape: {mfccs.shape}, MFCCs mean shape: {mfccs_mean.shape}")
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate))
            
            # Extract temporal features
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
            rms = np.mean(librosa.feature.rms(y=audio_data))
            
            # Combine all features
            features = np.concatenate([
                mfccs_mean,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth],
                [zero_crossing_rate, rms]
            ])
            logger.info(f"Final features shape: {features.shape}")
            
            return features
        except Exception as e:
            logger.error(f"Error in extract_features: {str(e)}")
            return None 