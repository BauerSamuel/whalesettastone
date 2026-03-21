"""
Test script for the whale audio analysis workflow.
"""

import sys
import logging
from pathlib import Path
import shutil

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.workflow import Workflow
from src.config import (
    BASE_DIR,
    WHALE_CODAS_DIR,
    UPLOADS_DIR,
    SAMPLE_AUDIO_DIR,
    LOGS_DIR,
    METADATA_DIR,
    PLOTS_DIR
)

def setup_test_data():
    """Set up test data in the sample audio directory."""
    print("\nSetting up test data...")
    
    # Create a test tone if it doesn't exist
    test_file = SAMPLE_AUDIO_DIR / "test_tone_1.wav"
    if not test_file.exists():
        try:
            import numpy as np
            import soundfile as sf
            
            # Generate a simple test tone
            sample_rate = 44100
            duration = 1.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            # Save as WAV file
            sf.write(test_file, tone, sample_rate)
            print(f"✓ Created test file: {test_file}")
        except Exception as e:
            print(f"✗ Failed to create test file: {e}")
            return False
    else:
        print(f"✓ Test file exists: {test_file}")
    
    return True

def test_directory_creation():
    """Test that all required directories exist."""
    print("\nTesting directory creation...")
    required_dirs = [
        BASE_DIR,
        WHALE_CODAS_DIR,
        UPLOADS_DIR,
        SAMPLE_AUDIO_DIR,
        LOGS_DIR,
        METADATA_DIR,
        PLOTS_DIR
    ]
    
    for directory in required_dirs:
        if directory.exists():
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  → Created directory: {directory}")

def test_file_import():
    """Test importing a sample file."""
    print("\nTesting file import...")
    
    # Create workflow instance
    workflow = Workflow()
    
    # Test importing single file
    sample_file = SAMPLE_AUDIO_DIR / "test_tone_1.wav"
    if not sample_file.exists():
        print("✗ Sample file not found")
        return False
    
    result = workflow.importer.import_file(sample_file)
    if result:
        print(f"✓ Successfully imported file: {result.name}")
        return True
    else:
        print("✗ File import failed")
        return False

def test_feature_extraction():
    """Test feature extraction from an imported file."""
    print("\nTesting feature extraction...")
    
    # Create workflow instance
    workflow = Workflow()
    
    # Get a test file
    test_file = WHALE_CODAS_DIR / "test_tone_1.wav"
    if not test_file.exists():
        print("✗ No test file available for feature extraction")
        return False
    
    # Extract features
    features = workflow.extractor.extract_features(test_file)
    if features:
        print("✓ Successfully extracted features:")
        for key in features.keys():
            print(f"  - {key}")
        return True
    else:
        print("✗ Feature extraction failed")
        return False

def main():
    """Run all tests."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    if setup_test_data():
        test_directory_creation()
        test_file_import()
        test_feature_extraction()
    else:
        print("\n✗ Test setup failed, skipping remaining tests")
    
    print("\nTest completed")

if __name__ == "__main__":
    main() 