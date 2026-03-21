"""
Script to create test WAV files with different frequencies.

Run from repository root:
    python scripts/create_test_audio.py
"""

import numpy as np
import wave
import struct
from pathlib import Path

def create_sine_wave(frequency, duration, sample_rate=44100):
    """Create a sine wave with the given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave_data = np.sin(2 * np.pi * frequency * t)
    return wave_data

def save_wav(filename, wave_data, sample_rate=44100):
    """Save wave data to a WAV file."""
    # Normalize the data to [-1, 1]
    normalized_data = np.int16(wave_data * 32767)
    
    with wave.open(filename, 'w') as wav_file:
        # Set parameters
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        
        # Write data
        for value in normalized_data:
            packed_value = struct.pack('h', value)
            wav_file.writeframes(packed_value)

def main():
    """Create test WAV files."""
    # Create sample_audio directory if it doesn't exist
    sample_dir = Path("data/sample_audio")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test files with different frequencies
    frequencies = [100, 200, 300]  # Hz
    duration = 2.0  # seconds
    
    for i, freq in enumerate(frequencies, 1):
        filename = sample_dir / f"test_tone_{i}.wav"
        wave_data = create_sine_wave(freq, duration)
        save_wav(str(filename), wave_data)
        print(f"Created {filename} with {freq} Hz tone")

if __name__ == "__main__":
    main()
