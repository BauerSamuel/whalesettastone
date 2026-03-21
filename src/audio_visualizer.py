"""
Module for visualizing audio features and playing audio files.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import ipywidgets as widgets
from IPython.display import Audio, display, clear_output
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import pandas as pd

class AudioVisualizer:
    """Class for visualizing audio features and playing audio files."""
    
    def __init__(self, whale_codas_dir: Path):
        """Initialize the visualizer with the directory containing audio files."""
        self.whale_codas_dir = whale_codas_dir
        self.current_file = None
        self.y = None
        self.sr = None
        
    def create_visualization_interface(self, features: Dict[str, Dict]) -> widgets.VBox:
        """Create an interactive interface for visualizing audio features."""
        # File selector dropdown
        file_dropdown = widgets.Dropdown(
            options=list(features.keys()),
            description='Audio File:',
            style={'description_width': 'initial'}
        )
        
        # Feature selector dropdown
        feature_dropdown = widgets.Dropdown(
            options=['waveform', 'spectrogram', 'mel_spectrogram', 'feature_plot'],
            description='Visualization:',
            style={'description_width': 'initial'}
        )
        
        # Play button
        play_button = widgets.Button(
            description='Play Audio',
            icon='play'
        )
        
        # Output widget for visualizations
        output = widgets.Output()
        
        # Feature values display
        feature_text = widgets.HTML(
            value='<h3>Audio Features:</h3>',
            layout={'border': '1px solid black', 'padding': '10px'}
        )
        
        def update_display(change=None):
            """Update the visualization and feature display."""
            with output:
                clear_output(wait=True)
                if not features or file_dropdown.value is None:
                    feature_text.value = "<h3>Audio Features</h3><p>No audio files loaded. Add files via Download or Upload.</p>"
                    return
                # Load audio file
                file_path = self.whale_codas_dir / file_dropdown.value
                self.current_file = file_path
                self.y, self.sr = librosa.load(str(file_path))
                
                # Create figure
                plt.figure(figsize=(12, 6))
                
                if feature_dropdown.value == 'waveform':
                    librosa.display.waveshow(self.y, sr=self.sr)
                    plt.title('Waveform')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                
                elif feature_dropdown.value == 'spectrogram':
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
                    librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Spectrogram')
                
                elif feature_dropdown.value == 'mel_spectrogram':
                    mel_spect = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
                    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
                    librosa.display.specshow(mel_spect_db, sr=self.sr, x_axis='time', y_axis='mel')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Mel Spectrogram')
                
                elif feature_dropdown.value == 'feature_plot':
                    # Create a bar plot of scalar features
                    scalar_features = {k: v for k, v in features[file_dropdown.value].items() 
                                    if isinstance(v, (int, float)) and k != 'duration'}
                    plt.bar(scalar_features.keys(), scalar_features.values())
                    plt.xticks(rotation=45)
                    plt.title('Audio Features')
                    plt.tight_layout()
                
                plt.show()
                
                # Update feature text
                feature_html = '<h3>Audio Features:</h3>'
                feature_html += '<table style="width:100%">'
                for feature, value in features[file_dropdown.value].items():
                    if feature != 'mfccs':  # Skip MFCCs as they're too long
                        disp = f"{value:.4f}" if isinstance(value, float) else value
                        feature_html += f'<tr><td><b>{feature}</b></td><td>{disp}</td></tr>'
                feature_html += '</table>'
                feature_text.value = feature_html
        
        def play_audio(b):
            """Play the currently selected audio file."""
            if self.current_file:
                display(Audio(str(self.current_file)))
        
        # Connect callbacks
        file_dropdown.observe(update_display, names='value')
        feature_dropdown.observe(update_display, names='value')
        play_button.on_click(play_audio)
        
        # Create layout
        controls = widgets.HBox([file_dropdown, feature_dropdown, play_button])
        
        # Initial display
        update_display()
        
        return widgets.VBox([
            controls,
            output,
            feature_text
        ])

class FeatureVisualizer:
    """
    Creates visualizations of whale audio features.
    """
    
    def __init__(self, output_dir: Path = Path("data/plots")):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_features(self, 
                     features: np.ndarray,
                     labels: List[str],
                     title: str = "Feature Visualization",
                     filename: str = "feature_plot.png") -> None:
        """
        Create a scatter plot of reduced-dimension features with improved readability.
        
        Args:
            features: Array of reduced-dimension features
            labels: List of labels for each point
            title: Plot title
            filename: Output filename
        """
        try:
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': features[:, 0],
                'y': features[:, 1],
                'label': labels
            })
            
            # Create figure with larger size
            plt.figure(figsize=(16, 10))
            
            # Create scatter plot with transparency
            scatter = sns.scatterplot(
                data=df,
                x='x',
                y='y',
                hue='label',
                s=100,
                alpha=0.7,  # Add transparency
                palette='husl'  # Use a more distinct color palette
            )
            
            plt.title(title, pad=20, fontsize=14)
            plt.xlabel('Dimension 1', fontsize=12)
            plt.ylabel('Dimension 2', fontsize=12)
            
            # Adjust legend
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0,
                fontsize=10,
                ncol=2  # Display legend in 2 columns
            )
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3)
            
            # Adjust layout to prevent label overlap
            plt.tight_layout()
            
            # Save plot with high DPI
            output_path = self.output_dir / filename
            plt.savefig(
                output_path,
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.5
            )
            logging.info(f"Visualization saved to {output_path}")
            
            # Show plot
            plt.show()
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating visualization: {str(e)}")
            
    def create_interactive_plot(self,
                              features: np.ndarray,
                              labels: List[str],
                              title: str = "Interactive Feature Visualization") -> None:
        """
        Create an interactive plot using Plotly for better exploration of many data points.
        
        Args:
            features: Array of reduced-dimension features
            labels: List of labels for each point
            title: Plot title
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create DataFrame
            df = pd.DataFrame({
                'x': features[:, 0],
                'y': features[:, 1],
                'label': labels
            })
            
            # Create interactive scatter plot
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='label',
                title=title,
                hover_name='label',
                opacity=0.7,
                width=1000,
                height=800
            )
            
            # Update layout
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            
            # Save plot
            output_path = self.output_dir / "interactive_plot.html"
            fig.write_html(str(output_path))
            logging.info(f"Interactive visualization saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error creating interactive visualization: {str(e)}")
            
    def plot_feature_comparison(self,
                              features_a: np.ndarray,
                              features_b: np.ndarray,
                              labels: List[str],
                              title_a: str = "Method A",
                              title_b: str = "Method B",
                              filename: str = "comparison_plot.png") -> None:
        """
        Create a side-by-side comparison of two feature reduction methods.
        
        Args:
            features_a: First set of reduced features
            features_b: Second set of reduced features
            labels: List of labels for each point
            title_a: Title for first plot
            title_b: Title for second plot
            filename: Output filename
        """
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Create DataFrames
            df_a = pd.DataFrame({
                'x': features_a[:, 0],
                'y': features_a[:, 1],
                'label': labels
            })
            
            df_b = pd.DataFrame({
                'x': features_b[:, 0],
                'y': features_b[:, 1],
                'label': labels
            })
            
            # Plot first method
            sns.scatterplot(
                data=df_a,
                x='x',
                y='y',
                hue='label',
                ax=ax1,
                s=100,
                alpha=0.7,
                palette='husl'
            )
            ax1.set_title(title_a, pad=20, fontsize=14)
            ax1.set_xlabel('Dimension 1', fontsize=12)
            ax1.set_ylabel('Dimension 2', fontsize=12)
            ax1.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0,
                fontsize=10
            )
            ax1.grid(True, alpha=0.3)
            
            # Plot second method
            sns.scatterplot(
                data=df_b,
                x='x',
                y='y',
                hue='label',
                ax=ax2,
                s=100,
                alpha=0.7,
                palette='husl'
            )
            ax2.set_title(title_b, pad=20, fontsize=14)
            ax2.set_xlabel('Dimension 1', fontsize=12)
            ax2.set_ylabel('Dimension 2', fontsize=12)
            ax2.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0,
                fontsize=10
            )
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / filename
            plt.savefig(
                output_path,
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.5
            )
            logging.info(f"Comparison visualization saved to {output_path}")
            
            # Show plot
            plt.show()
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating comparison visualization: {str(e)}") 