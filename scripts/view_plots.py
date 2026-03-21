"""
Display dimensionality reduction plots side by side (requires data/plots/*.png).

Run from repository root:
    python scripts/view_plots.py
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_plots():
    """Display t-SNE and UMAP plots side by side."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Load and display t-SNE plot
    tsne_img = mpimg.imread('data/plots/tsne_plot.png')
    ax1.imshow(tsne_img)
    ax1.set_title('t-SNE Visualization')
    ax1.axis('off')
    
    # Load and display UMAP plot
    umap_img = mpimg.imread('data/plots/umap_plot.png')
    ax2.imshow(umap_img)
    ax2.set_title('UMAP Visualization')
    ax2.axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_plots()
