import json
import os
import uuid
import pickle
from IPython.display import display, HTML, Javascript
import matplotlib.pyplot as plt

from .util import format_attention, format_attention_all, num_layers, num_heads


def head_map_view(
        attention=None,
        include_layers=None,
        include_heads=None,
        html_action='view'
):
    n_heads=num_heads(attention)
    n_layers=num_layers(attention)
    num_input_tokens = attention[0][0].shape[3]
    num_output_tokens = len(attention) - 1
    print(f"Number of input tokens: {num_input_tokens}")
    print(f"Number of output tokens: {num_output_tokens}")
    print(f"Number of layers: {n_layers}")
    print(f"Number of heads: {n_heads}")

    # Create a directory for the plots
    max_token_len = num_input_tokens + num_output_tokens 
    os.makedirs("concatenated_attention", exist_ok=True)
    # Set fixed values for plotting
    # Create a single large figure with subplots for all layers and heads
    if include_layers is None:
        include_layers = list(range(n_layers))
    if include_heads is None:
        include_heads = list(range(n_heads))
    attention = format_attention_all(attention, layers =include_layers, heads=include_heads)

    # Calculate figure size dynamically based on number of layers and heads
    width_per_head = 5
    height_per_layer = 4
    figsize = (len(include_heads) * width_per_head, len(include_layers) * height_per_layer)
    
    fig, axes = plt.subplots(len(include_layers), len(include_heads), figsize=figsize)
    fig.suptitle("Attention Visualization - Rows: Layers, Columns: Heads", fontsize=16)

    # Set color scale with max of 0.1 as requested
    vmin = 0
    vmax = 0.1  # Fixed maximum for better visualization of lower values

# Process all attention matrices
    all_matrices = attention
    for layer_idx, layer_index in enumerate(include_layers):
        for head_idx, head_index in enumerate(include_heads):
            # Handle case where there's only one subplot
            if len(include_layers) == 1 and len(include_heads) == 1:
                ax = axes
            elif len(include_layers) == 1:
                ax = axes[head_idx]
            elif len(include_heads) == 1:
                ax = axes[layer_idx]
            else:
                ax = axes[layer_idx, head_idx]
                
            # Plot the attention matrix
            im = ax.imshow(all_matrices[layer_idx][head_idx], vmin=vmin, vmax=vmax, cmap='viridis')
                # Add labels
            ax.set_title(f"Head {head_idx}")
            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx}")
                
    # Add a colorbar to the figure
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Attention Score')

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle
    plt.savefig("concatenated_attention/all_attention_layers_heads.png", dpi=300)
    print("All attention visualizations saved to single figure with vmax=0.3")
    plt.show()
