import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


def add_padding(attn):
    output_len = len(attn) - 1 # the first element is for the self attention
    input_len = attn[0][0].shape[3]
    total_len = attn[-1][0].shape[3]
    assert total_len == input_len + output_len
    num_layers = len(attn[0])
    num_heads = len(attn[0][0][0])
    all_layer_attn = []
    for layer_idx in range(0, num_layers):
        layer_attn_list = []
        for head_idx in range(0, num_heads):
            head_attn_list = []
            for token_idx in range(0, output_len + 1):
                head_attn = attn[token_idx][layer_idx][0][head_idx]
                head_attn = pad_2d_tensor(head_attn, total_len)
                head_attn_list.append(head_attn)
            head_attn = np.concatenate(head_attn_list, axis=0)
            # add batch 1 to the first dimension
            layer_attn_list.append(head_attn)
        #layer_attn = np.concatenate(layer_attn_list, axis=0)
        all_layer_attn.append(layer_attn_list)
    all_layer_attn = torch.tensor(all_layer_attn) 
    shape = all_layer_attn.shape
    # add batch 1 to third dimentiosn
    all_layer_attn = all_layer_attn.reshape(shape[0], 1, shape[1], shape[2], shape[3])
    return torch.tensor(all_layer_attn)

def pad_2d_tensor(tensor, max_len):
    """
    Pad a 2D tensor to max_len in both dimensions.
    
    Args:
        tensor: numpy array of shape (n, m)
        max_len: desired length for both dimensions
        
    Returns:
        padded tensor of shape (max_len, max_len)
    """
    if len(tensor.shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")
    dim0 = tensor.shape[0]
    padded = np.zeros((dim0, max_len))

    current_h, current_w = tensor.shape
    
    # Copy the original tensor into the top-left corner of the padded tensor
    padded[:current_h, :current_w] = tensor
    
    return padded

# Example usage:
# tensor = np.array([[1, 2], [3, 4]])  # 2x2 tensor
# padded = pad_2d_tensor(tensor, max_len=4)  # Pad to 4x4
# Result will be:
# [[1, 2, 0, 0],
#  [3, 4, 0, 0],
#  [0, 0, 0, 0],
#  [0, 0, 0, 0]]


file_name = "generation_attention.pkl"

with open(file_name, "rb") as f:
    attentions = pickle.load(f)

# Print information about attention scores
num_input_tokens = attentions[0][0].shape[3]
num_output_tokens = len(attentions) - 1
num_layers = len(attentions[0])
num_heads = attentions[0][0][0].shape[1]  # Assuming shape (1, num_heads, ...)
print(f"Number of input tokens: {num_input_tokens}")
print(f"Number of output tokens: {num_output_tokens}")
print(f"Number of layers: {num_layers}")
print(f"Number of heads: {num_heads}")

# Create a directory for the plots
max_token_len = num_input_tokens + num_output_tokens 
os.makedirs("concatenated_attention", exist_ok=True)

# Set fixed values for plotting
num_layers = 3
num_heads = 4

# Create a single large figure with subplots for all layers and heads
fig, axes = plt.subplots(num_layers, num_heads, figsize=(20, 15))
fig.suptitle("Attention Visualization - Rows: Layers, Columns: Heads", fontsize=16)

# Set color scale with max of 0.1 as requested
vmin = 0
vmax = 0.1  # Fixed maximum for better visualization of lower values

# Process all attention matrices
attentions = add_padding(attentions)
all_matrices = attentions[:num_layers,:,:num_heads]
# for layer_idx in range(num_layers):
#     for head_idx in range(num_heads):
#         attention_matrices = []
        
#         # Process each matrix
#         for token_idx in range(1, num_output_tokens):
#             attn = attentions[token_idx][layer_idx][0][head_idx]
#             attn = attn.cpu().numpy()
            
#             # Check if attn is 1D and reshape if needed
#             if len(attn.shape) == 1:
#                 # For 1D vectors, we need to ensure they're the right shape
#                 padded = np.zeros(max_token_len)
#                 # Copy values up to the length of attn
#                 padded[:attn.shape[0]] = attn
#                 attn = padded
#             else:
#                 # For 2D matrices
#                 if attn.shape[1] < max_token_len:
#                     padded = np.zeros((attn.shape[0], max_token_len))
#                     padded[:, :attn.shape[1]] = attn
#                     attn = padded
                
#             attention_matrices.append(attn)
        
#         # Concatenate all matrices
#         if attention_matrices:
#             # Check if matrices are 1D or 2D
#             if len(attention_matrices[0].shape) == 1:
#                 # For 1D vectors, stack them vertically
#                 concatenated_attention = np.vstack(attention_matrices)
#             else:
#                 # For 2D matrices
#                 concatenated_attention = np.concatenate(attention_matrices, axis=0)
            
#             all_matrices.append(concatenated_attention)

# Now plot all matrices with consistent color scale
matrix_idx = 0
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        ax = axes[layer_idx, head_idx]
        # Plot the attention matrix
        im = ax.imshow(all_matrices[layer_idx][0][head_idx], vmin=vmin, vmax=vmax, cmap='viridis')
        
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
plt.close()

