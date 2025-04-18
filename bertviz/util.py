import torch
import numpy as np

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
    # all_layer_attn = all_layer_attn.reshape(shape[0], 1, shape[1], shape[2], shape[3])
    return torch.tensor(all_layer_attn)



def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def format_attention_all(attention, layers=None, heads=None):
    if layers:
        attention = [ [token_attention[layer_index] for layer_index in layers] for token_attention in attention ]
    attention = add_padding(attention)
    if heads:
        attention = attention[:,heads,:, :]
    return attention


def num_layers(attention):
    return len(attention[0])


def num_heads(attention):
    return attention[0][0][0].size(0)


def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]
