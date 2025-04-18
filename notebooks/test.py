from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
# Save generation attention scores to file
import numpy as np
import os
import pickle

def to_cpu_recursive(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, tuple):
        return tuple(to_cpu_recursive(item) for item in data)
    elif isinstance(data, list):
        return [to_cpu_recursive(item) for item in data]
    else:
        return data


# Use an open-source small LLaMA variant (7B parameters)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~1.1B parameters
# MODEL_NAME="facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")


prompt = "what is the result of 1 + 1 ? "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=5,
        temperature=0.7,
        do_sample=True,
        output_attentions=True,
        # output_hidden_states=True,
        # output_scores=True,
        # output_logits=True,
        return_dict_in_generate=True,
    )


from bertviz import model_view

# Decode and print
generated_ids = outputs.sequences
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated response:")
print(response)

# Extract attention scores from generation
generation_attentions = outputs.attentions  # Tuple of attention tensors
attention_cpu = cpu_generation_attentions = to_cpu_recursive(generation_attentions)
print(inputs.input_ids[0])
input_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
print(f'generated_ids:{generated_ids}, inputs.input_ids[0]:{inputs.input_ids[0]}')
output_tokens= tokenizer.convert_ids_to_tokens(generated_ids[0])
output_index = 0

tokens =  output_tokens[:-1]
# tokens=input_tokens
print(f'tokens:{tokens}')

aa=model_view(attention_cpu, tokens, include_layers=[1,2], include_heads=[1,2])  # Display model view


# from bertviz import head_view
# head_view(attention_cpu, output_tokens[:-1], html_action='return')
