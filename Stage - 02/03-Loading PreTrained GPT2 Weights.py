pip install tensorflow>=2.15.0 tqdm>=4.66
import tensorflow as tf
import tqdm
print("TensorFlow version:", tf.__version__)
print("tqdm version:", tqdm.__version__)

from gpt_download3 import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# After the execution of the previous code has been completed, let's inspect the contents of settings and params:
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
# Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
# Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])

# Both settings and params are Python dictionaries. The settings dictionary stores the LLM architecture settings 
# similarly to our manually defined GPT_CONFIG_124M settings.
# The params dictionary contains the actual weight tensors.

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

# Careful readers may remember that we used a 256-token length earlier, but the original GPT-2 models from OpenAI were 
# trained with a 1,024-token length, so we have to update the NEW_CONFIG accordingly:

# However, since we are working with pretrained weights, we need to match the settings for consistency and enable these bias vectors:
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval();


# By default, the GPTModel instance is initialized with random weights for pretraining.

# The last step to using OpenAI's model weights is to override these random weights with the weights we loaded into 
# the params dictionary.

# For this, we will first define a small assign utility function that checks whether two tensors or arrays 
# (left and right) have the same dimensions or shape and returns the right tensor as trainable PyTorch parameters:
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))