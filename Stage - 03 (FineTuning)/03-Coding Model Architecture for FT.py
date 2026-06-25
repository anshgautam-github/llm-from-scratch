# We start with initializing the pretrained model we worked with in the previous chapter
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

# Next, we import the download_and_load_gpt function from the gpt_download3.py file we downloaded earlier.
# Furthermore, we also reuse the GPTModel class and load_weights_into_gpt function from chapter 5 to load the downloaded weights into the GPT model:
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
from gpt_download3 import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

# To ensure that the model was loaded correctly, let's double-check that it generates coherent text
text_1 = "Every effort moves you"

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
# Every effort moves you forward.
# The first step is to understand the importance of your work

# Now, before we start finetuning the model as a spam classifier, let's see if the model can perhaps already classify spam messages by by prompting it with instructions:
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))
# Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'
# The following text 'spam'? Answer with 'yes' or 'no': 'You are a winner

# Based on the output, it's apparent that the model struggles with following instructions.
# This is anticipated, as it has undergone only pretraining and lacks instruction-finetuning, which we will explore 


# ADDING A CLASSIFICATION HEAD

# To get the model ready for classification-finetuning, we first freeze the model, meaning that we make all layers non-trainable:
for param in model.parameters():
    param.requires_grad = False

# Then, we replace the output layer (model.out_head), which originally maps the layer inputs to 50,257 dimensions (the size of the vocabulary):
torch.manual_seed(123)

num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# Additionally, we configure the last transformer block and the final LayerNorm module, which connects this block to the output layer, to be trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

# Even though we added a new output layer and marked certain layers as trainable or nontrainable, we can still use this model in a similar way to previous chapters.
# For instance, we can feed it an example text identical to how we have done it in earlier chapters. For example, consider the following example text:
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)
# Inputs: tensor([[5211,  345,  423,  640]])
# Inputs dimensions: torch.Size([1, 4])

# Then, we can pass the encoded token IDs to the model as usual:
with torch.no_grad():
    outputs = model(inputs)

print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)
# Outputs:
#  tensor([[[-1.5854,  0.9904],
#          [-3.7235,  7.4548],
#          [-2.2661,  6.6049],
#          [-3.5983,  3.9902]]])
# Outputs dimensions: torch.Size([1, 4, 2])

#To extract the last output token, from the output tensor, we use the following code:
print("Last output token:", outputs[:, -1, :])
# Last output token: tensor([[-3.5983,  3.9902]]

# In this code, we have random initalized values because we have not trained it yet only classifier dataset.