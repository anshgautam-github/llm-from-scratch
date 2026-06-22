# PyTorch already implements a cross_entropy function that carries out the previous steps

# Before we apply the cross_entropy function, let's check the shape of the logits and targets

# Logits have shape (batch_size, num_tokens, vocab_size)
print("Logits shape:", logits.shape)
# Targets have shape (batch_size, num_tokens)
print("Targets shape:", targets.shape)
# Logits shape: torch.Size([2, 3, 50257])
# Targets shape: torch.Size([2, 3])

# For the cross_entropy function in PyTorch, we want to flatten these tensors by combining them over the batch dimension:
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
# Flattened logits: torch.Size([6, 50257])
# Flattened targets: torch.Size([6])

# The cross_entropy function in PyTorch will automatically take care of applying the softmax and log-probability
# computation internally over those token indices in the logits that are to be maximized
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# tensor(10.7940)