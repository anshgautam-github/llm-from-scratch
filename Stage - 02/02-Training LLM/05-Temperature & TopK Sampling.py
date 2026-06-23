# Step 1: For-loop is the same as before: Get logits, and only focus on last time step

# Step 2: In this new section, we filter logits with top_k sampling

# Step 3: This is the new section where we apply temperature scaling

# Step 4: Carry out greedy next-token selection as before when temperature scaling is disabled

# Step 5: Stop generating early if end-of-sequence token is encountered and eos_id is specified


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


# Let's now see this new generate function in action:
torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Output text:Every effort moves you stand to work on surprise, a one of us had gone with random-

# As we can see, the generated text is very different from the one we previously generated via the generate_simple 
# function at the beginning of section 5.3 ("Every effort moves you know," was one of the axioms he laid...!"), 
# which was a memorized passage from the training set.


# now model has become a generative model because it will not give the exact line that in the dataset you can see it. 
# There is not such any specific sentence. It's generated on its own.