# As discussed in the previous chapter, inside the generate_text_simple, we convert the logits into probabilities via 
# the softmax function and obtain the token ID corresponding the generated token via the argmax function, which we can
# then map back into text via the inverse vocabulary:
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id]) # forward

# To implement a probabilistic sampling, we can now replace the argmax with the multinomial function in PyTorch:
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id]) # forward

# The printed output is "forward" just like before. What happened? The multinomial function samples the next token
# proportional to its probability score.
# In other words, "forward" is still the most likely token and will be selected by multinomial most of the time
# but not all the time.

# To illustrate this, let's implement a function that repeats this sampling 1000 times:
def print_sampled_tokens(probas):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)
# 73 x closer
# 0 x every
# 0 x effort
# 582 x forward
# 2 x inches
# 0 x moves
# 0 x pizza
# 343 x toward

# This means that if we replaced the argmax function with the multinomial function inside the 
# generate_and_print_sample function, the LLM would sometimes generate texts such as "every effort moves you toward", 
# "every effort moves you inches", and "every effort moves you closer" instead of "every effort moves you forward".

# We can further control the distribution and selection process via a concept called temperature scaling, 
# where temperature scaling is just a fancy description for dividing the logits by a number greater than 0: