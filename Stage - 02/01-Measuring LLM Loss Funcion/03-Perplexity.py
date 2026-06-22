# The perplexity is simply the exponential of the cross-entropy loss.

perplexity = torch.exp(loss)
print(perplexity)
# tensor(48725.8203)


