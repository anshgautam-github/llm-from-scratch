# We can further control the distribution and selection process via a concept called temperature scaling, 
# where temperature scaling is just a fancy description for dividing the logits by a number greater than 0:
    
# Temperatures greater than 1 result in more uniformly distributed token probabilities, and Temperatures
# smaller than 1 will result in more confident (sharper or more peaky) distributions.

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


# Let's illustrate this by plotting the original probabilities alongside probabilities scaled with different 
# temperature values:

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

# Plotting
x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()

plt.tight_layout()
plt.savefig("temperature-plot.pdf")
plt.show()