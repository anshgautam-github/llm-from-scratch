# Next, let's use the GELU function to implement the small neural network module, FeedForward, that we will be using '
# 'in the LLM's transformer block later:

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768) #A
out = ffn(x)
print(out.shape)

# Although the input and output dimensions of this module are the same, it internally expands the embedding dimension 
# into a higher-dimensional space through the first linear layer.
# This expansion is followed by a non-linear GELU activation, and then a contraction back to the original dimension 
# with the second linear transformation.