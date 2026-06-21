GPT_CONFIG_124M = { # GPT 2 CONFIGURATION
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # For the flow, you can check the below forward function. Here. We will see what all things are created 
        # when we create instance of this transformer block.
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    # Here, we call normalisation layers -> pre-normalisation layer because they are used before ff and attention heads

    def forward(self, x):
        # To understand the sequence over here referred to a transformer block diagram
        # Shortcut connection for attention block
        shortcut = x

        # You can see in the block diagram, we have, firstly, normalisation then attention followed by dropout follw by shortcut
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        # Then in the block diagram, we have normalisation layer two then feed forward, neural network,drop, then shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x