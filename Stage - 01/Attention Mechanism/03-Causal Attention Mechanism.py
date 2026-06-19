
# Reuse the query and key weight matrices of the SelfAttention_v2 object from the previous section for convenience
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
    

# We can now use PyTorch's tril function to create a mask where the values above the diagonal are zero:
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# Now, we can multiply this mask with the attention weights to zero out the values above the diagonal:
masked_simple = attn_weights*mask_simple
print(masked_simple)

# The third step is to renormalize the attention weights to sum up to 1 again in each row.
# We can achieve this by dividing each element in each row by the sum in each row:
row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
# The result is an attention weight matrix where the attention weights above the diagonal are zeroed out and where the rows sum to 1.

# We can implement this more efficient masking "trick" by creating a mask with 1's above the diagonal and then '
# 'replacing these 1's with negative infinity (-inf) values:
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# Now, all we need to do is apply the softmax function to these masked results, and we are done.
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)