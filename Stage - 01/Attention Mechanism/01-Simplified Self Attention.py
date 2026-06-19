import torch

# Sppose we have embedding vector for each of them

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 2nd input token is the query
query = inputs[1]  

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)
print(attn_scores_2)


attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Below is a basic implementation of the softmax function for normalizing the attention scores:
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())


# Note that this naive softmax implementation (softmax_naive) may encounter numerical instability problems, 
# such as overflow and underflow, when dealing with large or small input values.
# Therefore, in practice, it's advisable to use the PyTorch implementation of softmax, which has been extensively
# optimized for performance
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
# In this case, we can see that it yields the same results as our previous softmax_naive function:


# The context vector z(2)is calculated as a weighted sum of all input vectors.
# This involves multiplying each input vector by its corresponding attention weight:
query = inputs[1] # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)


# Now, we can extend this computation to calculate attention weights and context vectors for all inputs

# First, we add an additional for-loop to compute the dot products for all pairs of inputs.
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# However, for-loops are generally slow, and we can achieve the same results using matrix multiplication:
attn_scores = inputs @ inputs.T
print(attn_scores)

# We now normalize each row so that the values in each row sum to 1:
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# Let's briefly verify that the rows indeed all sum to 1:
print("All row sums:", attn_weights.sum(dim=-1))

# In the third and last step, we now use these attention weights to compute all context vectors via matrix multiplication:
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# We can double-check that the code is correct by comparing the 2nd row with the context vector z(2) calculated previously
print("Previous 2nd context vector:", context_vec_2)