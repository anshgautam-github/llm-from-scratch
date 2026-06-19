# The embedding vectors are learned during training. Initially: E[5]=[0.12,−0.8,0.4,...] is random.
# During training, backpropagation adjusts it so that tokens used in similar contexts get similar vectors.
# Eventually: "cat" and "dog" embeddings become close , "king" and "queen" show relationships etc.
# So: Meaning lives in the embedding vectors, NOT in token IDs.

# Very Important Clarification
# People often say: “Convert token IDs into embeddings” But technically it is: “Use token IDs to LOOK UP embeddings.”
# Not “convert mathematically.” No computation like: f(5)→embedding Instead: embedding=E[5] a table lookup.

# Behind the scenes, the token ID is used as an index into the embedding matrix.
# So if the token ID is 5, the model literally does something conceptually like: embedding_vector = embedding_matrix[5]
# and gets: [0.6, 0.5, 0.1, -0.4] . That vector is the embedding.
# So it’s not really “converting” the number 5 into meaning through some formula.

input_ids = torch.tensor([2, 3, 5, 1])

# Before creating the embedding layer, we must define: Vocabulary size & Embedding dimension (output dimension)
# because these determine the size and structure of the embedding matrix.
vocab_size = 6
output_dim = 3

# An embedding layer is basically a big table (matrix). E∈(R)^V×D where: V = vocabulary size, D = embedding dimension
# So if: vocab_size = 10000 , output_dim = 768 then embedding matrix becomes: 10000×768 , Meaning: 10,000 rows
# each row is a 768-dimensional vector

#Initially embeddings are random.Later training updates them into meaningful vectors.
torch.manual_seed(123) #fixes the random number generator. Without it, every run creates different random embeddings.

embedding_layer = torch.nn.Embedding(vocab_size, output_dim) #It creates a trainable embedding matrix with random val.


print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3]))) #"Give me the embedding vector for token ID 3" looks inside Embed matrix
print(embedding_layer(input_ids))


# Why Must Embedding Dimension Be Defined?
# Because the model must know: “How much information capacity should each token vector have?”
# Example: If: output_dim = 3 then: "cat"→[0.2,−0.5,0.8] only 3 numbers.

# Why Must Vocabulary Size Be Defined?
# Because: the model needs one row per token.
# Suppose tokenizer knows these tokens:
# Token	ID
# "I"	    0
# "love"	1
# "cats"	2
# "AI"	3
# Vocabulary size: V=4 , Then embedding matrix must have 4 rows:
# E= [ 
#     embedding for token 0
#     embedding for token 1
#     embedding for token 2
#     embedding for token 3
# ]

# Why Random Initialization?
# If all embeddings started as zeros: [0,0,0] then every token would look identical.
# The model could not differentiate tokens. Random initialization breaks symmetry.

# What manual_seed(123) Does?
# torch.manual_seed(123) makes the random initialization reproducible.
# Without it: Every run creates different random embeddings.
# With seed: Same random matrix every time.

# Very Important Insight
# The embedding matrix is NOT built from data immediately. It is: Randomly initialized Gradually learned 
# from training data

# The embedding matrix is NOT built from data immediately. It is: Randomly initialized Gradually learned from 
# training  data, how is this thing diff from when we can create embedding for a word directly thenhow does it
# work no random initlisqation happens ?

# Random Initialization :
# Create empty learnable vectors
# ↓
# Train on text
# ↓
# Embeddings learn meaning

# Pretrained Embeddings :
# Someone already trained embeddings
# ↓
# You load learned vectors directly
# ↓
# No need to start random


# print(embedding_layer(torch.tensor([3]))) 
# You are thinking: “Shouldn’t it search data or text to create embedding?” NO. That already happened during training.

# embedding_layer([2,3,5,1]) , and internally does something conceptually like:
# [
#     embedding_matrix[2],
#     embedding_matrix[3],
#     embedding_matrix[5],
#     embedding_matrix[1]
# ]