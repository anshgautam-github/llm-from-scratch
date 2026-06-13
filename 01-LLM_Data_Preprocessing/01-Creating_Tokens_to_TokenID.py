
# Converting text into tokens is a crucial step in natural language processing (NLP). Tokens are the basic units of text, such as words, punctuation marks, and special characters. In this example, we will use regular expressions to split a given text into tokens while keeping punctuation as separate tokens.

#Sample Logic:
import re
text = "Hello, world. Is this-- a test?"
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:10])
print(len(preprocessed))

print("\n Now we will apply the same logic to a larger dataset. We will read the contents of a text file and preprocess it to create tokens. The file the-verdict.txt is assumed to contain the text data we want to process.\n")

#Applying the same logic to a larger data₹set, we will read the contents of a text file and preprocess it to create tokens. The file "the-verdict.txt" is assumed to contain the text data we want to process.:
with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read() # It returns the entire contents of the file as a single string.

print("Total number of character:", len(raw_text))
print(raw_text[:99])
print(type(raw_text))

print("\n\nPreprocessing the text to create tokens...")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

# Now that we have preprocessed the text and created tokens, we can create a vocabulary of unique tokens and assign each token a unique integer ID. This is an essential step in preparing the data for machine learning models, as it allows us to represent text data in a numerical format.

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)} 

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


