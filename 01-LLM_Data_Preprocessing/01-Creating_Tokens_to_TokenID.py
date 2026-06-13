
# Converting text into tokens is a crucial step in natural language processing (NLP). Tokens are the basic units of text, such as words, punctuation marks, and special characters. In this example, we will use regular expressions to split a given text into tokens while keeping punctuation as separate tokens.

#Sample Logic:
import re
text = "Hello, world. Is this-- a test?"
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:10])
print(len(preprocessed))


#Applying the same logic to a larger data₹set, we will read the contents of a text file and preprocess it to create tokens. The file "the-verdict.txt" is assumed to contain the text data we want to process.:
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read() # It returns the entire contents of the file as a single string.

print("Total number of character:", len(raw_text))
print(raw_text[:99])
print(type(raw_text))

