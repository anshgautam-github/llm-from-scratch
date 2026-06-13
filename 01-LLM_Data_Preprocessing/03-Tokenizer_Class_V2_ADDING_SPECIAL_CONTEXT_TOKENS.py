import re

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

tokens = [ token.strip() for token in tokens if token.strip() ]

all_tokens = sorted(set(tokens))

all_tokens.extend([
    "<|endoftext|>",
    "<|unk|>"
])

vocab = { token: idx for idx, token in enumerate(all_tokens) }

# print("Last 5 vocab items:")
# for item in list(vocab.items())[-5:]:
#     print(item)


class SimpleTokenizerV2:

    def __init__(self, vocab):
        self.str_to_int = vocab

        self.int_to_str = {
            i: s
            for s, i in vocab.items()
        }

    def _tokenize(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        return tokens

    def encode(self, text):
        tokens = self._tokenize(text)

        tokens = [
            token
            if token in self.str_to_int
            else "<|unk|>"
            for token in tokens
        ]

        ids = [
            self.str_to_int[token]
            for token in tokens
        ]

        return ids

    def decode(self, ids):
        text = " ".join(
            self.int_to_str[i]
            for i in ids
        )

        text = re.sub(
            r'\s+([,.:;?!"()\'])',
            r'\1',
            text
        )

        return text


# Create tokenizer
tokenizer = SimpleTokenizerV2(vocab)

# Test text
text = "Hello, do you like tea?"

# Encode
ids = tokenizer.encode(text)
print("\nEncoded IDs:")
print(ids)

# Decode
decoded_text = tokenizer.decode(ids)
print("\nDecoded Text:")
print(decoded_text)