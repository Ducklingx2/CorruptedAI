class Tokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}

    def train(self, text):
        tokens = text.split()
        vocab = sorted(set(tokens))

        self.word_to_id = {
            word: i for i, word in enumerate(vocab)
        }

        self.id_to_word = {
            i: word for word, i in self.word_to_id.items()
        }

    def encode(self, text):
        return [
            self.word_to_id[word]
            for word in text.split()
        ]

    def decode(self, ids):
        return " ".join(
            self.id_to_word[i]
            for i in ids
        )

with open("data.txt") as f:
    text = f.read()

tok = Tokenizer()

tok.train(text)

encoded = tok.encode("Hello AI!")

print(encoded)

print(tok.decode(encoded))

class Tokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}

    def train(self, text):
        tokens = text.split()
        vocab = sorted(set(tokens))

        self.word_to_id = {
            word: i for i, word in enumerate(vocab)
        }

        self.id_to_word = {
            i: word for word, i in self.word_to_id.items()
        }

    def encode(self, text):
        return [
            self.word_to_id[word]
            for word in text.split()
        ]

    def decode(self, ids):
        return " ".join(
            self.id_to_word[i]
            for i in ids
        )

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tok = Tokenizer()

tok.train(text)

encoded = tok.encode("Hello AI!")

print(encoded)

print(tok.decode(encoded))
