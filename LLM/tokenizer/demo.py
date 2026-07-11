from pathlib import Path
from tokenizer import Tokenizer

data_path = Path(__file__).resolve().parents[2] / "data" / "words.txt"

with data_path.open("r", encoding="utf-8") as f:
    text = f.read()

tok = Tokenizer()
tok.train(text)

encoded = tok.encode("hello")

print(encoded)
print(tok.decode(encoded))
