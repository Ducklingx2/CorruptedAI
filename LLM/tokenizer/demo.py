with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tok = Tokenizer()

tok.train(text)

encoded = tok.encode("Hello AI!")

print(encoded)

print(tok.decode(encoded))
