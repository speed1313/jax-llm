from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/aozora_10246/tokenizer.json")
output = tokenizer.encode("国境の長いトンネルを抜けると雪国であった")
print(output.tokens)
print(output.ids)
# vocab_size = len(tokenizer.get_vocab())
print(tokenizer.get_vocab_size())
output = tokenizer.encode("Hello, y'all! How are you <|endoftext|>?")
print(output.tokens)
print(output.ids)
