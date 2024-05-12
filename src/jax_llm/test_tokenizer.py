from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer-aozora.json")

output = tokenizer.encode("深いおどろきにうたれて、<|endoftext|>")
print(output.tokens)
print(output.ids)
