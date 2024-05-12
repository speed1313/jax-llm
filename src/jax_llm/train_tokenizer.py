from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["<|endoftext|>"])

# files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
files = ["aozora.txt"]
tokenizer.train(files, trainer)

tokenizer.save("data/tokenizer-aozora.json")
