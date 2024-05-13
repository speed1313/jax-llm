from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import click


@click.command()
@click.option("--data_path", type=str, default="aozora.txt")
def main(data_path: str):
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=30000, special_tokens=["<|endoftext|>"])
    files = [data_path]
    tokenizer.train(
        files, trainer
    )  # default vocab_size=30000 (ref: https://github.com/huggingface/tokenizers/blob/25aee8b88c8de3c5a52e2f9cb6281d6df00ad516/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py#L82)
    tokenizer.save("data/tokenizer.json")


if __name__ == "__main__":
    main()
