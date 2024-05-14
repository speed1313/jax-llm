from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import click


@click.command()
@click.option("--data_name", type=str, default="aozora")
@click.option("--vocab_size", type=int, default=50304)
def main(data_name: str, vocab_size: int):
    data_file_path = f"data/{data_name}/input.txt"
    import os

    save_dir = f"data/{data_name}"
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    files = [data_file_path]
    tokenizer.train(
        files, trainer
    )  # default vocab_size=30000 (ref: https://github.com/huggingface/tokenizers/blob/25aee8b88c8de3c5a52e2f9cb6281d6df00ad516/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py#L82)

    tokenizer.save(f"{save_dir}/tokenizer.json")

    # print token num
    with open(data_file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    print("Total tokens: ", len(tokenizer.encode(text_data)))


if __name__ == "__main__":
    main()
