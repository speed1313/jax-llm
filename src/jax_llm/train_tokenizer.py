from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import click
import jax.numpy as jnp


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
    tokenizer.train(files, trainer)

    tokenizer.save(f"{save_dir}/tokenizer.json")

    # print token num
    with open(data_file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    tokenized_text = jnp.array(tokenizer.encode(text_data).ids)
    total_tokens = len(tokenized_text)
    print("Total tokens: ", total_tokens)
    with open(f"{save_dir}/config.json", "w") as f:
        f.write(
            f'{{"vocab_size": {vocab_size}, "total_tokens": {total_tokens}, "data_file_path": "{data_file_path}"}}'
        )

    # store tokenized text as bin
    with open(f"{save_dir}/tokenized_text.bin", "wb") as f:
        f.write(tokenized_text.tobytes())


if __name__ == "__main__":
    main()
