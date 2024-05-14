import tiktoken
import jax
from model import NanoLM
from tokenizers import Tokenizer
import click
from utils import (
    AbstractTokenizer,
    text_to_token_ids,
    token_ids_to_text,
    top_k_generate,
)
import pickle
import json


@click.command()
@click.option("--data_name", type=str, default="aozora")
@click.option("--prompt", type=str, default="私は")
@click.option("--max_new_tokens", type=int, default=60)
@click.option("--temperature", type=float, default=1.0)
@click.option("--top_k", type=int, default=25)
def main(
    data_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    tokenizer_path = f"data/{data_name}/tokenizer.json"
    model_path = f"model/{data_name}"
    # load config json
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    # load params
    with open(f"{model_path}/params.pkl", "rb") as f:
        params = pickle.load(f)

    model = NanoLM(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        head_size=config["head_size"],
        dropout_rate=config["dropout_rate"],
        embed_size=config["embed_size"],
        block_size=config["block_size"],
    )
    key = jax.random.PRNGKey(0)

    if tokenizer_path == "gpt2":
        tokenizer = AbstractTokenizer(tiktoken.get_encoding(tokenizer_path), "gpt-2")
    else:
        tokenizer = AbstractTokenizer(
            Tokenizer.from_file(tokenizer_path), tokenizer_path
        )

    batch = text_to_token_ids(prompt, tokenizer)

    key, subkey = jax.random.split(key)
    token_ids = top_k_generate(
        model,
        subkey,
        params,
        max_new_tokens,
        batch,
        top_k=top_k,
        temperature=temperature,
    )

    print("Output:", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
