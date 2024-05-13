import tiktoken
import jax
from model import NanoLM
from tokenizers import Tokenizer
import click
from utils import AbstractTokenizer, text_to_token_ids, token_ids_to_text, generate
import pickle
import json


@click.command()
@click.option("--tokenizer_path", type=str, default="data/tokenizer.json")
@click.option("--params_path", type=str, default="model/params.pkl")
@click.option("--prompt", type=str, default="私は")
@click.option("--max_new_tokens", type=int, default=30)
@click.option("--temperature", type=float, default=None)
@click.option("--top_k", type=int, default=25)
def main(
    tokenizer_path: str,
    params_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    # load config json
    with open("model/config.json", "r") as f:
        config = json.load(f)
    # load params
    with open(params_path, "rb") as f:
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
    print("Params loaded")
    key = jax.random.PRNGKey(0)

    if tokenizer_path == "gpt2":
        tokenizer = AbstractTokenizer(tiktoken.get_encoding(tokenizer_path), "gpt-2")
    else:
        tokenizer = AbstractTokenizer(
            Tokenizer.from_file(tokenizer_path), tokenizer_path
        )
    print("Tokenizer loaded")

    batch = text_to_token_ids(prompt, tokenizer)

    key, subkey = jax.random.split(key)
    print(temperature)
    if temperature == None:
        print("Using fast generation")
        from utils import fast_generate
        token_ids = fast_generate(model, subkey, params, max_new_tokens, batch)
    else:
        token_ids = generate(
            model=model,
            params=params,
            key=subkey,
            idx=batch,
            max_new_tokens=max_new_tokens,
            context_size=config["block_size"],
            temperature=temperature,
            top_k=top_k,
        )

    print("Output:", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
