import tiktoken
import jax
from model import GPTModel
from train import GPTConfig
from tokenizers import Tokenizer
import click
from utils import AbstractTokenizer, text_to_token_ids, token_ids_to_text, generate
import pickle


@click.command()
@click.option("--tokenizer_path", type=str, default="data/tokenizer-aozora.json")
@click.option("--variables_path", type=str, default="model/aozora_variables.pkl")
@click.option("--prompt", type=str, default="私は")
@click.option("--max_new_tokens", type=int, default=30)
@click.option("--temperature", type=float, default=1.4)
@click.option("--top_k", type=int, default=25)
def main(
    tokenizer_path: str,
    variables_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    model = GPTModel(cfg=GPTConfig())
    with open(variables_path, "rb") as f:
        variables = pickle.load(f)

    key = jax.random.PRNGKey(0)

    if tokenizer_path == "gpt2":
        tokenizer = AbstractTokenizer(tiktoken.get_encoding(tokenizer_path), "gpt-2")
    else:
        tokenizer = AbstractTokenizer(
            Tokenizer.from_file(tokenizer_path), tokenizer_path
        )

    batch = text_to_token_ids(prompt, tokenizer)

    key, subkey = jax.random.split(key)
    token_ids = generate(
        model=model,
        variables=variables,
        key=subkey,
        idx=batch,
        max_new_tokens=max_new_tokens,
        context_size=GPTConfig.ctx_len,
        temperature=temperature,
        top_k=top_k,
    )

    print("Output:", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
