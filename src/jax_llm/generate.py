import tiktoken
import jax
import jax.numpy as jnp
from model import GPTModel
from dataclasses import dataclass
import click


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    ctx_len: int = 256
    emb_dim: int = 768
    n_heads: int = 8
    n_layers: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False


@dataclass
class OptimizerConfig:
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    batch_size: int = 4
    epochs: int = 10


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = jnp.array(encoded).reshape(1, -1)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.flatten()
    return tokenizer.decode(flat.tolist())


def print_sampled_tokens(probas):
    key = jax.random.PRNGKey(0)
    sample = jax.random.categorical(
        key, probas, shape=(1000,)
    )  # NOTE: probs should be logits.
    sampled_ids = jnp.bincount(jnp.array(sample), minlength=0)
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


def generate(
    model, variables, key, idx, max_new_tokens, context_size, temperature, top_k=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model.apply(variables, idx_cond, training=False)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = jax.lax.top_k(logits, top_k)
            min_val = top_logits[:, -1]
            logits = jnp.where(logits < min_val, -jnp.inf, logits)

        if temperature > 0.0:
            logits = logits / temperature

            idx_next = jax.random.categorical(key, logits, shape=(1,), axis=-1)
            idx_next = idx_next[:, None]

        else:
            idx_next = jnp.argmax(logits, axis=-1, keepdims=True)

        idx = jnp.concatenate([idx, idx_next], axis=-1)
    return idx


@click.command()
@click.option("--data_path", type=str, default="the-verdict.txt")
@click.option("--tokenizer_path", type=str, default="data/tokenizer-aozora.json")
def main():
    model = GPTModel(cfg=GPTConfig())

    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.ones((OptimizerConfig.batch_size, GPTConfig.ctx_len), dtype=jnp.int32),
        training=False,
    )
    key = jax.random.PRNGKey(0)

    start_context = "Every effort moves you"
    if tokenizer_path != "gpt2":
        start_context = "深いおどろきにうたれて、"
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = text_to_token_ids(start_context, tokenizer)

    import pickle

    with open(f"{data_path}_variables.pkl", "rb") as f:
        variables = pickle.load(f)

    key, subkey = jax.random.split(key)
    token_ids = generate(
        model=model,
        variables=variables,
        key=subkey,
        idx=batch,
        max_new_tokens=15,
        context_size=GPTConfig.ctx_len,
        temperature=1.4,
        top_k=25,
    )

    print("Output:", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
