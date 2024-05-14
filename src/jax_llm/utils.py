import jax
import jax.numpy as jnp


class AbstractTokenizer:
    def __init__(self, tokenizer, name):
        self.tokenizer = tokenizer
        self.name = name

    def encode(self, text):
        if self.name == "gpt2":
            return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        else:
            return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


def text_to_token_ids(text: str, tokenizer: AbstractTokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = jnp.array(encoded).reshape(1, -1)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer: AbstractTokenizer):
    flat = token_ids.flatten()
    return tokenizer.decode(flat.tolist())


# @functools.partial(jax.jit, static_argnames=("model", "length", "top_k", "temperature"))
def top_k_generate(
    model, rng, params, length, prompt, top_k: int = 30, temperature: float = 1.0
):
    def _scan_generate(carry, _):
        random_key, context = carry
        logits = model.apply(params, context, training=False)
        top_logits, _ = jax.lax.top_k(logits[:, -1, :], top_k)
        min_val = top_logits[:, -1]
        logits = jnp.where(logits < min_val, -jnp.inf, logits)
        logits = logits / temperature
        rng, rng_subkey = jax.random.split(random_key)
        new_token = jax.random.categorical(
            rng_subkey, logits[:, -1, :], axis=-1, shape=(1, 1)
        )
        context = jnp.concatenate([context[:, 1:], new_token], axis=1)
        return (rng, context), new_token

    context = prompt
    context = jnp.array(context).reshape(1, -1)
    _, new_tokens = jax.lax.scan(
        _scan_generate,
        (rng, context),
        (),
        length=length,
    )
    generated = jnp.concatenate([prompt.flatten(), new_tokens.flatten()])
    return generated


# @functools.partial(jax.jit, static_argnames=("model", "length"))
def fast_generate(model, rng, params, length, prompt):
    def _scan_generate(carry, _):
        random_key, context = carry
        logits = model.apply(params, context, training=False)
        rng, rng_subkey = jax.random.split(random_key)
        new_token = jax.random.categorical(
            rng_subkey, logits[:, -1, :], axis=-1, shape=(1, 1)
        )
        context = jnp.concatenate([context[:, 1:], new_token], axis=1)
        return (rng, context), new_token

    context = prompt
    context = jnp.array(context).reshape(1, -1)
    _, new_tokens = jax.lax.scan(
        _scan_generate,
        (rng, context),
        (),
        length=length,
    )

    return new_tokens
