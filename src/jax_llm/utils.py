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


def generate(
    model,
    variables,
    key,
    idx,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k=None,
):
    """
    Generate new tokens from a given context
        temperature: 0.0 means greedy sampling, > 0.0 means sampling with temperature
        top_k: if not None, only sample from the top k most likely tokens
    """
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