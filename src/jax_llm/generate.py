import tiktoken
import jax
import jax.numpy as jnp
from dataloader import create_dataset_v1
from model import GPTModel, generate_text_simple
import optax
from matplotlib import pyplot as plt

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # TODO: change to 50304
    "ctx_len": 256,  # gpt-2 uses 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model = GPTModel(cfg=GPT_CONFIG_124M)
key = jax.random.PRNGKey(0)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = jnp.array(encoded).reshape(1, -1)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.flatten()
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
batch = text_to_token_ids(start_context, tokenizer)
key, subkey = jax.random.split(key)
import pickle
with open("variables.pkl", "rb") as f:
    variables = pickle.load(f)

token_ids = generate_text_simple(
    model=model,
    variables=variables,
    idx=batch,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["ctx_len"],
)
print("Output:", token_ids_to_text(token_ids, tokenizer))

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}


inverse_vocab = {v: k for k, v in vocab.items()}

# Suppose input is "every effort moves you", and the LLM
# returns the following logits for the next token:
next_token_logits = jnp.array(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = jax.nn.softmax(next_token_logits, axis=-1)
next_token_id = jnp.argmax(probas)

# The next generated token is then as follows:
print(inverse_vocab[next_token_id.item()])

next_token_id = jax.random.categorical(key, next_token_logits, axis=-1)
print(inverse_vocab[next_token_id.item()])


def print_sampled_tokens(probas):
    key = jax.random.PRNGKey(0)
    sample = jax.random.categorical(key, probas, shape=(1000,)) # NOTE: probs should be logits.
    sampled_ids = jnp.bincount(jnp.array(sample), minlength=0)
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(next_token_logits)

def softmax_with_temperature(logits, temperature):
    return jax.nn.softmax(logits / temperature)

temperatures = [1, 0.1, 5]

scaled_probas = [softmax_with_temperature(next_token_logits, t) for t in temperatures]

x = jnp.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5,3))
for i, t in enumerate(temperatures):
    ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f"Temp: {t}")

ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.set_ylabel("Probability")
ax.legend()
plt.tight_layout()
plt.savefig("temperature_plot.png")
plt.show()


print_sampled_tokens(jnp.log(scaled_probas[0]))

print_sampled_tokens(jnp.log(scaled_probas[1]))
print_sampled_tokens(jnp.log(scaled_probas[2]))



top_k = 3
top_logits, top_pos = jax.lax.top_k(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = jnp.where(
    jnp.isin(jnp.arange(len(vocab)), top_pos),
    next_token_logits,
    -jnp.inf,
)

print("New logits:", new_logits)
topk_probas = jax.nn.softmax(new_logits, axis=-1)
print("Top-k probabilities:", topk_probas)

def generate(model, variables, key, idx, max_new_tokens, context_size, temperature, top_k=None):
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

key, subkey = jax.random.split(key)
token_ids = generate(
    model=model,
    variables=variables,
    key=subkey,
    idx=batch,
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["ctx_len"],
    temperature=1.4,
    top_k=25,
)

print("Output:", token_ids_to_text(token_ids, tokenizer))
