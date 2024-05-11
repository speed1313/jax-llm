import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataloader import create_dataset_v1
from model import GPTModel, generate_text_simple
import optax
import os

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # TODO: change to 50304
    "ctx_len": 256, # gpt-2 uses 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 1,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model = GPTModel(cfg=GPT_CONFIG_124M)
key = jax.random.PRNGKey(0)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = jnp.array(encoded).reshape(1, -1)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.flatten()
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
batch = text_to_token_ids(start_context, tokenizer)
key, subkey = jax.random.split(key)
variables = model.init(subkey, batch, training=False)
token_ids = generate_text_simple(
    model=model,
    variables=variables,
    idx=batch,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["ctx_len"],
)
print("Output:", token_ids_to_text(token_ids, tokenizer))
# Output: Every effort moves you effects Turtle Turtle Turtle Turtle exped Turtle Turtle Turtle Turtle

inputs = jnp.array([[16833, 3626, 6100], # ["every effort moves",
                    [40, 1107, 588]])     # "I really like"]
targets = jnp.array([[3626, 6100, 345],  # [" effort moves you",
                     [588, 428, 11311]]) # " really like chocolate"]

logits = model.apply(variables, inputs, training=False)
probas = jax.nn.softmax(logits, axis=-1)
print("probas.shape:", probas.shape)

token_ids = jnp.argmax(probas, axis=-1, keepdims=True)
print("Token IDs:", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(token_ids[0][0])
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0][0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0][1], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0][2], tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]] # (batch_size, num_tokens, vocab_size)
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]] # (batch_size, num_tokens, vocab_size)
print("Text 2:", target_probas_2)


log_probas = jnp.log(jnp.concatenate([target_probas_1, target_probas_2], axis=0))
print("Log probas:", log_probas)


avg_log_probas = jnp.mean(log_probas)
print("Average log probas:", avg_log_probas)

neg_avg_log_probas = -avg_log_probas
print("Negative average log probas:", neg_avg_log_probas)


print("Logits shape", logits.reshape(-1, logits.shape[-1]).shape)
print("Targets shape", targets.reshape(-1, 1).shape)
# one hot targets
targets = jax.nn.one_hot(targets, logits.shape[-1])
print(targets.shape)
loss = optax.losses.softmax_cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1, logits.shape[-1])).mean()
print(loss)
print(jnp.exp(loss))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

print(text_data[:99])

print(text_data[-99:])

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print(f"Total characters: {total_characters}")
print(f"Total tokens: {total_tokens}")

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataset_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M["ctx_len"], stride=GPT_CONFIG_124M["ctx_len"], shuffle=True, drop_last=True)
val_loader = create_dataset_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M["ctx_len"], stride=GPT_CONFIG_124M["ctx_len"], shuffle=False, drop_last=False)

if total_tokens * train_ratio < GPT_CONFIG_124M["ctx_len"]:
    raise ValueError("The training dataset is too small for the context length")

if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["ctx_len"]:
    raise ValueError("The validation dataset is too small for the context length")

print(len(train_data))
print("Train data:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("Validation data:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.size
val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.size
print(f"Train tokens: {train_tokens}")
print(f"Validation tokens: {val_tokens}")
print(f"Total tokens: {train_tokens + val_tokens}")


