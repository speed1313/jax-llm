import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import optax
from utils import AbstractTokenizer
from tokenizers import Tokenizer
from model import NanoLM
import pickle
import json

# platform check
print("JAX running on", jax.devices()[0].platform.upper())

# @markdown Random seed:
SEED = 42  # @param{type:"integer"}
# @markdown Learning rate passed to the optimizer:
LEARNING_RATE = 1e-4  # @param{type:"number"}
# @markdown Batch size:
BATCH_SIZE = 256  # @param{type:"integer"}
# @markdown Numer of training iterations:
N_ITERATIONS = 50000  # @param{type:"integer"}
# @markdown Number of training iterations between two consecutive evaluations:
N_FREQ_EVAL = 1000  # @param{type:"integer"}
# @markdown Rate for dropout in the transformer model
DROPOUT_RATE = 0.2  # @param{type:"number"}
# @markdown Context window for the transformer model
BLOCK_SIZE = 64  # @param{type:"integer"}
# @markdown Number of layer for the transformer model
NUM_LAYERS = 6  # @param{type:"integer"}
# @markdown Size of the embedding for the transformer model
EMBED_SIZE = 256  # @param{type:"integer"}
# @markdown Number of heads for the transformer model
NUM_HEADS = 8  # @param{type:"integer"}
# @markdown Size of the heads for the transformer model
HEAD_SIZE = 32  # @param{type:"integer"}


key = jax.random.PRNGKey(0)

tokenizer = AbstractTokenizer(
    Tokenizer.from_file("data/tokenizer.json"), "data/tokenizer.json"
)
print("Tokenizer loaded")
with open("input.txt", "r", encoding="utf-8") as f:
    text_data = f.read()
train_ratio = 0.90
total_tokens = len(tokenizer.encode(text_data))
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
train_data = tokenizer.encode(train_data)
train_data = jnp.array(train_data)
eval_data = text_data[split_idx:]
eval_data = tokenizer.encode(eval_data)
eval_data = jnp.array(eval_data)


dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))


@jax.jit
def get_batch(random_key, data):
    """Prepares a random batch of training data.

    Args:
        random_key: A random seed for sampling a batch.
        data: The complete training dataset.

    Returns:
        x: Input sequences.
        y: Target sequences (shifted inputs).
    """
    ix = jax.random.randint(
        random_key, shape=(BATCH_SIZE, 1), minval=0, maxval=len(data) - BLOCK_SIZE
    )
    x = dynamic_slice_vmap(data, ix, (BLOCK_SIZE,))
    y = dynamic_slice_vmap(data, ix + 1, (BLOCK_SIZE,))
    return x, y


model = NanoLM(
    vocab_size=30000,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    head_size=HEAD_SIZE,
    dropout_rate=DROPOUT_RATE,
    embed_size=EMBED_SIZE,
    block_size=BLOCK_SIZE,
)


def loss_fun(params, x, y, dropout_key):
    logits = model.apply(params, x, training=True, rngs={"dropout": dropout_key})
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()


@jax.jit
def eval_step(params, x, y):
    logits = model.apply(params, x, training=False)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()


key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)

var_params = model.init(
    key,
    jnp.ones((BATCH_SIZE, BLOCK_SIZE), dtype=jnp.int32),
    training=False,
)


n_params = sum(p.size for p in jax.tree_util.tree_leaves(var_params))

print(f"Total number of parameters: {n_params:_}")


# To run with SGD instead of adam, replace `adam` with `sgd`
opt = optax.adamw(learning_rate=LEARNING_RATE)

opt_state = opt.init(var_params)


all_train_losses = []
all_eval_losses = []


# we define one iteration of the optimizer and JIT this function
@jax.jit
def step(key, params, opt_state):
    key, subkey = jax.random.split(key)
    batch = get_batch(key, train_data)
    loss, grad = jax.value_and_grad(loss_fun)(params, *batch, subkey)
    updates, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, key, opt_state, loss


for i in range(N_ITERATIONS):
    var_params, key, opt_state, loss = step(key, var_params, opt_state)
    all_train_losses.append(loss)

    # once every N_FREQ_EVAL we compute loss on the validation set
    if i % N_FREQ_EVAL == 0:
        key, subkey = jax.random.split(key)
        eval_loss = eval_step(var_params, *get_batch(subkey, eval_data))
        all_eval_losses.append(eval_loss)
        print(f"Step: {i}\t train loss: {loss}\t eval loss: {eval_loss}")


plt.title("Convergence of adamw (train loss)")
plt.plot(all_train_losses, label="train", lw=3)
plt.plot(
    jnp.arange(0, len(all_eval_losses) * N_FREQ_EVAL, N_FREQ_EVAL),
    all_eval_losses,
    label="test",
    lw=3,
)
# display token seen num on the upper x-axis
ax2 = plt.gca().twiny()
tokens_seen = jnp.arange(0, len(all_train_losses)) * BATCH_SIZE * BLOCK_SIZE
ax2.plot(tokens_seen, all_train_losses, alpha=0)
ax2.set_xlabel("tokens seen")
plt.xlabel("steps")
plt.ylabel("loss")
plt.grid()
plt.legend(frameon=False)
plt.savefig("train_loss.png")
plt.show()


# Let's now generate some text
key, subkey = jax.random.split(key)
text = model.generate(key, var_params, 1000)[:, 0, 0].tolist()
print(tokenizer.decode(text))

# Store the model
import os

os.makedirs("model", exist_ok=True)
with open("model/params.pkl", "wb") as f:
    pickle.dump(var_params, f)
print("Params stored")

# store the moel config
config = {
    "vocab_size": 30000,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "head_size": HEAD_SIZE,
    "dropout_rate": DROPOUT_RATE,
    "embed_size": EMBED_SIZE,
    "block_size": BLOCK_SIZE,
}
with open("model/config.json", "w") as f:
    json.dump(config, f)
