
import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import optax
import tensorflow_datasets as tfds
from utils import AbstractTokenizer
from tokenizers import Tokenizer
from dataclasses import dataclass
# platform check
print("JAX running on", jax.devices()[0].platform.upper())

# @markdown Random seed:
SEED = 42  # @param{type:"integer"}
# @markdown Learning rate passed to the optimizer:
LEARNING_RATE = 1e-3 # @param{type:"number"}
# @markdown Batch size:
BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown Numer of training iterations:
N_ITERATIONS = 1000  # @param{type:"integer"}
# @markdown Number of training iterations between two consecutive evaluations:
N_FREQ_EVAL = 2_0 # @param{type:"integer"}
# @markdown Rate for dropout in the transformer model
DROPOUT_RATE = 0.2  # @param{type:"number"}
# @markdown Context window for the transformer model
BLOCK_SIZE = 64  # @param{type:"integer"}
# @markdown Number of layer for the transformer model
NUM_LAYERS = 4  # @param{type:"integer"}
# @markdown Size of the embedding for the transformer model
EMBED_SIZE = 128  # @param{type:"integer"}
# @markdown Number of heads for the transformer model
NUM_HEADS = 4  # @param{type:"integer"}
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


class NanoLM(nn.Module):
    """NanoLM model."""
    vocab_size: int
    num_layers: int = 4
    num_heads: int = 4
    head_size: int = 32
    dropout_rate: float = 0.2
    embed_size: int = 128
    block_size: int = 64

    @nn.compact
    def __call__(self, x, training: bool):
        seq_len = x.shape[1]

        x = nn.Embed(self.vocab_size, self.embed_size)(x) + nn.Embed(
            self.block_size, self.embed_size
        )(jnp.arange(seq_len))
        for _ in range(self.num_layers):
            x_norm = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.head_size,
                out_features=self.head_size * self.num_heads,
                dropout_rate=self.dropout_rate,
            )(
                x_norm,
                x_norm,
                mask=jnp.tril(jnp.ones((x.shape[-2], x.shape[-2]))),
                deterministic=not training,
            )

            x = x + nn.Sequential([
                nn.Dense(4 * self.embed_size),
                nn.relu,
                nn.Dropout(self.dropout_rate, deterministic=not training),
                nn.Dense(self.embed_size),
            ])(nn.LayerNorm()(x))

        x = nn.LayerNorm()(x)
        return nn.Dense(self.vocab_size)(x)

    @functools.partial(jax.jit, static_argnames=("self", "length"))
    def generate(self, rng, params, length):
        def _scan_generate(carry, _):
            random_key, context = carry
            logits = self.apply(params, context, training=False)
            rng, rng_subkey = jax.random.split(random_key)
            new_token = jax.random.categorical(
                rng_subkey, logits[:, -1, :], axis=-1, shape=(1, 1)
            )
            context = jnp.concatenate([context[:, 1:], new_token], axis=1)
            return (rng, context), new_token

        _, new_tokens = jax.lax.scan(
            _scan_generate,
            (rng, jnp.zeros((1, self.block_size), dtype=jnp.int32)),
            (),
            length=length,
        )
        return new_tokens


model = NanoLM(
    vocab_size=30000,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    head_size=HEAD_SIZE,
    dropout_rate=DROPOUT_RATE,
    embed_size=EMBED_SIZE,
    block_size=BLOCK_SIZE,
)
"""
from model import GPTModel

@dataclass
class GPTConfig:
    vocab_size: int = 30000
    ctx_len: int = 64
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    drop_rate: float = 0.0
    qkv_bias: bool = False

model = GPTModel(GPTConfig())
"""

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



plt.title(f"Convergence of adamw (train loss)")
plt.plot(all_train_losses, label="train", lw=3)
plt.plot(
    jnp.arange(0, len(all_eval_losses) * N_FREQ_EVAL, N_FREQ_EVAL),
    all_eval_losses,
    label="test",
    lw=3,
)
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