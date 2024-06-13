import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import optax
from utils import AbstractTokenizer
from tokenizers import Tokenizer
from model import NanoLM
import pickle
import json
import click
from tqdm import trange
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from flax.training import train_state
from flax.struct import dataclass
from typing import Any, Dict, Tuple
import numpy as np
from ml_collections import ConfigDict
import flax.linen as nn
import functools

PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


class TrainState(train_state.TrainState):
    rng: jax.Array


def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
    """Folds the random number generator over the given axis.

    This is useful for generating a different random number for each device
    across a certain axis (e.g. the model axis).

    Args:
        rng: The random number generator.
        axis_name: The axis name to fold the random number generator over.

    Returns:
        A new random number generator, different for each device index along the axis.
    """
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


@click.command()
@click.option("--data_name", type=str, default="aozora")
@click.option("--seed", type=int, default=42)
@click.option("--learning_rate", type=float, default=1e-4)
@click.option("--batch_size", type=int, default=256)
@click.option(
    "--n_iterations",
    type=int,
    default=50000,
    help="Number of training iterations (batch_size * block_size * n_iterations = total tokens seen)",
)
@click.option(
    "--n_freq_eval",
    type=int,
    default=1000,
    help="Number of training iterations between two consecutive evaluations",
)
@click.option("--dropout_rate", type=float, default=0.2)
@click.option(
    "--block_size",
    type=int,
    default=128,
    help="Context window for the transformer model",
)
@click.option(
    "--num_layers",
    type=int,
    default=8,
    help="Number of layer for the transformer model",
)
@click.option(
    "--embed_size",
    type=int,
    default=256,
    help="Size of the embedding for the transformer model",
)
@click.option(
    "--num_heads", type=int, default=8, help="Number of heads for the transformer model"
)
@click.option(
    "--head_size",
    type=int,
    default=32,
    help="Size of the heads for the transformer model",
)
@click.option(
    "--device_num",
    type=int,
    default=2,
    help="Number of devices to use",
)
def main(
    data_name: str,
    seed: int,
    learning_rate: float,
    batch_size: int,
    n_iterations: int,
    n_freq_eval: int,
    dropout_rate: float,
    block_size: int,
    num_layers: int,
    embed_size: int,
    num_heads: int,
    head_size: int,
    device_num: int,
):
    data_filename = f"data/{data_name}/tokenized_text.bin"
    tokenizer_path = f"data/{data_name}/tokenizer.json"
    model_path = f"model/{data_name}"
    figure_dir = f"figure/{data_name}"

    assert (
        embed_size == head_size * num_heads
    ), "embed_size must be equal to head_size * num_heads"
    import os

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    # platform check
    print("JAX running on", jax.devices()[0].platform.upper())

    tokenizer = AbstractTokenizer(
        Tokenizer.from_file(tokenizer_path), tokenizer_path.split("/")[-2]
    )
    print("Tokenizer loaded")
    # load tokenizer json

    tokenizer_json = json.load(open(tokenizer_path, "r"))
    vocab_size = len(tokenizer_json["model"]["vocab"])

    if not os.path.exists(data_filename):
        with open(f"data/{data_name}/input.txt", "r", encoding="utf-8") as f:
            text_data = f.read()
        tokenized_text = jnp.array(tokenizer.encode(text_data).ids)
    else:
        with open(data_filename, "rb") as f:
            tokenized_text = f.read()
    train_ratio = 0.90
    tokenized_text = jnp.frombuffer(tokenized_text, dtype=jnp.int32)
    total_tokens = len(tokenized_text)
    split_idx = int(train_ratio * total_tokens)
    train_data = tokenized_text[:split_idx]
    eval_data = tokenized_text[split_idx:]

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
            random_key, shape=(batch_size, 1), minval=0, maxval=len(data) - block_size
        )
        x = dynamic_slice_vmap(data, ix, (block_size,))
        y = dynamic_slice_vmap(data, ix + 1, (block_size,))
        return x, y

    model = NanoLM(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        dropout_rate=dropout_rate,
        embed_size=embed_size,
        block_size=block_size,
    )

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply(params, x, training=False)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=y
        ).mean()

    config = ConfigDict(
        dict(
            data_axis_name="data",
        )
    )
    device_array = np.array(jax.devices()[:device_num])
    mesh = Mesh(device_array, config.data_axis_name)

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    # To run with SGD instead of adam, replace `adam` with `sgd`
    opt = optax.adamw(learning_rate=learning_rate)

    def init_dp(rng: jax.random.PRNGKey, x: jax.Array, model: nn.Module) -> TrainState:
        init_rng, rng = jax.random.split(rng)
        variables = model.init(init_rng, x, training=False)
        n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables))
        print(f"Total number of parameters: {n_params:_}")

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables,
            tx=opt,
            rng=rng,
        )
        return state

    init_dp_fn = jax.jit(
        shard_map(
            functools.partial(init_dp, model=model),
            mesh,
            in_specs=(P(), P(config.data_axis_name)),
            out_specs=P(),
            check_rep=False,
        ),
    )

    model_init_rng, data_inputs_rng, data_labels_rng = jax.random.split(key, 3)
    state_dp = init_dp_fn(
        model_init_rng, jnp.ones((batch_size, block_size), dtype=jnp.int32)
    )
    print(device_array)
    print(mesh)

    def loss_fun(params, apply_fn, x, y, dropout_key):
        dropout_key = fold_rng_over_axis(dropout_key, config.data_axis_name)
        logits = apply_fn(params, x, training=True, rngs={"dropout": dropout_key})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        batch_size = x.shape[0]
        step_metrics = {"loss": loss, "batch_size": batch_size}
        return loss, step_metrics

    def train_step_dp(
        state: TrainState, metrics: Metrics | None, x, y
    ) -> Tuple[TrainState, Metrics]:
        rng, step_rng = jax.random.split(state.rng)
        (loss, step_metrics), grads = jax.value_and_grad(loss_fun, has_aux=True)(
            state.params, state.apply_fn, x, y, step_rng
        )

        # Update parameters. We need to sync the gradients across devices before updating.
        with jax.named_scope("sync_gradients"):
            grads = jax.tree.map(
                lambda g: jax.lax.pmean(g, axis_name=config.data_axis_name), grads
            )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        with jax.named_scope("sync_metrics"):
            step_metrics = jax.tree.map(
                lambda x: jax.lax.pmean(x, axis_name=config.data_axis_name),
                step_metrics,
            )

        return new_state, step_metrics

    train_step_dp_fn = jax.jit(
        shard_map(
            train_step_dp,
            mesh,
            in_specs=(P(), P(), P(config.data_axis_name), P(config.data_axis_name)),
            out_specs=(P(), P()),
            check_rep=False,
        ),
        donate_argnames=("state", "metrics"),
    )
    _, metric_shapes = jax.eval_shape(
        train_step_dp_fn,
        state_dp,
        None,
        jnp.ones((batch_size, block_size), dtype=jnp.int32),
        jnp.ones((batch_size, block_size), dtype=jnp.int32),
    )
    metrics_dp = jax.tree.map(
        lambda x: jnp.zeros(x.shape, dtype=x.dtype), metric_shapes
    )
    all_train_losses = []
    all_eval_losses = []
    for epoch in trange(n_iterations):
        key, subkey = jax.random.split(key)
        batch = get_batch(subkey, train_data)
        state_dp, metrics_dp = train_step_dp_fn(state_dp, metrics_dp, *batch)
        all_train_losses.append(metrics_dp["loss"])
        if epoch % n_freq_eval == 0:
            key, subkey = jax.random.split(key)
            eval_loss = eval_step(state_dp.params, *get_batch(subkey, eval_data))
            all_eval_losses.append(eval_loss)
            print(
                f"Step: {epoch}\t train loss: {metrics_dp['loss']}\t eval loss: {eval_loss}"
            )

    # Let's now generate some text
    params = state_dp.params
    key, subkey = jax.random.split(key)
    text = model.generate(key, params, 1000)[:, 0, 0].tolist()
    print(tokenizer.decode(text))

    plt.title("Loss dynamics")
    fig, ax1 = plt.subplots()
    ax1.plot(all_train_losses, label="train", lw=3)
    ax1.plot(
        jnp.arange(0, len(all_eval_losses) * n_freq_eval, n_freq_eval),
        all_eval_losses,
        label="test",
        lw=3,
    )
    ax1.set_xlabel("steps")
    ax1.set_ylabel("loss")

    ax2 = ax1.twiny()
    tokens_seen = jnp.arange(0, len(all_train_losses)) * batch_size * block_size
    ax2.plot(tokens_seen, all_train_losses, alpha=0)
    ax2.set_xlabel("tokens seen")
    ax1.legend()
    ax1.grid()
    fig.tight_layout()
    plt.savefig(f"{figure_dir}/loss_dynamics.png")
    plt.show()

    # Store the model
    with open(f"{model_path}/params.pkl", "wb") as f:
        pickle.dump(state_dp.params, f)
    print("Params stored")

    # store the moel config
    config = {
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_size": head_size,
        "dropout_rate": dropout_rate,
        "embed_size": embed_size,
        "block_size": block_size,
        "batch_size": batch_size,
        "n_iterations": n_iterations,
        "n_freq_eval": n_freq_eval,
        "total_tokens": total_tokens,
        "learning_rate": learning_rate,
    }
    with open(f"{model_path}/config.json", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
