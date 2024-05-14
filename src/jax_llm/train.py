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
):
    data_filename = f"data/{data_name}/input.txt"
    tokenizer_path = f"data/{data_name}/tokenizer.json"
    model_path = f"model/{data_name}"

    assert embed_size == head_size * num_heads, "embed_size must be equal to head_size * num_heads"
    import os

    os.makedirs(model_path, exist_ok=True)

    # platform check
    print("JAX running on", jax.devices()[0].platform.upper())

    tokenizer = AbstractTokenizer(
        Tokenizer.from_file(tokenizer_path), tokenizer_path.split("/")[-2]
    )
    print("Tokenizer loaded")
    # load tokenizer json

    tokenizer_json = json.load(open(tokenizer_path, "r"))
    vocab_size = len(tokenizer_json["model"]["vocab"])

    with open(data_filename, "r", encoding="utf-8") as f:
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

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    var_params = model.init(
        key,
        jnp.ones((batch_size, block_size), dtype=jnp.int32),
        training=False,
    )

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(var_params))

    print(f"Total number of parameters: {n_params:_}")

    # To run with SGD instead of adam, replace `adam` with `sgd`
    opt = optax.adamw(learning_rate=learning_rate)

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

    for i in trange(n_iterations):
        var_params, key, opt_state, loss = step(key, var_params, opt_state)
        all_train_losses.append(loss)

        # once every n_freq_eval we compute loss on the validation set
        if i % n_freq_eval == 0:
            key, subkey = jax.random.split(key)
            eval_loss = eval_step(var_params, *get_batch(subkey, eval_data))
            all_eval_losses.append(eval_loss)
            print(f"Step: {i}\t train loss: {loss}\t eval loss: {eval_loss}")

    plt.title("Convergence of adamw (train loss)")
    plt.plot(all_train_losses, label="train", lw=3)
    plt.plot(
        jnp.arange(0, len(all_eval_losses) * n_freq_eval, n_freq_eval),
        all_eval_losses,
        label="test",
        lw=3,
    )
    # display token seen num on the upper x-axis
    ax2 = plt.gca().twiny()
    tokens_seen = jnp.arange(0, len(all_train_losses)) * batch_size * block_size
    ax2.plot(tokens_seen, all_train_losses, alpha=0)
    ax2.set_xlabel("tokens seen")
    ax1 = plt.gca()
    ax1.set_xlabel("steps")
    plt.ylabel("loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"train_loss_{data_name}.png")
    plt.show()

    # Let's now generate some text
    key, subkey = jax.random.split(key)
    text = model.generate(key, var_params, 1000)[:, 0, 0].tolist()
    print(tokenizer.decode(text))

    # Store the model
    with open(f"{model_path}/params.pkl", "wb") as f:
        pickle.dump(var_params, f)
    print("Params stored")

    # Store optimizer state
    with open(f"{model_path}/opt_state.pkl", "wb") as f:
        pickle.dump(opt_state, f)

    # store the moel config
    config = {
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_size": head_size,
        "dropout_rate": dropout_rate,
        "embed_size": embed_size,
        "block_size": block_size,
        "n_params": n_params,
    }
    with open(f"{model_path}/config.json", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
