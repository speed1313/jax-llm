import tiktoken
import jax
import jax.numpy as jnp
from dataloader import create_jax_dataset
from model import GPTModel
import optax
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pickle
import click
from utils import AbstractTokenizer, text_to_token_ids, token_ids_to_text, generate
from tokenizers import Tokenizer

jax.config.update("jax_debug_nans", True)


@dataclass
class GPTConfig:
    vocab_size: int = 30000
    ctx_len: int = 64
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    drop_rate: float = 0.0
    qkv_bias: bool = False





def generate_and_print_sample(model, variables, start_context, tokenizer):
    batch = text_to_token_ids(start_context, tokenizer)
    token_ids = generate(
        model,
        variables,
        None,
        batch,
        max_new_tokens=50,
        context_size=GPTConfig.ctx_len,
    )
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    plt.savefig("loss_curve.png")
    plt.show()


@click.command()
@click.option("--data_path", type=str, default="input.txt")
@click.option("--tokenizer_path", type=str, default="data/tokenizer.json")
@click.option("--batch_size", type=int, default=128)
@click.option("--epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=0.0)
def main(
    data_path: str,
    tokenizer_path: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
):
    #model = GPTModel(cfg=GPTConfig())
    from model import NanoLM
    model = NanoLM(vocab_size=30000)
    # TODO: jit the loss function
    @jax.jit
    def loss_fn(variables, input_batch, target_batch, key=None):
        logits = model.apply(
            variables,
            input_batch,
            training=True,
            rngs={"dropout": key},
        )
        return optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=target_batch
        ).mean()

    def calc_loss_train(variables, input_batch, target_batch, training=True, key=None):
        loss = loss_fn(variables, input_batch, target_batch, key=key)
        return loss

    def step(input_batch, target_batch, variables, optimizer, optimizer_state, key):
        key, subkey = jax.random.split(key)
        grads = jax.grad(calc_loss_train)(
            variables, input_batch, target_batch, training=True, key=subkey
        )
        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, variables
        )
        variables = optax.apply_updates(variables, updates)
        return variables, optimizer_state, key

    def train_model_simple(
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
        optimizer,
        optimizer_state,
        variables,
        batch_size,
    ):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        key = jax.random.PRNGKey(0)
        for epoch in range(num_epochs):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, len(X_train))
            for i in range(0, len(X_train), batch_size):
                input_batch = X_train[perm[i : i + batch_size]]
                target_batch = Y_train[perm[i : i + batch_size]]
                variables, optimizer_state, subkey = step(
                    input_batch,
                    target_batch,
                    variables,
                    optimizer,
                    optimizer_state,
                    key,
                )
                tokens_seen += input_batch.size
                global_step += 1
                if global_step % eval_freq == 0:
                    train_loss = calc_loss_train(
                        variables,
                        input_batch,
                        target_batch,
                        training=False,
                        key=subkey,
                    )
                    val_loss = calc_loss_train(
                        variables,
                        input_batch,
                        target_batch,
                        training=False,
                        key=subkey,
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"
                    )
                    # generate_and_print_sample(model, variables, start_context, tokenizer)
        return train_losses, val_losses, track_tokens_seen, variables, optimizer_state


    key = jax.random.PRNGKey(0)

    if tokenizer_path == "gpt2":
        tokenizer = AbstractTokenizer(tiktoken.get_encoding("gpt2"), "gpt2")
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AbstractTokenizer(
            Tokenizer.from_file(tokenizer_path), tokenizer_path
        )
        print("Tokenizer loaded")
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    train_ratio = 0.90
    total_tokens = len(tokenizer.encode(text_data))
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    X_train, Y_train = create_jax_dataset(
        train_data,
        tokenizer,
        batch_size,
        GPTConfig.ctx_len,
        GPTConfig.ctx_len,
        False,
        True,
    )
    print("Tokenized train data shape:", X_train.shape, Y_train.shape)

    X_val, Y_val = create_jax_dataset(
        val_data,
        tokenizer,
        batch_size,
        GPTConfig.ctx_len,
        GPTConfig.ctx_len,
        False,
        True,
    )
    print("Tokenized val data shape:", X_val.shape, Y_val.shape)

    if total_tokens * train_ratio < GPTConfig.ctx_len:
        raise ValueError("The training dataset is too small for the context length")
    if total_tokens * (1 - train_ratio) < GPTConfig.ctx_len:
        raise ValueError("The validation dataset is too small for the context length")

    optimizer = optax.adamw(learning_rate, weight_decay)
    variables = model.init(
        key,
        jnp.ones((batch_size, GPTConfig.ctx_len), dtype=jnp.int32),
        training=False,
    )
    # print("Total tokens:", len(tokenizer.encode(text_data)))
    print("Total parameters:", sum(x.size for x in jax.tree.leaves(variables)))

    opt_state = optimizer.init(variables)
    train_losses, val_losses, track_tokens_seen, variables, opt_state = (
        train_model_simple(
            model,
            X_train,
            Y_train,
            X_val,
            Y_val,
            num_epochs=epochs,
            eval_freq=5,
            eval_iter=5,
            start_context="深いおどろきにうたれて、",
            tokenizer=tokenizer,
            optimizer=optimizer,
            optimizer_state=opt_state,
            variables=variables,
            batch_size=batch_size,
        )
    )

    epochs_tensor = jnp.linspace(0, epochs, num=len(train_losses))
    plot_losses(
        epochs_tensor,
        jnp.array(track_tokens_seen),
        jnp.array(train_losses),
        jnp.array(val_losses),
    )

    # store variables
    with open("model/variables.pkl", "wb") as f:
        pickle.dump(variables, f)
    print("Variables stored")


if __name__ == "__main__":
    main()
