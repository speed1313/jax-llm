import tiktoken
import jax
import jax.numpy as jnp
from dataloader import create_dataset_v1
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
    vocab_size: int = 50304
    ctx_len: int = 64
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    drop_rate: float = 0.
    qkv_bias: bool = False


def calc_loss_train(
    variables, input_batch, target_batch, model, training=True, key=None
):
    @jax.jit
    def loss_fn(variables):
        logits = model.apply(
            variables,
            input_batch,
            training=True,
            rngs={"dropout": key},
        )
        return optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=target_batch
        ).mean()
    loss = loss_fn(variables)
    return loss


def calc_loss_val(
    variables, input_batch, target_batch, model, training=False, key=None
):
    @jax.jit
    def loss_fn(variables):
        logits = model.apply(variables, input_batch, training=False)
        return optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=target_batch
        ).mean()
    loss = loss_fn(variables)
    return loss


def calc_loss_loader(
    data_loader, model, num_batches=None, variables=None, training=False, key=None
):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_val(
                variables, input_batch, target_batch, model, training=training, key=key
            )
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def train_model_simple(
    model,
    train_loader,
    val_loader,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    optimizer,
    optimizer_state,
    variables,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    key = jax.random.PRNGKey(0)
    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            key, subkey = jax.random.split(key)
            grads = jax.grad(calc_loss_train)(
                variables, input_batch, target_batch, model, training=True, key=subkey
            )
            updates, optimizer_state = optimizer.update(
                grads, optimizer_state, variables
            )
            variables = optax.apply_updates(variables, updates)
            tokens_seen += input_batch.size
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(
                    train_loader,
                    model,
                    variables=variables,
                    training=False,
                    num_batches=eval_iter,
                )
                val_loss = calc_loss_loader(
                    val_loader,
                    model,
                    variables=variables,
                    training=False,
                    num_batches=eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"
                )
                #generate_and_print_sample(model, variables, start_context, tokenizer)
    return train_losses, val_losses, track_tokens_seen, variables, optimizer_state


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
@click.option("--data_path", type=str, default="the-verdict.txt")
@click.option("--tokenizer_path", type=str, default="data/tokenizer-aozora.json")
@click.option("--batch_size", type=int, default=16)
@click.option("--epochs", type=int, default=1)
@click.option("--learning_rate", type=float, default=6e-4)
@click.option("--weight_decay", type=float, default=0.1)
def main(
    data_path: str,
    tokenizer_path: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
):
    model = GPTModel(cfg=GPTConfig())
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
    train_loader = create_dataset_v1(
        train_data,
        tokenizer,
        batch_size,
        max_length=GPTConfig.ctx_len,
        stride=GPTConfig.ctx_len,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataset_v1(
        val_data,
        tokenizer,
        batch_size,
        max_length=GPTConfig.ctx_len,
        stride=GPTConfig.ctx_len,
        shuffle=False,
        drop_last=False,
    )

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
            train_loader,
            val_loader,
            num_epochs=epochs,
            eval_freq=5,
            eval_iter=5,
            start_context="深いおどろきにうたれて、",
            tokenizer=tokenizer,
            optimizer=optimizer,
            optimizer_state=opt_state,
            variables=variables,
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
    with open(f"model/{data_path.split('.')[0]}_variables.pkl", "wb") as f:
        pickle.dump(variables, f)
    print("Variables stored")


if __name__ == "__main__":
    main()
