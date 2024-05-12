import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn
from torch.utils import data

from utils import AbstractTokenizer


class GPTDatasetV1(data.Dataset):
    def __init__(self, txt, tokenizer: AbstractTokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        token_ids = jnp.array(token_ids)

        @jax.jit
        def chunk_processing(i):
            input_chunk = jax.lax.dynamic_slice(token_ids, (i,), (max_length,))
            target_chunk = jax.lax.dynamic_slice(token_ids, (i + 1,), (max_length,))
            return input_chunk, target_chunk

        input_chunks, target_chunks = jax.vmap(chunk_processing)(jnp.arange(0, len(token_ids) - max_length, stride))

        self.input_ids = jnp.array(input_chunks)
        self.target_ids = jnp.array(target_chunks)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def numpy_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


def create_dataset_v1(
    txt,
    tokenizer: AbstractTokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=numpy_collate,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = AbstractTokenizer(tiktoken.get_encoding("gpt2"), "gpt2")
    encoded_text = tokenizer.encode(raw_text)

    vocab_size = 50257
    output_dim = 256
    block_size = 1024  # 1024 tokens per block
    max_length = 4
    token_embedding_layer = nn.Embed(vocab_size, output_dim)
    pos_embedding_layer = nn.Embed(block_size, output_dim)
    token_embedding_variables = token_embedding_layer.init(
        jax.random.PRNGKey(0), jnp.arange(max_length)
    )
    pos_embedding_variables = pos_embedding_layer.init(
        jax.random.PRNGKey(0), jnp.arange(max_length)
    )

    dataloader = create_dataset_v1(
        raw_text,
        tokenizer,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False,
        drop_last=True,
    )

    for batch in dataloader:
        inputs, targets = batch
        print(inputs.shape)
        token_embeddings = token_embedding_layer.apply(
            token_embedding_variables, inputs
        )
        pos_embeddings = pos_embedding_layer.apply(
            pos_embedding_variables, jnp.arange(max_length)
        )
        input_embeddings = token_embeddings + pos_embeddings
        break
    print(input_embeddings.shape)
