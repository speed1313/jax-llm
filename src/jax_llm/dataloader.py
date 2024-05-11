import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn


class GPTDatasetV1:
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(jnp.array(input_chunk))
            self.target_ids.append(jnp.array(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataset_v1(
    txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    key = jax.random.PRNGKey(0)
    num_batches = len(dataset) // batch_size
    if not drop_last and len(dataset) % batch_size:
        num_batches += 1
    while True:
        if shuffle:
            key, subkey = jax.random.split(key)
            indices = jax.random.permutation(subkey, len(dataset))
        else:
            indices = jnp.arange(len(dataset))
        for i in range(num_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            batch_inputs = [dataset[j][0] for j in batch_indices]
            batch_targets = [dataset[j][1] for j in batch_indices]
            yield jnp.stack(batch_inputs), jnp.stack(batch_targets)


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
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
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False, drop_last=True
    )

    for batch in dataloader:
        inputs, targets = batch
        token_embeddings = token_embedding_layer.apply(
            token_embedding_variables, inputs
        )
        pos_embeddings = pos_embedding_layer.apply(
            pos_embedding_variables, jnp.arange(max_length)
        )
        input_embeddings = token_embeddings + pos_embeddings
        break
    print(input_embeddings.shape)
