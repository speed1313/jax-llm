import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn
from torch.utils import data

class GPTDatasetV1(data.Dataset):
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



def numpy_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


def create_dataset_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=numpy_collate, drop_last=drop_last)

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
