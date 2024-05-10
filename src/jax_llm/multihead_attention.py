import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataloader import create_dataset_v1

class CausalAttention(nn.Module):
    d_in: int
    d_out: int
    block_size: int
    dropout: float
    qkv_bias: bool = False

    def setup(self):
        self.W_query = nn.Dense(features=self.d_out, use_bias=self.qkv_bias)
        self.W_key = nn.Dense(features=self.d_out, use_bias=self.qkv_bias)
        self.W_value = nn.Dense(features=self.d_out, use_bias=self.qkv_bias)
        self.mask = jnp.triu(jnp.ones((self.block_size, self.block_size)), k=1)
    @nn.compact
    def __call__(self, x, training: bool):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        attn_scores = queries @ keys.transpose((0, 2, 1))
        attn_scores = jnp.where(self.mask == 1, -jnp.inf, attn_scores)
        attn_weights = jax.nn.softmax(attn_scores / jnp.sqrt(keys.shape[-1]), axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout, deterministic=not training)(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    num_heads: int
    d_in: int
    d_out: int
    block_size: int
    dropout: float
    qkv_bias: bool = False

    def setup(self):
        self.heads = [CausalAttention(d_in=self.d_in, d_out=self.d_out, block_size=self.block_size, dropout=self.dropout, qkv_bias=self.qkv_bias) for _ in range(self.num_heads)]

    def __call__(self, x, training: bool):
        return jnp.concatenate([head(x, training) for head in self.heads], axis=-1)


class MultiHeadAttention(nn.Module):
    d_out : int
    block_size: int
    num_heads: int
    head_dim : int
    dropout_rate: float = 0.0
    qkv_bias: bool = False

    def setup(self):
        self.W_query = nn.Dense(features=self.d_out, use_bias=self.qkv_bias)
        self.W_key = nn.Dense(features=self.d_out, use_bias=self.qkv_bias)
        self.W_value = nn.Dense(features=self.d_out, use_bias=self.qkv_bias)
        self.W_out = nn.Dense(features=self.d_out, use_bias=self.qkv_bias)
        self.out_proj = nn.Dense(features=self.d_out)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.mask = jnp.triu(jnp.ones((self.block_size, self.block_size)), k=1)

    def __call__(self, x, training: bool):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)


        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose((0, 2, 1, 3))
        values = values.transpose((0, 2, 1, 3))
        queries = queries.transpose((0, 2, 1, 3))

        attn_scores = queries @ keys.transpose((0, 1, 3, 2))


        attn_scores = jnp.where(self.mask == 1, -jnp.inf, attn_scores)

        attn_weights = jax.nn.softmax(attn_scores / jnp.sqrt(keys.shape[-1]), axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=not training)

        context_vec = (attn_weights @ values).transpose((0, 2, 1, 3))

        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    datalodaer = create_dataset_v1(raw_text, batch_size=8, max_length=4, stride=5, shuffle=True, drop_last=True)
    vocab_size = 50257
    output_dim = 256
    block_size = 1024 # 1024 tokens per block
    max_length = 4
    token_embedding_layer = nn.Embed(vocab_size, output_dim)
    pos_embedding_layer = nn.Embed(block_size, output_dim)
    token_embedding_variables = token_embedding_layer.init(jax.random.PRNGKey(0), jnp.arange(max_length))
    pos_embedding_variables = pos_embedding_layer.init(jax.random.PRNGKey(0), jnp.arange(max_length))

    print("data loaded")
    for batch in datalodaer:
        x, y = batch
        token_embeddings = token_embedding_layer.apply(token_embedding_variables, x)
        pos_embeddings = pos_embedding_layer.apply(pos_embedding_variables, jnp.arange(max_length))
        input_embeddings = token_embeddings + pos_embeddings
        break
    print("input", input_embeddings.shape)

    block_size = max_length
    d_in = output_dim
    num_heads = 2
    d_out = d_in
    mha = MultiHeadAttention(d_out=d_out, block_size = block_size, num_heads=num_heads, head_dim=d_out//num_heads, dropout_rate=0.0, qkv_bias=False)
    batch = input_embeddings
    params = mha.init(jax.random.PRNGKey(0), batch, True)
    context_vec = mha.apply(params, batch, True)
    print(context_vec.shape)


