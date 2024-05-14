import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from utils import AbstractTokenizer, fast_generate
import functools


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    ctx_len: int = 256
    emb_dim: int = 768
    n_heads: int = 8
    n_layers: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False


class FeerForward(nn.Module):
    cfg: GPTConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.cfg.emb_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.cfg.emb_dim)(x)
        return x


class TransformerBlock(nn.Module):
    cfg: GPTConfig

    def setup(self):
        self.head_dim = self.cfg.emb_dim // self.cfg.n_heads
        self.ff = FeerForward(cfg=self.cfg)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.drop_resid = nn.Dropout(rate=self.cfg.drop_rate)

    @nn.compact
    def __call__(self, x, training: bool):
        B, T, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x, training)
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.cfg.n_heads,
            qkv_features=self.head_dim,
            out_features=self.cfg.emb_dim,
            dropout_rate=self.cfg.drop_rate,
        )(
            x,
            x,
            mask=jnp.tril(jnp.ones((T, T), dtype=jnp.bool_), 0),
            deterministic=not training,
        )
        assert x.shape == (B, T, C)
        x = self.drop_resid(x, deterministic=not training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x, deterministic=not training)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    cfg: dict

    def setup(self):
        self.tok_emb = nn.Embed(self.cfg.vocab_size, self.cfg.emb_dim)
        self.pos_emb = nn.Embed(self.cfg.ctx_len, self.cfg.emb_dim)
        self.drop_emb = nn.Dropout(rate=self.cfg.drop_rate)

        self.trf_blocks = [
            TransformerBlock(cfg=self.cfg) for _ in range(self.cfg.n_layers)
        ]

        self.final_norm = nn.LayerNorm(self.cfg.emb_dim)
        self.out_head = nn.Dense(self.cfg.vocab_size, use_bias=False)

    def __call__(self, x, training: bool):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(jnp.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x, deterministic=not training)
        for block in self.trf_blocks:
            x = block(x, training)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class NanoLM(nn.Module):
    """NanoLM model."""

    vocab_size: int = 50304
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

            x = x + nn.Sequential(
                [
                    nn.Dense(4 * self.embed_size),
                    nn.relu,
                    nn.Dropout(self.dropout_rate, deterministic=not training),
                    nn.Dense(self.embed_size),
                ]
            )(nn.LayerNorm()(x))

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


if __name__ == "__main__":
    tokenizer = AbstractTokenizer(tiktoken.get_encoding("gpt2"), "gpt2")
    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(jnp.array(tokenizer.encode(txt1)))
    batch.append(jnp.array(tokenizer.encode(txt2)))
    batch = jnp.stack(batch)

    model = GPTModel(cfg=GPTConfig())
    variables = model.init(jax.random.PRNGKey(0), batch, training=False)
    logits = model.apply(
        variables, batch, training=False, rngs={"dropout": jax.random.key(2)}
    )
    print("input batch", batch)
    print("logits.shape:", logits.shape)
    print("logits:", logits)

    total_params = sum(x.size for x in jax.tree_leaves(variables))
    print("total params:", total_params)

    print(
        "Token embedding layer shape:",
        variables["params"]["tok_emb"]["embedding"].shape,
    )
    print("Output layer shape:", variables["params"]["out_head"]["kernel"].shape)
    out_head_params = sum(
        [p.size for p in jax.tree.leaves(variables["params"]["out_head"])]
    )
    tok_emb_params = sum(
        [p.size for p in jax.tree.leaves(variables["params"]["tok_emb"])]
    )
    print(total_params + out_head_params + tok_emb_params)
    print(variables["params"].keys())
    print(sum([p.size for p in jax.tree.leaves(variables["params"]["trf_blocks_0"])]))
    total_params_gpt2 = total_params - sum(
        [p.size for p in jax.tree.leaves(variables["params"]["out_head"])]
    )
    print("Total params in GPT2:", total_params_gpt2)

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / 1024 / 1024
    print("Total size in MB:", total_size_mb)

    print(logits)
    b = logits[0, -1, :]
    b = b.at[0].set(-1.4929)
    b = b.at[1].set(4.4812)
    b = b.at[2].set(-1.6093)
    print(b[:3])

    print(jax.nn.softmax(b, axis=0))

    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = jnp.array(encoded)[None, :]
    print("encoded_tensor:", encoded_tensor.shape)

    out = generate(
        model,
        variables,
        key=None,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPTConfig.ctx_len,
    )
    print("out:", out)
    print(len(out[0]))
    decoded_text = tokenizer.decode(out[0])
    print("decoded_text:", decoded_text)
