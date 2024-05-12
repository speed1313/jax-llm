import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn
from multihead_attention import MultiHeadAttention
from dataclasses import dataclass
from utils import AbstractTokenizer, generate


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    ctx_len: int = 256
    emb_dim: int = 768
    n_heads: int = 8
    n_layers: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False


class LayerNorm(nn.Module):
    emb_dim: int

    def setup(self):
        self.eps = 1e-5
        self.scale = self.param("scale", nn.initializers.ones, (self.emb_dim,))
        self.bias = self.param("bias", nn.initializers.zeros, (self.emb_dim,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)  # for compatibility with gpt2
        out = (x - mean) / jnp.sqrt(var + self.eps)
        out = out * self.scale + self.bias
        return out


class FeerForward(nn.Module):
    cfg: GPTConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.cfg.emb_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.cfg.emb_dim)(x)
        return x


class ExampleDeepNeuralNetwork(nn.Module):
    layer_sizes: list
    use_shortcut: bool

    def setup(self):
        self.layers = [
            nn.Sequential([nn.Dense(self.layer_sizes[1]), nn.gelu]),
            nn.Sequential([nn.Dense(self.layer_sizes[2]), nn.gelu]),
            nn.Sequential([nn.Dense(self.layer_sizes[3]), nn.gelu]),
            nn.Sequential([nn.Dense(self.layer_sizes[4]), nn.gelu]),
            nn.Sequential([nn.Dense(self.layer_sizes[5]), nn.gelu]),
        ]

    def __call__(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, variable, x):
    output = model.apply(variable, x)
    target = jnp.ones_like(output)
    loss_fn = lambda params, x: jnp.mean((model.apply(params, x) - target) ** 2)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(variable, x)
    # show layer's weight grads mean.
    print(jax.tree.map(lambda x: jnp.mean(x), grad))


class TransformerBlock(nn.Module):
    cfg: GPTConfig

    def setup(self):
        self.att = MultiHeadAttention(
            d_out=self.cfg.emb_dim,
            block_size=self.cfg.ctx_len,
            num_heads=self.cfg.n_heads,
            head_dim=self.cfg.emb_dim // self.cfg.n_heads,
            dropout_rate=self.cfg.drop_rate,
            qkv_bias=self.cfg.qkv_bias,
        )
        self.ff = FeerForward(cfg=self.cfg)
        self.norm1 = LayerNorm(self.cfg.emb_dim)
        self.norm2 = LayerNorm(self.cfg.emb_dim)
        self.drop_resid = nn.Dropout(rate=self.cfg.drop_rate)

    @nn.compact
    def __call__(self, x, training: bool):
        shortcut = x
        x = self.norm1(x)
        # x = self.att(x, training)
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.cfg.n_heads,
            qkv_features=self.cfg.emb_dim // self.cfg.n_heads,
            out_features=self.cfg.n_heads * (self.cfg.emb_dim // self.cfg.n_heads),
            dropout_rate=self.cfg.drop_rate,
        )(
            x,
            x,
            mask=jnp.tril(jnp.ones((x.shape[-2], x.shape[-2]))),
            deterministic=not training,
        )
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

        self.final_norm = LayerNorm(self.cfg.emb_dim)
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

    def get_total_params(self, variables):
        return sum(x.size for x in jax.tree_leaves(variables))


def test_gpt_model():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(jnp.array(tokenizer.encode(txt1)))
    batch.append(jnp.array(tokenizer.encode(txt2)))
    batch = jnp.stack(batch)
    model = GPTModel(cfg=GPTConfig())
    variables = model.init(jax.random.PRNGKey(0), batch, training=False)
    assert 163009536 == model.get_total_params(variables)


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
