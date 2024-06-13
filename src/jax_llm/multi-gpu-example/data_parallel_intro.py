import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


a = jnp.arange(8)
print("Array", a)
print("Device", a.devices())
print("Sharding", a.sharding)

mesh = Mesh(np.array(jax.devices()), ("i",))
print(mesh)

sharding = NamedSharding(
    mesh,
    P("i"),
)

print("Sharding", sharding)

a_sharded = jax.device_put(a, sharding)
print("Sharded array", a_sharded)
print("Device", a_sharded.devices())
print("Sharding", a_sharded.sharding)

jax.debug.visualize_array_sharding(a_sharded)

out = nn.tanh(a_sharded)
print("Output array", out)
jax.debug.visualize_array_sharding(out)


mesh = Mesh(np.array(jax.devices()).reshape(1, 2), ("i", "j"))
mesh

batch_size = 192
input_dim = 64
output_dim = 128
x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))
w = jax.random.normal(jax.random.PRNGKey(1), (input_dim, output_dim))
b = jax.random.normal(jax.random.PRNGKey(2), (output_dim,))

x_sharded = jax.device_put(x, NamedSharding(mesh, P("i", None)))
w_sharded = jax.device_put(w, NamedSharding(mesh, P(None, "j")))
b_sharded = jax.device_put(b, NamedSharding(mesh, P("j")))

out = jnp.dot(x_sharded, w_sharded) + b_sharded
print("Output shape", out.shape)
jax.debug.visualize_array_sharding(out)


def matmul_fn(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    print("Local x shape", x.shape)
    print("Local w shape", w.shape)
    print("Local b shape", b.shape)
    return jnp.dot(x, w) + b


# input specs correspond to the sharding pattern of the input arguments
# output specs correspond to the sharding pattern of the output
matmul_sharded = shard_map(
    matmul_fn,
    mesh,
    in_specs=(P("i", None), P(None, "j"), P("j")),
    out_specs=P("i", "j"),
)

y = matmul_sharded(x_sharded, w_sharded, b_sharded)
print("Output shape", y.shape)
jax.debug.visualize_array_sharding(y)


@functools.partial(shard_map, mesh=mesh, in_specs=P("i", "j"), out_specs=P("i", "j"))
def parallel_normalize(x: jax.Array) -> jax.Array:
    mean = jax.lax.pmean(x, axis_name="j")
    std = jax.lax.pmean((x - mean) ** 2, axis_name="j") ** 0.5
    return (x - mean) / std


out = parallel_normalize(x)
out = jax.device_get(out)
print("Mean", out.mean())
print("Std", out.std())
