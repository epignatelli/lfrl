import os
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )

# os.environ.update({
#     "NCCL_LL128_BUFFSIZE": "-2",
#     "NCCL_LL_BUFFSIZE": "-2",
#     "NCCL_PROTO": "SIMPLE,LL,LL128",
# })

import jax
import jax.numpy as jnp


def mpmd_test():
    def compute_1(x):
        x = x[:10_000, :10_000]
        x = jnp.pad(x, ((0, 10_000), (0, 10_000)))
        x = jnp.matmul(x, x.T)
        return x

    def compute_2(x):
        x = x[:10_000, :10_000]
        return jnp.matmul(x, x.T)

    devices = jax.devices("cuda")
    f1 = jax.jit(compute_1, device=devices[0])
    f2 = jax.jit(compute_2, device=devices[1])

    def combine(x):
        x = f1(x)
        x = f2(x)
        return x

    x = jnp.ones((10_000, 10_000))
    while True:
        x = f1(x)
        x = f2(x)
    # jax.lax.while_loop(lambda x: jnp.asarray(True), combine, jnp.ones((10_000, 10_000)))


if __name__ == "__main__":
    mpmd_test()