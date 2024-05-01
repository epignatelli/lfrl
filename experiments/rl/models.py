from __future__ import annotations
from functools import partial
from typing import Tuple

import jax
from jax import Array
import jax.numpy as jnp
import flax.linen as nn
from helx.base.modules import Flatten, Merge, Lambda


class NetHackEncoder(nn.Module):
    @nn.compact
    def __call__(self, x: Tuple[Array, Array, Array]) -> Array:
        # format inputs into channel-last image format
        glyphs, chars_crop, blstats = x

        chars_embedding = nn.Sequential(
            [
                Lambda(partial(jnp.expand_dims, axis=-1)),
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                Flatten(),
            ]
        )(chars_crop)
        # return chars_embedding
    
        glyphs_embedding = nn.Sequential(
            [
                Lambda(partial(jnp.expand_dims, axis=-1)),
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                nn.Conv(16, kernel_size=(3, 3), strides=1, padding=1),
                nn.elu,
                Flatten(),
            ]
        )(glyphs)

        blstats_embedding = nn.Sequential(
            [
                nn.Dense(32),
                jax.nn.relu,
                nn.Dense(32),
                jax.nn.relu,
            ]
        )(blstats)

        embedding = nn.Sequential(
            [
                Merge(aggregate=partial(jnp.concatenate, axis=-1)),
                nn.Dense(512),
                jax.nn.relu,
                nn.Dense(512),
                jax.nn.relu,
            ]
        )(glyphs_embedding, chars_embedding, blstats_embedding)

        return embedding
