from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from helx.base.modules import Flatten, Split, Merge, Parallel


def get_nethack_encoder() -> nn.Module:
    # define network
    glyphs_enc = nn.Sequential(
        [
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
    )

    chars_crop_env = nn.Sequential(
        [
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
    )

    blstats_enc = nn.Sequential(
        [
            nn.Dense(32),
            jax.nn.relu,
            nn.Dense(32),
            jax.nn.relu,
        ]
    )

    encoder = nn.Sequential(
        [
            Parallel((glyphs_enc, chars_crop_env, blstats_enc)),
            Merge(aggregate=partial(jnp.concatenate, axis=-1)),
            nn.Dense(512),
            jax.nn.relu,
            nn.Dense(512),
            jax.nn.relu,
        ]
    )

    return encoder
