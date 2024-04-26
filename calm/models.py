from __future__ import annotations
from typing import Dict, Tuple, Union

import flax.linen as nn
from jax import Array
import jax


class Network(nn.Module):
    """An encoder for the MiniHack environment."""

    encoder: nn.Module
    head: nn.Module
    recurrent: bool = True
    n_hidden: int = 512

    @nn.compact
    def __call__(
        self,
        x: Array | Tuple[Array, ...] | Dict[str, Array],
        hidden_state: Tuple[Array, Array] | None = None,
    ) -> Tuple[Union[Tuple[Array, Array], None], Array]:

        # apply the backbone
        y: Array = self.encoder(x)

        # apply the recurrent layer or a dense layer
        if self.recurrent:
            lstm = nn.OptimizedLSTMCell(self.n_hidden)
            if hidden_state is None:
                key_unused = jax.random.key(0)
                hidden_state = lstm.initialize_carry(key_unused, y.shape)
            hidden_state, y = nn.OptimizedLSTMCell(self.n_hidden)(hidden_state, y)
        else:
            y = nn.Dense(self.n_hidden)(x)
            y = nn.relu(y)

        # apply the head
        y = self.head(y)

        return hidden_state, y
