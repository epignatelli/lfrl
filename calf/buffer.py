from __future__ import annotations

from typing import Generic, List,TypeVar
from flax import struct

import jax
from jax.random import KeyArray
import jax.numpy as jnp

from helx.base.mdp import Timestep


class RingBuffer(struct.PyTreeNode):
    """A circular buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    Use the `CircularBuffer.init` method to construct a buffer."""

    capacity: int
    """Returns the capacity of the buffer."""
    elements: Timestep
    """The elements currently stored in the buffer."""
    idx: jax.Array
    """The index of the next element to be added to the buffer."""

    @classmethod
    def create(cls, element: Timestep, capacity: int, n_steps: int=1) -> RingBuffer:
        """Constructs a RingBuffer class."""
        # reserve memory
        uninitialised_elements = jax.tree_map(
            lambda x: jnp.broadcast_to(
                jnp.asarray(x * 0, dtype=x.dtype),
                (capacity, n_steps + 1, *jnp.asarray(x).shape),
            ),
            element,
        )
        return cls(
            capacity=capacity,
            elements=uninitialised_elements,
            idx=jnp.asarray(0),
        )

    def __len__(self) -> int:
        return self.idx  # type: ignore

    def add(self, item: Timestep) -> RingBuffer:
        """Adds a single element to the buffer. If the buffer is full,
        the oldest element is overwritten."""
        idx = self.idx % self.capacity
        elements = jax.tree_map(lambda x, y: x.at[idx].set(y), self.elements, item)
        return self.replace(
            idx=self.idx + 1,
            elements=elements,
        )

    def sample(self, key: KeyArray, n: int = 1) -> Timestep:
        """Samples `n` elements uniformly at random from the buffer,
        and stacks them into a single pytree.
        If `n` is greater than state.idx,
        the function returns uninitialised elements"""
        k1, k2 = jax.random.split(key)
        seq_len = self.elements.t.shape[-1]
        batch_idx = jax.random.randint(key=k1, shape=(n,), minval=0, maxval=self.idx)
        time_idx = jax.random.randint(key=k2, shape=(n,), minval=0, maxval=seq_len)
        items = jax.tree_map(lambda x: x[batch_idx, time_idx], self.elements)
        return items
