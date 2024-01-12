from __future__ import annotations

from typing import Generic,TypeVar
from flax import struct

import jax
from jax.random import KeyArray
import jax.numpy as jnp


T = TypeVar("T")


class RingBuffer(struct.PyTreeNode, Generic[T]):
    """A circular buffer used for Experience Replay (ER):
    Li, L., 1993, https://apps.dtic.mil/sti/pdfs/ADA261434.pdf.
    Use the `CircularBuffer.init` method to construct a buffer."""

    capacity: int = struct.field(pytree_node=False)
    """Returns the capacity of the buffer."""
    elements: T = struct.field(pytree_node=True)
    """The elements currently stored in the buffer."""
    idx: jax.Array = struct.field(pytree_node=True)
    """The index of the next element to be added to the buffer."""

    @classmethod
    def create(cls, element: T, capacity: int, n_steps: int=1) -> RingBuffer:
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

    def size(self) -> jax.Array:
        """Returns the number of elements currently stored in the buffer."""
        return self.idx

    def add(self, item: T) -> RingBuffer:
        """Adds a single element to the buffer. If the buffer is full,
        the oldest element is overwritten."""
        idx = self.idx % self.capacity
        elements = jax.tree_map(lambda x, y: x.at[idx].set(y), self.elements, item)
        return self.replace(
            idx=self.idx + 1,
            elements=elements,
        )

    def sample(self, key: KeyArray, n: int = 1) -> T:
        """Samples `n` elements uniformly at random from the buffer,
        and stacks them into a single pytree.
        If `n` is greater than state.idx,
        the function returns uninitialised elements"""
        indices = jax.random.randint(key=key, shape=(n,), minval=0, maxval=self.idx)
        items = jax.tree_map(lambda x: x[indices], self.elements)
        return items
