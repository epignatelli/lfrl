from __future__ import annotations

from flax import struct

import jax
from jax import Array
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
    def init(cls, element: Timestep, capacity: int) -> RingBuffer:
        """Constructs a RingBuffer class."""
        # reserve memory
        uninitialised_elements = jax.tree_map(
            lambda x: jnp.zeros((capacity, *x.shape), dtype=x.dtype),
            element
        )
        return cls(
            capacity=capacity,
            elements=uninitialised_elements,
            idx=jnp.asarray(0),
        )

    def __getitem__(self, idx) -> Timestep:
        return self.elements[idx]

    def length(self) -> Array:
        return jnp.asarray(self.idx, dtype=jnp.int32)

    def add(self, timestep: Timestep, axis=None) -> RingBuffer:
        """Adds a single element to the buffer. If the buffer is full,
        the oldest element is overwritten."""
        if axis is None:
            if timestep.t.ndim == 1 or timestep.t.ndim == 0:
                n_items = 1
            else:
                # assume batch axis is the first axis
                n_items = timestep.t.shape[0]
        else:
            n_items = timestep.t.shape[axis]
        idx = (self.idx % self.capacity) + jnp.arange(n_items)
        elements = jax.tree_map(lambda x, y: x.at[idx].set(y), self.elements, timestep)
        return self.replace(
            idx=self.idx + n_items,
            elements=elements,
        )
