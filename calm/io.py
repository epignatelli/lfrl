import os
import pickle
from typing import Generator, List

import jax.numpy as jnp
from helx.base.mdp import Timestep

from .annotate import Annotation


def split(episode: Timestep, size: int) -> List[Timestep]:
    """Splits an array into a List of arrays including the last remaining items
    whose chunk length could be less than size"""
    # mask
    episode = episode[episode.info["mask"] != jnp.asarray(0)]
    if len(episode) <= size:
        return [episode]
    upper = (len(episode) // size + 1) * size + 1
    indices = jnp.arange(0, upper, size)
    return [episode[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]


def make_transitions(episode: Timestep) -> List[Timestep]:
    transitions = []
    for i in range(len(episode.t) - 2):
        transition = episode[i : i + 3]
        if transition.info["mask"].sum() > 0:
            transitions.append(transition)
    return transitions


def get_next_valid_path(filepath: str, extension: str = "") -> str:
    basedir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)

    if extension:
        if extension[0] != ".":
            extension = "." + extension
        ext = extension

    i = 0
    while True:
        candidate = f"{os.path.join(basedir, name)}_{i}{ext}"
        if not os.path.exists(candidate):
            break
        i += 1
    os.makedirs(basedir, exist_ok=True)
    return candidate


def load_pickle_stream(filepath: str) -> List[Annotation]:
    with open(filepath, "rb") as file:
        content = []
        while True:
            try:
                segment = pickle.load(file)
                content.extend(segment)
            except EOFError:
                break  # End of file
    return content


def load_pickle_minibatches(
    filename: str, batch_size: int, seq_len: int
) -> Generator[List[Timestep], None, None]:
    with open(filename, "rb") as f:
        queue = []

        while True:
            while len(queue) < batch_size:
                try:
                    segment = pickle.load(f)
                except EOFError:
                    break  # End of file

                for episode in segment:
                    try:
                        chunks = split(episode, seq_len)
                        queue.extend(chunks)
                    except:
                        print("Corrupted chunk. skipping.")
                        continue

            batch = []
            if len(queue) >= batch_size:
                for _ in range(batch_size):
                    batch.append(queue.pop(0))
                yield batch
            else:
                break
