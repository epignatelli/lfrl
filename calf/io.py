import os
import pickle
from typing import Generator, List

from helx.base.mdp import Timestep


def get_next_valid_path(filepath: str, extension: str = "") -> str:
    if not os.path.exists(filepath):
        return filepath

    basedir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)

    if extension:
        ext = extension

    i = 0
    while True:
        candidate = f"{os.path.join(basedir, filename)}_{i}.{ext}"
        if not os.path.exists(candidate):
            break
        i += 1
    os.makedirs(basedir, exist_ok=True)
    return candidate


def load_pickle_stream(filepath: str):
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
    filename: str, batch_size: int
) -> Generator[List[Timestep], None, None]:
    with open(filename, "rb") as f:
        remainder = []  # Store segment remainders in memory

        while True:
            batch = remainder  # Start with any existing remainder
            current_batch_size = len(batch)

            while current_batch_size < batch_size:
                try:
                    segment = pickle.load(f)
                except EOFError:
                    break  # End of file

                if current_batch_size + len(segment) > batch_size:
                    batch.extend(segment[: batch_size - current_batch_size])
                    remainder = segment[
                        batch_size - current_batch_size :
                    ]  # Store remainder in memory
                    break
                else:
                    batch.extend(segment)
                    current_batch_size += len(segment)

            if batch:
                yield batch
                remainder = []  # Reset remainder after yielding
            else:
                break
