from __future__ import annotations

import argparse
import time
import os
import pickle

import jax.tree_util as jtu
import jax.numpy as jnp

from calf.annotate import query_llm
from calf.prompts import (
    PROMPT_REDISTRIBUTION,
    PROMPT_REDISTRIBUTION_ALT,
    PROMPT_IDENTIFY_SUBGOALS,
)


argparser = argparse.ArgumentParser()
argparser.add_argument("--max_new_tokens", type=int, default=512)
argparser.add_argument("--url", type=str, default="http://localhost:5000/respond")
argparser.add_argument("--batch_size", type=int, default=16)
argparser.add_argument("--filepath", type=str, default="./demonstrations.pkl")
args = argparser.parse_args()


def main():
    tasks = [
        PROMPT_REDISTRIBUTION,
        PROMPT_REDISTRIBUTION_ALT,
        PROMPT_IDENTIFY_SUBGOALS,
    ]
    filenames = {
        PROMPT_REDISTRIBUTION: "redistribution",
        PROMPT_REDISTRIBUTION_ALT: "redistribution_alt",
        PROMPT_IDENTIFY_SUBGOALS: "subgoals",
    }

    filepath = args.filepath
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    url = args.url

    with open(filepath, "rb") as f:
        demonstrations = pickle.load(f)
        # demonstrations = jtu.tree_map(lambda *x: jnp.stack(x, axis=0), *demonstrations)

    task = tasks[1]
    annotated = []
    for i in range(0, len(demonstrations), batch_size):
        print("Annotating batch", i)
        start_time = time.time()
        start = i
        end = min(i + batch_size, len(demonstrations))
        demo = demonstrations[start:end]
        # get LLM rewards
        llm_output = query_llm(
            demo, url=url, max_new_tokens=max_new_tokens, llm_task_prompt=task
        )
        # replace demo.reward with LLM rewards
        if llm_output is not None:
            llm_rewards, valid = llm_output
            for idx in valid:
                annotated.append(demo[idx].replace(reward=llm_rewards[idx]))
        print("Batch annotated in {}s".format(time.time() - start_time))

    # pack into pytree
    annotated = jtu.tree_map(lambda *x: jnp.concatenate(x, axis=0), annotated)
    # save to disk
    filename = filenames[task]
    filepath = os.path.join(f"annotations/{filename}.pkl")
    print("Saving annotations {} to disk...".format(filename))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(annotated, f)


if __name__ == "__main__":
    main()
