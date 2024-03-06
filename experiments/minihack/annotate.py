from __future__ import annotations

import argparse
from functools import partial
import time
import os
import pickle

from calf.annotate import (
    annotate_redistribution,
    annotate_subgoals,
)
from calf.io import load_pickle_minibatches
from calf.prompts import (
    PROMPT_REDISTRIBUTION,
    PROMPT_REDISTRIBUTION_ALT,
    PROMPT_IDENTIFY_SUBGOALS,
)


argparser = argparse.ArgumentParser()
argparser.add_argument("--max_new_tokens", type=int, default=1024)
argparser.add_argument("--url", type=str, default="http://localhost:5000/respond")
argparser.add_argument("--batch_size", type=int, default=4)
argparser.add_argument("--seq_len", type=int, default=20)
argparser.add_argument("--task", type=int, default=2)
argparser.add_argument(
    "--filepath", type=str, default="/scratch/uceeepi/calf/demonstrations_0.pkl"
)
args = argparser.parse_args()


def main():
    # args
    source_path = args.filepath
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    url = args.url
    seq_len = args.seq_len
    task_idx = args.task

    # static
    tasks = [
        PROMPT_REDISTRIBUTION,
        PROMPT_REDISTRIBUTION_ALT,
        PROMPT_IDENTIFY_SUBGOALS,
    ]
    annotation_fns = {
        PROMPT_REDISTRIBUTION: annotate_redistribution,
        PROMPT_REDISTRIBUTION_ALT: annotate_redistribution,
        PROMPT_IDENTIFY_SUBGOALS: partial(annotate_subgoals, seq_len=seq_len)
    }
    filenames = {
        PROMPT_REDISTRIBUTION: "redistribution",
        PROMPT_REDISTRIBUTION_ALT: "redistribution_alt",
        PROMPT_IDENTIFY_SUBGOALS: "subgoals",
    }

    # select task
    task = tasks[task_idx]
    annotate = annotation_fns[task]

    # prepare for saving
    dirname = os.path.dirname(source_path)
    out_dir = os.path.join(dirname, filenames[task])
    out_name = os.path.basename(source_path).replace("demonstrations", "annotations")
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        print(f"File {out_path} exists already")
        out_path += "2"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print("Annotations will be saved to {}".format(out_path))
    file = open(out_path, "ab")

    i = 0
    for demo_batch in load_pickle_minibatches(source_path, batch_size):
        if len(demo_batch) == 0:
            break
        demo_batch = [x[-seq_len:] for x in demo_batch]
        print(f"Annotating episodes {i * batch_size}-{(i + 1) * batch_size}")
        start_time = time.time()
        try:
            annotated = annotate(demo_batch, max_new_tokens=max_new_tokens, url=url)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(
                f"The LLM encountered an error. Skipping episodes {i * batch_size}-{(i + 1) * batch_size}"
            )
            print(repr(e))
            i += 1
            continue
        print(
            f"Saving annotations {i * batch_size}-{(i + 1) * batch_size} to {out_path}"
        )
        pickle.dump(annotated, file)
        i += 1
        print("Batch annotated in {}s".format(time.time() - start_time))
    file.close()


if __name__ == "__main__":
    main()
