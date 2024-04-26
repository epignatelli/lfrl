from __future__ import annotations

import argparse
from functools import partial
import time
import os
import pickle

import wandb

from calm.annotate import annotate_subgoals
from calm.io import load_pickle_minibatches

argparser = argparse.ArgumentParser()
argparser.add_argument("--max_new_tokens", type=int, default=1024)
argparser.add_argument("--url", type=str, default="http://localhost:5000/respond")
argparser.add_argument("--batch_size", type=int, default=3)
argparser.add_argument("--seq_len", type=int, default=3)
argparser.add_argument("--root", type=str, default="/scratch/uceeepi/calf/")
argparser.add_argument("--experiment", type=str, default="experiment_2")
argparser.add_argument("--out_name", type=str, default="ann_debug.pkl")
argparser.add_argument("--ablation", type=str, default="")
args = argparser.parse_args()


def main():
    # args
    root = args.root
    expeirment_name = args.experiment
    out_name = args.out_name
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    url = args.url
    seq_len = args.seq_len

    # load from
    source_path = os.path.join(root, expeirment_name, "demo.pkl")

    # save into
    dest_path = os.path.join(root, expeirment_name, out_name)
    print("Annotations will be saved to {}".format(dest_path))

    # overwrite
    if os.path.exists(dest_path):
        os.remove(dest_path)

    # create new
    file = open(dest_path, "ab")

    # log
    config = {
        **args.__dict__,
        **{"source_path": source_path, "dest_path": dest_path, "phase": "annotation"},
    }
    wandb.init(project="calf", config=config)

    # annotate
    i = 0
    annotation_size = 0
    table = wandb.Table(columns=["Prompts", "Response", "Parsed"])
    for demo_batch in load_pickle_minibatches(source_path, batch_size, seq_len):
        if len(demo_batch) == 0:
            break

        print(f"\nAnnotating episodes {i * batch_size}-{(i + 1) * batch_size}")
        start_time = time.time()
        try:
            annotations = annotate_subgoals(
                demo_batch, max_new_tokens=max_new_tokens, url=url
            )
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error in annotating batch {i * batch_size}-{(i + 1) * batch_size}")
            continue

        for ann in annotations:
            table.add_data(
                [str(ann.prompt)],
                [str(ann.response)],
                [str(ann.parsed).encode().decode("unicode_escape")],
            )
            annotation_size += len(ann.parsed.values())
            wandb.log({"annotate/annotations": table}, commit=False)
        print(
            f"Saving annotations {i * batch_size}-{(i + 1) * batch_size} to {dest_path}"
        )
        wandb.log({"annotate/annotation_size": annotation_size})
        pickle.dump(annotations, file)
        i += 1
        print("Batch annotated in {}s".format(time.time() - start_time))

    file.close()


if __name__ == "__main__":
    main()
