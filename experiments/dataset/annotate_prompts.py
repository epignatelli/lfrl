from __future__ import annotations

import argparse
import os
import glob

from calm.annotate import batch_query_llm_with_name


def main(argv):
    # argv
    prompts_dir = argv.prompts_dir

    root = os.path.normpath(os.path.join(prompts_dir, ".."))

    ablation = os.path.basename(os.path.normpath(prompts_dir)).split("prompts")[-1]

    for filename in glob.glob(os.path.join(prompts_dir, "*.txt")):
        filename = os.path.basename(filename)
        with open(os.path.join(prompts_dir, filename), "r") as file:
            prompt = file.read()

        print(f"Annotating prompt {filename}...\t", end="")
        # annotate
        responses, model_name = batch_query_llm_with_name([prompt], 1024)
        response = responses[0]
        model_name = model_name.split("/")[-1]

        out_path = os.path.join(root, model_name + ablation, filename)
        if not os.path.exists(os.path.dirname(out_path)):
            os.mkdir(os.path.dirname(out_path))
        print(f"Writing annotation to {out_path}...\t", end="")
        with open(out_path, "w") as file:
            file.write(response)
        print("Done")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--prompts_dir",
        type=str,
        default="/scratch/uceeepi/calf/dataset/dataset-3/prompts-base/",
    )
    args = argparser.parse_args()

    main(args)
