import argparse
import os
import glob


def rename(filename):
    dir = os.path.dirname(filename)
    name = os.path.basename(filename)
    name, ext = os.path.splitext(name)
    num = name.split("_")[-1]
    return os.path.join(dir, f"prompt_{num}{ext}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dir", type=str, default="/scratch/uceeepi/calf/dataset/dataset-2/"
    )
    args = argparser.parse_args()

    for name in glob.glob(os.path.join(args.dir, "cohere-*")):
        name 
        os.rename(name, rename(name))
