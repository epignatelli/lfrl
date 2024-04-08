import argparse
import os
import glob


def extract_trajectory(filename):
    with open(filename, "r") as file:
        prompt = file.read()

    traj = prompt.split("Observation Sequence:")[1]
    return traj



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dir", type=str, default="/scratch/uceeepi/calf/dataset"
    )
    args = argparser.parse_args()

    traj_dir = os.path.join(args.dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    prompts_dir = os.path.join(args.dir, "prompts")
    n_prompts = len(os.listdir(prompts_dir))
    for i in range(n_prompts):
        name = os.path.join(prompts_dir, f"prompt_{i}.txt")
        print(f"Extracting trajectory from {name}")

        traj = extract_trajectory(name)

        with open(os.path.join(traj_dir, f"traj_{i}.txt"), "w") as file:
            file.write(traj)


