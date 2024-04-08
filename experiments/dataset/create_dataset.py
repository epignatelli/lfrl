from __future__ import annotations

import argparse
import os
import json
import random
from typing import Tuple

from helx.base.mdp import Timestep

from calm.decode import decode_message, decode_action, decode_observation
from calm.annotate import batch_compose_prompts
from calm.io import load_pickle_minibatches
from calm.prompts import (
    prompt_subgoals,
    ROLE_ANALYST,
    ROLE_GENERIC,
    SYMSET_KEYROOM,
    SYMSET_KEYROOM_EXPLAINED,
    TASK_KEYROOM,
    TASK_KEYROOM_MANUAL,
    TASK_WIN,
    SUBGOALS_IDENTIFY,
    SUBGOALS_PRESET,
    INSTRUCTION_IDENTIFY,
    INSTRUCTION_PROGRESS,
    INSTRUCTION_OPTIMALITY,
    INSTRUCTION_TRANSITION,
    REMARK_MESSAGE_ACTION,
    REMARK_ACTION,
    REMARK_MESSAGE,
    REMARK_NONE,
    OUTPUT_FORMAT_DIC,
    OUTPUT_FORMAT_GENERIC,
    OUTPUT_FORMAT_TRANSITION,
    INPUT_TRAJ,
    OUTPUT_REMARK,
)


def select_prompt(
    role_type,
    symset_type,
    task_type,
    subgoals_type,
    instruction_type,
    remark_type,
    output_type,
    ablate_action,
    ablate_message,
):
    role_type = role_type.lower()
    if role_type == "generic":
        role = ROLE_GENERIC
    elif role_type == "analyst":
        role = ROLE_ANALYST
    else:
        raise ValueError(f"Unknown role_type {role_type}")

    symset_type = symset_type.lower()
    if symset_type == "keyroom":
        symset = SYMSET_KEYROOM
    elif symset_type == "keyroom_explained":
        symset = SYMSET_KEYROOM_EXPLAINED
    else:
        raise ValueError(f"Unknown symset_type {symset_type}")

    task_type = task_type.lower()
    if task_type == "keyroom":
        task_prompt = TASK_KEYROOM
    elif task_type == "win":
        task_prompt = TASK_WIN
    elif task_type == "manual":
        task_prompt = TASK_KEYROOM_MANUAL
    else:
        raise ValueError(f"Unknown task_type {task_type}")

    subgoals_type = subgoals_type.lower()
    if subgoals_type == "identify":
        subgoals = SUBGOALS_IDENTIFY
    elif subgoals_type == "preset":
        subgoals = SUBGOALS_PRESET
    else:
        raise ValueError(f"Unknown subgoals_type {subgoals_type}")

    instruction_type = instruction_type.lower()
    if instruction_type == "identify":
        instructions = INSTRUCTION_IDENTIFY
    elif instruction_type == "progress":
        instructions = INSTRUCTION_PROGRESS
    elif instruction_type == "optimality":
        instructions = INSTRUCTION_OPTIMALITY
    elif instruction_type == "transition":
        instructions = INSTRUCTION_TRANSITION
    else:
        raise ValueError(f"Unknown instruction_type {instruction_type}")

    remark_type = remark_type.lower()
    if remark_type == "none":
        remark = REMARK_NONE
    elif remark_type == "action":
        remark = REMARK_ACTION
    elif remark_type == "message":
        remark = REMARK_MESSAGE
    elif remark_type == "message_action" or remark_type == "action_message":
        remark = REMARK_MESSAGE_ACTION
    else:
        if ablate_action and not ablate_message:
            remark = REMARK_ACTION
        elif ablate_message and not ablate_action:
            remark = REMARK_MESSAGE
        else:
            remark = REMARK_NONE

    output_type = output_type.lower()
    if output_type == "dic":
        output_format = OUTPUT_FORMAT_DIC
    elif output_type == "generic":
        output_format = OUTPUT_FORMAT_GENERIC
    elif output_type == "transition":
        output_format = OUTPUT_FORMAT_TRANSITION
    else:
        raise ValueError(f"Unknown output_type {output_type}")

    return prompt_subgoals(
        role=role,
        symset=symset,
        task=task_prompt,
        subgoals=subgoals,
        instructions=instructions,
        remark=remark,
        output_format=output_format,
        trajectory=INPUT_TRAJ,
        output_remark=OUTPUT_REMARK,
    )


def load_traj(
    filepath,
    ablate_action=False,
    ablate_observation=False,
    ablate_message=False,
    ablate_smart_tokenisation=False,
    gamescreen=False,
) -> str:
    content = ""
    with open(filepath, "r") as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if ablate_action and (
            line.startswith("Last action:") or line.startswith("Next action:")
        ):
            i += 1
            continue

        if gamescreen and line.startswith("Current observation:"):
            i += 1
            continue
        if gamescreen and line.startswith("Current message:"):
            line = line.split(": ")[-1]

        if ablate_observation and line.startswith("Current observation:"):
            i += 9
            continue

        if ablate_smart_tokenisation and line.startswith("Current observation:"):
            i += 1
            for j in range(8):
                line = lines[i + j].split("\n")[0]
                line = line[::2]
                content += line + "\n"
            i += j + 1
        if ablate_message and line.startswith("Current message:"):
            i += 1
            continue
        content += line
        i += 1
    return content


def why_rewarding(demo: Timestep) -> str:
    # check if key is picked up
    action = decode_action(int(demo.action[0]))
    key_picked_msg = " - a key named The Master Key of Thievery"
    
    if action == "pickup" and key_picked_msg in decode_message(
        demo.info["observations"]["message"][1]
    ):
        return "key"

    # check if door has been unlocked
    if action == "apply":
        obs_t = decode_observation(demo.observation[0])
        obs_tp1 = decode_observation(demo.observation[1])
        if obs_t.count("+") == obs_tp1.count("+") + 1:
            return "door"
    return "non-rewarding"


def main(argv):
    # argv
    source_path = argv.source_path
    dest_dir = argv.dest_dir
    seq_len = argv.seq_len
    num_annotations = argv.num_annotations
    ablation = argv.ablation

    role = argv.role
    symset = argv.symset
    task = argv.task
    subgoals = argv.subgoals
    instruction = argv.instruction
    token_separator = argv.token_separator
    remark = argv.remark
    ablate_action = argv.ablate_action
    ablate_message = argv.ablate_message
    tty = argv.tty
    output = argv.output

    ablation = f"-{ablation}" if ablation != "" else ""
    dest_dir = os.path.join(dest_dir, "prompts" + ablation)
    os.makedirs(dest_dir, exist_ok=True)

    random.seed(argv.seed)

    config = argv.__dict__
    with open(os.path.join(dest_dir, "config.json"), "w") as file:
        json.dump(config, file)
    
    # read indices
    indices_path = os.path.join(dest_dir, "demo_indices.txt")
    if os.path.exists(indices_path):
        with open(indices_path, "r") as file:
            indices = file.readlines()
    else:
        indices = None

    bins = {
        "key": 0,
        "door": 0,
        "non-rewarding": 0
    }
    i = 0
    loader = load_pickle_minibatches(source_path, 1, seq_len)
    while i < num_annotations:
        # get a demonstration
        demo_batch = next(loader)

        if indices is None:
            # skip with 80% probability
            if random.random() > 0.2:
                continue

            if len(demo_batch[0].t) < seq_len:
                continue
        else:
            if str(i) not in indices:
                continue
        
        print(bins)

        # balance rewarding and non-rewarding prompts
        why = why_rewarding(demo_batch[0])
        # if bin is fill, continue
        if bins[why] >= num_annotations // len(bins):
            print("Full bin for", why, f"{bins[why]}/{num_annotations // len(bins)}")
            continue
        bins[why] += 1

        # append index to indices file
        with open(indices_path, "a") as file:
            file.write(f"{i}\n")

        print(f"Writing prompt {i} to {dest_dir}...\t", end="")
        with open(os.path.join(dest_dir, f"prompt_{i}.txt"), "w") as file:
            # save prompt
            instr = select_prompt(
                role_type=role,
                symset_type=symset,
                task_type=task,
                subgoals_type=subgoals,
                instruction_type=instruction,
                remark_type=remark,
                output_type=output,
                ablate_action=ablate_action,
                ablate_message=ablate_message,
            )
            prompt = batch_compose_prompts(
                episodes=demo_batch,
                prompt=instr,
                ablate_action=ablate_action,
                ablate_message=ablate_message,
                token_separator=token_separator,
                tty=tty,
            )
            file.write(prompt[0])
        print("Done")
        i += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--seq_len", type=int, default=20)
    argparser.add_argument("--num_annotations", type=int, default=10)

    # paths
    argparser.add_argument(
        "--source_path",
        type=str,
        default="/scratch/uceeepi/calf/demonstrations/demo_2.pkl",
    )
    argparser.add_argument(
        "--dest_dir", type=str, default="/scratch/uceeepi/calf/dataset/"
    )
    argparser.add_argument("--ablation", type=str, default="")

    # prompt options
    argparser.add_argument(
        "--role", type=str, default="generic", help="(generic, analyst)"
    )
    argparser.add_argument(
        "--symset", type=str, default="keyroom", help="(keyroom, keyroom_explained)"
    )
    argparser.add_argument(
        "--task", type=str, default="keyroom", help="(keyroom, manual, win)"
    )
    argparser.add_argument(
        "--subgoals", type=str, default="identify", help="(identify, preset)"
    )
    argparser.add_argument(
        "--instruction",
        type=str,
        default="identify",
        help="(identify, progress, optimality)",
    )
    argparser.add_argument(
        "--remark",
        type=str,
        default="none",
        help="(none, action, message, message_action)",
    )
    argparser.add_argument(
        "--ablate_action",
        default=False,
        action="store_true",
    )
    argparser.add_argument(
        "--ablate_message",
        default=False,
        action="store_true",
    )
    argparser.add_argument(
        "--token_separator",
        type=str,
        default=" ",
        help="character to separate obs glyphs",
    )
    argparser.add_argument(
        "--tty",
        default=False,
        action="store_true",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="generic",
        help="(generic, dic)",
    )

    args = argparser.parse_args()
    main(args)
