from __future__ import annotations

import argparse
import os
import re

from calm.annotate import batch_query_llm
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
    REMARK_MESSAGE_ACTION,
    REMARK_ACTION,
    REMARK_MESSAGE,
    REMARK_NONE,
    OUTPUT_FORMAT_DIC,
    OUTPUT_FORMAT_GENERIC,
    INPUT_TRAJ,
    OUTPUT_REMARK
)


def select_prompt(
    ablate_role,
    ablate_symset,
    ablate_message,
    ablate_action,
    task_type,
    instruction_type,
    subgoals_type,
    remark_type,
    output_type
):
    role = ROLE_ANALYST
    symset = SYMSET_KEYROOM
    remark = REMARK_MESSAGE_ACTION
    output_format = OUTPUT_FORMAT_DIC
    trajectory = INPUT_TRAJ

    task_type = task_type.lower()
    if ablate_role:
        role = ROLE_GENERIC
    if ablate_symset:
        symset = SYMSET_KEYROOM_EXPLAINED

    if task_type == "keyroom":
        task_prompt = TASK_KEYROOM
    elif task_type == "win":
        task_prompt = TASK_WIN
    elif task_type == "manual":
        task_prompt = TASK_KEYROOM_MANUAL
    else:
        raise ValueError(f"Unknown task_type {task_type}")

    if subgoals_type == "identify":
        subgoals = SUBGOALS_IDENTIFY
    elif subgoals_type == "preset":
        subgoals = SUBGOALS_PRESET
    else:
        raise ValueError(f"Unknown subgoals_type {subgoals_type}")

    if instruction_type == "identify":
        instructions = INSTRUCTION_IDENTIFY
    elif instruction_type == "progress":
        instructions = INSTRUCTION_PROGRESS
    elif instruction_type == "optimality":
        instructions = INSTRUCTION_OPTIMALITY
    else:
        raise ValueError(f"Unknown instruction_type {instruction_type}")

    if remark_type == "none":
        remark = REMARK_NONE
    if ablate_action and not ablate_message:
        remark = REMARK_ACTION
    if ablate_message and not ablate_action:
        remark = REMARK_MESSAGE
    if ablate_message and ablate_action:
        remark = ""

    if output_type == "dic":
        output_format = OUTPUT_FORMAT_DIC
    elif output_type == "generic":
        output_format = OUTPUT_FORMAT_GENERIC
    else:
        raise ValueError(f"Unknown output_format {output_type}")

    return prompt_subgoals(
        role=role,
        symset=symset,
        task=task_prompt,
        subgoals=subgoals,
        instructions=instructions,
        remark=remark,
        output_format=output_format,
        trajectory=trajectory,
        output_remark=OUTPUT_REMARK
    )


def parse_response(response: str) -> dict:
    pattern = r"```(python)?(.*)({.*})"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            raise
        dict_str = match.group(3)
        response_parsed = eval(dict_str)
        return response_parsed
    except:
        return {}


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


def main(argv):
    # argv
    source_dir = argv.source_dir
    model_name = argv.model_name
    ablation = argv.ablation

    ablate_role = argv.ablate_role
    ablate_symset = argv.ablate_symset
    ablate_action = argv.ablate_action
    ablate_message = argv.ablate_message
    ablate_observation = argv.ablate_observation
    ablate_tokenisation = argv.ablate_tokenisation
    gamescreen = argv.gamescreen

    task = argv.task
    subgoals = argv.subgoals
    instruction = argv.instruction
    remark = argv.remark
    output = argv.output

    instructions = select_prompt(
        ablate_role=ablate_role,
        ablate_symset=ablate_symset,
        ablate_message=ablate_message,
        ablate_action=ablate_action,
        task_type=task,
        instruction_type=instruction,
        subgoals_type=subgoals,
        remark_type=remark,
        output_type=output
    )

    traj_dir = os.path.join(source_dir, "trajectories")
    ablation = f"-{ablation}" if ablation != "" else ""
    ann_dest_dir = os.path.join(source_dir, model_name + ablation)
    os.makedirs(ann_dest_dir, exist_ok=True)

    print(instructions)
    n_prompts = len(os.listdir(traj_dir))
    for i in range(n_prompts):
        # get a demonstration
        traj = load_traj(
            os.path.join(traj_dir, f"traj_{i}.txt"),
            ablate_action=ablate_action,
            ablate_message=ablate_message,
            ablate_observation=ablate_observation,
            ablate_smart_tokenisation=ablate_tokenisation,
            gamescreen=gamescreen
        )
        # print(traj)

        prompt = instructions.format(traj)

        print(f"Annotating prompt {i}...\t", end="")
        # annotate
        response = batch_query_llm([prompt], 1024)[0]

        print(f"Writing annotation {i} to file...\t", end="")
        with open(os.path.join(ann_dest_dir, f"prompt_{i}.txt"), "w") as file:
            file.write(response)
        print("Done")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    # ablations
    argparser.add_argument("--ablate_role", default=False, action="store_true")
    argparser.add_argument("--ablate_symset", default=False, action="store_true")
    argparser.add_argument("--ablate_action", default=False, action="store_true")
    argparser.add_argument("--ablate_message", default=False, action="store_true")
    argparser.add_argument("--ablate_observation", default=False, action="store_true")
    argparser.add_argument("--ablate_tokenisation", default=False, action="store_true")
    argparser.add_argument("--gamescreen", default=False, action="store_true")

    # modes
    argparser.add_argument("--task", type=str, default="keyroom")
    argparser.add_argument("--subgoals", type=str, default="identify")
    argparser.add_argument("--instruction", type=str, default="identify")
    argparser.add_argument("--remark", type=str, default="")
    argparser.add_argument("--output", type=str, default="dic")

    # loading/saving
    argparser.add_argument("--ablation", type=str, default="")
    argparser.add_argument("--model_name", type=str, default="gemma-7B-it")
    argparser.add_argument(
        "--source_dir", type=str, default="/scratch/uceeepi/calf/dataset"
    )
    args = argparser.parse_args()

    main(args)
