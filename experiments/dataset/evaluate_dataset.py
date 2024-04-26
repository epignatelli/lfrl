from __future__ import annotations

import argparse
from functools import partial
import os
import re
import glob

import numpy as np
import pandas as pd


def parse_response(response: str, strict: bool = False) -> dict:
    if strict:
        pattern = r"```(python)?(.*)({.*})"
    else:
        pattern = r"({.*})"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            raise
        if strict:
            dict_str = match.group(3)
        else:
            dict_str = match.group(1)
        response_parsed = eval(dict_str)
        if isinstance(response_parsed, dict):
            return response_parsed
        else:
            return {}
    except:
        # print(f"Invalid response {response}.\nDiscarding prompt")
        return {}


def postprocess_response(response):
    goals_dic = parse_response(response)

    times = []
    reasons = []
    for reason, v in goals_dic.items():
        if v is None:
            continue
        # identified a single timestep
        elif isinstance(v, int):
            times.append(v)
            reasons.append(reason)
        elif isinstance(v, float):
            times.append(int(v))
            reasons.append(reason)
        # identified a list of timesteps for each subgoal
        elif isinstance(v, (list, tuple)):
            try:
                times.extend(list(map(int, v)))
                reasons.extend([reason] * len(v))
            except:
                continue
        # llm inverted keys and values
        elif isinstance(v, str):
            v, reason = reason, v
            try:
                times.append(int(v))
                reasons.append(str(reason))
            except:
                continue
        else:
            print("Cannot extract information from {}".format(goals_dic))
    return dict(zip(times, reasons))


def evaluate(pred, truth):
    tp = 0
    fp = 0
    fn = 0
    # true positive
    for time in pred:
        # correct, TP
        if time in truth:
            tp += 1
        # wrong, FP
        if time not in truth:
            fp += 1
    for time in truth:
        # missed, FN
        if time not in pred:
            fn += 1
    n = len(set(list(pred) + list(truth)))
    return tp, fp, fn, n


def evaluate_transitions(pred, truth):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # we only check if the LLM flagged the transition as rewarding
    # we do not check the reason
    # (maybe LLM flagged because pickup but the true goal was unlock)
    pred_detected = any(pred.keys())
    truth_detected = any(truth.keys())

    if pred_detected and truth_detected:
        tp += 1
    elif pred_detected and not truth_detected:
        fp += 1
    elif not pred_detected and truth_detected:
        fn += 1
    elif not pred_detected and not truth_detected:
        tn += 1
    return tp, fp, fn, tn


def _highligh_n_best(
    data: pd.DataFrame | pd.Series, op: str, props: str, n: int = 2
) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(n)
    # if isinstance(data, pd.DataFrame):  # min/max must be done twice to return scalar
    # value = getattr(value, op)(n, skipna=True)
    if op == "nlargest":
        cond = data >= value[-1]
    elif op == "nsmallest":
        cond = data <= value[-1]
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, "")


def main(argv):
    # argv
    source_dir = argv.source_dir
    sort = argv.sort
    evaluation = argv.evaluation
    latex = argv.latex

    pattern = re.compile(f"^(?!.*({argv.exclude})).*({argv.match})")
    evaluate_folders = [x for x in os.listdir(source_dir) if re.search(pattern, x)]
    evaluate_folders += glob.glob(os.path.join(source_dir, "human-*"))

    # iterate over each annotation agent
    results = {}
    for folder in evaluate_folders:
        ann_folder = os.path.join(source_dir, folder)
        truth_folder = os.path.join(source_dir, "human-0")
        optimal_folder = os.path.join(source_dir, "optimal")
        tp = fp = fn = tn = 0
        for filepath in glob.glob(os.path.join(ann_folder, "*.txt")):
            filename = os.path.basename(filepath)
            with open(os.path.join(ann_folder, f"{filename}"), "r") as file:
                prediction = postprocess_response(file.read())

            if evaluation == "optimal":
                with open(os.path.join(optimal_folder, f"{filename}"), "r") as file:
                    truth = parse_response(file.read())
            elif evaluation == "human":
                with open(os.path.join(truth_folder, f"{filename}"), "r") as file:
                    truth = postprocess_response(file.read())
            else:
                raise ValueError(f"Unknown evaluation method {evaluation}")

            out = evaluate_transitions(prediction, truth)
            tp += out[0]
            fp += out[1]
            fn += out[2]
            tn += out[3]

        n = len(glob.glob(os.path.join(ann_folder, "*.txt")))
        # evaluate
        acc = (tp + tn) / n
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
        key = os.path.basename(os.path.normpath(folder))
        results[key] = {
            "Subgoals": "Predefined" if "preset" in folder else "Discovered",
            "Observation": "NetHack tty" if "tty" in folder else "9x9 chars",
            "Accuracy": acc,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Total": n,
        }

    data = pd.DataFrame(results).transpose().convert_dtypes()
    # put "sort" column first
    front = data[sort]
    data.drop(labels=[sort], axis=1, inplace=True)
    data.insert(2, sort, front)

    if sort == "index":
        data.sort_index(ascending=False, inplace=True)
    else:
        data.sort_values(sort, ascending=False, inplace=True)

    prompts_folder = list(filter(lambda x: "prompts" in x, evaluate_folders))
    if len(prompts_folder) > 0:
        prompts_folder = prompts_folder[0]
    else:
        raise ValueError("No prompts folder found")
    
    data.drop(index=prompts_folder, inplace=True)

    print(data)

    ablation = prompts_folder.split("prompts-")[-1]
    print(ablation)
    for model_name in data.index:
        # remove ablation name from model name
        data.rename(index={model_name: model_name[: -len(ablation) - 1]}, inplace=True)
    print()
    if latex:
        data = data.round(2)
        if argv.highlight:
            data = data.style.apply(
                partial(_highligh_n_best, n=2, op="nlargest"),  # type: ignore
                props="textbf:--rwrap;",
                subset=["TP", "Precision", "Recall", "F1", "Accuracy"],
            ).apply(
                partial(_highligh_n_best, n=2, op="nsmallest"),  # type: ignore
                props="textbf:--rwrap;",
                subset=["FP", "FN"],
            )
        else:
            data = data.style

        print(data.format(precision=2).to_latex(hrules=True))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--match", type=str, default="gemma-7b-it-subgoals-preset-win"
    )
    argparser.add_argument("--exclude", type=str, default=" $^")
    argparser.add_argument("--sort", type=str, default="F1")
    argparser.add_argument("--evaluation", type=str, default="human")
    argparser.add_argument("--latex", default=False, action="store_true")
    argparser.add_argument("--highlight", default=False, action="store_true")
    argparser.add_argument(
        "--source_dir", type=str, default="/scratch/uceeepi/calf/dataset/dataset-3/"
    )
    args = argparser.parse_args()

    main(args)
