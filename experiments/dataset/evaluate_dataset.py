from __future__ import annotations

import argparse
from functools import partial
import os
import re
import glob

import numpy as np
import pandas as pd


def parse_response(response: str) -> dict:
    pattern = r"```(python)?(.*)({.*})"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            raise
        dict_str = match.group(3)
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

    evaluate_folders = [x for x in os.listdir(source_dir) if re.search(argv.match, x)]
    evaluate_folders += glob.glob(os.path.join(source_dir, "human-*"))

    # iterate over each annotation agent
    results = {}
    for folder in evaluate_folders:
        ann_folder = os.path.join(source_dir, folder)
        truth_folder = os.path.join(source_dir, "human-0")
        optimal_folder = os.path.join(source_dir, "optimal")
        tp = fp = fn = n = 0
        n_annotations = len(glob.glob(os.path.join(ann_folder, "*.txt")))
        for i in range(n_annotations):
            with open(os.path.join(ann_folder, f"prompt_{i}.txt"), "r") as file:
                prediction = postprocess_response(file.read())

            if evaluation == "optimal":
                with open(os.path.join(optimal_folder, f"prompt_{i}.txt"), "r") as file:
                    truth = parse_response(file.read())
            elif evaluation == "human":
                with open(os.path.join(truth_folder, f"prompt_{i}.txt"), "r") as file:
                    truth = postprocess_response(file.read())
            else:
                raise ValueError(f"Unknown evaluation method {evaluation}")

            out = evaluate(prediction, truth)
            tp += out[0]
            fp += out[1]
            fn += out[2]
            n += out[3]

        # evaluate
        acc = tp / (n + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
        key = os.path.basename(os.path.normpath(folder))
        results[key] = {
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Accuracy": acc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Total": n,
        }

    data = pd.DataFrame(results).transpose().convert_dtypes()

    if sort == "index":
        data.sort_index(ascending=False, inplace=True)
    else:
        data.sort_values(sort, ascending=False, inplace=True)

    print(data)
    # pprint(results, sort_dicts=False)
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
    argparser.add_argument("--match", type=str, default="c4ai-*")
    argparser.add_argument("--sort", type=str, default="F1")
    argparser.add_argument("--evaluation", type=str, default="human")
    argparser.add_argument("--latex", default=False, action="store_true")
    argparser.add_argument("--highlight", default=False, action="store_true")
    argparser.add_argument(
        "--source_dir", type=str, default="/scratch/uceeepi/calf/dataset/dataset-2/"
    )
    args = argparser.parse_args()

    main(args)
