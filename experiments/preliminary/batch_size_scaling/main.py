#!/usr/bin/env python
# encoding: utf-8
"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""


import time
import torch
import torch.backends.cuda
import wandb
from calf.models import (
    Llama7B,
    Mistral7B,
    OpenOrcaMistral7B,
    Falcon7B,
    Llama13B,
    Llama70B,
    Fuyu8B,
)

MODELS = [
    Llama7B(),
    Mistral7B(),
    OpenOrcaMistral7B(),
    Falcon7B(),
]

MODELS_EXTRA = [
    Llama13B(),
    Llama70B(),
]

MODELS_MULTIMODAL = [
    Fuyu8B(),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]

QUANTISATIONS = [
    32,
    8,
    4,
]


def run_one(model, dtype, quantisation):
    print(f"Loading {model.model_name}-{dtype}-{quantisation}_bit...")
    config = {
        "model": model.model_name,
        "dtype": str(dtype),
        "quantisation": quantisation,
        "experiment": "batch_size_scaling",
    }
    run_name = "-".join(list(map(str, config.values())))
    wandb.finish()
    wandb.init(project="lfrl", name=run_name)
    wandb.config.update(config)

    model.init()
    PROMPT = "Write a long text about a topic of your choice."
    MAX_NEW_TOKENS = 128
    exp = 0
    while True:
        batch_size = 2**exp
        prompt = [PROMPT] * batch_size
        start_time = time.time()
        try:
            print(f"Running with batch size: {batch_size}")
            start_time = time.time()
            response = model.forward(prompt, max_new_tokens=MAX_NEW_TOKENS)
            response_time = time.time() - start_time
            wandb.log({"Batch size": batch_size, "Response time": response_time})
            print(f"Response time: {response_time:.4f}s")
            exp += 1
        except Exception as e:
            print(f"Failed to run {run_name} with batch size {batch_size}")
            print("Exiting...")
            return


def run():
    for model in MODELS:
        for dtype in DTYPES:
            for quantisation in QUANTISATIONS:
                run_one(model, dtype, quantisation)
    return

if __name__ == "__main__":
    run()
