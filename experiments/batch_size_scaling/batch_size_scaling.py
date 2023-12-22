#!/usr/bin/env python
# encoding: utf-8
"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""


import time
import torch
import torch.backends.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb


# MODELS = [
#     "meta-llama/Llama-2-7b-chat-hf",
#     "mistralai/Mistral-7B-Instruct-v0.2",
# ]
# DTYPES = [
#     torch.float32,
#     torch.float16,
#     torch.bfloat16,
# ]


# MODEL_NAME = MODELS[0]
# DTYPE = DTYPES[2]
# QUANTISATION = 8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--dtype", type=str)
parser.add_argument("--quantisation", type=int)
args = parser.parse_args()

MODEL_NAME = args.model_name
DTYPE = eval(args.dtype)
QUANTISATION = args.quantisation

def run():
    print(f"Loading {MODEL_NAME}-{DTYPE}-{QUANTISATION}_bit...")
    config = {"model": MODEL_NAME, "dtype": str(DTYPE), "quantisation": QUANTISATION, "multi_gpu": torch.cuda.device_count() > 1}
    run_name = '-'.join(list(map(str, config.values())))
    wandb.init(project="lfrl", name=run_name)
    wandb.config.update(config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        use_flash_attention_2=DTYPE in [torch.float16, torch.bfloat16],
        device_map="auto",
        load_in_8bit=QUANTISATION == 8,
        load_in_4bit=QUANTISATION == 4,
    )
    print(f"{MODEL_NAME} loaded")

    PROMPT = "Write a long text about a topic of your choice."
    MAX_NEW_TOKENS = 128
    exp = 0
    while True:
        batch_size = 2**exp
        print(f"Running with batch size: {batch_size}")
        prompt = [PROMPT] * batch_size
        start_time = time.time()
        try:
            with torch.no_grad():
                encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
                generated_tokens = model.generate(**encoding, max_new_tokens=MAX_NEW_TOKENS)
                generated_text = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True,
                )
            response_time = time.time() - start_time
            wandb.log({"Batch size": batch_size, "Response time": response_time})
            print(f"Response time: {response_time:.4f}s")
            exp += 1
        except Exception as e:
            print(repr(e))
            print(f"Failed to run {run_name} with batch size {batch_size}")
            print("Exiting...")
            break
    return


if __name__ == "__main__":
    run()
