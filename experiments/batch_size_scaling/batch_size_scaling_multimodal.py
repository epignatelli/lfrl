"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""


import time
import requests
from PIL import Image
import torch
import torch.backends.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import FuyuProcessor, FuyuForCausalLM

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="adept/fuyu-8b")
parser.add_argument("--dtype", type=str, default="torch.float32")
parser.add_argument("--quantisation", type=int, default=0)
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

    processor = FuyuProcessor.from_pretrained(MODEL_NAME)
    model = FuyuForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        # use_flash_attention_2=DTYPE in [torch.float16, torch.bfloat16],
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
        url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
        image = [Image.open(requests.get(url, stream=True).raw)] * batch_size
        start_time = time.time()
        try:
            with torch.no_grad():
                encoding = processor(text=prompt, images=image, return_tensors="jax").to(model.device)  # type: ignore
                generated_tokens = model.generate(**encoding, max_new_tokens=MAX_NEW_TOKENS)  # type: ignore
                generated_text = processor.batch_decode(
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
