import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FuyuProcessor,
    FuyuForCausalLM,
)
from prompts import SYSTEM_PROMPT, PROMPT_CREDIT_ASSIGNMENT

ENVS = [
    "Room",
    "Corridor",
    "KeyRoom",
    "MazeWalk",
    "River",
    "HideNSeek",
    "CorridorBattle",
    "Memento",
    "MazeExplore",
]

MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-7b-instruct",
    # "adept/fuyu-8b",
]


def load_model(model_name):
    print(f"Loading {model_name}...")
    if model_name.endswith("fuyu-8b"):
        tokenizer = FuyuProcessor.from_pretrained(model_name)
        model = FuyuForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_flash_attention_2=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    print(f"{model_name} loaded")
    return tokenizer, model


def run_model(tokenizer, model, prompt):
    with torch.no_grad():
        conversation = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        tokenised = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        generated_tokens = model.generate(tokenised, max_new_tokens=256)
        generated_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
    return generated_text[0]


def run():
    for model_name in reversed(MODELS):
        print("###################")
        tokenizer, model = load_model(model_name)
        prompt = SYSTEM_PROMPT + "\n" + PROMPT_CREDIT_ASSIGNMENT
        description = run_model(tokenizer, model, prompt)
        print("-------------------")
        print(description)
        print("\n\n\n\n\n")


if __name__ == "__main__":
    run()
