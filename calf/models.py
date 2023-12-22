import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

DEFAULT_SYSTEM_PROMPT = """\
Welcome to the MiniHack data collection and evaluation interface.
This interface is designed to collect data from MiniHack and evaluate the performance of reinforcement learning agents.
You are a helpful and honest judge of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while being truthful.
If you don't know the answer to a question, please don't share false information."""


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def init(self, **kwargs):
        print(f"Loading {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=kwargs.pop("device_map", "auto"),
            torch_dtype=kwargs.pop("torch_dtype", torch.bfloat16),
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.model = model
        print(f"{self.model_name} loaded")

    def forward(
        self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, **kwargs
    ) -> str:
        with torch.no_grad():
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            encoding = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(  # type: ignore
                self.model.device
            )
            max_new_tokens = kwargs.pop("max_new_tokens", 256)
            generated_tokens = self.model.generate(
                encoding,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            generated_text = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            return generated_text[0]


class Llama7B(LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-7B-chat-hf")


class Llama13B(LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-13B-chat-hf")


class Llama70B(LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-70B-chat-hf")


class Mistral7B(LLM):
    def __init__(self):
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1")


class Falcon7B(LLM):
    def __init__(self):
        super().__init__("tiiuae/falcon-7b-instruct")


class Fuyu8B(LLM):
    def __init__(self):
        super().__init__("adept/fuyu-8b")
