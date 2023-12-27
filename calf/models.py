import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import requests

from .prompts import DEFAULT_SYSTEM_PROMPT


class LLM:
    def __init__(self, model_name: str, template_url: str = ""):
        self.model_name = model_name
        self.template_url = template_url

    def patch_chat_template(self):
        if self.template_url == "":
            print("No patch template url provided, skip patching chat template")
            return
        res = requests.get(self.template_url)
        template = res.content.decode("utf-8")
        self.chat_template = template
        self.tokenizer.chat_template = template
        print(f"Patch chat template for Falcon7B with template.")

    def init(self, **kwargs):
        print("o-----------------------------")
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
        self.patch_chat_template()
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
        super().__init__("meta-llama/Llama-2-7b-chat-hf")


class Llama13B(LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-13b-chat-hf")


class Llama70B(LLM):
    def __init__(self):
        super().__init__("meta-llama/Llama-2-70b-chat-hf")


class Mistral7B(LLM):
    def __init__(self):
        super().__init__(
            "mistralai/Mistral-7B-Instruct-v0.1",
            "https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/mistral.jinja",
        )

class OpenOrcaMistral7B(LLM):
    def __init__(self):
        super().__init__(
            "Open-Orca/Mistral-7B-OpenOrca",
        )


class Falcon7B(LLM):
    def __init__(self):
        super().__init__(
            "tiiuae/falcon-7b-instruct",
            "https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/falcon.jinja",
        )


class Fuyu8B(LLM):
    def __init__(self):
        super().__init__("adept/fuyu-8b")
