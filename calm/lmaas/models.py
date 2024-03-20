from __future__ import annotations

from typing import List, Dict
from jax import Array
import jax
import torch
import jax.numpy as jnp
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    FlaxGemmaForCausalLM,
    BitsAndBytesConfig,
)
import requests


class Roles:
    USER: str = "user"
    SYSTEM: str = "system"
    ASSISTANT: str = "assistant"


class LLM:
    def __init__(self, model_name: str, template_url: str = ""):
        self.model_name = model_name
        self.template_url = template_url

    def patch_chat_template(self):
        if self.template_url == "":
            print("No patch template url provided, skipping chat template patch")
            return
        res = requests.get(self.template_url)
        template = res.content.decode("utf-8")
        self.chat_template = template
        self.tokenizer.chat_template = template
        print(f"Patch chat template for {self.model_name} with template:\n {template}.")

    def init(self, **kwargs):
        print("o" + "-" * 79)
        print(f"Loading {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_in_4bit = kwargs.pop("load_in_4bit", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        if load_in_4bit or load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit
            )
            torch_dtype = None
        else:
            bnb_config = None
            torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=kwargs.pop("device_map", "auto"),
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            quantization_config=bnb_config,
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.patch_chat_template()
        print(f"{self.model_name} loaded")

    def chat(
        self,
        conversation: List[Dict[str, str]] | List[List[Dict[str, str]]],
        *,
        append_prompt: bool = False,
        **kwargs,
    ):
        with torch.no_grad():
            # encode text to tokens
            is_batched = isinstance(conversation, list) and isinstance(
                conversation[0], list
            )
            if not is_batched:
                conversation = [conversation]  # type: ignore

            chats = [
                self.tokenizer.apply_chat_template(
                    convo,  # type: ignore
                    tokenize=False,
                    padding=True,
                    truncation=True,
                    add_generation_prompt=True,
                )
                for convo in conversation
            ]
            encoding = self.tokenizer(
                chats,  # type: ignore
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            # prompt the model
            max_new_tokens = kwargs.pop("max_new_tokens", 256)
            generated_tokens = self.model.generate(
                **encoding,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            # get only the new tokens
            if not append_prompt:
                input_len = encoding["input_ids"].shape[-1]  # type: ignore
                generated_tokens = generated_tokens[:, input_len:]
            # decode the tokens to text
            generated_text = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            return generated_text

    def forward(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> List[str]:
        conversation = [
            {"role": Roles.SYSTEM, "content": system_prompt},
            {"role": Roles.USER, "content": prompt},
        ]
        return self.chat(conversation, **kwargs)


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


class Gemma7B(LLM):
    def __init__(self):
        super().__init__("google/gemma-7b-it")


class Parrot(LLM):
    def __init__(self):
        self.model_name = "Parrot"

    def init(self, **kwargs):
        print("o" + "-" * 79)
        print(f"Loading {self.model_name}...")
        print(f"{self.model_name} loaded")

    def chat(self, conversation: List[Dict[str, str]], **kwargs):
        return [conversation[-1]["content"]]
