import json
from typing import Dict, List
import requests
import re

from .prompts import PROMPT_INTRINSIC_REWARD, PROMPT_ENV_DESCRIPTION, PROMPT_CREDIT_ASSIGNMENT


CA_PATTERN = re.compile(r"\{\s*\"timestep-\d\"\s*:\s*\d*.\d*\s*\}", re.IGNORECASE)


def request_chat(
    prompts: List[str], *, host: str = "localhost", port: int = 5000
) -> List[str]:
    url = "http://{}:{}/chat".format(host, port)
    response = requests.get(url, data=json.dumps(prompts))
    if not response.ok:
        return []

    content = response.json()
    if len(content) <= 0:
        return []

    text_batch = map(lambda x: x["generated_text"], content)
    return list(text_batch)


def get_env_description(env_name: str) -> str:
    prompt = PROMPT_ENV_DESCRIPTION.format(env_name)
    response = request_chat([prompt]) or ""
    return response[0]


def ask_credit(
    observations: str,
    env_description: str,
    *,
    host: str = "localhost",
    port: int = 5000,
) -> List[str]:
    prompt = PROMPT_CREDIT_ASSIGNMENT.format(env_description)
    prompts = ["\n\n".join([prompt, obs]) for obs in observations]
    return request_chat(prompts, host=host, port=port)


def parse_credit(text: str) -> Dict[str, float]:
    matches = CA_PATTERN.findall(text)
    result = {}
    for match in matches:
        try:
            x = json.loads(match)
        except:
            x = {}
        result = {**result, **x}
    return result

