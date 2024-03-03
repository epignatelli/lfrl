from __future__ import annotations

from typing import List, Tuple
import time
import requests
import re

from jax import Array
import jax.numpy as jnp
from helx.base.mdp import Timestep

from nle import nethack
from .prompts import PROMPT_REDISTRIBUTION
from .nethack import ACTIONS_MAP


def query_llm(
    episodes: List[Timestep],
    max_new_tokens: int,
    llm_task_prompt: str = PROMPT_REDISTRIBUTION,
    url: str = "http://localhost:5000/respond",
) -> Tuple[List[Array], List[int]] | None:
    """Evaluate the model with a large language model."""
    # compose prompt
    print("Composing prompts...\t", end="")
    start_time = time.time()
    prompts = list(map(lambda x: _compose_prompt(x, llm_task_prompt), episodes))
    print(f"{time.time() - start_time}s")

    print("Prompting LLM...\t", end="")
    start_time = time.time()
    response = requests.post(
        url,
        data={"prompts[]": [prompt.encode() for prompt in prompts]},
        params={"max_new_tokens": max_new_tokens},
    )
    responses = response.json()
    print(f"{time.time() - start_time}s")

    # parse response
    print("Parsing responses...\t", end="")
    start_time = time.time()
    rewards, valid = _parse_redistribution(responses, seq_len=len(episodes[0].t))
    print(f"{time.time() - start_time}s")
    return rewards, valid


def get_action_desc(action: int):
    actions = [
        nethack.CompassCardinalDirection.N,  # 0
        nethack.CompassCardinalDirection.E,  # 1
        nethack.CompassCardinalDirection.S,  # 2
        nethack.CompassCardinalDirection.W,  # 3
        nethack.Command.PICKUP,  # 4
        nethack.Command.APPLY,  # 5
    ]
    nle_action = actions[action]
    return ACTIONS_MAP[nle_action][0]


def _compose_prompt(
    episode: Timestep, llm_task_prompt: str = PROMPT_REDISTRIBUTION
) -> str:
    def arr_to_string(obs) -> str:
        rows, cols = obs.shape
        result = ""
        for i in range(rows):
            result += "\n" + "".join(map(chr, obs[i]))
        return result

    observations = episode.info["observations"]["chars"]
    return_ = jnp.sum(episode.reward)
    prompt = ""
    mask_length = int(episode.info["mask"].sum())
    for t in range(0, mask_length):
        prompt += f"Timestep: {int(episode.t[t])}\n"
        action = get_action_desc(int(episode.action[t]))
        prompt += f"Action: {action}\n"
        # TODO: too many tokens for uncropped obs. Perhaps use cropped?
        obs_as_string = arr_to_string(observations[t])
        prompt += f"Observation: {obs_as_string}\n\n"

    prompt = llm_task_prompt.format(prompt, return_)
    return prompt


def _parse_redistribution(
    responses: List[str], seq_len: int
) -> Tuple[List[Array], List[int]]:
    pattern = r"```(python)?(.*)({.*})"
    rewards = []
    valid = []
    for i, response in enumerate(responses):
        match = re.search(pattern, response, re.DOTALL)
        if match is None or len(match.groups()) < 3:
            print("Invalid response. Discarding prompt", i)
            rewards.append(jnp.ones((seq_len,)) / seq_len)
        else:
            dict_str = match.group(3)
            try:
                r = eval(dict_str)
                credits = list(r.values())
                assert len(credits) == seq_len
                rewards.append(jnp.asarray(credits))
                valid.append(i)
            except:
                print("Invalid response. Discarding prompt", i)
                rewards.append(jnp.ones((seq_len,)) / seq_len)
    # TODO: allow uniform distribution rather than discarding it?
    return rewards, valid


def parse_subgoals(): ...
