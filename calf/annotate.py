from __future__ import annotations
import json
import pickle

from typing import Dict, Generator, List, Tuple
import time
import jax
import requests
import re

import jax.numpy as jnp
from helx.base.mdp import Timestep

from nle import nethack
from .environment import ACTIONS_MAP

from calf.prompts import PROMPT_IDENTIFY_SUBGOALS


def split(array, size):
    """Splits an array into a List of arrays including the last remaining items
    whose chunk length could be less than size"""
    upper = (len(array) // size + 1) * size + 1
    indices = jnp.arange(0, upper, size)
    return [array[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]


def annotate_subgoals(
    episodes_batch: List[Timestep],
    max_new_tokens: int,
    url: str = "http://localhost:5000/respond",
    seq_len: int | None= None
) -> Dict[Tuple[int, int, int], int]:
    # split into chunks of seq_len
    if seq_len is not None:
        episodes_chunks = []
        for episode in episodes_batch:
            chunks = [x for x in split(episode, seq_len) if len(x) > 0]
            episodes_chunks.extend(chunks)
        episodes_batch = episodes_chunks

    # compose prompt
    prompts = batch_compose_prompts(episodes_batch, PROMPT_IDENTIFY_SUBGOALS)

    # query LLM
    responses = batch_query_llm(prompts, max_new_tokens, url)

    # parse subgoals
    parsed = batch_parse_response(responses)
    rewarding_states = {}
    for dic in parsed:
        if dic is not None:
            times = list(dic.values())
            for t in times:
                stats = episodes_batch[t].info["observations"]["blstats"]
                key = (stats[0], stats[1], stats[12])  # x, y, d
                rewarding_states[key] = 1
    return rewarding_states


def annotate_redistribution(
    episodes: List[Timestep],
    max_new_tokens: int,
    url: str = "http://localhost:5000/respond",
) -> List[Timestep]:
    # compose prompt
    prompts = batch_compose_prompts(episodes, PROMPT_IDENTIFY_SUBGOALS)

    # query LLM
    responses = batch_query_llm(prompts, max_new_tokens, url)

    # parse rewards
    parsed = batch_parse_response(responses)
    for i, reward_dict in enumerate(parsed):
        if reward_dict is None:
            continue
        rewards = list(reward_dict.values())
        if len(rewards) != len(episodes[i].t):
            continue
        rewards = jnp.asarray(rewards)
        episodes[i] = episodes[i].replace(reward=rewards)
    return episodes


def batch_query_llm(
    prompts: List[str],
    max_new_tokens: int,
    url: str = "http://localhost:5000/respond",
) -> List[str]:
    print("Prompting LLM...\t", end="", flush=True)
    start_time = time.time()
    response = requests.post(
        url,
        data={"prompts[]": [prompt.encode() for prompt in prompts]},
        params={"max_new_tokens": max_new_tokens},
    )
    responses = response.json()
    print(f"Done in {time.time() - start_time}s")
    return responses


def batch_compose_prompts(episodes: List[Timestep], prompt: str) -> List[str]:
    print("Composing prompts...\t", end="", flush=True)
    start_time = time.time()
    prompts = list(map(lambda x: compose_prompt(x, PROMPT_IDENTIFY_SUBGOALS), episodes))
    print(f"Done in {time.time() - start_time}s")
    return prompts


def batch_parse_response(responses: List[str]) -> List[dict | None]:
    print("Parsing responses...\t", end="", flush=True)
    start_time = time.time()
    dictionaries = list(map(parse_response, responses))
    print(f"Done in {time.time() - start_time}s")
    return dictionaries


def compose_prompt(episode: Timestep, llm_task_prompt: str) -> str:
    def decode_observation(obs) -> str:
        rows, cols = obs.shape
        result = ""
        for i in range(rows):
            result += "\n" + "".join(map(chr, obs[i]))
        return result

    def decode_message(message):
        if int(message.sum()) == 0:
            return ""
        return "".join(map(chr, message))

    def decode_action(action: int):
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

    # TODO: too many tokens for uncropped obs. Perhaps use cropped?
    observations = episode.info["observations"]["chars_crop"]
    observations = jnp.asarray(observations, dtype=jax.numpy.int32)
    messages = episode.info["observations"]["message"]
    messages = jnp.asarray(messages, dtype=jax.numpy.int32)
    prompt = ""
    mask = episode.info["mask"]
    # timesteps = []
    for t in range(0, len(episode.t)):
        if mask[t] == jnp.asarray(0):
            continue
        prompt += f"Time: {int(episode.t[t])}\n"
        # action = decode_action(int(episode.action[t]))
        # prompt += f"Action: {action}\n"
        obs_as_string = decode_observation(observations[t])
        # obs_as_string = obs_as_string.replace("    ", "\t")
        # obs_as_string = obs_as_string.replace("\t\t\t\t\t\t\t\t", "\t")
        prompt += f"Message: {decode_message(messages[t])}\n"
        prompt += f"Observation: {obs_as_string}\n\n"

        # timesteps.append(
        #     {
        #         "Timestep": int(episode.t[t]),
        #         # "Action": decode_action(int(episode.action[t])),
        #         "Observation": obs_as_string,
        #     }
        # )

    # prompt = json.dumps(timesteps).encode().decode("unicode_escape")
    prompt = llm_task_prompt.format(prompt, 1.0)
    return prompt


def parse_response(response: str) -> dict | None:
    pattern = r"```(python)?(.*)({.*})"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            raise
        dict_str = match.group(3)
        response_parsed = eval(dict_str)
        return response_parsed
    except:
        print(f"Invalid response {response}.\nDiscarding prompt")
        return None


# def batch_parse_redistribution(
#     responses: List[str], episodes: List[Timestep]
# ) -> List[Array]:
#     return list(map(parse_redistribution, responses, episodes))


# def parse_redistribution(response: str, episode: Timestep) -> Array:
#     pattern = r"```(python)?(.*)({.*})"
#     try:
#         seq_len = int(episode.info["mask"].sum())
#         match = re.search(pattern, response, re.DOTALL)
#         if match is None:
#             raise
#         dict_str = match.group(3)
#         r = eval(dict_str)
#         credits = list(r.values())
#         assert len(credits) == seq_len
#         return jnp.asarray(credits)
#     except:
#         print(f"Invalid response {response}.\nDiscarding prompt.")
#         return jnp.ones((seq_len,)) / seq_len  # distribute uniformly


# def annotate_rewards(
#     episodes: List[Timestep],
#     max_new_tokens: int,
#     llm_task_prompt: str,
#     url: str = "http://localhost:5000/respond",
# ) -> Tuple[List[Array], List[int]] | None:
#     """Evaluate the model with a large language model."""
#     # compose prompt
#     print("Composing prompts...\t", end="", flush=True)
#     start_time = time.time()
#     prompts = list(map(lambda x: compose_prompt(x, llm_task_prompt), episodes))
#     print(f"Done in {time.time() - start_time}s")

#     print("Prompting LLM...\t", end="", flush=True)
#     start_time = time.time()
#     response = requests.post(
#         url,
#         data={"prompts[]": [prompt.encode() for prompt in prompts]},
#         params={"max_new_tokens": max_new_tokens},
#     )
#     responses = response.json()
#     print(f"Done in {time.time() - start_time}s")

#     # parse response
#     print("Parsing responses...\t", end="", flush=True)
#     start_time = time.time()
#     rewards, valid = parse_redistribution(responses, episodes)
#     print(f"Done in {time.time() - start_time}s")
#     return rewards, valid
