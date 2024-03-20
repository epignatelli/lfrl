from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
import time
import requests
import re

import jax
import jax.numpy as jnp
from helx.base.mdp import Timestep

from .prompts import (
    PROMPT_IDENTIFY_SUBGOALS_BASE,
    PROMPT_IDENTIFY_SUBGOALS_TASK_ABLATION,
    PROMPT_IDENTIFY_SUBGOALS_ROLE_ABLATION,
    PROMPT_IDENTIFY_SUBGOALS_IDENTIFICAION_ABLATION
)
from .decode import decode_action, decode_message, decode_observation


PROMPT = PROMPT_IDENTIFY_SUBGOALS_BASE


@dataclass
class Annotation:
    prompt: str
    response: str
    parsed: Dict[Tuple[str, str], List[str]]


def split(episode: Timestep, size: int) -> List[Timestep]:
    """Splits an array into a List of arrays including the last remaining items
    whose chunk length could be less than size"""
    upper = (len(episode) // size + 1) * size + 1
    indices = jnp.arange(0, upper, size)
    return [episode[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]


def make_transitions(episode: Timestep) -> List[Timestep]:
    transitions = []
    for i in range(len(episode.t) - 2):
        transition = episode[i : i + 3]
        if transition.info["mask"].sum() > 0:
            transitions.append(transition)
    return transitions


def annotate_subgoals(
    episodes_batch: List[Timestep],
    max_new_tokens: int,
    url: str = "http://localhost:5000/respond",
    seq_len: int | None = None,
) -> List[Annotation]:
    # split into chunks of seq_len
    if seq_len is not None:
        episodes_chunks = []
        for episode in episodes_batch:
            # chunks = make_transitions(episode)
            chunks = [x for x in split(episode, seq_len) if len(x) > 0]
            episodes_chunks.extend(chunks)
        episodes_batch = episodes_chunks

    # compose prompt
    prompts = batch_compose_prompts(episodes_batch, PROMPT)

    # query LLM
    responses = batch_query_llm(prompts, max_new_tokens, url)

    # parse subgoals
    annotations = parse_subgoals(prompts, responses, episodes_batch)
    return annotations


def parse_subgoals(
    prompts: List[str], responses: List[str], episodes: List[Timestep]
) -> List[Annotation]:
    parsed_response = batch_parse_response(responses)
    annotations = []
    # for each response in the batch
    for i, _ in enumerate(parsed_response):
        goals_dic = parsed_response[i]

        # if the response is invalid, append None and move on
        if goals_dic is None:
            continue

        episode = episodes[i]
        times = []
        reasons = []
        # for each subgoal identified in the trajectory
        for reason, v in goals_dic.items():
            if v is None:
                continue
            # identified a single timestep
            elif isinstance(v, int):
                times.append(v)
                reasons.append(reason)
            elif isinstance(v, float):
                times.append(int(v))
                reasons.append(reason)
            # identified a list of timesteps for each subgoal
            elif isinstance(v, (list, tuple)):
                try:
                    times.extend(list(map(int, v)))
                    reasons.extend([reason] * len(v))
                except:
                    continue
            else:
                print("Cannot extract information from {}".format(goals_dic))

        # otherwise, retrieve the obs and action at the time of subgoal achievement
        rewarding_states = {}
        for j, time in enumerate(times):
            reason = reasons[j]
            idx = jnp.where(episode.t == jnp.asarray(time - 1))[0]
            if len(idx) > 0:
                idx = idx[0]
                obs = jnp.asarray(
                    episode.info["observations"]["chars"][idx], dtype=jnp.int32
                )
                action = jnp.asarray(episode.action[idx], dtype=jnp.int32)
                key = (decode_observation(obs), str(int(action)))
                if key in rewarding_states:
                    rewarding_states[key].append(reason)
                else:
                    rewarding_states[key] = [reason]

        # append the retrieved info to the annotation
        if len(times) > 0:
            annotation = Annotation(
                prompt=prompts[i], response=responses[i], parsed=rewarding_states
            )
            annotations.append(annotation)

    return annotations


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
    prompts = list(map(lambda x: compose_prompt(x, prompt), episodes))
    print(f"Done in {time.time() - start_time}s")
    return prompts


def batch_parse_response(responses: List[str]) -> List[dict | None]:
    print("Parsing responses...\t", end="", flush=True)
    start_time = time.time()
    dictionaries = list(map(parse_response, responses))
    print(f"Done in {time.time() - start_time}s")
    return dictionaries


def compose_prompt(episode: Timestep, llm_task_prompt: str) -> str:
    observations = jnp.asarray(
        episode.info["observations"]["chars_crop"], dtype=jax.numpy.int32
    )
    messages = jnp.asarray(
        episode.info["observations"]["message"], dtype=jax.numpy.int32
    )
    prompt = ""
    for t in range(0, len(episode.t)):
        if episode.info["mask"][t] == jnp.asarray(0):
            continue
        prompt += f"Time: {int(episode.t[t])}\n"
        action = decode_action(int(episode.action[t]))
        obs_as_string = decode_observation(observations[t])
        prompt += f"Observation: {obs_as_string}\n\n"
        prompt += f"Message: {decode_message(messages[t])}\n"
        prompt += f"Planned action: {action}\n\n\n"

    prompt = llm_task_prompt.format(prompt)
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


# def annotate_redistribution(
#     episodes: List[Timestep],
#     max_new_tokens: int,
#     url: str = "http://localhost:5000/respond",
# ) -> List[Timestep]:
#     # compose prompt
#     prompts = batch_compose_prompts(episodes, PROMPT_IDENTIFY_SUBGOALS)

#     # query LLM
#     responses = batch_query_llm(prompts, max_new_tokens, url)

#     # parse rewards
#     parsed = batch_parse_response(responses)
#     for i, reward_dict in enumerate(parsed):
#         if reward_dict is None:
#             continue
#         rewards = list(reward_dict.values())
#         if len(rewards) != len(episodes[i].t):
#             continue
#         rewards = jnp.asarray(rewards)
#         episodes[i] = episodes[i].replace(reward=rewards)
#     return episodes


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
