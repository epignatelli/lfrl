from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import requests
import re

import jax
import jax.numpy as jnp
from helx.base.mdp import Timestep

from .prompts import (
    PROMPT_BASE,
)
from .decode import decode_action, decode_message, decode_observation


PROMPT = PROMPT_BASE


@dataclass
class Annotation:
    prompt: str
    response: str
    parsed: Dict[Tuple[str, str], List[str]]


def annotate_subgoals(
    episodes_batch: List[Timestep],
    max_new_tokens: int,
    url: str = "http://localhost:5000/respond",
    instructions: str = PROMPT,
    ablate_action=False,
    ablate_message=False,
    token_separator="",
    tty=False,
) -> List[Annotation]:
    # compose prompt
    prompts = batch_compose_prompts(
        episodes=episodes_batch,
        prompt=instructions,
        ablate_action=ablate_action,
        ablate_message=ablate_message,
        token_separator=token_separator,
        tty=tty,
    )

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
            print(f"Discarding response {i}. No dictionary to parse.")
            continue
        elif len(goals_dic) == 0:
            print(f"Discarding response {i}. Empty dictionary.")
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
        print(f"Found {len(rewarding_states)} useful annotations in response {i}.")
        if len(rewarding_states) > 0:
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


def batch_query_llm_with_name(
    prompts: List[str],
    max_new_tokens: int,
    url: str = "http://localhost:5000/respond",
) -> Tuple[List[str], str]:
    print("Prompting LLM...\t", end="", flush=True)
    start_time = time.time()
    response = requests.post(
        url,
        data={"prompts[]": [prompt.encode() for prompt in prompts]},
        params={"max_new_tokens": max_new_tokens},
    )
    responses = response.json()
    print(f"Done in {time.time() - start_time}s")
    return responses, response.headers.get("X-Model-Name", "unknown")


def batch_compose_prompts(
    episodes: List[Timestep],
    prompt: str,
    ablate_action=False,
    ablate_message=False,
    token_separator="",
    tty=False,
) -> List[str]:
    print("Composing prompts...\t", end="", flush=True)
    start_time = time.time()
    prompts = list(
        map(
            lambda x: compose_prompt(
                x,
                prompt,
                ablate_action=ablate_action,
                ablate_message=ablate_message,
                token_separator=token_separator,
                tty=tty,
            ),
            episodes,
        )
    )
    print(f"Done in {time.time() - start_time}s")
    return prompts


def batch_parse_response(responses: List[str]) -> List[dict]:
    print("Parsing responses...\t", end="", flush=True)
    start_time = time.time()
    dictionaries = list(map(parse_response, responses))
    print(f"Done in {time.time() - start_time}s")
    return dictionaries


def compose_prompt(
    episode: Timestep,
    llm_task_prompt: str,
    ablate_action=False,
    ablate_message=False,
    token_separator="",
    tty=False,
) -> str:
    if tty:
        ttys = jnp.asarray(
            episode.info["observations"]["tty_chars"], dtype=jax.numpy.int32
        )
    else:
        observations = jnp.asarray(
            episode.info["observations"]["chars_crop"], dtype=jax.numpy.int32
        )
        messages = jnp.asarray(
            episode.info["observations"]["message"], dtype=jax.numpy.int32
        )

    prompt = ""
    last_action = "None"
    for t in range(len(episode.t)):
        time = int(t)
        prompt += f"Time: {time}\n"

        # tty
        if tty:
            obs = decode_observation(ttys[t], separator="")
            obs = obs.split("\n")
            grid = [token_separator.join(x) for x in obs[2:22]]
            obs = "\n".join(obs[0:2] + grid + obs[22:])
            prompt = prompt[:-2] # remove new line
            prompt += obs + "\n\n"
        else:
            if not ablate_action:
                next_action = decode_action(int(episode.action[t]))
                prompt += f"Last action: {last_action}\n"

            if not ablate_message:
                current_message = decode_message(messages[t])
                prompt += f"Current message: {current_message}\n"

            current_observation = decode_observation(
                observations[t], separator=token_separator
            )
            prompt += f"Current observation: {current_observation}\n\n"

            if not ablate_action:
                prompt += f"Next action: {next_action}\n\n\n"
                last_action = next_action

    prompt = llm_task_prompt.format(prompt)
    return prompt


def parse_response(response: str) -> dict:
    pattern = r"```(python)?(.*)({.*})"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            raise
        dict_str = match.group(3)
        response_parsed = eval(dict_str)
        return response_parsed
    except:
        # print(f"Invalid response {response}.\nDiscarding prompt")
        return {}
