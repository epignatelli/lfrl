from __future__ import annotations


import gym
import minihack
from nle.env.base import NLE
import numpy as np
from dataclasses import dataclass
from llama import Llama
from transformers import pipeline
from actions import all_nle_action_map


CHECKPOINT_DIR = "/scratch/uceeepi/llama/llama-2-7b-chat"
TOKENISER_PATH = "/scratch/uceeepi/llama/tokenizer.model"
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 1
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_SYSTEM_PROMPT = """\
You are an expert in Reinforcement Learning, and in particular in the game of MiniHack. \
When asked questions, always answer truthfully and never lie about your knowledge. \
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
If you don't know the answer to a question, please don't share false information."""

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful and honest judge of good gameplaying and progress in the MiniHack game. Always answer as helpfully as possible while being truthful.

If you don't know the answer to a question, please don't share false information."""

PROMPT_INTRO = """\
I will present you with a sequence of observations from the gameplay of MiniHack.
Recall that a single observation in MiniHack has three main parts: a) a *message* appearing on top of the screen; b) a *grid of symbols* showing the positions of entities in the room and c) a *set of statistics* at the bottom of the screen.
I will present you with a sequence of these."""


PROMPT_PRELIMINARY_KNOWLEDGE = """\
First, tell me about your knowledge of MiniHack.
Title this paragraph "Preliminary knowledge"."""

PROMPT_OBS_ANALYSIS = """\
Write an analysis describing the semantics of each observation strictly using information from the observations and your knowledge of MiniHack.
Title this paragraph **Observation analysis**."""

PROMPT_TRAJ_ANALYSIS = """\
Then, write an analysis describing the semantics of the sequence of observations focusing on the reasons that could have led to the final observation.
End this analysis by writing whether the agent should avoid or repeat the action at its next encounter with the same state.
Recall that the goal of the agent is find the staircase up, denoted by ">" and do not confound that with the staircase down symbol "<".
Title this paragraph **Reasoning Leading to Final Observation**."""

# PROMPT_CREDIT = """\
# Finally, for each timestep, respond with a number between 0.0 and 1.0 representing the credit assigned to each observation in order to reach the final observation.
# A credit of 1.0 means that the action taken in that timestep was absolutely necessary to get to the final state.
# A credit of 0.0 means that the action is irrelevant and the agent would have arrived at the same timestep even without taking it."""

PROMPT_CREDIT = """\
Finally, for each timestep, respond by providing the number of the timestep that you evaluate to be the most significant to reach the final observation.
Title this paragraph **Action recommendation**."""

PROMPT_ANSWER = """\
Synthetise the action recommendation into a dictionary of the form `{"Timestep 5": True}`, writing `True` if you recommend to take same action next time, `False` if you do not recommend and `None` if the action does not matter."""

PROMPT_END = """\
Now begins the sequence of observations:"""

PROMPT = f"""
{PROMPT_INTRO}

{PROMPT_PRELIMINARY_KNOWLEDGE}

{PROMPT_OBS_ANALYSIS}

{PROMPT_TRAJ_ANALYSIS}

{PROMPT_CREDIT}

{PROMPT_ANSWER}

{PROMPT_END}
"""


def tty_render(chars, cursor=None):
    rows, cols = chars.shape
    if cursor is None:
        cursor = (-1, -1)
    cursor = tuple(cursor)
    result = ""
    for i in range(rows):
        result += "\n"
        for j in range(cols):
            entry = chr(chars[i, j])
            if cursor != (i, j):
                result += entry
            else:
                result += entry
    return result


def render(env) -> str:
    obs = env.last_observation
    tty_chars = obs[env._observation_keys.index("tty_chars")]
    tty_cursor = obs[env._observation_keys.index("tty_cursor")]
    window = tty_render(tty_chars, tty_cursor)
    return window


def ord_to_char(array):
    chars = np.char.mod('%c', array).tolist()
    obs = ""
    for line in chars:
        obs += "".join(line) + "\n"
    return obs


@dataclass
class Timestep:
    t: int
    observation: str
    action: int | None

    def to_prompt(self, env) -> str:
        timestep = "Timestep {}".format(self.t)
        if self.action is not None:
            action = all_nle_action_map[env.actions[self.action]][0]
        else:
            action = ""
        action = "Action: {}".format(action)
        return "\n".join([timestep, action, self.observation])


def unroll():
    env = gym.make(
        "MiniHack-KeyRoom-Fixed-S5-v0",
        observation_keys=("chars", "chars_crop",),
        max_episode_steps=100,
    )
    print("Available actions:", env.action_space)

    obs = env.reset()
    timesteps = []
    done = False
    i = 0
    # while not done:
    for i in range(10):
        # print the state
        obs = render(env)

        # choose action
        print(obs)
        # action = int(input())  # manual action
        action = env.action_space.sample()  # random action

        # pack and store timestep
        timestep = Timestep(i, obs, action)
        timestep = timestep.to_prompt(env)
        timesteps.append(timestep)

        # update env
        _, _, _, done = env.step(action)
        done = done['end_status'] != 0
        i += 1

    # print the state
    obs = render(env)
    print(obs)
    timestep = Timestep(i, obs, None)
    timestep = timestep.to_prompt(env)
    timesteps.append(timestep)

    return timesteps


def tiemsteps_to_prompt(timesteps):
    traj_description = "\n".join(timesteps)
    return "\n".join([PROMPT, traj_description])


def save_prompt(prompt):
    with open("./prompt.txt", "w") as f:
        f.write(prompt)


class LLM:
    def __init__(self,
        checkpoint_dir: str=CHECKPOINT_DIR,
        tokenizer_path: str=TOKENISER_PATH,
        max_seq_len: int=MAX_SEQ_LEN,
        max_batch_size: int=MAX_BATCH_SIZE,
    ):
        self.model = Llama.build(
            ckpt_dir=checkpoint_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def __call__(self, prompt: str, temperature: float=DEFAULT_TEMPERATURE, system_prompt: str = DEFAULT_SYSTEM_PROMPT, top_p: float=DEFAULT_TOP_P) -> str:
        dialog = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        dialogs = [dialog]

        response = self.model.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=None,
            temperature=temperature,
            top_p=top_p,
        )

        return response['generation']['content']  # type: ignore


if __name__ == "__main__":
    print("Loading model...")
    # pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
    # llm = LLM()
    print("Model loaded")

    print("Collect trajectory")
    timesteps = unroll()
    print("Trajectory collected")

    print("Creating prompt...")
    prompt = tiemsteps_to_prompt(timesteps)
    save_prompt(prompt)
    print("Prompt created")

    print("Prompting model...")
    # response = pipe(prompt)
    # response = llm(prompt)
    print("Model prompted")

    print(prompt)
    print(prompt)

