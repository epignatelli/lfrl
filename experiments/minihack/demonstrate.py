from __future__ import annotations
import argparse
from typing import List

argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
# argparser.add_argument("--env_name", type=str, default="MiniHack-Room-5x5-v0")
argparser.add_argument("--env_name", type=str, default="MiniHack-KeyRoom-S5-v0")

argparser.add_argument("--budget", type=int, default=10_000_000)
argparser.add_argument("--n_actors", type=int, default=2)
argparser.add_argument("--n_epochs", type=int, default=4)
argparser.add_argument("--batch_size", type=int, default=128)
argparser.add_argument("--iteration_size", type=int, default=2048)
argparser.add_argument("--discount", type=float, default=0.99)
argparser.add_argument("--lambda_", type=float, default=0.95)
argparser.add_argument("--observation_key", type=str, default="pixel_crop")
argparser.add_argument("--max_buffer_size", type=int, default=10_000)
argparser.add_argument("--llm_transition_len", type=int, default=100)
argparser.add_argument("--filename", type=str, default="demonstrations")
args = argparser.parse_args()

import random

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import torch

torch.manual_seed(args.seed)

import os
from dataclasses import asdict
import pickle

import wandb
import gym
import gym.vector
import minihack
from nle import nethack

import jax
from jax.random import KeyArray
import jax.tree_util as jtu
import jax.numpy as jnp
import flax.linen as nn

from helx.base.modules import Flatten
from helx.envs.environment import Environment
from helx.base.mdp import Timestep, StepType

from calf.trial import Agent
from calf.calf import HParams, PPO
from calf.nethack import UndictWrapper, MiniHackWrapper


def extract_episode_from_stream(experience: Timestep):
    start_indices = jnp.where(experience.is_first())
    end_indices = jnp.where(experience.step_type == StepType.TERMINATION)

    segments = []
    i = j = 0

    while i < len(start_indices[0]) and j < len(end_indices[0]):
        batch_s, start_idx = start_indices[0][i], start_indices[1][i]
        batch_e, end_idx = end_indices[0][j], end_indices[1][j]

        if batch_s == batch_e and start_idx < end_idx:
            segment = experience[batch_s, start_idx : end_idx + 1]
            if segment.reward.sum() > 0:
                segments.append(segment)
        i += 1
        j += 1
    return segments

# TODO: Resolve:
# a) the slice goes back and crosses the boundaries of another env
# b) unpadded sequences are 102 long, padded are 100
def extract_n_steps_from_stream(
    experience: Timestep, transition_len: int
) -> List[Timestep]:
    if experience.reward.max() <= 0:
        return []

    end_indices = jnp.where(experience.step_type == StepType.TERMINATION)
    segments = []
    for batch_idx, end_idx in zip(*end_indices):
        # if the terminal reward is not 1, skip
        if experience.reward[batch_idx, end_idx] < 1:
            continue

        start_idx = max(0, end_idx - transition_len)
        segment = experience[batch_idx, start_idx + 1: end_idx + 1]

        # if there is a start in segment, take the latest one
        start_idx = jnp.where(segment.is_first())[0]
        if len(start_idx) > 0:
            segment = segment[start_idx[-1]:]

        # set mask
        segment.info["mask"] = jnp.ones_like(segment.t)

        # make sure there is at most 1 start in the segment
        assert len(jnp.where(segment.t == jnp.asarray(0))[0]) <= 1

        # Pad with zeros if the segment is too short
        def pad(x, length):
            padding = jnp.zeros((length, *x.shape[1:]))
            return jnp.concatenate([padding, x], axis=0)

        if segment.shape[0] < transition_len:
            length = transition_len - segment.shape[0]
            segment = jtu.tree_map(lambda x: pad(x, length), segment)

        assert len(segment.t) == transition_len
        if len(jnp.where(segment.step_type == StepType.TERMINATION)[0]) != 1:
            print("Multiple ends in segment", segment)
        # assert len(jnp.where(segment.step_type == StepType.TERMINATION)[0]) == 1

        segments.append(segment)

    return segments


def run_experiment(
    agent: Agent,
    env: Environment,
    key: KeyArray,
    project_name: str | None = None,
    **kwargs,
):
    device_cpu = jax.devices("cpu")[0]

    # init wandb
    config = {**asdict(agent.hparams), **kwargs, "phase": "demo"}
    wandb.init(project=project_name, config=config)

    llm_transition_len = kwargs.pop("llm_transition_len")

    # prepare for saving
    filename, i = args.filename, 0
    while os.path.exists(f"{filename}_{i}.pkl"):
        i += 1
    filename = f"{filename}_{i}.pkl"
    out_path = os.path.join("demonstrations", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # init values
    budget = kwargs["budget"]
    frames = jnp.asarray(0)
    iteration = jnp.asarray(0)
    timestep = env.reset(key)
    if "return" not in timestep.info:
        timestep.info["return"] = timestep.reward
    buffer = []

    # run experiment
    while frames < budget:
        # step
        k1, k2, key = jax.random.split(key, num=3)
        experience, timestep = agent.collect_experience(env, timestep, key=k1)
        agent, log = agent.update(experience, key=k2)

        # add rewarding episodes to buffer
        episodes = extract_n_steps_from_stream(experience, llm_transition_len)
        print("Adding episodes with returns:", [x.reward.sum() for x in episodes])
        episodes = [jax.device_put(x, device_cpu) for x in episodes]
        buffer.extend(episodes)

        # log frames and iterations
        log["buffer_size"] = jnp.asarray(len(buffer))
        log["frames"] = frames
        log["iteration"] = iteration

        # log episode length
        final_t = experience.t[experience.is_last()]
        if final_t.size > 0:
            log["train/episode_length"] = jnp.mean(final_t)
            log["train/min_episode_length"] = jnp.min(final_t)
            log["train/max_episode_length"] = jnp.max(final_t)

        # log rewards
        log["train/average_reward"] = jnp.mean(experience.reward)
        log["train/min_reward"] = jnp.min(experience.reward)
        log["train/max_reward"] = jnp.max(experience.reward)

        # log returns
        if "return" in experience.info:
            return_ = experience.info["return"]
            return_ = return_[experience.is_last()]
            if return_.size > 0:
                log["train/return"] = jnp.mean(return_)
                log["train/min_return"] = jnp.min(return_)
                log["train/max_return"] = jnp.max(return_)

        # log success rates
        success_hits = jnp.sum(experience.reward == 1.0, dtype=jnp.int32)
        log["train/success_hits"] = success_hits
        log["train/success_rate"] = success_hits / jnp.sum(experience.is_last())

        # log render
        if frames % 1_000_000 <= (agent.hparams.iteration_size + 1):
            start_t = experience.t[0][experience.is_first()[0]]
            end_t = experience.t[0][experience.is_last()[0]]
            if start_t.size > 0 and end_t.size > 0:
                render = experience.observation[0, start_t[0] : end_t[0]]
                log["train/render"] = wandb.Video(  #  type: ignore
                    np.asarray(render.transpose(0, 3, 1, 2)), fps=1
                )

        # print and push log
        print(log)
        wandb.log(log)

        frames += experience.t.size
        iteration += 1

        # save buffer to file
        # with jax.default_device(device_cpu):
        #     buffer = jtu.tree_map(lambda *x: jnp.stack(x), *buffer)
        with open(out_path, "wb") as f:
            pickle.dump(buffer, f)


def main():
    hparams = HParams(
        beta=0.01,
        clip_ratio=0.2,
        n_actors=args.n_actors,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        discount=args.discount,
        lambda_=args.lambda_,
        iteration_size=args.iteration_size,
    )
    actions = [
        nethack.CompassCardinalDirection.N,
        nethack.CompassCardinalDirection.E,
        nethack.CompassCardinalDirection.S,
        nethack.CompassCardinalDirection.W,
        nethack.Command.PICKUP,
        nethack.Command.APPLY,
    ]
    env = gym.vector.make(
        args.env_name,
        observation_keys=(args.observation_key, "chars", "chars_crop", "message"),
        actions=actions,
        max_episode_steps=100,
        num_envs=args.n_actors,
        asynchronous=True,
    )
    env = UndictWrapper(env, key=args.observation_key)
    env = MiniHackWrapper.wraps(env)
    encoder = nn.Sequential(
        [
            # greyscale,
            nn.Conv(16, (3, 3), (1, 1)),
            nn.tanh,
            nn.Conv(32, (3, 3), (1, 1)),
            nn.tanh,
            nn.Conv(64, (3, 3), (1, 1)),
            nn.tanh,
            Flatten(),
            nn.Dense(256),
            nn.tanh,
            nn.Dense(128),
            nn.tanh,
        ]
    )
    key = jax.random.PRNGKey(args.seed)
    agent = PPO.init(env, hparams, encoder, key=key)

    # run experiment
    run_experiment(agent, env, key, "calf", **args.__dict__)


if __name__ == "__main__":
    main()
