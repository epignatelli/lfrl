from __future__ import annotations
import argparse
from typing import List
import random
import os
from stat import S_IREAD, S_IRGRP, S_IROTH
import sys

import numpy as np
import torch
import pickle
import gym
import gym.vector
import minihack
from nle import nethack
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import flax.linen as nn
from helx.base.modules import Flatten
from helx.base.mdp import Timestep, StepType

from calm.trial import Experiment
from calm.ppo import HParams, PPO
from calm.environment import UndictWrapper, MiniHackWrapper


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
        # experience has canonical structure, so we index it via `experience.at`
        segment = experience[batch_idx].at_time[start_idx:end_idx]
        # arrays in `segment` are now aligned in time (s_0, a_0, ..., s_T, a_T)

        # if there is a start in segment, take the latest one
        start_idx = jnp.where(segment.is_first())[0]
        if len(start_idx) > 0:
            segment = segment[start_idx[-1] :]

        # if there is a truncation, take the lastest one
        trunc_idx = jnp.where(segment.step_type == jnp.asarray(1))[0]
        if len(trunc_idx) > 0:
            segment = segment[trunc_idx[-1] :]

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
            print("Multiple ends in segment", segment.step_type)

        segments.append(segment)

    return segments


class DemoExperiment(Experiment):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        # prepare for saving
        out_path = config["dest_path"]
        # init
        self.out_path = out_path
        if os.path.exists(out_path):
            remove = input(f"Overwrite file {out_path} [y/N]?  ")
            if remove.lower() in ["y", "yes"]:
                print(f"Overwriting file {out_path}")
                os.remove(out_path)
            else:
                sys.exit()
        self.file = open(out_path, "ab")
        self.buffer_len = jnp.asarray(0)

    def update(self, agent, experience, key):
        # store demonstrations
        llm_transition_len = self.config["llm_transition_len"]
        episodes = extract_n_steps_from_stream(experience, llm_transition_len)
        self.buffer_len += len(episodes)
        if len(episodes) > 0:
            pickle.dump(episodes, self.file)
        # update agent
        agent, log = super().update(agent, experience, key=key)
        # log buffer size
        log["demonstrate/buffer_size"] = self.buffer_len
        return agent, log

    def close(self):
        super().close()
        # close the file
        self.file.close()
        # set demo file in readonly mode to avoid disasters (yes, it happened, once!)
        os.chmod(self.out_path, S_IREAD | S_IRGRP | S_IROTH)
        return


def main(argv):
    hparams = HParams(
        beta=0.01,
        clip_ratio=0.2,
        n_actors=argv.n_actors,
        n_epochs=argv.n_epochs,
        batch_size=argv.batch_size,
        discount=argv.discount,
        lambda_=argv.lambda_,
        iteration_size=argv.iteration_size,
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
        argv.env_name,
        observation_keys=(
            argv.observation_key,
            "chars",
            "chars_crop",
            "message",
            "blstats",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
        ),
        actions=actions,
        max_episode_steps=100,
        num_envs=argv.n_actors,
        asynchronous=True,
        seeds=[[argv.seed]] * argv.n_actors,
    )
    env = UndictWrapper(env, key=argv.observation_key)
    env = MiniHackWrapper.wraps(env)
    encoder = nn.Sequential(
        [
            Flatten(),
            nn.Dense(2048),
            nn.tanh,
            nn.Dense(1024),
            nn.tanh,
            nn.Dense(512),
            nn.tanh,
        ]
    )
    key = jnp.asarray(jax.random.PRNGKey(argv.seed))
    agent = PPO.init(env, hparams, encoder, key=key)

    # run experiment
    config = {**argv.__dict__, **{"phase": "demo"}}
    experiment = DemoExperiment("calf", config)
    experiment.run(agent, env, key)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--env_name", type=str, default="MiniHack-KeyRoom-S5-v0")

    argparser.add_argument("--budget", type=int, default=10_000_000)
    argparser.add_argument("--n_actors", type=int, default=2)
    argparser.add_argument("--n_epochs", type=int, default=4)
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--iteration_size", type=int, default=2048)
    argparser.add_argument("--discount", type=float, default=0.99)
    argparser.add_argument("--lambda_", type=float, default=0.95)
    argparser.add_argument("--observation_key", type=str, default="glyphs")
    argparser.add_argument("--max_buffer_size", type=int, default=10_000)
    argparser.add_argument("--llm_transition_len", type=int, default=100)

    argparser.add_argument("--dest_path", type=str, default="/scratch/uceeepi/calf/demonstrations/demo_3.pkl")
    args = argparser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
