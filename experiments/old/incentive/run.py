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
argparser.add_argument("--out_dir", type=str, default="/scratch/uceeepi/calf")
args = argparser.parse_args()

import random

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import torch

torch.manual_seed(args.seed)

import os
from stat import S_IREAD, S_IRGRP, S_IROTH
from collections import deque
import time
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

from calm.trial import Agent, run_experiment
from calm.calf import HParams, PPO
from calm.environment import UndictWrapper, MiniHackWrapper


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
    run_experiment(agent, env, key, "incentive", **args.__dict__)


if __name__ == "__main__":
    main()
