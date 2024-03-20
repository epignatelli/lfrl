from __future__ import annotations
import argparse
from typing import List, Tuple

from calm.io import get_next_valid_path

argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="MiniHack-Room-5x5-v0")
# argparser.add_argument("--env_name", type=str, default="MiniHack-KeyRoom-S5-v0")

argparser.add_argument("--budget", type=int, default=10_000_000)
argparser.add_argument("--n_actors", type=int, default=2)
argparser.add_argument("--n_epochs", type=int, default=4)
argparser.add_argument("--batch_size", type=int, default=128)
argparser.add_argument("--iteration_size", type=int, default=2048)
argparser.add_argument("--discount", type=float, default=0.99)
argparser.add_argument("--lambda_", type=float, default=0.95)
argparser.add_argument("--observation_key", type=str, default="pixel_crop")
args = argparser.parse_args()

import random

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import torch

torch.manual_seed(args.seed)

import gym
import gym.vector
import minihack
from nle import nethack

import jax
import jax.numpy as jnp
import flax.linen as nn

from helx.base.modules import Flatten
from helx.base.mdp import Timestep
from helx.envs.environment import Environment

from calm.trial import Agent, Experiment
from calm.calf import HParams, PPO
from calm.environment import UndictWrapper, MiniHackWrapper


class UniformExperiment(Experiment):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    def redistribute_rewards(self, experience: Timestep) -> Timestep:
        """Redistributes terminal returns uniformly across the episode"""
        time_mask = jnp.cumsum(experience.t == 0)

        episodes_id = jnp.unique(time_mask, size=20)
        for i in episodes_id:
            episode_mask = time_mask == i
            reward = experience.reward * (episode_mask)
            reward = jnp.where(episode_mask, reward.sum() / sum(episode_mask), reward)
        return experience.replace(reward=reward)

    def collect_experience(
        self, agent: Agent, env: Environment, timestep: Timestep, *, key: jax.Array
    ) -> Tuple[Timestep, Timestep]:
        # collect experience
        experience, timestep = super().collect_experience(agent, env, timestep, key=key)
        # redistribute rewards
        experience = jax.jit(jax.vmap(self.redistribute_rewards))(experience)
        return experience, timestep

    def update(self, agent, experience, key):

        # update agent
        agent, log = super().update(agent, experience, key=key)

        return agent, log


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
        observation_keys=(
            args.observation_key,
            # "chars",
            # "chars_crop",
            # "message",
            # "blstats",
        ),
        actions=actions,
        max_episode_steps=100,
        num_envs=args.n_actors,
        asynchronous=True,
        seeds=[args.seed] * args.n_actors,
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
    key = jnp.asarray(jax.random.PRNGKey(args.seed))
    agent = PPO.init(env, hparams, encoder, key=key)

    # run experiment
    config = args.__dict__
    experiment = UniformExperiment("uniform", config)
    experiment.run(agent, env, key)


if __name__ == "__main__":
    main()
