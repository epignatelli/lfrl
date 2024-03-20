from __future__ import annotations
import argparse
import random


import numpy as np
import torch
import gym
import gym.vector
from gym.wrappers.autoreset import AutoResetWrapper
import minihack
from nle import nethack
import jax
import flax.linen as nn
from helx.base.modules import Flatten

from calm.trial import Experiment
from calm.dqn import HParams, DQN
from calm.environment import UndictWrapper, MiniHackWrapper


def main(argv):
    hparams = HParams(
        batch_size=argv.batch_size,
        discount=argv.discount,
        initial_exploration=argv.initial_exploration,
        final_exploration=argv.final_exploration,
        final_exploration_frame=argv.final_exploration_frame,
        replay_start=argv.replay_start,
        replay_memory_size=argv.replay_memory_size,
        update_frequency=argv.update_frequency,
        target_network_update_frequency=argv.target_network_update_frequency,
        num_envs=argv.num_envs
    )
    actions = [
        nethack.CompassCardinalDirection.N,
        nethack.CompassCardinalDirection.E,
        nethack.CompassCardinalDirection.S,
        nethack.CompassCardinalDirection.W,
        nethack.Command.PICKUP,
        nethack.Command.APPLY,
    ]
    # env = gym.make(
    #     argv.env_name,
    #     observation_keys=(
    #         argv.observation_key,
    #         "chars",
    #         "message",
    #     ),
    #     actions=actions,
    #     max_episode_steps=100,
    #     seeds=[argv.seed],
    # )
    # env = AutoResetWrapper(env)
    env = gym.vector.make(
        args.env_name,
        observation_keys=(
            args.observation_key,
            "chars",
            "chars_crop",
            "message",
            "blstats",
        ),
        actions=actions,
        max_episode_steps=100,
        num_envs=args.num_envs,
        asynchronous=True,
        seeds=[[args.seed]] * args.num_envs,
    )
    env = UndictWrapper(env, key=argv.observation_key)
    env = MiniHackWrapper.wraps(env)
    encoder = nn.Sequential(
        [
            # greyscale,
            # nn.Conv(16, (5, 5), (1, 1)),
            # nn.tanh,
            # nn.Conv(32, (2, 2), (1, 1)),
            # nn.tanh,
            # nn.Conv(64, (2, 2), (1, 1)),
            # nn.tanh,
            Flatten(),
            nn.Dense(2048),
            nn.tanh,
            nn.Dense(1024),
            nn.tanh,
            nn.Dense(512),
            nn.tanh,
        ]
    )
    key = jax.numpy.asarray(jax.random.PRNGKey(argv.seed))
    agent = DQN.init(env, hparams, encoder, key=key)

    # run experiment
    config = argv.__dict__
    experiment = Experiment("dqn", config)
    experiment.run(agent, env, key)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--env_name", type=str, default="MiniHack-Room-5x5-v0")
    # argparser.add_argument("--env_name", type=str, default="MiniHack-KeyRoom-S5-v0")

    argparser.add_argument("--initial_exploration", type=float, default=1.0)
    argparser.add_argument("--final_exploration", type=float, default=0.01)
    argparser.add_argument("--final_exploration_frame", type=int, default=1000000)
    argparser.add_argument("--replay_start", type=int, default=1000)
    argparser.add_argument("--replay_memory_size", type=int, default=1000)
    argparser.add_argument("--update_frequency", type=int, default=1)
    argparser.add_argument("--target_network_update_frequency", type=int, default=10000)

    argparser.add_argument("--budget", type=int, default=10_000_000)
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--discount", type=float, default=0.99)
    argparser.add_argument("--observation_key", type=str, default="glyphs")
    argparser.add_argument("--num_envs", type=int, default=2)
    args = argparser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
