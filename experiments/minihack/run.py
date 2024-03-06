from __future__ import annotations
import argparse

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
argparser.add_argument(
    "--buffer_path", type=str, default="/scratch/uceeepi/calf/redistribution_alt/annotations_1.pkl2"
)
args = argparser.parse_args()

import random

random.seed(args.seed)

import numpy as np

np.random.seed(args.seed)

import gym
import gym.vector
import minihack
from nle import nethack

import jax
import flax.linen as nn

from helx.base.modules import Flatten

from calf.trial import Experiment
from calf.calf import HParams, CALF
from calf.environment import UndictWrapper, MiniHackWrapper


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
        buffer_path=args.buffer_path,
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
        observation_keys=(args.observation_key, "chars"),
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
    agent = CALF.init(env, hparams, encoder, key=key)

    # run experiment
    config = {**args.__dict__, **{"phase": "calf"}}
    experiment = Experiment("calf", config)
    experiment.run(agent, env, key)


if __name__ == "__main__":
    main()
