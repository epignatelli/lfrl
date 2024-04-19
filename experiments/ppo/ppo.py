from __future__ import annotations

import random

import numpy as np
import torch
import gym
import gym.vector
import minihack
from nle import nethack
import jax

from calm.trial import Experiment
from calm.ppo import HParams, PPO
from calm.environment import UndictWrapper, MiniHackWrapper
from models import get_nethack_encoder


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
            "glyphs",
            "chars_crop",
            "blstats",
        ),
        actions=actions,
        max_episode_steps=100,
        num_envs=argv.n_actors,
        asynchronous=True,
        seeds=[[argv.seed]] * argv.n_actors,
    )
    env = UndictWrapper(env, key=argv.observation_key)
    env = MiniHackWrapper.wraps(env)

    encoder = get_nethack_encoder()
    key = jax.numpy.asarray(jax.random.PRNGKey(argv.seed))
    agent = PPO.init(env, hparams, encoder, key=key)

    # run experiment
    config = argv.__dict__
    config["phase"] = "baselines"
    config["algo"] = "ppo"
    experiment = Experiment("calm", config)
    experiment.run(agent, env, key)


if __name__ == "__main__":
    import argparse

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
    argparser.add_argument("--observation_key", type=str, default="message")
    argparser.add_argument("--log_compiles", action="store_true", default=False)
    args = argparser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
