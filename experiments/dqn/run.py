from __future__ import annotations
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
argparser.add_argument("--observation_key", type=str, default="glyphs")
argparser.add_argument("--ablation", type=str, default="full")
argparser.add_argument("--beta", type=float, default=0.1)
argparser.add_argument(
    "--annotations_path",
    type=str,
    default="/scratch/uceeepi/calf/experiment_2/ann_full.pkl",
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
import jax.numpy as jnp
import flax.linen as nn

from helx.base.modules import Flatten
from helx.base.mdp import Timestep
import rlax

from calm.trial import Experiment, Agent
from calm.ppo import HParams, PPO
from calm.environment import UndictWrapper, MiniHackWrapper, LLMTableWrapper


def batch_returns(rewards, discounts):
    return jax.jit(jax.vmap(rlax.discounted_returns))(
        rewards, discounts, jnp.zeros_like(rewards)
    )


class CalfExperiment(Experiment):
    def update(self, agent: Agent, experience: Timestep, key: jax.Array):
        agent, log = super().update(agent, experience, key=key)
        log["calf/avg_intrinsic_reward"] = jnp.mean(experience.info["intrinsic_reward"])
        log["calf/avg_extrinsic_reward"] = jnp.mean(experience.info["extrinsic_reward"])
        rewards = experience.info["extrinsic_reward"]
        discounts = agent.hparams.discount**experience.t * (
            experience.step_type != jnp.asarray(2)
        )
        extrinsic_returns = jnp.asarray(
            jax.vmap(rlax.discounted_returns)(
                rewards, discounts, jnp.zeros_like(rewards)
            )
        )
        extrinsic_returns = extrinsic_returns[experience.is_last()]
        if extrinsic_returns.size > 0:
            log["calf/avg_extrinsic_return"] = jnp.mean(extrinsic_returns)
            log["calf/min_extrinsic_return"] = jnp.min(extrinsic_returns)
            log["calf/max_extrinsic_return"] = jnp.max(extrinsic_returns)
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
            "chars",
        ),
        actions=actions,
        max_episode_steps=100,
        num_envs=args.n_actors,
        asynchronous=True,
        seeds=[[args.seed] * args.n_actors],
    )
    env = UndictWrapper(env, key=args.observation_key)
    env = LLMTableWrapper(env, table_path=args.annotations_path, beta=args.beta)
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
    key = jnp.asarray(jax.random.PRNGKey(args.seed))
    agent = PPO.init(env, hparams, encoder, key=key)

    # run experiment
    config = {**args.__dict__, **{"phase": "calf", "ablation": args.ablation}}
    experiment = CalfExperiment("calf", config)
    experiment.run(agent, env, key)


if __name__ == "__main__":
    main()
