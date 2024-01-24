import argparse

import random
from dataclasses import asdict
import numpy as np
import gym
import gym.vector
import minihack
from nle import nethack
import wandb

import jax
import jax.numpy as jnp
from jax.random import KeyArray
import flax.linen as nn

from helx.base.modules import Flatten

from calf.ppo import HParams, PPO
from calf.nethack import UndictWrapper, MiniHackWrapper


argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="MiniHack-Room-5x5-v0")

argparser.add_argument("--budget", type=int, default=10_000_000)
argparser.add_argument("--n_actors", type=int, default=3)
argparser.add_argument("--n_epochs", type=int, default=4)
argparser.add_argument("--batch_size", type=int, default=4)
argparser.add_argument("--iteration_size", type=int, default=30)
argparser.add_argument("--discount", type=float, default=1.0)
argparser.add_argument("--lambda_", type=float, default=1.0)
argparser.add_argument("--observation_key", type=str, default="pixel_crop")
args = argparser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)


def run_experiment(agent: PPO, env: MiniHackWrapper, key: KeyArray, **kwargs):
    config = {**asdict(agent.hparams), **kwargs}
    wandb.init(project="calf", config=config)
    budget = kwargs["budget"]

    frames = 0
    iteration = 0
    timestep = env.reset(key)
    timestep.info["return"] = timestep.reward
    while True:
        # step
        k1, k2, key = jax.random.split(key, num=3)
        experience, timestep = agent.collect_experience(env, timestep, key=k1)
        agent, log = agent.update(experience, key=k2)

        # extract returns
        if "return" in experience.info:
            return_ = experience.info["return"]
            return_ = return_[experience.is_last()]
            if return_.size > 0:
                log["train/return"] = jnp.mean(return_)

        # extraxt episode length
        final_t = experience.t[experience.is_last()]
        if final_t.size > 0:
            log["train/episode_length"] = jnp.mean(final_t)

        # log
        log["frames"] = frames
        log["iteration"] = iteration
        log["train/average_reward"] = jnp.mean(experience.reward)
        log["train/min_reward"] = jnp.mean(jnp.min(experience.reward, axis=-1))
        log["train/max_reward"] = jnp.mean(jnp.max(experience.reward, axis=-1))
        print(log)
        if frames % 1_000_000 <= (agent.hparams.iteration_size + 1):
            start_t = experience.t[0][experience.is_first()[0]]
            end_t = experience.t[0][experience.is_last()[0]]
            if start_t.size > 0 and end_t.size > 0:
                render = experience.observation[0, start_t[0] : end_t[0]]
                log["train/render"] = wandb.Video(
                    np.asarray(render.transpose(0, 3, 1, 2)), fps=1
                )
        wandb.log(log)

        if frames > budget:
            break

        frames += experience.t.size
        iteration += 1


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
        observation_keys=(args.observation_key,),
        actions=actions,
        max_episode_steps=100,
        num_envs=args.n_actors,
        asynchronous=True,
    )
    env = UndictWrapper(env, key=args.observation_key)
    env = MiniHackWrapper.wraps(env)
    network = nn.Sequential(
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
    agent = PPO.init(env, hparams, network, key=key)

    # run experiment
    run_experiment(agent, env, key, **args.__dict__)


if __name__ == "__main__":
    main()
