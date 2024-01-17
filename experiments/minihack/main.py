import argparse

import gym
import gym.vector
import minihack
from nle import nethack
import wandb

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from helx.base.modules import Flatten

from calf.ppo import HParams, PPO
from calf.nethack import UndictWrapper, MiniHackWrapper


argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="MiniHack-KeyRoom-S5-v0")
argparser.add_argument("--num_envs", type=int, default=3)
argparser.add_argument("--num_epochs", type=int, default=4)
argparser.add_argument("--num_minibatches", type=int, default=4)
argparser.add_argument("--num_steps", type=int, default=1)
argparser.add_argument("--max_episodes", type=int, default=10000)
args = argparser.parse_args()


def main():
    hparams = HParams(
        beta=0.01,
        clip_ratio=0.2,
        n_actors=args.num_envs,
        n_epochs=args.num_epochs,
        batch_size=args.num_minibatches,
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
        "MiniHack-KeyRoom-Fixed-S5-v0",
        observation_keys=("pixel_crop",),
        max_episode_steps=100,
        actions=actions,
        num_envs=args.num_envs,
        asynchronous=True,
    )
    env = UndictWrapper(env, key="pixel_crop")
    env = MiniHackWrapper.wraps(env)
    optimiser = optax.adam(learning_rate=hparams.learning_rate)
    network = nn.Sequential(
        [
            nn.Conv(64, (3, 3), (1, 1)),
            nn.tanh,
            nn.Conv(32, (3, 3), (1, 1)),
            nn.tanh,
            nn.Conv(16, (3, 3), (1, 1)),
            nn.tanh,
            Flatten(),
            nn.Dense(256),
            nn.tanh,
            nn.Dense(128),
            nn.tanh,
        ]
    )
    key = jax.random.PRNGKey(args.seed)
    agent = PPO.init(env, hparams, optimiser, network, key=key)

    # run experiment
    wandb.init(mode="disabled")
    iteration = 0
    for _ in range(args.max_episodes):
        k1, k2, key = jax.random.split(key, num=3)
        experience = agent.collect_experience(env, key=k1)
        agent, log = agent.update(experience, key=k2)

        # inject iteration into log
        timestep_elapsed = jnp.prod(jnp.asarray(experience.t.shape))
        iteration += timestep_elapsed
        log["iteration"] = iteration
        print(log)
        wandb.log(log)


if __name__ == "__main__":
    main()
