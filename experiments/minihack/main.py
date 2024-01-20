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
import jax.tree_util as jtu
from jax.random import KeyArray
import flax.linen as nn

from helx.base.modules import Flatten
from helx.base.mdp import Timestep

from calf.ppo import HParams, PPO
from calf.nethack import UndictWrapper, MiniHackWrapper



argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="MiniHack-KeyRoom-S5-v0")
argparser.add_argument("--num_envs", type=int, default=3)
argparser.add_argument("--num_epochs", type=int, default=4)
argparser.add_argument("--num_minibatches", type=int, default=4)
argparser.add_argument("--num_steps", type=int, default=1)
argparser.add_argument("--budget", type=int, default=100_000)
argparser.add_argument("--discount", type=float, default=1.0)
argparser.add_argument("--lambda_", type=float, default=1.0)
args = argparser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)


def run_episode(agent: PPO, env: MiniHackWrapper, key: KeyArray) -> Timestep:
    def sample_action(params, observation, key):
        action_distribution = agent.policy(params, observation)
        action = action_distribution.sample(
            seed=key, sample_shape=env.action_space.shape[1:]
        )
        return action

    sample_action = jax.jit(jax.vmap(sample_action, in_axes=(None, 0, 0)))
    timestep = env.reset(key)
    episode = []
    timestep.info["return"] = timestep.reward
    while True:
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        return_ = timestep.info["return"] * (timestep.is_mid())
        action = sample_action(agent.params, timestep.observation, k1)
        timestep = env.step(k2, timestep, action)
        timestep.info["return"] = return_ + (
            timestep.reward * agent.hparams.discount**timestep.t
        )

        episode.append(timestep)
        if timestep.is_last().all():
            break

    return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode)


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
        agent, log = agent.update(experience, timestep, key=k2)

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
        wandb.log(log)

        # evaluate
        if iteration % 1000 == 0:
            num_eval_episodes = 5
            log = {}
            for _ in range(num_eval_episodes):
                episode = run_episode(agent, env, key)
                returns = jnp.mean(episode[episode.is_last()].info["return"])
                log["val/return"] = returns
                log["val/episode_length"] = episode.t[episode.is_last()].mean()
                log["val/average_reward"] = jnp.mean(episode.reward)
                log["val/min_reward"] = jnp.mean(jnp.min(episode.reward, axis=-1))
                log["val/max_reward"] = jnp.mean(jnp.max(episode.reward, axis=-1))
                log["val/episode"] = wandb.Video(
                    np.asarray(episode.observation[0]).transpose(0, 3, 1, 2), fps=1
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
        n_actors=args.num_envs,
        n_epochs=args.num_epochs,
        batch_size=args.num_minibatches,
        discount=args.discount,
        lambda_=args.lambda_,
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
        observation_keys=("pixel_crop",),
        actions=actions,
        max_episode_steps=100,
        num_envs=args.num_envs,
        asynchronous=True,
    )
    env = UndictWrapper(env, key="pixel_crop")
    env = MiniHackWrapper.wraps(env)
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
    agent = PPO.init(env, hparams, network, key=key)

    # run experiment
    run_experiment(agent, env, key, **args.__dict__)


if __name__ == "__main__":
    main()
