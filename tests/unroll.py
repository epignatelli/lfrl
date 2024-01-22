import gym
import gym.vector
from gym.wrappers.autoreset import AutoResetWrapper

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from calf.nethack import UndictWrapper, MiniHackWrapper
from nle import nethack


def test_unroll():
    action_set = [
        nethack.CompassCardinalDirection.N,
        nethack.CompassCardinalDirection.E,
        nethack.CompassCardinalDirection.S,
        nethack.CompassCardinalDirection.W,
        nethack.Command.PICKUP,
        nethack.Command.APPLY,
    ]
    env = gym.vector.make(
        "MiniHack-Room-5x5-v0",
        observation_keys=("pixel_crop",),
        actions=action_set,
        max_episode_steps=100,
        num_envs=2,
        asynchronous=True,
    )
    env = UndictWrapper(env, key="pixel_crop")
    # env = AutoResetWrapper(env)
    env = MiniHackWrapper.wraps(env)
    key = jax.random.PRNGKey(0)
    actions = jnp.asarray([
        [2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 4, 4],
        [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4]
    ])
    actions = jnp.asarray([
        [2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
        [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4]
    ])
    discount = 1.00
    episode_length = actions.shape[1] + 2
    timestep = env.reset(key)
    timestep.info["return"] = timestep.reward
    episodes = []
    for t in range(episode_length):
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        episodes.append(timestep)
        # step the environment
        next_timestep = env.step(k2, timestep, actions.T[t])
        # update return
        next_timestep.info["return"] = timestep.info["return"] * timestep.is_mid() + (
            next_timestep.reward * discount**next_timestep.t
        )
        timestep = next_timestep
    # first axis is the number of actors, second axis is time
    trajectories = jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episodes)

    return trajectories

if __name__ == "__main__":
    test_unroll()