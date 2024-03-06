import gym
import gym.vector

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array
import rlax
import jax.tree_util as jtu
from nle import nethack
from calf.environment import UndictWrapper, MiniHackWrapper
from calf.ppo import PPO, HParams
import flax.linen as nn
from helx.base.mdp import Timestep, StepType
from helx.base.modules import Flatten
from jax.random import KeyArray


# compute the advantage for each timestep
def value_targets(self, episode: Timestep) -> Tuple[Array, Array]:
    """Calculate targets for multiple concatenated episodes.
    For episodes with a total sum of T steps, it computes values and advantage
    in the range [0, T-1]"""
    action = episode.action[1:]  # a_t
    obs = episode.observation[:-1]  # s_t
    step_type = episode.step_type[1:]  # d_t(s_t, a_t)
    q_values = jax.vmap(self.value_fn, in_axes=(None, 0))(
        self.params, obs
    )  # q(s_t, a'_t) \\forall a'
    value = jax.vmap(lambda x, i: x[i])(q_values, action) # q(s_t, a_t)
    value = value * (step_type != StepType.TERMINATION)
    advantage = truncated_gae(
        episode, value, self.hparams.discount, self.hparams.lambda_
    )
    if self.hparams.normalise_advantage:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
    # values and advantages from [0, T-1]
    return value, advantage


def sample_experience(
    self, episodes: Timestep, *, key: KeyArray
) -> Tuple[Array, Array, Array]:
    """Samples a minibatch of transitions from the collected experience with
    the "Shuffle transitions (recompute advantages)" method: see
    https://arxiv.org/pdf/2006.05990.pdf
    Args:
        episodes (Timestep): a Timestep representing a batch of trajectories.
            The first axis is the number of actors, the second axis is time.
        key (Array): a random key to sample actions
    Returns:
        Timestep: a minibatch of transitions (2-steps sars tuples) where the
        first axis is the number of actors * n_steps // n_minibatches
    """
    msg = "`episodes` must be a batch of trajectories with at least two-steps,\
    got `episode.t.ndim` = {} instead".format(
        episodes.t.ndim
    )
    assert episodes.t.ndim == 2, msg
    assert "log_prob" in episodes.info
    values, advantages = jax.vmap(self.value_targets)(episodes)
    episodes.info["value"] = values
    episodes.info["advantage"] = advantages

    # sample transitions
    batch_size = self.hparams.batch_size
    actor_idx = jax.random.randint(
        key, shape=(batch_size,), minval=0, maxval=self.hparams.n_actors
    )
    # exclude last timestep (-1) as it was excluded from the advantage computation
    episode_length = episodes.t.shape[1]
    time_idx = jax.random.randint(
        key, shape=(batch_size,), minval=0, maxval=episode_length - 1
    )
    transitions_t = episodes[actor_idx, time_idx]  # contains s_{t+1}
    transitions_tp1 = episodes[actor_idx, time_idx + 1]  # contains a_t, \pi_t(a_t)
    transitions = jtu.tree_map(
        lambda *x: jnp.stack(x, axis=1), transitions_t, transitions_tp1
    )  # (batch_size, 2)
    return transitions, actor_idx, time_idx


def truncated_gae(
    episode: Timestep, values: Array, discount: float, lambda_: float
) -> Array:
    advantage = rlax.truncated_generalized_advantage_estimation(
        episode.reward[:-1],
        episode.is_mid()[:-1] * discount ** episode.t[:-1],
        lambda_,
        values,
    )
    return jnp.asarray(advantage * episode.is_mid()[:-1])


def collect_experience(env, timestep, *, key):
    key = jax.random.PRNGKey(0)
    actions = jnp.asarray(
        [[2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 4, 4], [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4]]
    )
    actions = jnp.asarray(
        [
            [2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
            [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4],
        ]
    )
    discount = 1.00
    lambda_ = 1.0
    episode_length = actions.shape[1] + 2
    timestep = env.reset(key)
    timestep.info["return"] = timestep.reward

    episode = []
    for t in range(episode_length):
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        episode.append(timestep)
        # step the environment
        action = actions.T[t]
        log_prob = jnp.ones((*action.shape, 6))
        timestep.info["log_prob"] = log_prob
        next_timestep = env.step(k2, timestep, action)
        # log return, if available
        if "return" in timestep.info:
            next_timestep.info["return"] = timestep.info["return"] * (
                timestep.is_mid()
            ) + (next_timestep.reward * discount**next_timestep.t)

        timestep = next_timestep
    # first axis is the number of actors, second axis is time
    trajectories = jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode)

    trajectories = trajectories[:, 2:-1]
    values = jnp.zeros_like(trajectories.reward)
    advantages = jax.vmap(truncated_gae, in_axes=(0, 0, None, None))(
        trajectories, values, discount, lambda_
    )
    trajectories.info["advantage"] = advantages
    return trajectories, timestep


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
    actions = jnp.asarray(
        [[2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 4, 4], [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4]]
    )
    actions = jnp.asarray(
        [
            [2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
            [0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4],
        ]
    )
    discount = 1.00
    lambda_ = 1.0
    episode_length = actions.shape[1] + 2
    timestep = env.reset(key)
    timestep.info["return"] = timestep.reward

    episode = []
    for t in range(episode_length):
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        episode.append(timestep)
        # step the environment
        action = actions.T[t]
        log_prob = jnp.ones((*action.shape, 6))
        timestep.info["log_prob"] = log_prob
        next_timestep = env.step(k2, timestep, action)
        # log return, if available
        if "return" in timestep.info:
            next_timestep.info["return"] = timestep.info["return"] * (
                timestep.is_mid()
            ) + (next_timestep.reward * discount**next_timestep.t)

        timestep = next_timestep
    # first axis is the number of actors, second axis is time
    trajectories = jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode)

    trajectories = trajectories[:, 2:-1]
    values = jnp.zeros_like(trajectories.reward)
    advantages = jax.vmap(truncated_gae, in_axes=(0, 0, None, None))(
        trajectories, values, discount, lambda_
    )
    trajectories.info["advantage"] = advantages

    # episodes = []
    # for t in range(episode_length):
    #     k1, k2, key = jax.random.split(key, num=3)
    #     k1 = jax.random.split(k1, num=env.action_space.shape[0])
    #     action, log_prob = batch_policy(self.params, timestep.observation, k1)
    #     timestep.info["log_prob"] = log_prob
    #     episodes.append(timestep)
    #     # cache return
    #     return_ = timestep.info["return"] * (timestep.is_mid())
    #     # step the environment
    #     timestep = env.step(k2, timestep, action)
    #     # update return
    #     timestep.info["return"] = return_ + (
    #         timestep.reward * self.hparams.discount**timestep.t
    #     )
    # # first axis is the number of actors, second axis is time
    # trajectories = jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episodes)

    # # update current return
    # return trajectories, timestep

    return trajectories


def test_step():
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
    env = MiniHackWrapper.wraps(env)
    key = jax.random.PRNGKey(0)
    hparams = HParams(advantage_normalisation=False)
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
    agent = PPO.init(env, hparams, network, key=key)

    timestep = env.reset(key)

    for i in range(10):
        experience, timestep = collect_experience(env, timestep, key=key)
        transitions = sample_experience(agent, experience, key=key)
        print(transitions)


if __name__ == "__main__":
    # test_unroll()
    test_step()
