from __future__ import annotations
from functools import partial
from typing import Dict, Tuple

import distrax
import flax.linen as nn
from flax import struct
from flax.core.scope import VariableDict as Params
from jax import Array
import jax
from jax.random import KeyArray
from jax.lax import stop_gradient
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import rlax

from helx.base.spaces import Discrete
from helx.base.mdp import Timestep, StepType
from helx.envs.environment import Environment


class HParams(struct.PyTreeNode):
    discount: float = 0.99
    """The MDP discount factor."""
    lambda_: float = 0.95
    """The lambda parameter for the TD(lambda) algorithm."""
    learning_rate: float = 0.00025
    """The learning rate of the gradient descent algorithm."""
    beta: float = 0.01
    """Entropy bonus coefficient."""
    clip_ratio: float = 0.2
    """The epsilon parameter in the PPO paper."""
    n_actors: int = 2
    """The number of actors to use."""
    n_epochs: int = 4
    """The number of epochs to train for each update."""
    iteration_size: int = 128
    """The number of steps to collect in total from all environments at update."""
    batch_size: int = 16
    """The number of minibatches to run at each epoch."""
    gradient_clip_norm: float = 0.5
    """The maximum norm of the gradient"""
    advantage_normalisation: bool = False
    """Whether to normalise the advantage function as per
    https://arxiv.org/pdf/2006.05990.pdf"""
    value_clipping: bool = True
    """Whether to clip the value function estimates as per
    https://arxiv.org/pdf/2005.12729.pdf."""
    value_loss_coefficient: float = 0.5
    """The multiplying coefficient for the value loss as per
    https://arxiv.org/pdf/2006.05990.pdf"""


class PPO(struct.PyTreeNode):
    """Proximal Policy Optimisation as described
    in https://arxiv.org/abs/1707.06347
    Implementation details as per recommendations
    in https://arxiv.org/pdf/2006.05990.pdf and https://arxiv.org/pdf/2005.12729.pdf
    """

    # static:
    hparams: HParams = struct.field(pytree_node=False)
    optimiser: optax.GradientTransformation = struct.field(pytree_node=False)
    actor: nn.Module = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)
    # dynamic:
    params: Params = struct.field(pytree_node=True)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        backbone: nn.Module,
        *,
        key: KeyArray,
    ) -> PPO:
        assert isinstance(env.action_space, Discrete)
        assert (
            env.observation_space.shape[0]
            == env.action_space.shape[0]
            == hparams.n_actors
        )
        actor = nn.Sequential(
            [
                backbone.clone(),
                nn.Dense(
                    env.action_space.maximum + 1,
                    kernel_init=nn.initializers.orthogonal(scale=0.01),
                ),
            ]
        )
        critic = nn.Sequential([backbone.clone(), nn.Dense(1)])
        unbatched_obs_sample = env.observation_space.sample(key)[0]
        params_actor = actor.init(key, unbatched_obs_sample)
        params_critic = critic.init(key, unbatched_obs_sample)
        params = {"actor": params_actor, "critic": params_critic}
        optimiser = optax.chain(
            optax.clip_by_global_norm(hparams.gradient_clip_norm),
            optax.adam(hparams.learning_rate),
        )
        opt_state = optimiser.init(params)
        return cls(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            params=params,
            opt_state=opt_state,
        )

    def policy(self, params: Params, observation: Array) -> distrax.Softmax:
        logits = jnp.asarray(self.actor.apply(params["actor"], observation))
        return distrax.Softmax(logits=logits)

    def value_fn(self, params: Params, observation: Array) -> Array:
        return jnp.asarray(self.critic.apply(params["critic"], observation))[0]

    def collect_experience(
        self, env: Environment, timestep: Timestep, *, key: KeyArray
    ) -> Tuple[Timestep, Timestep]:
        """Collects `n_actors` trajectories of experience of length `n_steps`
        from the environment. This method is the only one that interacts with the
        environment, and cannot be jitted unless the environment is JAX-compatible.
        Args:
            env (Environment): the environment must be a batched environment,
                such that `env.step` returns `n_actors` tim
            esteps
            key (Array): a random key to sample actions
        Returns:
            Timestep: a Timestep representing a batch of trajectories.
            The first axis is the number of actors, the second axis is time.
        """
        episode_length = self.hparams.iteration_size // self.hparams.n_actors
        return run_n_steps(env, timestep, self, episode_length, key=key)

    def evaluate_experience(self, episode: Timestep) -> Tuple[Array, Array]:
        """Calculate targets for multiple concatenated episodes.
        For episodes with a total sum of T steps, it computes values and advantage
        in the range [0, T-1]"""
        assert episode.t.ndim == 1, "episode.t.ndim must be 1, got {} instead.".format(
            episode.t.ndim
        )
        obs = episode.observation  # s_t \\forall t \\in [0, T]
        reward = episode.reward[1:]  # r_t \\forall t \\in [0, T-1]
        discount = (episode.is_mid() * self.hparams.discount**episode.t)[
            1:
        ]  # \\gamma^t \\forall t \\in [0, T-1]
        value = jax.vmap(self.value_fn, in_axes=(None, 0))(self.params, obs) * (
            episode.step_type != StepType.TERMINATION
        )  # q(s_t, a'_t) \\forall a'
        advantage = jnp.asarray(
            rlax.truncated_generalized_advantage_estimation(
                reward,
                discount,
                self.hparams.lambda_,
                value,
            )
        )
        # values and advantages from [0, T-1]
        return value, advantage

    def sample_experience(self, episodes: Timestep, *, key: KeyArray) -> Timestep:
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
        assert episodes.t.ndim == 2, "episodes.ndim must be 2, got {} instead.".format(
            episodes.t.ndim
        )
        value, advantage = jax.vmap(self.evaluate_experience)(episodes)
        episodes.info["value"] = value
        episodes.info["advantage"] = advantage

        # sample transitions
        batch_size = self.hparams.batch_size
        actor_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.hparams.n_actors
        )
        # exclude last timestep (-1) as it was excluded from the advantage computation
        episode_length = episodes.t.shape[1]
        time_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=episode_length
        )
        transitions_t = episodes[actor_idx, time_idx]  # contains s_{t+1}
        transitions_tp1 = episodes[actor_idx, time_idx + 1]  # contains s_{t+2}
        return jtu.tree_map(
            lambda *x: jnp.stack(x, axis=1), transitions_t, transitions_tp1
        )  # (batch_size, 2)

    def loss(
        self, params: Params, transition: Timestep
    ) -> Tuple[Array, Dict[str, Array]]:
        # make sure the transition has the required info
        assert "advantage" in transition.info
        assert "log_prob" in transition.info
        # stop gradient on advantage and log_prob
        observation = transition.observation[0]  # s_t
        advantage = stop_gradient(transition.info["advantage"][0])  # A(s_t, a_t)
        action_value_old = stop_gradient(transition.info["value"][0])  # v_{k-1}(s_t)
        action = stop_gradient(transition.action[1])  # a_t
        log_prob_old = stop_gradient(transition.info["log_prob"][0])
        # critic loss
        value = self.value_fn(params, observation)  # v(s_t)
        critic_loss = 0.5 * jnp.square(value - advantage)
        if self.hparams.value_clipping:
            value_clipped = jnp.clip(
                value,
                action_value_old - self.hparams.clip_ratio,
                action_value_old + self.hparams.clip_ratio,
            )
            critic_loss_clipped = 0.5 * jnp.square(value_clipped - advantage)
            critic_loss = jnp.maximum(critic_loss, critic_loss_clipped)
        critic_loss = self.hparams.value_loss_coefficient * critic_loss
        # actor loss
        action_distribution = self.policy(params, observation)
        log_prob = action_distribution.log_prob(action)
        ratio = jnp.exp(log_prob - log_prob_old)
        clipped_ratio = jnp.clip(
            ratio, 1 - self.hparams.clip_ratio, 1 + self.hparams.clip_ratio
        )
        actor_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        # entropy
        entropy = action_distribution.entropy()
        entropy_loss = -jnp.asarray(entropy) * jnp.asarray(self.hparams.beta)
        # total loss
        loss = actor_loss + critic_loss + entropy_loss
        # logs
        log = {
            "losses/loss": loss,
            "losses/critic_loss": critic_loss,
            "losses/actor_loss": actor_loss,
            "losses/entropy_bonus": entropy_loss,
            "losses/entropy": entropy,
        }
        return loss, log

    @jax.jit
    def update(
        self, trajectories: Timestep, *, key: KeyArray
    ) -> Tuple[PPO, Dict[str, Array]]:
        def batch_loss(params, transitions):
            out = jax.vmap(self.loss, in_axes=(None, 0))(params, transitions)
            out = jtu.tree_map(lambda x: x.mean(axis=0), out)
            return out

        params, opt_state, log = self.params, self.opt_state, {}
        for _ in range(self.hparams.n_epochs):
            transitions = self.sample_experience(trajectories, key=key)
            (_, log), grads = jax.value_and_grad(batch_loss, has_aux=True)(
                params, transitions
            )
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            log["losses/grad_norm"] = optax.global_norm(updates)
        return self.replace(params=params, opt_state=opt_state), log


def run_n_steps(
    env: Environment, timestep: Timestep, agent: PPO, n_steps: int, *, key: KeyArray
) -> Tuple[Timestep, Timestep]:
    """Runs `n_steps` in the environment using the agent's policy and returns a
    partial trajectory.

    Args:
        env (Environment): the environment to step in.
        timestep (Timestep): the timestep to start from.
        agent (PPO): the agent.
        n_steps (int): the number of steps to run.
        key (KeyArray): a random key to sample actions.

    Returns:
        Tuple[Timestep, Timestep]: a partial trajectory of length `n_steps` and the
        final timestep.
        The trajectory can contain multiple episodes.
        Each timestep in the trajectory has the following structure:
        $(t, o_t, a_{t-1}, r_{t-1}, step_type_{t-1}, info_{t-1})$
        where:
        $a_{t} \\sim \\pi(s_t)$ is the action sampled from the policy conditioned on
        $s_t$; $r_{t} = R(s_t, a_t)$ is the reward after taking the action $a_t$ in
        $s_t$.

    Example:
    0    (0, s_0,  -1, 0.0,      0, info_0, G_0)  RESET
    1    (1, s_1, a_0, r_0, term_0, info_0, G_1 = G_0 + r_0 * gamma ** 1)
    2    (2, s_2, a_1, r_1, term_1, info_1, G_2 = G_1 + r_1)
    3    (3, s_3, a_2, r_2, term_2, info_2, G_3 = G_2 + r_2)  TERMINATION
    4    (0, s_0,  -1, 0.0,      0, info_3, G_4 = G_3 + 0.0)  RESET
    5    (1, s_1, a_0, r_0, term_0, info_0, G_0 = r_0)
    6    (2, s_2, a_1, r_1, term_1, info_1, G_1 = G_1 + r_1)  TERMINATION
    7    (0, s_0,  -1, 0.0,      0, info_2, G_2 = G_1 + 0.0)  RESET

    idx = 3:
        timestep_t =   (3, s_3, a_2, r_2, term_2, info_2, G_2 = G_1 + r_2)  TERMINATION
        timestep_tp1 = (0, s_0,  -1, 0.0,      0, info_3, G_3 = G_2 + 0.0)  RESET
    """

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def policy_sample(params, observation, key):
        action_distribution = agent.policy(params, observation)
        action, log_prob = action_distribution.sample_and_log_prob(
            seed=key, sample_shape=env.action_space.shape[1:]
        )
        return action, log_prob

    episode = []
    for _ in range(n_steps):
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        episode.append(timestep)
        # step the environment
        action, log_prob = policy_sample(agent.params, timestep.observation, k1)
        timestep.info["log_prob"] = log_prob
        next_timestep = env.step(k2, timestep, action)
        # log return, if available
        if "return" in timestep.info:
            next_timestep.info["return"] = timestep.info["return"] * (
                timestep.is_mid()
            ) + (next_timestep.reward * agent.hparams.discount**timestep.t)

        timestep = next_timestep

    return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode), timestep


def run_episode(env: Environment, agent: PPO, *, key: KeyArray) -> Timestep:
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def policy_sample(params, observation, key):
        action_distribution = agent.policy(params, observation)
        action, log_prob = action_distribution.sample_and_log_prob(
            seed=key, sample_shape=env.action_space.shape[1:]
        )
        return action, log_prob

    timestep = env.reset(key)
    timestep.info["return"] = timestep.reward
    final = timestep.is_last()
    episode = []
    while True:
        k1, k2, key = jax.random.split(key, num=3)
        k1 = jax.random.split(k1, num=env.action_space.shape[0])
        episode.append(timestep)
        # step the environment
        action, log_prob = policy_sample(agent.params, timestep.observation, k1)
        timestep.info["log_prob"] = log_prob
        next_timestep = env.step(k2, timestep, action)
        # log return, if available
        if "return" in timestep.info:
            next_timestep.info["return"] = timestep.info["return"] * (
                timestep.is_mid()
                + (next_timestep.reward * agent.hparams.discount**timestep.t)
            )
        final = jnp.logical_or(final, timestep.is_last())
        if final.all():
            break
        timestep = next_timestep

    return jtu.tree_map(lambda *x: jnp.stack(x, axis=1), *episode)
