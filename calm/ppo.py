from __future__ import annotations
from functools import partial
from typing import Any, Dict, Sequence, Tuple

import distrax
import flax.linen as nn
from flax import struct
import flax.training.train_state
from flax.typing import VariableDict as Params
from jax import Array
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import rlax

from helx.base.spaces import Discrete
from helx.base.mdp import Timestep, StepType
from helx.envs.environment import Environment

from .models import Network
from .trial import Agent, HParams as HParamsBase, RNNState, AgentState


class HParams(HParamsBase):
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
    iteration_size: int = 2048
    """The number of steps to collect in total from all environments at update."""
    batch_size: int = 128
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
    recurrent: bool = True
    """Whether to use a recurrent encoder."""
    n_hidden: int = 512
    """The number of hidden units in the recurrent encoder."""
    prioritised_sampling: bool = False
    """Whether to use td-error-based prioritised sampling"""


class PPO(Agent):
    """Proximal Policy Optimisation as described
    in https://arxiv.org/abs/1707.06347
    Implementation details as per recommendations
    in https://arxiv.org/pdf/2006.05990.pdf and https://arxiv.org/pdf/2005.12729.pdf
    """

    hparams: HParams = struct.field(pytree_node=False)
    optimiser: optax.GradientTransformation = struct.field(pytree_node=False)
    actor: nn.Module = struct.field(pytree_node=False)
    critic: nn.Module = struct.field(pytree_node=False)

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        encoder: nn.Module,
        *,
        key: Array,
    ) -> Tuple[PPO, AgentState]:
        assert isinstance(env.action_space, Discrete)
        assert (
            env.observation_space.shape[0]
            == env.action_space.shape[0]
            == hparams.n_actors
        )
        # agent functions
        actor = Network(
            encoder,
            head=nn.Dense(
                env.action_space.maximum + 1,
                kernel_init=nn.initializers.orthogonal(scale=0.01),
            ),
            recurrent=hparams.recurrent,
            n_hidden=hparams.n_hidden,
        )

        critic = Network(
            encoder,
            head=nn.Dense(1, kernel_init=nn.initializers.zeros_init()),
            recurrent=hparams.recurrent,
            n_hidden=hparams.n_hidden,
        )

        # ppo state
        timestep = env.reset(key)
        obs = (
            timestep.info["observations"]["glyphs"],
            timestep.info["observations"]["chars_crop"],
            timestep.info["observations"]["blstats"],
        )
        unbatched_obs = jtu.tree_map(lambda x: x[0], obs)
        params_actor = actor.init(key, unbatched_obs)
        params_critic = critic.init(key, unbatched_obs)
        params = {"actor": params_actor, "critic": params_critic}
        optimiser = optax.chain(
            optax.clip_by_global_norm(hparams.gradient_clip_norm),
            optax.adam(hparams.learning_rate),
        )
        hidden_state = {}
        if hparams.recurrent:
            (actor_hidden, _), _ = actor.init_with_output(key, unbatched_obs)
            (critic_hidden, _), _ = critic.init_with_output(key, unbatched_obs)
            hidden_state = jtu.tree_map(
                lambda x: jnp.stack([x] * hparams.n_actors),
                {
                    "actor": actor_hidden,
                    "critic": critic_hidden,
                },
            )

        # pack and return
        ppo_state = AgentState(
            opt_state=optimiser.init(params),
            iteration=jnp.asarray(0),
            params=params,
            hidden_state=hidden_state,
        )
        agent = cls(hparams=hparams, optimiser=optimiser, actor=actor, critic=critic)
        return agent, ppo_state

    @jax.jit
    def policy(
        self,
        params: Params,
        observation: Tuple[Array, Array, Array],
        *,
        hidden_state: RNNState,
    ) -> Tuple[RNNState, distrax.Softmax]:
        hidden_state_actor, logits = self.actor.apply(
            params["actor"], observation, hidden_state.get("actor", None)
        )
        hidden_state["actor"] = hidden_state_actor
        return hidden_state, distrax.Softmax(logits=jnp.asarray(logits))

    @jax.jit
    def value_fn(
        self,
        params: Params,
        observation: Tuple[Array, Array, Array],
        *,
        hidden_state: RNNState,
    ) -> Tuple[RNNState, Array]:
        hidden_state_critic, value = self.critic.apply(
            params["critic"], observation, hidden_state.get("critic", None)
        )
        hidden_state["critic"] = hidden_state_critic
        return hidden_state, jnp.asarray(value)[0]

    def collect_experience(
        self,
        env: Environment,
        timestep: Timestep,
        params: Params,
        *,
        key: Array,
        hidden_state: RNNState,
    ) -> Tuple[Timestep, Timestep, RNNState]:
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
        action_shape = env.action_space.shape[1:]

        def policy_sample(
            observation: Tuple[Array, Array, Array],
            key: Array,
            hidden_state: RNNState,
        ) -> Tuple[RNNState, Array, Array]:
            hidden_state, action_distribution = self.policy(
                params, observation, hidden_state=hidden_state
            )
            action, log_prob = action_distribution.sample_and_log_prob(
                seed=key, sample_shape=action_shape
            )
            return hidden_state, jnp.asarray(action), jnp.asarray(log_prob)

        episode = []
        for _ in range(episode_length):
            k1, k2, key = jax.random.split(key, num=3)
            k1 = jax.random.split(k1, num=env.action_space.shape[0])
            episode.append(timestep)
            obs = (
                timestep.info["observations"]["glyphs"],
                timestep.info["observations"]["chars_crop"],
                timestep.info["observations"]["blstats"],
            )
            # step the environment
            hidden_state, action, log_prob = jax.vmap(policy_sample)(
                observation=obs,
                key=jnp.asarray(k1),
                hidden_state=hidden_state,
            )

            # log action log_prob
            timestep.info["log_prob"] = log_prob
            next_timestep = env.step(k2, timestep, action)

            # depure return log from intrinsic rewards
            reward = next_timestep.reward
            if "intrinsic_reward" in timestep.info:
                reward = reward - timestep.info["intrinsic_reward"]

            # log returns
            if "return" in timestep.info:
                next_timestep.info["return"] = timestep.info["return"] * (
                    timestep.is_mid()
                ) + (reward * self.hparams.discount**timestep.t)

            timestep = next_timestep

        batched = jtu.tree_map(lambda *x: jnp.stack(x, axis=env.ndim), *episode)
        return batched, timestep, hidden_state

    def evaluate_experience(self, params: Params, episodes: Timestep) -> Timestep:
        """Calculate targets for multiple concatenated episodes.
        For episodes with a total sum of T steps, it computes values and advantage
        in the range [0, T-1]"""
        assert (
            episodes.t.ndim == 1
        ), "episodes.t.ndim must be 1, got {} instead.".format(episodes.t.ndim)
        # s_t \\forall t \\in [0, T]
        obs = (
            episodes.info["observations"]["glyphs"],
            episodes.info["observations"]["chars_crop"],
            episodes.info["observations"]["blstats"],
        )
        discount = episodes.is_mid() * self.hparams.discount**episodes.t

        hidden_state = episodes.info.get("hidden_state", {})
        _, value = jax.vmap(lambda o, h: self.value_fn(params, o, hidden_state=h))(
            obs, hidden_state
        )

        # mask the value
        value = value * (episodes.step_type != StepType.TERMINATION)

        # calculate GAE
        advantage = jnp.asarray(
            rlax.truncated_generalized_advantage_estimation(
                episodes.reward[1:],  # r_{1}, ..., r_{T}
                discount[1:],  # γ_{1}, ..., γ_{T}
                self.hparams.lambda_,  # scalar
                value,  # v(s_0), ..., v(s_T)
            )
        )  # y(s_0), ..., y(s_{T-1})

        # store the value and advantage in the info dict
        episodes.info["value"] = value  # v(s_0), ..., y(s_T)
        if self.hparams.advantage_normalisation:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        episodes.info["target"] = advantage  # y(s_0), ..., y(s_{T-1})
        return episodes

    def sample_experience(self, episodes: Timestep, *, key: Array) -> Timestep:
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
        batch_size = self.hparams.batch_size
        # sample transitions
        batch_len = episodes.t.shape[0]
        actor_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=batch_len
        )
        # exclude last timestep (-1) as it was excluded from the advantage computation
        episode_length = episodes.t.shape[1]
        time_idx = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=episode_length
        )
        return episodes.at_time[actor_idx, time_idx]  # (batch_size,)

    def prioritised_sample_experience(
        self, episodes: Timestep, *, key: Array
    ) -> Timestep:
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
        td_error = episodes.info["value"][:, :-1] - episodes.info["target"]
        sampling_logits = jnp.reshape(jnp.abs(td_error) + 1e-6, (-1,))
        idx = jax.random.categorical(
            key, sampling_logits, shape=(self.hparams.batch_size,)
        )
        batch_idx, time_idx = jnp.divmod(idx, episodes.length)
        return episodes.at_time[batch_idx, time_idx]

    def loss(
        self, params: Params, transition: Timestep, value_fn, policy, hparams
    ) -> Tuple[Array, Dict[str, Array]]:
        # make sure the transition has the required info
        assert "target" in transition.info
        assert "log_prob" in transition.info
        # o_t
        observation = (
            transition.info["observations"]["glyphs"],
            transition.info["observations"]["chars_crop"],
            transition.info["observations"]["blstats"],
        )
        # A^{\pi}(s_t, a_t)
        advantage = stop_gradient(transition.info["target"])
        # v_{k-1}(s_t)
        action_value_old = stop_gradient(transition.info["value"])
        # a_t)
        action = stop_gradient(transition.action)  # a_t
        # pi_{old}(A_t|s_t)
        log_prob_old = stop_gradient(transition.info["log_prob"])
        # h_t
        hidden_state = transition.info.get("hidden_state", {})
        hidden_state = jtu.tree_map(lambda x: x * ~transition.is_first(), hidden_state)

        # critic loss
        value = value_fn(params, observation, hidden_state=hidden_state)[1]  # v(s_t)
        critic_loss = 0.5 * jnp.square(value - advantage)
        if hparams.value_clipping:
            value_clipped = jnp.clip(
                value,
                action_value_old - hparams.clip_ratio,
                action_value_old + hparams.clip_ratio,
            )
            critic_loss_clipped = 0.5 * jnp.square(value_clipped - advantage)
            critic_loss = jnp.maximum(critic_loss, critic_loss_clipped)
        critic_loss = hparams.value_loss_coefficient * critic_loss

        # actor loss
        action_distribution = policy(params, observation, hidden_state=hidden_state)[1]
        log_prob = action_distribution.log_prob(action)
        ratio = jnp.exp(log_prob - log_prob_old)
        clipped_ratio = jnp.clip(ratio, 1 - hparams.clip_ratio, 1 + hparams.clip_ratio)
        actor_loss = -jnp.minimum(ratio * advantage, clipped_ratio * advantage)

        # entropy
        entropy = action_distribution.entropy()
        entropy_loss = -jnp.asarray(entropy) * jnp.asarray(hparams.beta)

        # total loss
        loss = actor_loss + critic_loss + entropy_loss

        # logs
        log = {
            "losses/loss": loss,
            "losses/critic_loss": critic_loss,
            "losses/actor_loss": actor_loss,
            "losses/entropy_bonus": entropy_loss,
            "policy/entropy": entropy,
        }
        return loss, log

    @partial(jax.jit, static_argnums=(0, 3))
    def update(
        self,
        agent_state: AgentState,
        episodes: Timestep,
        *,
        key: Array,
    ) -> Tuple[AgentState, Dict[str, Array]]:
        def batch_loss(params, transitions):
            out = jax.vmap(self.loss, in_axes=(None, 0, None, None, None))(
                params, transitions, self.value_fn, self.policy, self.hparams
            )
            out = jtu.tree_map(lambda x: x.mean(axis=0), out)
            return out

        params, log = agent_state.params, {}
        for _ in range(self.hparams.n_epochs):
            # calculate GAE with new (updated) value function and inject in timestep
            episodes = jax.vmap(self.evaluate_experience, in_axes=(None, 0))(
                agent_state.params, episodes
            )
            # sample batch of transitions
            if self.hparams.prioritised_sampling:
                transitions = self.prioritised_sample_experience(episodes, key=key)
            else:
                transitions = self.sample_experience(episodes, key=key)
            # SGD with PPO loss
            (_, log), grads = jax.value_and_grad(batch_loss, has_aux=True)(
                params, transitions
            )
            updates, opt_state = self.optimiser.update(grads, agent_state.opt_state)
            params = optax.apply_updates(params, updates)
            # update agent_state object
            agent_state = agent_state.replace(
                iteration=agent_state.iteration + 1,
                params=params,
                opt_state=opt_state,
            )

            # log gradients norm
            log["losses/grad_norm"] = optax.global_norm(updates)
            log["train/avg_samples_return"] = transitions.info["return"].mean()
        return agent_state, log
