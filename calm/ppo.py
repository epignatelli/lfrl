from __future__ import annotations
from functools import partial
from typing import Any, Dict, Sequence, Tuple, Union

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

from .trial import run_n_steps, Agent, HParams as HParamsBase, RNNState


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


class Network(nn.Module):
    """An encoder for the MiniHack environment."""

    encoder: nn.Module
    head: nn.Module
    recurrent: bool = True
    n_hidden: int = 512

    @nn.compact
    def __call__(
        self,
        x: Tuple[Array, Array, Array],
        hidden_state: Tuple[Array, Array] | None = None,
    ) -> Tuple[Union[Tuple[Array, Array], None], Array]:
        # format inputs into channel-last image format
        glyphs, chars_crop, blstats = x
        glyphs = jnp.expand_dims(glyphs, axis=-1)
        chars_crop = jnp.expand_dims(chars_crop, axis=-1)
        x = (glyphs, chars_crop, blstats)

        # apply the backbone
        y: Array = self.encoder(x)

        # apply the recurrent layer or a dense layer
        if self.recurrent:
            lstm = nn.OptimizedLSTMCell(self.n_hidden)
            if hidden_state is None:
                key_unused = jax.random.key(0)
                hidden_state = lstm.initialize_carry(key_unused, y.shape)
            hidden_state, y = nn.OptimizedLSTMCell(self.n_hidden)(hidden_state, y)
        else:
            y = nn.Dense(self.n_hidden)(x)
            y = nn.relu(y)

        # apply the head
        y = self.head(y)

        return hidden_state, y


class PPO(Agent):
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
    train_state: Dict[str, Any] = struct.field(pytree_node=True)

    @classmethod
    def init(
        cls,
        env: Environment,
        hparams: HParams,
        encoder: nn.Module,
        *,
        key: Array,
    ) -> PPO:
        assert isinstance(env.action_space, Discrete)
        assert (
            env.observation_space.shape[0]
            == env.action_space.shape[0]
            == hparams.n_actors
        )

        actor = Network(
            encoder=encoder.clone(),
            head=nn.Dense(
                env.action_space.maximum + 1,
                kernel_init=nn.initializers.orthogonal(scale=0.01),
            ),
            recurrent=hparams.recurrent,
            n_hidden=hparams.n_hidden,
        )
        critic = Network(
            encoder=encoder.clone(),
            head=nn.Dense(1, kernel_init=nn.initializers.zeros_init()),
            recurrent=hparams.recurrent,
            n_hidden=hparams.n_hidden,
        )

        timestep = env.reset(key)
        unbatched_obs_sample = (
            timestep.info["observations"]["glyphs"][0],
            timestep.info["observations"]["chars_crop"][0],
            timestep.info["observations"]["blstats"][0],
        )
        actor_out, params_actor = actor.init_with_output(key, unbatched_obs_sample)
        critic_out, params_critic = critic.init_with_output(key, unbatched_obs_sample)
        params = {"actor": params_actor, "critic": params_critic}
        optimiser = optax.chain(
            optax.clip_by_global_norm(hparams.gradient_clip_norm),
            optax.adam(hparams.learning_rate),
        )
        train_state = {
            "opt_state": optimiser.init(params),
            "iteration": jnp.asarray(0),
            "params": params,
        }
        if hparams.recurrent:
            train_state["hidden_state"] = {
                "actor": actor_out[0],
                "critic": critic_out[0],
            }
        return cls(
            hparams=hparams,
            optimiser=optimiser,
            actor=actor,
            critic=critic,
            train_state=train_state,
        )

    def policy(
        self,
        params: Params,
        observation: Tuple[Array, Array, Array],
        *,
        hidden_state: RNNState,
    ) -> Tuple[RNNState, distrax.Softmax]:
        hidden_state_actor = hidden_state.get("actor", {})
        hidden_state_actor, logits = self.actor.apply(
            params["actor"], observation, hidden_state_actor
        )
        hidden_state["actor"] = hidden_state_actor
        return hidden_state, distrax.Softmax(logits=jnp.asarray(logits))

    def value_fn(
        self,
        params: Params,
        observation: Tuple[Array, Array, Array],
        *,
        hidden_state: RNNState,
    ) -> Tuple[RNNState, Array]:
        hidden_state_critic = hidden_state.get("critic", {})
        hidden_state_critic, value = self.critic.apply(
            params["critic"], observation, hidden_state_critic
        )
        hidden_state["critic"] = hidden_state_critic
        return hidden_state, jnp.asarray(value)[0]

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

        def policy_sample(
            params: Params,
            observation: Tuple[Array, Array, Array],
            shape: Sequence[int],
            key: Array,
            hidden_state: RNNState,
        ) -> Tuple[RNNState, Array, Array]:
            hidden_state, action_distribution = self.policy(
                params, observation, hidden_state=hidden_state
            )
            action, log_prob = action_distribution.sample_and_log_prob(
                seed=key, sample_shape=shape
            )
            return hidden_state, jnp.asarray(action), jnp.asarray(log_prob)

        action_shape = env.action_space.shape[1:]
        policy_sample = jax.jit(
            jax.vmap(
                partial(policy_sample, shape=action_shape, params=self.params),
                in_axes=(None, 0, None, 0, 0),
            )
        )

        hidden_state = timestep.info.get("hidden_state", None)

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
            hidden_state, action, log_prob = policy_sample(
                observation=obs,
                key=k1,
                hidden_state=hidden_state,
            )
            # log action log_prob and hidden state
            timestep.info["log_prob"] = log_prob
            timestep.info["hidden_state"] = hidden_state
            next_timestep = env.step(k2, timestep, action)

            # depure return log from intrinsic rewards
            reward = next_timestep.reward
            if "intrinsic_reward" in timestep.info:
                timestep.info["total_reward"] = reward
                reward = reward - timestep.info["intrinsic_reward"]

            # log returns
            if "return" in timestep.info:
                next_timestep.info["return"] = timestep.info["return"] * (
                    timestep.is_mid()
                ) + (reward * self.hparams.discount**timestep.t)

            timestep = next_timestep

        batched = jtu.tree_map(lambda *x: jnp.stack(x, axis=env.ndim), *episode)
        return batched, timestep

    def evaluate_experience(self, episodes: Timestep, timestep: Timestep) -> Timestep:
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
        # r_t \\forall t \\in [0, T-1]
        reward = episodes.reward[1:]
        # \\gamma^t \\forall t \\in [0, T-1]
        discount = (episodes.is_mid() * self.hparams.discount**episodes.t)[1:]

        hidden_state = timestep.info.get("hidden_state", {})
        _, value = jax.vmap(self.value_fn, in_axes=(None, 0, None))(
            self.params, obs, hidden_state=hidden_state
        )

        # mask the value
        value = value * (
            episodes.step_type != StepType.TERMINATION
        )  # q(s_t, a'_t) \\forall a'

        # calculate GAE
        advantage = jnp.asarray(
            rlax.truncated_generalized_advantage_estimation(
                reward,
                discount,
                self.hparams.lambda_,
                value,
            )
        )

        # store the value and advantage in the info dict
        episodes.info["value"] = value
        if self.hparams.advantage_normalisation:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        episodes.info["advantage"] = advantage
        # values and advantages from [0, T-1]
        return episodes

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

    def loss(
        self, params: Params, transition: Timestep
    ) -> Tuple[Array, Dict[str, Array]]:
        # make sure the transition has the required info
        assert "advantage" in transition.info
        assert "log_prob" in transition.info
        # o_t
        observation = (
            transition.info["observations"]["glyphs"],
            transition.info["observations"]["chars_crop"],
            transition.info["observations"]["blstats"],
        )
        # A^{\pi}(s_t, a_t)
        advantage = stop_gradient(transition.info["advantage"])
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
        value = self.value_fn(params, observation, hidden_state=hidden_state)[
            1
        ]  # v(s_t)
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
        action_distribution = self.policy(
            params, observation, hidden_state=hidden_state
        )[1]
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
            "policy/entropy": entropy,
        }
        return loss, log

    @partial(jax.jit, static_argnums=2)
    def update(
        self,
        episodes: Timestep,
        timestep: Timestep,
        n_epochs: int | None = None,
        *,
        key: KeyArray,
    ) -> Tuple[PPO, Dict[str, Array]]:
        def batch_loss(params, transitions):
            out = jax.vmap(self.loss, in_axes=(None, 0))(params, transitions)
            out = jtu.tree_map(lambda x: x.mean(axis=0), out)
            return out

        if n_epochs is None:
            n_epochs = self.hparams.n_epochs

        params, train_state, log = self.params, self.train_state, {}
        for _ in range(n_epochs):
            # calculate GAE with new (updated) value function and inject in timestep
            hidden_states, episodes = jax.vmap(self.evaluate_experience)(
                episodes, timestep
            )
            # sample batch of transitions
            transitions = self.sample_experience(episodes, key=key)
            # SGD with PPO loss
            (_, log), grads = jax.value_and_grad(batch_loss, has_aux=True)(
                params, transitions
            )
            opt_state = train_state["opt_state"]
            updates, opt_state = self.optimiser.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # update train_state object
            train_state["iteration"] += 1
            train_state["opt_state"] = opt_state
            train_state["params"] = params

            # log gradients norm
            log["losses/grad_norm"] = optax.global_norm(updates)
            if "buffer" in train_state:
                log["buffer_size"] = len(train_state["buffer"])
        return self.replace(train_state=train_state), log
