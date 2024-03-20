from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import gym
import gym.spaces
import gym.wrappers
import gym.vector
from gym.utils.step_api_compatibility import (
    TerminatedTruncatedStepType as GymTimestep,
    DoneStepType,
    convert_to_terminated_truncated_step_api,
)
import gym.core
import numpy as np

import jax
from jax import Array
from jax.random import KeyArray
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
from helx.base.mdp import Timestep, StepType
from helx.envs.gym import GymWrapper
from helx.base.spaces import Space, Discrete, Continuous

from .io import load_pickle_stream
from .decode import decode_observation
from .annotate import Annotation


class LLMTableWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, table_path: str, beta: float):
        super().__init__(env)
        # load table
        print(f"Loading LLM reward table from {table_path}...\t", end="")
        stream: List[Annotation] = load_pickle_stream(table_path)

        # postprocess table
        table = {}
        maximum = 0
        minimum = 1_000_000
        for ann in stream:
            for key in ann.parsed:
                count = table.get(key, 0) + len(ann.parsed[key])
                maximum = max(maximum, count)
                minimum = min(minimum, count)
                table[key] = count

        # normalise
        for key in table:
            table[key] = (table[key] - minimum) / (maximum - minimum)
        print("Done")
        # set fields
        self.beta = beta
        self.why = stream
        self.table = table
        self.n_actors = getattr(self, "num_envs", 1)

    def reset(self, seed, options={}):
        obs, info = self.env.reset(seed, options)  # type: ignore
        info["extrinsic_reward"] = np.zeros(self.n_actors)
        info["intrinsic_reward"] = np.zeros(self.n_actors)
        return obs, info

    def step(self, action):
        timestep = self.env.step(action)
        timestep = convert_to_terminated_truncated_step_api(timestep)
        obs, reward, term, trunc, info = timestep
        chars = info["observations"]["chars"]  # type: ignore
        r_int = []
        for actor in range(self.n_actors):
            key = (decode_observation(chars[actor]), str(int(action[actor])))
            r_int.append(self.table.get(key, 0))
        r_int = np.asarray(r_int)
        info["extrinsic_reward"] = reward  # type: ignore
        info["intrinsic_reward"] = r_int  # type: ignore
        reward += r_int * self.beta
        return obs, reward, term, trunc, info


class UndictWrapper(gym.core.Wrapper):
    def __init__(self, env: gym.Env, key: str):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.main_key = key
        self.observation_space = env.observation_space[key]

    def reset(self, seed, options={}):
        obs, info = self.env.reset()
        assert isinstance(
            obs, dict
        ), "UndictWrapper requires observations to be a dictionary, got {}".format(
            type(obs)
        )
        main_obs = obs.pop(self.main_key)
        info["observations"] = obs
        return main_obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)  # type: ignore
        assert isinstance(
            obs, dict
        ), "UndictWrapper requires observations to be a dictionary, got {}".format(
            type(obs)
        )
        main_obs = obs.pop(self.main_key)
        info["observations"] = obs
        return (main_obs, reward, term, trunc, info)


class MiniHackWrapper(GymWrapper):
    __transparent_info__: Tuple[str, ...] = struct.field(
        pytree_node=False,
        default=(
            "observations",
            "extrinsic_reward",
            "intrinsic_reward",
            "extrinsic_return",
            "intrinsic_return",
        ),
    )

    def reset(self, key: KeyArray) -> Timestep:
        # compute new timestep
        seeds = jax.random.randint(key, shape=self.shape, minval=0, maxval=1_000_000)
        seeds = np.array(seeds)
        obs, info = self.env.reset(seed=seeds)  # type: ignore
        # placeholder for missing info
        rewards = np.zeros(self.shape)
        dones = np.zeros(self.shape, dtype=jnp.bool_)
        timestep = (obs, rewards, dones, info)
        t = jnp.zeros(self.shape)
        action = jnp.zeros(self.shape) - 1
        timestep = self._wrap_timestep(timestep, action=action, t=t)
        timestep.info["return"] = timestep.reward
        return timestep

    def step(self, key: KeyArray, timestep: Timestep, action: Array) -> Timestep:
        next_timestep = self.env.step(np.asarray(action))
        t = jnp.asarray((timestep.t + 1) * timestep.is_mid(), dtype=jnp.int32)
        action = (action * timestep.is_mid()) - (timestep.is_last())
        # (t, s_{t+1}, r_t, d_t, a_t, g_t)
        next_timestep = self._wrap_timestep(next_timestep, action, t)
        return next_timestep

    @classmethod
    def wraps(cls, env: gym.Env) -> MiniHackWrapper:
        num_envs = getattr(env, "num_envs", 1)
        env_shape = (num_envs,) if num_envs > 1 else ()
        helx_env = cls(
            env=env,
            observation_space=cls._wrap_space(env.observation_space),
            action_space=cls._wrap_space(env.action_space),
            reward_space=Continuous(
                shape=env_shape,
                minimum=env.reward_range[0],
                maximum=env.reward_range[1],
            ),
        )
        assert helx_env.shape == env_shape
        return helx_env

    @classmethod
    def _wrap_space(cls, gym_space: gym.spaces.Space) -> Space:
        if isinstance(gym_space, gym.spaces.Discrete):
            return Discrete(gym_space.n)
        elif isinstance(gym_space, gym.spaces.Box):
            return Continuous(
                shape=gym_space.shape,
                minimum=gym_space.low.min().item(),
                maximum=gym_space.high.max().item(),
            )
        elif isinstance(gym_space, gym.spaces.MultiDiscrete):
            # gym.vector.VectorEnv returns MultiDiscrete
            upper = np.array(gym_space.nvec)
            assert np.sum(upper - upper) <= np.array(0)
            return Discrete(upper[0], shape=gym_space.shape)
        else:
            raise NotImplementedError(
                "Cannot convert dm_env space of type {}".format(type(gym_space))
            )

    def _wrap_timestep(
        self, gym_step: GymTimestep | DoneStepType, action: Array, t: Array
    ) -> Timestep:
        gym_step = convert_to_terminated_truncated_step_api(gym_step, self.is_batched())
        obs, reward, terminated, truncated, info = gym_step

        selector = jnp.asarray(terminated + truncated * 2, dtype=jnp.int32)
        step_type = jnp.asarray(
            [StepType.TRANSITION, StepType.TERMINATION, StepType.TRUNCATION]
        )[selector]

        obs = jtu.tree_map(lambda x: jnp.asarray(x, self.observation_space.dtype), obs)
        reward = jnp.asarray(reward, dtype=self.reward_space.dtype)
        action = jnp.asarray(action, dtype=self.action_space.dtype)

        clean_info = {k: v for k, v in info.items() if k in self.__transparent_info__}  # type: ignore

        return Timestep(
            observation=obs,
            reward=reward,
            step_type=step_type,
            action=action,
            t=jnp.asarray(t, dtype=jnp.int32),
            state=None,
            info=clean_info,
        )


SYMSET = {
    "-": "open door (in vertical wall)",
    "|": "open door (in horizontal wall)",
    "<space>": "dark part of a room or solid rock",
    ".": "doorway (with no or broken door)",
    "+": "closed door (in vertical wall)",
    ">": "staircase down",
    "<": "staircase up",
}
