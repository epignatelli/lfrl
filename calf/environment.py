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
from nle.nethack import actions as nethack_actions
import numpy as np

import jax
from jax import Array
from jax.random import KeyArray
import jax.numpy as jnp
import jax.tree_util as jtu
from helx.base.mdp import Timestep, StepType
from helx.envs.gym import GymWrapper
from helx.base.spaces import Space, Discrete, Continuous

from .io import load_pickle_stream
from .decode import decode_observation


class LLMTableWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, table_path: str, beta: float):
        super().__init__(env)
        # load table
        print(f"Loading LLM reward table from {table_path}...\t", end="")
        stream: Dict[Tuple[str, int], List[str]] = load_pickle_stream(table_path)
        print("Done")

        # postprocess table
        print(f"Postprocessing LLM reward table\t", end="")
        table = {}
        maximum = 0
        minimum = 1_000_000
        for key in stream:
            count = table.get(key, 0) + len(stream[key])
            maximum = max(maximum, count)
            minimum = min(minimum, count)
            table[key] = count
        # normalise
        for key in table:
            table[key] = (table[key] - minimum) / (maximum - minimum)
        print("Done")
        # set fields
        self.beta = beta
        self.stream = stream
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
            key = (decode_observation(chars[actor]), str(int(action)))
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
    def reset(self, key: KeyArray) -> Timestep:
        # get num envs
        num_envs = getattr(self.env, "num_envs", 1)
        # compute new timestep
        seeds = jax.random.randint(key, shape=(num_envs,), minval=0, maxval=1_000_000)
        seeds = np.array(seeds)
        obs, info = self.env.reset(seed=seeds)  # type: ignore
        # placeholder for missing info
        rewards = np.zeros((num_envs,))
        dones = np.asarray([False] * num_envs)
        timestep = (obs, rewards, dones, info)
        t = jnp.zeros((num_envs,))
        action = jnp.zeros((num_envs,)) - 1
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
        helx_env = cls(
            env=env,
            observation_space=cls._wrap_space(env.observation_space),
            action_space=cls._wrap_space(env.action_space),
            reward_space=Continuous(
                minimum=env.reward_range[0],  # type: ignore
                maximum=env.reward_range[1] // 100,  # type: ignore
            ),
        )
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
        is_vector_env = hasattr(self.env, "num_envs") or isinstance(
            self.env, gym.vector.VectorEnv
        )
        gym_step = convert_to_terminated_truncated_step_api(gym_step, is_vector_env)
        obs, reward, terminated, truncated, info = gym_step

        step_type = jnp.asarray(
            [StepType.TRANSITION, StepType.TERMINATION, StepType.TRUNCATION]
        )[terminated + truncated * 2]

        obs = jtu.tree_map(lambda x: jnp.asarray(x, self.observation_space.dtype), obs)
        reward = jnp.asarray(reward, dtype=self.reward_space.dtype)
        action = jnp.asarray(action, dtype=self.action_space.dtype)

        __allowed_info__ = (
            "observations",
            "extrinsic_reward",
            "intrinsic_reward",
            "extrinsic_return" "intrinsic_return",
        )
        clean_info = {k: v for k, v in info.items() if k in __allowed_info__}  # type: ignore
        # if "observations" in info:
        #     clean_info["observations"] = info["observations"]  # type: ignore
        # if "extrinsic_reward" in info:
        #     clean_info["extrinsic_reward"] = info["extrinsic_reward"]
        # if "intrinsic_reward" in info:
        #     clean_info["intrinsic_reward"] = info["intrinsic_reward"]

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
