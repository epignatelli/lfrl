from __future__ import annotations
from typing import Any, Dict, List, Tuple
import time

from minihack import MiniHack
import numpy as np
from gymnasium import register
import gym
import gym.spaces
import gym.wrappers
import gym.vector
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.utils.step_api_compatibility import (
    TerminatedTruncatedStepType as GymTimestep,
    DoneStepType,
    convert_to_terminated_truncated_step_api,
)
import gym.core
import numpy as np
from nle import nethack
import jax
from jax import Array
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
from helx.base.mdp import Timestep, StepType
from helx.envs.gym import GymWrapper
from helx.base.spaces import Space, Discrete, Continuous

from .io import load_pickle_stream
from .decode import decode_observation
from .annotate import Annotation

CLOSED_DOOR_ID = 2375
OPEN_DOOR_ID = 2373
KEY_PICKED_MSG_EXTRACT = np.asarray(
    [
        45,
        32,
        97,
        32,
        107,
        101,
        121,
        32,
        110,
        97,
        109,
        101,
        100,
        32,
        84,
        104,
        101,
        32,
        77,
        97,
        115,
        116,
        101,
        114,
        32,
        75,
        101,
        121,
        32,
        111,
        102,
        32,
        84,
        104,
        105,
        101,
        118,
        101,
        114,
        121,
        46,
    ],
    dtype=np.uint8,
)
NEVER_MIND_MSG_EXTRACT = np.asarray(
    [
        78,
        101,
        118,
        101,
        114,
        32,
        109,
        105,
        110,
        100,
        46,
    ],
    dtype=np.uint8,
)


class ShaperWrapper(gym.Wrapper):
    def __init__(self, env: MiniHack):
        super().__init__(env)
        self.is_batched = hasattr(env, "num_envs")
        self.key_id = np.asarray(2102, dtype=int)
        self.is_door_opened = False

        # AttributeError: accessing private attribute '_observation_keys' is prohibited
        # self.inv_glyphs_idx = env._observation_keys.index("inv_glyphs")
        self.glyphs_idx = 0
        self.inv_glyphs_idx = 6
        self.message_idx = 5

        # override reward range
        self.reward_range = (0.0, 3.0)

    def reset(self, **kwargs) -> Tuple[Any | Dict]:
        self.is_door_opened = False
        obs, info = super().reset(**kwargs)
        info["intrinsic_reward"] = 0
        return obs, info  # type: ignore

    def step(self, action):
        env: MiniHack = self.env  # type: ignore

        timestep = env.step(action)

        timestep = convert_to_terminated_truncated_step_api(timestep)
        obs, reward, term, trunc, info = timestep  # type: ignore
        info: Dict[str, Any] = info

        # key pickup
        r_int = self.key_picked() + self.door_unlocked(action)
        reward += r_int
        info["intrinsic_reward"] = r_int
        return obs, reward, term, trunc, info

    def key_picked(self):
        message = self.last_observation[self.message_idx]
        return np.array_equal(message[2:43], KEY_PICKED_MSG_EXTRACT)

    def door_unlocked(self, action) -> bool:
        if self.is_door_opened:
            # if already open, exit
            return False
        env: MiniHack = self.env  # type: ignore
        if action != env.actions.index(nethack.Command.APPLY):
            # if action is not open, exit
            return False
        message = self.last_observation[self.message_idx]
        if not np.array_equal(message[:11], NEVER_MIND_MSG_EXTRACT):
            # if message is not the one we are looking for, exit
            return False
        # check if door has been opened
        glyphs = self.last_observation[self.glyphs_idx]
        opened = bool(
            np.all(glyphs[glyphs == CLOSED_DOOR_ID] == OPEN_DOOR_ID)
        )
        self.is_door_opened = opened
        return opened


class LLMShaperWrapper(gym.Wrapper):
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
        self.reward_range = (0.0, 1.0)

    def reset(self, seed, options={}):
        obs, info = self.env.reset(seed, options)  # type: ignore
        info["intrinsic_reward"] = 0.0
        return obs, info

    def step(self, action):
        timestep = self.env.step(action)
        timestep = convert_to_terminated_truncated_step_api(timestep)
        obs, reward, term, trunc, info = timestep  # type: ignore
        info: Dict[str, Any] = info
        # lookup
        chars = info["observations"]["chars"]  # type: ignore
        key = (decode_observation(chars), str(int(action)))
        r_int = self.table.get(key, 0)
        # store
        info["intrinsic_reward"] = r_int
        reward += r_int * self.beta
        return obs, reward, term, trunc, info


class UndictWrapper(gym.core.Wrapper):

    def __init__(self, env: gym.Env, key: str):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.main_key = key
        self.observation_space = env.observation_space[key]

    @property
    def single_observation_space(self):
        return self.env.single_observation_space[self.main_key]  # type: ignore

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
    env: MiniHack | AsyncVectorEnv

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

    def reset(self, key: Array) -> Timestep:
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

    def step(self, key: Array, timestep: Timestep, action: Array) -> Timestep:
        next_timestep = self.env.step(np.asarray(action))  # type: ignore
        t = jnp.asarray((timestep.t + 1) * timestep.is_mid(), dtype=jnp.int32)
        action = (action * timestep.is_mid()) - (timestep.is_last())
        # (t, s_{t+1}, r_t, d_t, a_t, g_t)
        next_timestep = self._wrap_timestep(next_timestep, action, t)  # type: ignore
        return next_timestep

    @classmethod
    def wraps(cls, env: gym.Env) -> MiniHackWrapper:
        num_envs = getattr(env, "num_envs", 1)
        env_shape = (num_envs,) if num_envs > 1 else ()
        helx_env = cls(
            env=env,  # type: ignore
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

        obs = jtu.tree_map(lambda x: jnp.asarray(x, self.observation_space.dtype), obs)  # type: ignore
        reward = jnp.asarray(reward, dtype=self.reward_space.dtype)  # type: ignore
        action = jnp.asarray(action, dtype=self.action_space.dtype)  # type: ignore

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


register(
    id="MiniHack-KeyRoom-Fixed-S5-v0",
    entry_point="minihack.envs.keyroom:MiniHackKeyRoom5x5Fixed",
)
register(
    id="MiniHack-KeyRoom-S5-v0",
    entry_point="minihack.envs.keyroom:MiniHackKeyRoom5x5",
)
register(
    id="MiniHack-KeyRoom-S15-v0",
    entry_point="minihack.envs.keyroom:MiniHackKeyRoom15x15",
)
register(
    id="MiniHack-KeyRoom-Dark-S5-v0",
    entry_point="minihack.envs.keyroom:MiniHackKeyRoom5x5Dark",
)
register(
    id="MiniHack-KeyRoom-Dark-S15-v0",
    entry_point="minihack.envs.keyroom:MiniHackKeyRoom15x15Dark",
)

# class ShaperWrapper(gym.Wrapper):
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         self.is_batched = hasattr(env, "num_envs")
#         self.key_id = np.asarray(2102, dtype=int)

#     def step(self, action):
#         env: MiniHack = self.env  # type: ignore

#         # env.last_observation is mutable and overwritten by env.step, so do pre-checks
#         key_prechecks = self.key_picked_prechecks()
#         door_prechecks = self.door_unlocked_prechecks()

#         timestep = env.step(action)
#         timestep = convert_to_terminated_truncated_step_api(timestep)
#         obs, reward, term, trunc, info = timestep  # type: ignore
#         info: Dict[str, Any] = info

#         info["intrinsic_reward"] = reward * 0
#         # key pickup
#         r_int = self.key_picked(key_prechecks, action) + self.door_unlocked(
#             door_prechecks, action
#         )
#         reward += r_int
#         info["intrinsic_reward"] = info["intrinsic_reward"] + r_int

#         return obs, reward, term, trunc, info

#     def key_picked_prechecks(self):
#         env: MiniHack = self.env  # type: ignore
#         inv_glyphs_idx = env._observation_keys.index("inv_glyphs")
#         inventory = self.last_observation[inv_glyphs_idx]
#         return jnp.logical_not(np.isin(self.key_id, inventory))

#     def door_unlocked_prechecks(self):
#         env: MiniHack = self.env  # type: ignore
#         return env.screen_contains("closed door")

#     def key_picked(self, prechecks, action):
#         env: MiniHack = self.env  # type: ignore
#         inv_glyphs_idx = env._observation_keys.index("inv_glyphs")

#         # key already in inventory
#         not_in_inventory = prechecks
#         # did not use pickup action
#         correct_action = action == env.actions.index(nethack.Command.PICKUP)
#         # key has not been picked up
#         now_in_inventory = np.isin(self.key_id, env.last_observation[inv_glyphs_idx])

#         cond = np.stack([not_in_inventory, correct_action, now_in_inventory], axis=0)
#         cond = jnp.all(cond, axis=0)
#         return cond

#     def door_unlocked(self, prechecks, action) -> bool:
#         env: MiniHack = self.env  # type: ignore
#         # door before action is closed
#         is_closed = prechecks
#         # actoin is open
#         correct_action = action == env.actions.index(nethack.Command.APPLY)
#         # door open after action
#         now_open = env.screen_contains("open door", env.last_observation)

#         cond = np.stack([is_closed, correct_action, now_open], axis=0)
#         cond = np.all(cond, axis=0)
#         return cond


# class LLMTableWrapper(gym.Wrapper):
#     def __init__(self, env: gym.Env, table_path: str, beta: float):
#         super().__init__(env)
#         # load table
#         print(f"Loading LLM reward table from {table_path}...\t", end="")
#         stream: List[Annotation] = load_pickle_stream(table_path)

#         # postprocess table
#         table = {}
#         maximum = 0
#         minimum = 1_000_000
#         for ann in stream:
#             for key in ann.parsed:
#                 count = table.get(key, 0) + len(ann.parsed[key])
#                 maximum = max(maximum, count)
#                 minimum = min(minimum, count)
#                 table[key] = count

#         # normalise
#         for key in table:
#             table[key] = (table[key] - minimum) / (maximum - minimum)
#         print("Done")
#         # set fields
#         self.beta = beta
#         self.why = stream
#         self.table = table
#         self.n_actors = getattr(self, "num_envs", 1)

#     def reset(self, seed, options={}):
#         obs, info = self.env.reset(seed, options)  # type: ignore
#         info["extrinsic_reward"] = np.zeros(self.n_actors)
#         info["intrinsic_reward"] = np.zeros(self.n_actors)
#         return obs, info

#     def step(self, action):
#         timestep = self.env.step(action)
#         timestep = convert_to_terminated_truncated_step_api(timestep)
#         obs, reward, term, trunc, info = timestep
#         chars = info["observations"]["chars"]  # type: ignore
#         r_int = []
#         for actor in range(self.n_actors):
#             key = (decode_observation(chars[actor]), str(int(action[actor])))
#             r_int.append(self.table.get(key, 0))
#         r_int = np.asarray(r_int)
#         info["extrinsic_reward"] = reward  # type: ignore
#         info["intrinsic_reward"] = r_int  # type: ignore
#         reward += r_int * self.beta
#         return obs, reward, term, trunc, info


# class UndictWrapper(gym.core.Wrapper):
#     def __init__(self, env: gym.Env, key: str):
#         super().__init__(env)
#         assert isinstance(env.observation_space, gym.spaces.Dict)
#         self.main_key = key
#         self.observation_space = env.observation_space[key]

#     def reset(self, seed, options={}):
#         obs, info = self.env.reset()
#         assert isinstance(
#             obs, dict
#         ), "UndictWrapper requires observations to be a dictionary, got {}".format(
#             type(obs)
#         )
#         main_obs = obs.pop(self.main_key)
#         info["observations"] = obs
#         return main_obs, info

#     def step(self, action):
#         obs, reward, term, trunc, info = self.env.step(action)  # type: ignore
#         assert isinstance(
#             obs, dict
#         ), "UndictWrapper requires observations to be a dictionary, got {}".format(
#             type(obs)
#         )
#         main_obs = obs.pop(self.main_key)
#         info["observations"] = obs
#         return (main_obs, reward, term, trunc, info)
