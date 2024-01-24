from __future__ import annotations
from typing import Tuple

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
from minihack import MiniHack

import jax
from jax import Array
from jax.random import KeyArray
import jax.numpy as jnp
import jax.tree_util as jtu
from helx.base.mdp import Timestep, StepType
from helx.envs.gym import GymWrapper
from helx.base.spaces import Space, Discrete, Continuous


class UndictWrapper(gym.core.ObservationWrapper):
    def __init__(self, env: gym.Env, key: str):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.key = key
        self.observation_space = self.observation_space[key]  # type: ignore

    def observation(self, obs):
        return obs[self.key]


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
        timestep.info['return'] = timestep.reward
        return timestep

    def _step(self, key: KeyArray, timestep: Timestep, action: Array) -> Timestep:
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
        info = {}
        return Timestep(
            observation=obs,
            reward=reward,
            step_type=step_type,
            action=action,
            t=jnp.asarray(t, dtype=jnp.int32),
            state=None,
            info=info,
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


ACTIONS_MAP = {
    None: ["none", ""],
    nethack_actions.UnsafeActions.HELP: ["help", "?"],
    nethack_actions.UnsafeActions.PREVMSG: ["previous message", "^p"],
    nethack_actions.CompassDirection.N: ["north", "k"],
    nethack_actions.CompassDirection.E: ["east", "l"],
    nethack_actions.CompassDirection.S: ["south", "j"],
    nethack_actions.CompassDirection.W: ["west", "h"],
    nethack_actions.CompassIntercardinalDirection.NE: ["northeast", "u"],
    nethack_actions.CompassIntercardinalDirection.SE: ["southeast", "n"],
    nethack_actions.CompassIntercardinalDirection.SW: ["southwest", "b"],
    nethack_actions.CompassIntercardinalDirection.NW: ["northwest", "y"],
    nethack_actions.CompassCardinalDirectionLonger.N: ["far north", "K"],
    nethack_actions.CompassCardinalDirectionLonger.E: ["far east", "L"],
    nethack_actions.CompassCardinalDirectionLonger.S: ["far south", "J"],
    nethack_actions.CompassCardinalDirectionLonger.W: ["far west", "H"],
    nethack_actions.CompassIntercardinalDirectionLonger.NE: ["far northeast", "U"],
    nethack_actions.CompassIntercardinalDirectionLonger.SE: ["far southeast", "N"],
    nethack_actions.CompassIntercardinalDirectionLonger.SW: ["far southwest", "B"],
    nethack_actions.CompassIntercardinalDirectionLonger.NW: ["far northwest", "Y"],
    nethack_actions.MiscDirection.UP: ["up", "<"],
    nethack_actions.MiscDirection.DOWN: ["down", ">"],
    nethack_actions.MiscDirection.WAIT: ["wait", "."],
    nethack_actions.MiscAction.MORE: ["more", "\r", r"\r"],
    nethack_actions.Command.EXTCMD: ["extcmd", "#"],
    nethack_actions.Command.EXTLIST: ["extlist", "M-?"],
    nethack_actions.Command.ADJUST: ["adjust", "M-a"],
    nethack_actions.Command.ANNOTATE: ["annotate", "M-A"],
    nethack_actions.Command.APPLY: ["apply", "a"],
    nethack_actions.Command.ATTRIBUTES: ["attributes", "^x"],
    nethack_actions.Command.AUTOPICKUP: ["autopickup", "@"],
    nethack_actions.Command.CALL: ["call", "C"],
    nethack_actions.Command.CAST: ["cast", "Z"],
    nethack_actions.Command.CHAT: ["chat", "M-c"],
    nethack_actions.Command.CLOSE: ["close", "c"],
    nethack_actions.Command.CONDUCT: ["conduct", "M-C"],
    nethack_actions.Command.DIP: ["dip", "M-d"],
    nethack_actions.Command.DROP: ["drop", "d"],
    nethack_actions.Command.DROPTYPE: ["droptype", "D"],
    nethack_actions.Command.EAT: ["eat", "e"],
    nethack_actions.Command.ESC: ["esc", "^["],
    nethack_actions.Command.ENGRAVE: ["engrave", "E"],
    nethack_actions.Command.ENHANCE: ["enhance", "M-e"],
    nethack_actions.Command.FIRE: ["fire", "f"],
    nethack_actions.Command.FIGHT: ["fight", "F"],
    nethack_actions.Command.FORCE: ["force", "M-f"],
    nethack_actions.Command.GLANCE: ["glance", ";"],
    nethack_actions.Command.HISTORY: ["history", "V"],
    nethack_actions.Command.INVENTORY: ["inventory", "i"],
    nethack_actions.Command.INVENTTYPE: ["inventtype", "I"],
    nethack_actions.Command.INVOKE: ["invoke", "M-i"],
    nethack_actions.Command.JUMP: ["jump", "M-j"],
    nethack_actions.Command.KICK: ["kick", "^d"],
    nethack_actions.Command.KNOWN: ["known", "\\"],
    nethack_actions.Command.KNOWNCLASS: ["knownclass", "`"],
    nethack_actions.Command.LOOK: ["look", ":"],
    nethack_actions.Command.LOOT: ["loot", "M-l"],
    nethack_actions.Command.MONSTER: ["monster", "M-m"],
    nethack_actions.Command.MOVE: ["move", "m"],
    nethack_actions.Command.MOVEFAR: ["movefar", "M"],
    nethack_actions.Command.OFFER: ["offer", "M-o"],
    nethack_actions.Command.OPEN: ["open", "o"],
    nethack_actions.Command.OPTIONS: ["options", "O"],
    nethack_actions.Command.OVERVIEW: ["overview", "^o"],
    nethack_actions.Command.PAY: ["pay", "p"],
    nethack_actions.Command.PICKUP: ["pickup", ","],
    nethack_actions.Command.PRAY: ["pray", "M-p"],
    nethack_actions.Command.PUTON: ["puton", "P"],
    nethack_actions.Command.QUAFF: ["quaff", "q"],
    nethack_actions.Command.QUIT: ["quit", "M-q"],
    nethack_actions.Command.QUIVER: ["quiver", "Q"],
    nethack_actions.Command.READ: ["read", "r"],
    nethack_actions.Command.REDRAW: ["redraw", "^r"],
    nethack_actions.Command.REMOVE: ["remove", "R"],
    nethack_actions.Command.RIDE: ["ride", "M-R"],
    nethack_actions.Command.RUB: ["rub", "M-r"],
    nethack_actions.Command.RUSH: ["rush", "g"],
    nethack_actions.Command.RUSH2: ["rush2", "G"],
    nethack_actions.Command.SAVE: ["save", "S"],
    nethack_actions.Command.SEARCH: ["search", "s"],
    nethack_actions.Command.SEEALL: ["seeall", "*"],
    nethack_actions.Command.SEEAMULET: ["seeamulet", '"'],
    nethack_actions.Command.SEEARMOR: ["seearmor", "["],
    nethack_actions.Command.SEEGOLD: ["seegold", "dollar", "$"],
    nethack_actions.Command.SEERINGS: ["seerings", "="],
    nethack_actions.Command.SEESPELLS: ["seespells", "plus", "+"],
    nethack_actions.Command.SEETOOLS: ["seetools", "("],
    nethack_actions.Command.SEETRAP: ["seetrap", "^"],
    nethack_actions.Command.SEEWEAPON: ["seeweapon", ")"],
    nethack_actions.Command.SHELL: ["shell", "!"],
    nethack_actions.Command.SIT: ["sit", "M-s"],
    nethack_actions.Command.SWAP: ["swap", "x"],
    nethack_actions.Command.TAKEOFF: ["takeoff", "T"],
    nethack_actions.Command.TAKEOFFALL: ["takeoffall", "A"],
    nethack_actions.Command.TELEPORT: ["teleport", "^t"],
    nethack_actions.Command.THROW: ["throw", "t"],
    nethack_actions.Command.TIP: ["tip", "M-T"],
    nethack_actions.Command.TRAVEL: ["travel", "_"],
    nethack_actions.Command.TURN: ["turnundead", "M-t"],
    nethack_actions.Command.TWOWEAPON: ["twoweapon", "X"],
    nethack_actions.Command.UNTRAP: ["untrap", "M-u"],
    nethack_actions.Command.VERSION: ["version", "M-v"],
    nethack_actions.Command.VERSIONSHORT: ["versionshort", "v"],
    nethack_actions.Command.WEAR: ["wear", "W"],
    nethack_actions.Command.WHATDOES: ["whatdoes", "&"],
    nethack_actions.Command.WHATIS: ["whatis", "/"],
    nethack_actions.Command.WIELD: ["wield", "w"],
    nethack_actions.Command.WIPE: ["wipe", "M-w"],
    nethack_actions.Command.ZAP: ["zap", "z"],
    nethack_actions.TextCharacters.MINUS: ["minus", "-"],
    nethack_actions.TextCharacters.SPACE: ["space", " "],
    nethack_actions.TextCharacters.APOS: ["apos", "'"],
    nethack_actions.TextCharacters.NUM_0: ["zero", "0"],
    nethack_actions.TextCharacters.NUM_1: ["one", "1"],
    nethack_actions.TextCharacters.NUM_2: ["two", "2"],
    nethack_actions.TextCharacters.NUM_3: ["three", "3"],
    nethack_actions.TextCharacters.NUM_4: ["four", "4"],
    nethack_actions.TextCharacters.NUM_5: ["five", "5"],
    nethack_actions.TextCharacters.NUM_6: ["six", "6"],
    nethack_actions.TextCharacters.NUM_7: ["seven", "7"],
    nethack_actions.TextCharacters.NUM_8: ["eight", "8"],
    nethack_actions.TextCharacters.NUM_9: ["nine", "9"],
    nethack_actions.WizardCommand.WIZDETECT: ["wizard detect", "^e"],
    nethack_actions.WizardCommand.WIZGENESIS: ["wizard genesis", "^g"],
    nethack_actions.WizardCommand.WIZIDENTIFY: ["wizard identify", "^i"],
    nethack_actions.WizardCommand.WIZLEVELPORT: ["wizard teleport", "^v"],
    nethack_actions.WizardCommand.WIZMAP: ["wizard map", "^f"],
    nethack_actions.WizardCommand.WIZWHERE: ["wizard where", "^o"],
    nethack_actions.WizardCommand.WIZWISH: ["wizard wish", "^w"],
}
