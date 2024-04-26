import jax
from calm.environment import MiniHackWrapper, UndictWrapper


if __name__ == "__main__":
    import numpy as np
    import gym
    from gym.wrappers.autoreset import AutoResetWrapper
    import gym.vector
    import minihack
    from nle import nethack
    from calm.environment import ShaperWrapper

    actions = [
        nethack.CompassCardinalDirection.N,
        nethack.CompassCardinalDirection.E,
        nethack.CompassCardinalDirection.S,
        nethack.CompassCardinalDirection.W,
        nethack.Command.PICKUP,
        nethack.Command.APPLY,
    ]
    observation_key = "chars_crop"
    env = gym.make(
        "MiniHack-KeyRoom-S5-v0",
        actions=actions,
        observation_keys=(
            observation_key,
            "chars",
            "chars_crop",
            "message",
            "inv_glyphs",
            "blstats",
        ),
        max_episode_steps=100,
        seeds=[0],
    )
    env = ShaperWrapper(env)