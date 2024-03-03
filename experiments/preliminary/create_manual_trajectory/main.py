from dataclasses import dataclass
from typing import Union
import minihack
import gym
import numpy as np
from calf.nethack import ACTIONS_MAP
from nle import nethack


def render(env) -> str:
    obs = env.last_observation
    chars = obs[env._observation_keys.index("tty_chars")]
    cursor = obs[env._observation_keys.index("tty_cursor")]
    rows, cols = chars.shape
    if cursor is None:
        cursor = (-1, -1)
    cursor = tuple(cursor)
    result = ""
    for i in range(rows):
        result += "\n"
        for j in range(cols):
            entry = chr(chars[i, j])
            if cursor != (i, j):
                result += entry
            else:
                result += entry
    return result


@dataclass
class Timestep:
    t: int
    observation: str
    action: Union[int, None]
    reward: float

    def to_prompt(self, env) -> str:
        timestep = "Timestep {}".format(self.t)
        if self.action is not None:
            action = ACTIONS_MAP[env.actions[self.action]][0]
        else:
            action = ""
        action = "Action: {}".format(action)
        reward = "Reward: {}".format(self.reward)
        return "\n".join([timestep, action, reward, self.observation])


def tiemsteps_to_prompt(timesteps):
    return "\n\n".join(timesteps)


def save_prompt(prompt, filename):
    with open(filename, "w") as f:
        f.write(prompt)


def unroll():
    actions = [
        nethack.CompassCardinalDirection.N,  # 0
        nethack.CompassCardinalDirection.E,  # 1
        nethack.CompassCardinalDirection.S,  # 2
        nethack.CompassCardinalDirection.W,  # 3
        nethack.Command.PICKUP,  # 4
        nethack.Command.APPLY,  # 5
    ]
    env = gym.make(
        "MiniHack-KeyRoom-Fixed-S5-v0",
        observation_keys=("chars_crop",),
        max_episode_steps=100,
        actions=actions,
    )
    print("Available actions:", env.action_space)

    obs = env.reset()
    timesteps = []
    reward = 0.0
    cumulative_reward = reward
    i = 0
    while True:
        # for i in range(10):
        obs = render(env)
        # obs = ord_to_char(obs['chars_crop'])

        # choose action
        print(obs)
        action = int(input())  # manual action
        # action = env.action_space.sample()  # random action

        # pack and store timestep
        timestep = Timestep(i, obs, action, reward)
        timestep = timestep.to_prompt(env)
        timesteps.append(timestep)

        # update env
        obs, reward, done, info = env.step(action)
        done = info["end_status"] != 0
        cumulative_reward += reward
        if done or i >= 15:
            break
        i += 1

    # print the state
    obs = render(env)
    print(obs)
    print("Total cumulative reward:", cumulative_reward)
    timestep = Timestep(i, obs, None, reward)
    timestep = timestep.to_prompt(env)
    timesteps.append(timestep)

    return timesteps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="./prompt.txt")
    args = parser.parse_args()

    timesteps = unroll()
    prompt = tiemsteps_to_prompt(timesteps)
    save_prompt(prompt, args.filename)
