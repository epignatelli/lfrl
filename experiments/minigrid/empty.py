import gymnasium as gym

import optax

import helx.envs
from helx.agents import DQN, DQNHParams
from calf import LDQN, LDQNHParams


from helx.base import config
from helx.experiment import run


config.define_flags_from_hparams(DQNHParams)


def main():
    budget = int(1e8)  # 100M frames
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = helx.envs.interop.to_helx(env)

    dqn_hparams = config.hparams_from_flags(DQNHParams, env.observation_space, env.action_space)
    ldqn_hparams = config.hparams_from_flags(LDQNHParams, env.observation_space, env.action_space)
    optimiser = optax.adamw(learning_rate=dqn_hparams.learning_rate)
    repr_net = helx.base.modules.CNN(
        features=(16, 32, 64),
        strides=((1, 1), (1, 1), (1, 1)),
        kernel_sizes=((2, 2), (4, 4), (8, 8)),
        paddings=("SAME", "SAME", "SAME"),
        flatten=True
    )

    agents = [
        DQN(dqn_hparams, optimiser, repr_net),
        LDQN(ldqn_hparams, optimiser, repr_net),
    ]
    for agent in agents:
        run(seed=agent.hparams.seed, agent=agent, env=env, budget=budget)