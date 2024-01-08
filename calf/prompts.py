DEFAULT_SYSTEM_PROMPT = """\
Welcome to the MiniHack data collection and evaluation interface.
This interface is designed to collect data from MiniHack and evaluate the performance of reinforcement learning agents.
You are a helpful and honest judge of good gameplaying and progress in the MiniHack game.
Always answer as helpfully as possible, while being truthful.
If you don't know the answer to a question, please don't share false information.

"""

PROMPT_ENV_DESCRIPTION = """\
Briefly and concisely describe the `{}` environment of MiniHack in a few words.
Mention its goal and its challenges."""


PROMPT_INTRINSIC_REWARD = """\
I will present you with an observations from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

First, tell me about your knowledge of MiniHack.
The goal of the agent is find the staircase up, denoted by ">".
Do not confound that with the staircase down symbol "<".

Then, write an analysis describing the semantics of the observation strictly using information from the observations and your knowledge of MiniHack.

Finally, write a score for the observation that describes its contribution to achieve the goal.
You can choose any score between 0.0 and 1.0, writing {"score": X}, where X is the score you choose.
"""

PROMPT_CREDIT_ASSIGNMENT = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

Write a brief and concise analysis describing the semantics of the trajectory strictly using information from the observations and your knowledge of MiniHack.

Finally, write a score for each action that describes the contribution of that action to reach the final observation.
For example, if a different action also would have led to the same observation at the next timestep, then the score of the action is 0.0.
You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep of the action and X is the score you choose.
The sum of all scores must be 1.0.

We aim to find those unique actions that are crucial to the point that without them, the agent would not be able to reach the goal.
"""

PROMPT_CREDIT_ASSIGNMENT = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

The actions available to the agent at each timestep are the following:
- "North": the agent moves north
- "East": the agent moves east
- "South": the agent moves south
- "West": the agent moves west
- "Pickup": the agent picks up an item
- "Drop": the agent drops an item
- "Search": the agent interacts with the item in front of it

Your objective is two-fold:
1. To identify the actions in the trajecotry that are crucial to reach the goal.
2. To evaluate counterfactuals actions: if the agent had taken a different action at a given timestep, would it have reached the goal?

To do this, you will write a score for each action that describes the contribution (influence) of that action to reach the final observation.
Had that action changed, would the agent have reached the goal?
For example, if a different action also would have led to the same observation at the next timestep, then the score of the action is 0.0.
You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep of the action and X is the score you choose.
The sum of all scores must be 1.0.
"""


# PROMPT_INTRO = """\
# I will present you with a sequence of observations from the gameplay of MiniHack.
# Recall that a single observation in MiniHack has three main parts: a) a *message* appearing on top of the screen; b) a *grid of symbols* showing the positions of entities in the room and c) a *set of statistics* at the bottom of the screen.
# I will present you with a sequence of these."""

# PROMPT_PRELIMINARY_KNOWLEDGE = """\
# First, tell me about your knowledge of MiniHack.
# Title this paragraph "Preliminary knowledge"."""

# PROMPT_OBSERVATION = """\
# Write an analysis describing the semantics of each observation strictly using information from the observations and your knowledge of MiniHack.
# Title this paragraph **Observation analysis**."""

# PROMPT_REASONING = """\
# Then, write an analysis describing the semantics of the sequence of observations focusing on the reasons that could have led to the final observation.
# End this analysis by writing whether the agent should avoid or repeat the action at its next encounter with the same state.
# Recall that the goal of the agent is find the staircase up, denoted by ">" and do not confound that with the staircase down symbol "<".
# Title this paragraph **Reasoning Leading to Final Observation**."""

# PROMPT_ACTION_RECOMMENDATION = """\
# Finally, for each timestep, respond by providing the number of the timestep that you evaluate to be the most significant to reach the final observation.
# Title this paragraph **Action recommendation**.
# Synthetise the action recommendation into a dictionary of the form `{"Timestep 0": True}`, writing `True` if you recommend to take same action next time, `False` if you do not recommend and `None` if the action does not matter."""

# PROMPT_REDISTRIBUTION = """\
# The current sequence of observations and actions gathered a total of {} reward.
# Redistribute this reward among the timesteps you evaluated to be the most significant.
# Title this paragraph **Reward redistribution** and write the reward redistribution into a dictionary of the form `{"Timestep 0": 0.0}`."""

# PROMPT_CONCLUSION = """\
# Now begins the sequence of observations:
# {observations}"""
