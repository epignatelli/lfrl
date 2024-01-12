DEFAULT_SYSTEM_PROMPT = """\
Welcome to the MiniHack data collection and evaluation interface.
This interface is designed to collect data from MiniHack and evaluate the performance of reinforcement learning agents.
You are a helpful and honest judge of good gameplaying and progress in the MiniHack game.
Always answer as helpfully as possible, while being truthful.
If you don't know the answer to a question, please don't share false information.

"""

SYMBOLS_INDEX = """\
A dot "." represents an empty space.
A pipe "|" represent a walls.
A dash "-" can represent either a wall or an open door.
A plus sign "+" represents a closed door.
A parenthesis ")" represents a key.
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

The goal of the agent is to solve the maze and find the staircase up, denoted by the greater than sign ">".

Write a brief and concise analysis describing the semantics of the trajectory strictly using information from the observations and your knowledge of MiniHack.

Finally, write a score for each action that describes the contribution of that action to reach the final observation.
For example, if a different action also would have led to the same observation at the next timestep, then the score of the action is 0.0.
You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep of the action and X is the score you choose.
The sum of all scores must be 1.0.

We aim to find those unique actions that are crucial to the point that without them, the agent would not be able to reach the goal.

"""


PROMPT_COUNTERFACTUALS = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

Your objective is to identify the actions in the trajectory that have been crucial to achieve the goal.

To do this, you will write a score for the action taken at each timestep.
The score describes the contribution (influence) of that action to reach the final observation.
Useful questions to ask yourself are:
"Had that action changed, would the agent have reached the goal?",
"Would the agent have reached the goal if it had taken a different action at that timestep?",
"How important was that action to reach the goal?".

You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep and X is the score you choose.
The sum of all scores in the timestep must be 1.0.

Finally, explain why you chose that score and, if an action is crucial why that is so.
"""


PROMPT_REDISTRIBUTION = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;s
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

Your objective is to evaluate the *influence* of each action in the trajectory.
To do this, decompose the final return (the sum of rewards) and redistribute it to each event in the trajectory.
If an event is key to obtain that return, then it should receive a higher share of influence.
On the contrary, if an action has been irrelevant to achieve the goal, then it should receive a lower share of influence.

Report the influence in the following format: {"timestep-t": X}, where t is the timestep and X is the score you choose, for each timestep.
The influence must be between 0.0 and 1.0, and the sum of the all influences must be 1.0.

Finally, explain why you chose to attribute that amount of influence.
"""

PROMPT_EXPERIMENT = """\
Credit assignment refers to the process of determining how much credit or responsibility each action or decision in a sequence should receive for a particular outcome.
In the context of reinforcement learning, credit assignment is crucial for understanding which actions led to positive or negative outcomes and adjusting the model accordingly.

In simpler terms, it's about figuring out which steps or decisions in a series of actions contributed to the overall success or failure of a task.
This is essential for reinforcement learning agents to learn and improve their behavior over time by attributing the correct amount of credit to each action.

Here we will present you with a sequence of observations and actions from the gameplay of MiniHack.
Your objective is to assign the right credit to each action in the trajectory.

To do this, you will write a score for the action taken at each timestep that describes the credit (influence) of that action.
Report the credit in the following format: {"timestep-t": X}, where t is the timestep and X is the credit.
The credit must be between 0.0 and 1.0, and the sum of the all credits must be 1.0.

Finally, explain why you chose to attribute that amount of credit.
"""