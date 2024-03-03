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


TASK_KEYROOM = """\
These tasks require the agent to pickup a key, navigate to a door, and use the key to unlock the door, reaching the staircase down within the locked room."""


PROMPT_RETRY = """\
Please assign credit to all the actions and return the result as a python dictionary.
"""

PROMPT_TD_WEIGHTING = """\
I will present you with a sequence of observations and actions from the gameplay of MiniHack.
Recall that a single observation in MiniHack has four main parts:
a) The number of the timestep and the last action;
b) a *message* appearing on top of the screen;
c) a *grid of symbols* showing the positions of entities in the room and
d) a *set of statistics* at the bottom of the screen.

{}

Write a brief and concise analysis describing the semantics of the trajectory strictly using information from the observations and your knowledge of MiniHack.

Finally, write a score for each action that describes the contribution of that action to reach the final observation.
We aim to find those unique actions that are crucial to the point that without them, the agent would not be able to reach the goal.
For example, if a different action also would have led to the same observation at the next timestep, then the score of the action is 0.0.
You can choose any score between 0.0 and 1.0, writing {"timestep-t": X}, where t is the timestep of the action and X is the score you choose.
The sum of all scores must be 1.0.

Report the score in a python dictionary whose keys are the time of the timestep and the value is the corresponding score.

"""


PROMPT_INTRINSIC_REWARD = """\
I will present you with a trajectory from the gameplay of MiniHack, comprised of the time, actions, observations and rewards.
A dot "." represents an empty space.
A pipe "|" represent a walls.
A dash "-" can represent either a wall or an open door.
A plus sign "+" represents a closed door.
A parenthesis ")" represents a key.

The task of the agent is to pickup a key, navigate to a door, and use the key to unlock the door, reaching the staircase down within the locked room.

Write a brief and concise analysis describing the semantics of the observation strictly using information from the observations and your knowledge of MiniHack.
Rely only on your knowledge, and do not make stuff up. If you do not know the answer to the question, explicitly say so.

Finally, write a score for the observation that describes the influence of the action upon solving the task.
You can choose any score between 0.0 and 1.0.

Report the score in a python dictionary whose keys are the time of the timestep and the value is the corresponding score.
"""

PROMPT_IDENTIFY_SUBGOALS = """\
Environment: The environment is MiniHack. In each observation, symbols represent the following items:
# "." represents a floor tile.
# "|" represent a walls.
# "-" can represent either a wall or an open door.
# "+" represents a closed door. Doors can be locked, and require a key to open.
# ")" represents a key. Keys can open doors.

Observation Sequence:
{}

Final Return: {}

Task: Your task is to identify key sub-goals in the trajectory below, if there is any.\
Return the timestep at which these are achieved in a python dictionary."""


PROMPT_REDISTRIBUTION_ALT = """\
Environment: The environment is MiniHack. In each observation, symbols represent the following items:
# A dot "." represents a floor tile.
# A pipe "|" represent a walls.
# A dash "-" can represent either a wall or an open door.
# A plus sign "+" represents a closed door. Doors can be locked, and require a key to open.
# A parenthesis ")" represents a key. Keys can open doors.

Observation Sequence:
{}

Final Return: {}

Task: Your task is to redistribute the final return among the actions taken in this \
episode. Assign higher credit to actions that you believe had a significant positive \
impact on the final outcome, even if their contribution was delayed. Provide a \
justification for your credit assignment.

Return the results in a python dictionary with the timesteps as keys and the credits as values.
"""


PROMPT_REDISTRIBUTION = """\
I will present you with a sequence of observations from the gameplay of MiniHack.
In the observations, symbols represent the following items:
A dot "." represents an empty space.
A pipe "|" represent a walls.
A dash "-" can represent either a wall or an open door.
A plus sign "+" represents a closed door.
A parenthesis ")" represents a key.

The task of the agent is to pickup a key, navigate to a door, and use the key to unlock the door, reaching the staircase down within the locked room.

Write a brief and concise analysis describing the semantics of the trajectory strictly using information from the observations and your knowledge of MiniHack.
Rely only on your knowledge, and do not make stuff up. If you do not know the answer to the question, explicitly say so.

Finally, take the final return of the episode and redistribute it to each timestep in the trajectory.
Pay particular attention to actions that are significantly delayed from the final return and that have made the greatest difference in contributing to it.
Do not credit exploration: we are only interested in reditributing the final return into individual timesteps.

Report the credit in a python dictionary whose keys are the time t of the timestep and the value is the corresponding score.

{}

{}
"""


PROMPT_COUNTERFACTUALS = """\
I will present you with a sequence of observations from the gameplay of MiniHack.
In the observations, symbols represent the following items:
A dot "." represents an empty space.
A pipe "|" represent a walls.
A dash "-" can represent either a wall or an open door.
A plus sign "+" represents a closed door.
A parenthesis ")" represents a key.

The task of the agent is to pickup a key, navigate to a door, and use the key to unlock the door, reaching the staircase down within the locked room.

Write a brief and concise analysis describing the semantics of the trajectory strictly using information from the observations and your knowledge of MiniHack.
Rely only on your knowledge, and do not make stuff up. If you do not know the answer to the question, explicitly say so.

Finally, let's reason by counterfactuals.
Imagine that the agent, at timestep {t}, had choosen the action "{counterfactual_action}" instead of the action "{previous_action}".
Write a score that describes the advantage of taking the action "{counterfactual_action}", instead of the action "{previous_action}".
The advantage can be comprised between -1.0 and 1.0 and be positive it "{counterfactual_action}" would have been better than "{previous_action}" and negative otherwise.
Choose 0 if the change in actions does not matter.

Report the advantage in a python dictionary whose key is the time t of the timestep and the value is the corresponding advantage.
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