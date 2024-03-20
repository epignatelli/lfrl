##############################################################################
###### ROLES
##############################################################################

ROLE_GENERIC = """The environment is MiniHack.
"""


ROLE_ANALYST = """You are a helpful and honest analyst of good gameplaying and \
progress in the NetHack game. Always answer as helpfully as possible, while being \
truthful.

If you don’t know the answer to a question, please don’t share false information.
"""

##############################################################################

##############################################################################
###### SYMSETS
##############################################################################

SYMSET_KEYROOM = """\
I will present you with a short extract of a gameplay. At each timestep, symbols \
represent the following items:
- "." represents a floor tile.
- "|" can represent either a wall, a vertical wall, an open door.
- "-" can represent either the bottom left corner (of a room), bottom right corner \
(of a room), wall, horizontal wall, wall, top left corner (of a room), op right corner \
(of a room).
- "+" represents a closed door. Doors can be locked, and require a key to open.
- "(" represents a useful item (pick-axe, key, lamp...)
- "<" represents a ladder or staircase up.
- ">" represents a ladder or staircase down.
"""

##############################################################################


##############################################################################
###### TASKS
##############################################################################

TASK_WIN = """The task of the agent is to win the game.
"""


TASK_KEYROOM = """The task of the agent is to pickup a key, navigate to a door, and \
use the key to unlock the door, reaching the staircase down within the locked room.
"""

##############################################################################


##############################################################################
###### INSTRUCTIONS
##############################################################################

INSTRUCTION_SUBGOAL = """First, Based on your knowledge of NetHack, break down the \
task of the agent into subgoals.

Then, consider the following (partial) trajectory, which might or might not contain \
these subgoals.
Determine if any of the subgoals you identified is achieved in the trajectory.

Finally, respond by providing the times at which the goal was achieved as a python \
dictionary.
Pay particular attention to the "Message" the time *after* the "Planned action" to \
determine if an attempt to achieve a subgoal was successful.
"""


INSTRUCTION_PREDEFINED_GOAL_SET = """Given a pre-defined list of subgoals, first \
consider the following (partial) trajectory, which might or might not contain these \
subgoals.

Then, determine if any of the subgoals is achieved in the trajectory.

Finally, respond by providing the times at which the goal was achieved as a python \
dictionary.
Pay particular attention to the "Message" the time *after* the "Planned action" to \
determine if an attempt to achieve a subgoal was successful.
"""

##############################################################################


##############################################################################
###### OUTPUT FORMATS
##############################################################################

OUTPUT_FORMAT_DIC = """Finally, respond by providing the times at which the goal was \
achieved as a python dictionary.
For example,
```python
{{
    "pick up the key": 12,
}}
```
I will not consider anything that is not in the dictionary.
You have only one shot at this, and you cannot ask for clarifications.
"""


INPUT_TRAJ = """Observation Sequence:
{}
"""

##############################################################################


##############################################################################
###### PROMPTS
##############################################################################

def prompt_subgoals(
    role: str,
    symset: str,
    task: str,
    instructions: str,
    output_format: str,
    trajectory: str,
) -> str:
    return "\n".join([role, symset, task, instructions, output_format, trajectory])


PROMPT_IDENTIFY_SUBGOALS_BASE = prompt_subgoals(
    ROLE_ANALYST,
    SYMSET_KEYROOM,
    TASK_KEYROOM,
    INSTRUCTION_SUBGOAL,
    OUTPUT_FORMAT_DIC,
    INPUT_TRAJ,
)


PROMPT_IDENTIFY_SUBGOALS_TASK_ABLATION = prompt_subgoals(
    ROLE_ANALYST,
    SYMSET_KEYROOM,
    TASK_WIN,
    INSTRUCTION_SUBGOAL,
    OUTPUT_FORMAT_DIC,
    INPUT_TRAJ,
)


PROMPT_IDENTIFY_SUBGOALS_ROLE_ABLATION = prompt_subgoals(
    ROLE_GENERIC,
    SYMSET_KEYROOM,
    TASK_KEYROOM,
    INSTRUCTION_SUBGOAL,
    OUTPUT_FORMAT_DIC,
    INPUT_TRAJ,
)


PROMPT_IDENTIFY_SUBGOALS_IDENTIFICAION_ABLATION = prompt_subgoals(
    ROLE_GENERIC,
    SYMSET_KEYROOM,
    TASK_KEYROOM,
    INSTRUCTION_SUBGOAL,
    OUTPUT_FORMAT_DIC,
    INPUT_TRAJ,
)


##############################################################################

# PROMPT_IDENTIFY_SUBGOALS_OPENENDED = (
#     """You are a helpful and honest analyst of good gameplaying and progress in the \
# NetHack game. Always answer as helpfully as possible, while being truthful.

# If you don’t know the answer to a question, please don’t share false information.

# """
#     + """I will present you with a short extract of a gameplay. At each timestep, \
# symbols represent the following items:\n"""
#     + SYMSET_KEYROOM
#     + """\

# The task of the agent is to win the game.

# Let's reason step by step.

# First, Based on your knowledge of NetHack, bsreak down the task of the agent into subgoals.

# Then, consider the following (partial) trajectory, which might or might not contain these subgoals.
# Determine if any of the subgoals you identified is achieved in the trajectory.

# Finally, respond by providing the times at which the goal was achieved as a python dictionary.
# Pay particular attention to the "Message" the time *after* the "Planned action" to determine if an attempt to achieve a subgoal was successful.
# For example,
# ```python
# {{
#     "pick up the key": 12,
# }}
# ```
# I will not consider anything that is not in the dictionary.
# You have only one shot at this, and you cannot ask for clarifications.

# Observation Sequence:
# {}
# """
# )


# PROMPT_IDENTIFY_SUBGOALS_ROLE = (
#     """You are a helpful and honest analyst of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while
# being truthful.

# If you don’t know the answer to a question, please don’t share false information.

# """
#     + """I will present you with a short extract of a gameplay. At each timestep, symbols represent the following items:\n"""
#     + SYMSET_KEYROOM
#     + """\

# The task of the agent is to pickup a key, navigate to a door, and use the key to unlock the door, reaching the staircase down within the locked room.

# Break down the task of the agent into subgoals.
# Now, consider the following (partial) trajectory, which might or might not contain these subgoals.
# Determine if any of the subgoals you identified is achieved in the trajectory and identify which action in the trajectory is responsible for it.

# Observation Sequence:
# {}

# Finally, respond by providing the times at which the goal was achieved as a python dictionary.
# Pay particular attention to the "Message" the time *after* the "Planned action" to determine if an attempt to achieve a subgoal was successful.
# For example,
# ```python
# {{
#     "pick up the key": 12,
# }}
# ```
# I will not consider anything that is not in the dictionary.
# """
# )


# PROMPT_IDENTIFY_SUBGOALS = (
#     """You are a helpful and honest judge of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while
# being truthful.

# If you don’t know the answer to a question, please don’t share false information.

# """
#     + """You are presented with the MiniHack environment. In each observation, symbols represent the following items:\n"""
#     + SYMSET_KEYROOM
#     + """\

# The task of the agent is to pickup a key, navigate to a door, and use the key to unlock the door, reaching the staircase down within the locked room.

# Break down the task of the agent into subgoals.
# Now, consider the following (partial) trajectory, which might or might not contain these subgoals.
# Determine if any of the subgoals you identified is achieved in the trajectory and identify which action in the trajectory is responsible for it.

# Observation Sequence:
# {}

# Finally, provide the times at which the goal was achieved as a python dictionary.
# Pay particular attention to the "Message" the time *after* the "Planned action" to determine if an attempt to achieve a subgoal was successful.
# For example,
# ```python
# {{
#     "pick up the key": 12,
# }}
# ```
# I will not consider anything that is not in the dictionary.
# """
# )


# PROMPT_IDENTIFY_PREDEFINED_SUBGOALS = (
#     """\
# The environment is MiniHack. In each observation, symbols represent the following items:"""
#     + SYMSET_KEYROOM
#     + """\
# The task of the agent is to pickup a key, navigate to a door, and use the key to unlock\
#  the door, reaching the staircase down within the locked room.

# I will provide you with a list of subgoals for the task.

# Now, consider the following (partial) trajectory, which might or might not contain these subgoals.
# Determine if any of the subgoals is achieved in the trajectory and identify which action in the trajectory is responsible for it.
# Specify also a confidence score and do not report subgoals identified low confidence.


# Observation Sequence:
# {}

# List of subboals.
# ```python
# subgoals = [
#     "navigate to the key",
#     "pick up the key",
#     "navigate to the door",
#     "use the key to unlock the door",
#     "navigate to the staircase down"
# ]
# ```


# Finally, provide the times at which the goal was achieved as a python dictionary.
# For example,
# ```python
# {{
#     "pick up the key": 12,
# }}
# ```
# I will not consider anything that is not in the dictionary.
# """
# )
