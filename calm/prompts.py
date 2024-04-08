##############################################################################
###### ROLES
##############################################################################

ROLE_GENERIC = """The environment is MiniHack.
"""


ROLE_ANALYST = """You are a helpful and honest analyst of good gameplaying and \
progress in the NetHack game. Always answer as helpfully as possible, while being \
truthful.
If you don't know the answer to a question, please don't share false information.
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


SYMSET_KEYROOM_EXPLAINED = """\
I will present you with a short extract of a gameplay made of a sequence of timesteps.
Each timestep includes:
- "Time": the time at which it happened.
- "Last action": the action performed by the agent in the previous timestep.
- "Current Message": the on-screen message emitted by NetHack.
- "Current observation": the screen of the game at the current timstep.
- "Next action": the action planned for execution in the next timestep.

Each observation is made of the following symbols:
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


TASK_KEYROOM_MANUAL = """The task of the agent is to pick up a key, navigate to a \
door, use the key to unlock the door and navigate to the staircaise down.
"""

##############################################################################


##############################################################################
###### SUBGOALS
##############################################################################

SUBGOALS_IDENTIFY = """First, based on your knowledge of NetHack, break down the \
task of the agent into subgoals.
"""


SUBGOALS_PRESET = """First, recall your knowledge of NetHack with respect to the \
following subgoals:
```python
subgoals = {{
    "pick up the key": None,
    "navigate to the door": None,
    "unlock the door": None,
    "navigate to the staircase down": None,
}}
```
"""

##############################################################################


##############################################################################
###### INSTRUCTIONS
##############################################################################

INSTRUCTION_IDENTIFY = """Then, consider the following (partial) trajectory, which \
might or might not contain these subgoals.
Determine if any of the subgoals you identified appears in the trajectory.

Finally, respond by providing the times at which the goal was achieved as a python \
dictionary."""


INSTRUCTION_PROGRESS = """Then, consider the following (partial) trajectory, which \
might or might not contain these subgoals.
Identify the actions in the trajecory that make satisfactory progress \
towards achieving the subgoals you identified (any of them).

Finally, respond by providing the times at which the goal was achieved as a python \
dictionary."""


INSTRUCTION_OPTIMALITY = """Then, consider the following (partial) trajectory, which \
might or might not contain these subgoals.
For each time in the trajectory, classify whether the *chosen action* is optimal \
or not, and explain the reasons of your choice.

Finally, respond by filtering only the optimal actions an return them in a python \
dictionary."""


INSTRUCTION_TRANSITION = """Then, consider the following game transition, which \
might or might not contain these subgoals.
Determine if any of the subgoals you identified has been achieved at Time: 1 or not.
"""

##############################################################################


##############################################################################
###### REMARKS
##############################################################################

REMARK_MESSAGE_ACTION = """Pay particular attention to the "Message" the time *after* \
the "Last action" to determine if an attempt to achieve a subgoal was successful.
"""


REMARK_MESSAGE = """Pay particular attention to the "Last action" to determine if an \
attempt to achieve a subgoal was successful.
"""


REMARK_ACTION = """Pay particular attention to the "Message" to determine if an \
attempt to achieve a subgoal was successful.
"""


REMARK_NONE = ""

##############################################################################


##############################################################################
###### OUTPUT FORMATS
##############################################################################

OUTPUT_FORMAT_DIC = """For example, if the first observation after the key was picked \
up is at Time: 12, respond with:
```python
{{
    "pick up the key": 12,
}}
```
"""


OUTPUT_FORMAT_GENERIC = """For example,
```python
{{
    <goal-name as str>: <time as int>,
}}
```
"""


OUTPUT_FORMAT_TRANSITION = """Report your response in a dictionary containing the name \
of the subgoals as keys and booleans as value. For example:
```python
{{
    <name of goal>: <bool>,
}} 
"""


INPUT_TRAJ = """Observation Sequence:

<gameplay>
{}
</gameplay>
"""

##############################################################################


##############################################################################
###### OUTPUT REMARKS
##############################################################################

OUTPUT_REMARK = """I will not consider anything that is not in the dictionary.
You have only one shot at this, and you cannot ask for clarifications.
"""

##############################################################################


##############################################################################
###### PROMPTS
##############################################################################


def prompt_subgoals(
    role: str,
    symset: str,
    task: str,
    subgoals: str,
    instructions: str,
    remark: str,
    output_format: str,
    trajectory: str,
    output_remark: str,
) -> str:
    return "\n".join(
        [
            role,
            symset,
            task,
            subgoals,
            instructions,
            remark,
            output_format,
            trajectory,
            output_remark,
        ]
    )


PROMPT_BASE = prompt_subgoals(
    ROLE_ANALYST,
    SYMSET_KEYROOM,
    TASK_KEYROOM,
    SUBGOALS_IDENTIFY,
    INSTRUCTION_IDENTIFY,
    REMARK_MESSAGE_ACTION,
    OUTPUT_FORMAT_DIC,
    INPUT_TRAJ,
    OUTPUT_REMARK
)

##############################################################################
