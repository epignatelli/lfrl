import requests
from calm.prompts import PROMPT_TD_WEIGHTING, PROMPT_COUNTERFACTUALS, PROMPT_REDISTRIBUTION, PROMPT_EXPERIMENT
from calm.example_trajectories import TRAJECTORY_1, TRAJECTORY_2, TRAJECTORY_3, TRAJECTORY_4, TRAJECTORY_5, TRAJECTORY_6, OBSERVATION_1


def query(prompt):
    """Assumes that there is a server running on localhost:5000."""
    url = "http://localhost:5000/respond"
    prompt = prompt
    response = requests.post(url, data=prompt.encode(), params={"max_new_tokens": 2048})
    response = response.json()[0]
    print(response)


if __name__ == "__main__":
#     PROMPT = """\
# I will present you with a sequence of observations from the gameplay of MiniHack.
# In the observations, symbols represent the following items:
# A dot "." represents an empty space.
# A pipe "|" represent a walls.
# A dash "-" can represent either a wall or an open door.
# A plus sign "+" represents a closed door.
# A parenthesis ")" represents a key.
# The goal of the agent is to pickup a key, navigate to a door, and use the key to unlock the door, reaching the staircase down within the locked room.

# TASK:
# Take the total return of the episode and redistribute it to each timestep in the trajectory.
# Pay particular attention to actions that are significantly delayed from the final return and that have made the greatest difference in contributing to it.
# Report the credit in a python dictionary whose keys are the time t of the timestep and the value is the corresponding score.

# """
    prompt = """
Environment: The environment is MiniHack. In each observation, symbols represent the following items:
# "." represents a floor tile.
# "|" represent a walls.
# "-" can represent either a wall or an open door.
# "+" represents a closed door. Doors can be locked, and require a key to open.
# ")" represents a key. Keys can open doors.

Observation Sequence:
{}

Final Return: {}

Task: Your task is to identify key sub-goals in the trajectory below.\
Return the timestep at which these are achieved in a python dictionary.
Provide a justification for choice."""

    prompt = prompt.format(TRAJECTORY_5, "1")

    query(prompt)