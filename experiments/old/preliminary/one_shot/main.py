import requests
from calm.prompts import PROMPT_TD_WEIGHTING, PROMPT_COUNTERFACTUALS, PROMPT_REDISTRIBUTION, PROMPT_EXPERIMENT
from calm.example_trajectories import TRAJECTORY_1, TRAJECTORY_2, TRAJECTORY_3, TRAJECTORY_4, TRAJECTORY_6, OBSERVATION_1


def query(prompt, trajectory):
    """Assumes that there is a server running on localhost:5000."""
    url = "http://localhost:5000/respond"
    prompt = prompt + trajectory
    response = requests.post(url, data=prompt.encode(), params={"max_new_tokens": 512})
    response = response.json()[0]
    print(response)


def counterfactual():
    return query(PROMPT_COUNTERFACTUALS, TRAJECTORY_4)


def credit_assignment():
    return query(PROMPT_TD_WEIGHTING, TRAJECTORY_6)


def redistribute():
    return query(PROMPT_EXPERIMENT, TRAJECTORY_4)


if __name__ == "__main__":
    credit_assignment()
