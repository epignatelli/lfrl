from calf.prompts import SYSTEM_PROMPT, PROMPT_ENV_DESCRIPTION
from calf.models import LLM
import wandb


ENVS = [
    "Room",
    "Corridor",
    "KeyRoom",
    "MazeWalk",
    "River",
    "HideNSeek",
    "CorridorBattle",
    "Memento",
    "MazeExplore",
]

MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-7b-instruct",
    # "adept/fuyu-8b",
]


def run():
    for model_name in reversed(MODELS):
        wandb.init(project="lfrl", tags=["env-description"], name=model_name)
        wandb.config.model = model_name
        print("###################")
        print("###################")
        print("###################")
        model = LLM(model_name)
        model.init()
        for env in ENVS:
            prompt = SYSTEM_PROMPT + "\n" + PROMPT_ENV_DESCRIPTION.format(env)
            description = model.forward(prompt)
            result = f"Environment: {env}\n{description}"
            wandb.log({"any_key": wandb.Html(description)})
            print("-------------------")
            print(result)


if __name__ == "__main__":
    run()
