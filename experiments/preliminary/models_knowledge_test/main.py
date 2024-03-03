from calf.prompts import PROMPT_ENV_DESCRIPTION
from calf.language_models import (
    Llama7B,
    Llama13B,
    Llama70B,
    Mistral7B,
    OpenOrcaMistral7B,
    Falcon7B,
    Fuyu8B,
)
import wandb

from example_prompts import PROMPT_CREDIT_ASSIGNMENT_EXAMPLE


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
    Llama7B(),
    Mistral7B(),
    OpenOrcaMistral7B(),
    Falcon7B(),
]

MODELS_EXTRA = [
    Llama13B(),
    Llama70B(),
]

MODELS_MULTIMODAL = [
    Fuyu8B(),
]


def run_env_description():
    for model in MODELS_EXTRA:
        wandb.finish()
        wandb.init(project="lfrl", name=model.model_name)
        wandb.config.update({"model": model.model_name, "experiment": "ask_env_description"})
        model.init()
        for env_name in ENVS:
            html = f"<h3>{model.model_name}</h3><br>"
            prompt = PROMPT_ENV_DESCRIPTION.format(env_name)
            description = model.forward(prompt)
            result = f"Environment: {env_name}\n{description}"
            print("-------------------")
            print(result)
            html += f"<h4>MiniHack {env_name}</h4><br>"
            html += f"<p>{description}</p><br>"
            wandb.log({f"ask_env_description_{env_name}": wandb.Html(html)})


def run_credit():
    for model in MODELS_EXTRA:
        wandb.finish()
        wandb.init(project="lfrl", tags=["exp: ask_credit"], name=model.model_name)
        wandb.config.update({"model": model.model_name, "experiment": "ask_credit"})
        html = f"<h3>{model.model_name}</h3>"
        model.init()
        prompt = PROMPT_CREDIT_ASSIGNMENT_EXAMPLE
        response = model.forward(prompt, max_new_tokens=512)
        html += f"<p>{response}</p>"
        print("-------------------")
        print(response)
        html += "<hr>"
        wandb.log({"ask_credit": wandb.Html(html)})


if __name__ == "__main__":
    run_env_description()
    run_credit()
