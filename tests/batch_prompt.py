from __future__ import annotations

import requests


def main():
    len_prompts = 64
    prompts = ["This is prompt number {}".format(i) for i in range(len_prompts)]

    url = "http://localhost:5000/respond"
    max_new_tokens = 512
    response = requests.post(
        url=url,
        data={"prompts[]": [prompt.encode() for prompt in prompts]},
        params={"max_new_tokens": max_new_tokens},
    )

    print(response)

if __name__ == "__main__":
    main()

