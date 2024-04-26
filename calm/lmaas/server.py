"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""

import time
from typing import Dict, List
from flask import Flask, jsonify, request

from calm.lmaas.models import LLM, Roles, Gemma7B


class LLMServer:
    def __init__(
        self,
        name: str = "",
        revision: str = "",
        host: str = "localhost",
        port: int = 5000,
        template_url: str = "",
        system_prompt: str = "",
    ):
        self.host = host
        self.port = port

        # init server
        self.app = Flask(__name__)
        self.app.route("/respond", methods=["POST"])(self.respond)
        self.app.route("/ready", methods=["GET"])(self.ready)

        # init LLM
        if name != "":
            self.llm = LLM(name, revision, template_url)
        else:
            self.llm = Gemma7B()

        # init conversation registry
        self.system_prompt = system_prompt

    def serve(self, **init_kwargs):
        self.llm.init(**init_kwargs)
        return self.app.run(host=self.host, port=self.port, debug=False)

    def ready(self):
        return jsonify({"ready": True})

    def respond(self):
        # get the prompt from the request
        max_new_tokens = int(request.args.get("max_new_tokens", 256))

        prompts = request.form.getlist("prompts[]")
        conversations = [
            [{"role": Roles.USER, "content": prompt}] for prompt in prompts
        ]

        print(f"Received {len(prompts)} prompts. Processing...")

        # prompt the LLM
        start_time = time.time()
        response = self.llm.chat(conversations, max_new_tokens=max_new_tokens)
        response_time = time.time() - start_time
        print(f"Response time: {response_time:.4f}s")

        # format the response
        response = jsonify(response)
        response.headers["X-Response-Time"] = str(response_time)
        response.headers["X-Model-Name"] = str(self.llm.model_name)
        return response


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    argparser.add_argument("--revision", type=str, default="")
    argparser.add_argument("--host", type=str, default="localhost")
    argparser.add_argument("--port", type=int, default=5000)
    argparser.add_argument("--template_url", type=str, default="")
    argparser.add_argument("--load_in_4bit", default=False, action="store_true")
    argparser.add_argument("--load_in_8bit", default=False, action="store_true")
    args = argparser.parse_args()
    server = LLMServer(
        name=args.name,
        revision=args.revision,
        host=args.host,
        port=args.port,
        template_url=args.template_url,
    )
    if args.load_in_4bit:
        kwargs = {"load_in_4bit": True}
    elif args.load_in_8bit:
        kwargs = {"load_in_8bit": True}
    else:
        kwargs = {}
    server.serve(**kwargs)
