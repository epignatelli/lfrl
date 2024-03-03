"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""
import time
from typing import Dict, List
from flask import Flask, jsonify, request

from calf.language_models import OpenOrcaMistral7B, Roles, Gemma7B
from calf.prompts import DEFAULT_SYSTEM_PROMPT


class LLMServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.host = host
        self.port = port

        # init server
        self.app = Flask(__name__)
        self.app.route("/respond", methods=["POST"])(self.respond)

        # init LLM
        self.llm = Gemma7B()
        self.busy = False

        # init conversation registry
        self.system_prompt = system_prompt

    def start(self):
        self.llm.init()
        return self.app.run(host=self.host, port=self.port, debug=False)

    def respond(self):
        # get the prompt from the request
        max_new_tokens = int(request.args.get("max_new_tokens", 256))

        prompts = request.form.getlist("prompts[]")
        conversations = [[{"role": Roles.USER, "content": prompt}] for prompt in prompts]

        print(f"Received {len(prompts)} prompts. Processing...")

        # prompt the LLM
        start_time = time.time()
        response = self.llm.chat(conversations, max_new_tokens=max_new_tokens)
        response_time = time.time() - start_time
        print(f"Response time: {response_time:.4f}s")

        # format the response
        response = jsonify(response)
        response.headers["X-Response-Time"] = str(response_time)
        return response

if __name__ == "__main__":
    server = LLMServer()
    server.start()
