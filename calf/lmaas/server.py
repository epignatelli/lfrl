"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""
import time
from typing import Dict, List
from flask import Flask, jsonify, request

from calf.models import OpenOrcaMistral7B, Roles
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
        self.llm = OpenOrcaMistral7B()
        self.busy = False

        # init conversation registry
        self.system_prompt = system_prompt
        self.conversations_registry: Dict[str, List[Dict[str, str]]] = {}

    def start(self):
        self.llm.init()
        return self.app.run(host=self.host, port=self.port, debug=True)

    def respond(self):
        # get the prompt from the request
        prompt = request.data.decode()
        conv_id = request.args.get("conv_id")
        print(f"Received prompt: {prompt} for conversation ID: {conv_id}")

        conversation = [
            {"role": Roles.SYSTEM, "content": self.system_prompt},
            {"role": Roles.USER, "content": prompt},
        ]

        # if conversation recording is requested
        if conv_id is not None:
            if conv_id not in self.conversations_registry:
                # if this is a new conversation, add it to the registry
                self.conversations_registry[conv_id] = conversation
            else:
                # otherwise, append the prompt to the existing conversation
                self.conversations_registry[conv_id].append(conversation[-1])
                conversation = self.conversations_registry[conv_id]

        # prompt the LLM
        start_time = time.time()
        response = self.llm.chat(conversation)
        response_time = time.time() - start_time
        print(f"Response time: {response_time:.4f}s")
        print(
            f"{len(self.conversations_registry)} conversations in registry with IDs: {list(self.conversations_registry.keys())}"
        )

        # update registry with llm response
        if conv_id is not None:
            self.conversations_registry[conv_id].append(
                {"role": "assistant", "content": response[0]}
            )

        # format the response
        response = jsonify(response)
        response.headers["X-Response-Time"] = str(response_time)
        return response


if __name__ == "__main__":
    server = LLMServer()
    server.start()
