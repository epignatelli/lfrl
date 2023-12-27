#!/usr/bin/env python
# encoding: utf-8
"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""


import json
import time
from typing import Dict, List
import uuid
from flask import Flask, jsonify, request
from calf.models import OpenOrcaMistral7B, Roles, Parrot
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
        self.system_prompt = ""
        self.conversations_registry: Dict[str, List[Dict[str, str]]] = {}

    def start(self):
        self.llm.init()
        return self.app.run(host=self.host, port=self.port, debug=True)

    def respond(self):
        # set red light
        self.busy = True

        # get the prompt from the request
        prompt = request.data.decode()
        conv_id = request.args.get("conv_id")
        print(f"Received prompt: {prompt} for conversation ID: {conv_id}")

        if conv_id is None:
            conv_id = str(uuid.uuid4())

        # new conversation
        if conv_id not in self.conversations_registry:
            self.conversations_registry[conv_id] = [
                {"role": Roles.SYSTEM, "content": self.system_prompt},
                {"role": Roles.USER, "content": prompt},
            ]
        # existing conversation
        else:
            self.conversations_registry[conv_id].append(
                {"role": Roles.ASSISTANT, "content": prompt}
            )

        # prompt the LLM
        conversation = self.conversations_registry[conv_id]
        start_time = time.time()
        response = self.llm.chat(conversation)
        response_time = time.time() - start_time
        print(f"Response time: {response_time:.4f}s")
        print(f"{len(self.conversations_registry)} conversations in registry with IDs: {list(self.conversations_registry.keys())}")

        # update registry
        self.conversations_registry[conv_id].append(
            {"role": "assistant", "content": response[0]}
        )

        # format the response
        response = jsonify(response)
        response.headers["X-Response-Time"] = str(response_time)

        # set green light
        self.busy = False

        return response


# class InteractiveChatServer(LLMServer):
#     def respond(self):
#         response = super().respond()
#         response_time = float(response.headers["X-Response-Time"])
#         print(f"Response time: {response_time:.4f}s")
#         return response


if __name__ == "__main__":
    server = LLMServer()
    server.start()
