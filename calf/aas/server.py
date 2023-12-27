#!/usr/bin/env python
# encoding: utf-8
"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""


import json
import time
from flask import Flask, jsonify, request
from ..models import LLM



class LLMServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        model_name: str = "adept/fuyu-8b",
    ):
        self.host = host
        self.port = port

        # init server
        self.app = Flask(__name__)
        self.app.route("/chat", methods=["GET"])(self.chat)

        # init LLM
        self.llm = LLM(model_name)

    def start(self):
        self.llm.init()
        return self.app.run(host=self.host, port=self.port, debug=True)

    def chat(self):
        start_time = time.time()
        # get the prompt from the request
        prompt = json.loads(request.data)

        # prompt the LLM
        response = self.llm.forward(prompt)

        # return the response
        response = jsonify(response)
        response_time = time.time() - start_time
        response.headers["X-Response-Time"] = str(response_time)
        print(f"Response time: {response_time:.4f}s")
        return response


class InteractiveChatServer(LLMServer):
    def chat(self):
        start_time = time.time()
        # get the prompt from the request
        prompt = request.data.decode()
        print("> ", prompt)

        # return empty response if no prompt
        if prompt is None:
            return jsonify({})

        # parse the inference parameters
        temperature = request.args.get("temperature", 0.9)
        top_p = request.args.get("top_p", 0.9)
        temperature = float(temperature)
        top_p = float(top_p)

        # prompt the LLM
        prompt = prompt
        response = self.llm.forward(prompt)
        print("\t>", response[0])

        # check for a valid response
        if response is None:
            return jsonify({})

        # return the response
        response = jsonify(response)
        response_time = time.time() - start_time
        response.headers["X-Response-Time"] = str(response_time)
        print(f"Response time: {response_time:.4f}s")
        return response


if __name__ == "__main__":
    server = InteractiveChatServer()
    server.start()
