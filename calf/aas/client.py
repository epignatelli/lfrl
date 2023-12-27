"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""


import uuid
import requests

WELCOME_MESSAGE = """
Welcome to the Large Language Model as a Service (LLMaaS)!
This is a simple command line client to interact with the LLMaaS server.
Type your prompt and press enter to get a response.
"""

class HTTPClient:
    def __init__(self):
        self.conv_id = None

    def start(self, prompt: str = ""):
        if self.conv_id is None:
            # if this is a new conversation, set the conversation ID
            self.conv_id = str(uuid.uuid4())
        url = "http://localhost:5000/respond"
        print(WELCOME_MESSAGE)
        while True:
            prompt = input("> ") or ""
            response = requests.post(
                url, params={"conv_id": self.conv_id}, data=prompt.encode()
            )
            response = response.json()
            if isinstance(response, list) and len(response) > 0:
                response = response[0]

            print(response)

    def start_new(self):
        self.conv_id = None
        self.start()


if __name__ == "__main__":
    client = HTTPClient()
    client.start()
