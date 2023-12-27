"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""


import requests


class HTTPClient:
    def start(self, prompt: str = "", temperature: float = 0.9, top_p: float = 0.9):
        url = "http://localhost:5000/chat"
        while True:
            prompt = input("> ")
            params = {"temperature": temperature, "top_p": top_p}
            response = requests.get(url, params=params, data=prompt.encode())
            response = response.json()
            if isinstance(response, list) and len(response) > 0:
                response = response[0]

            if "generated_text" not in response:
                response = ""

            print("\t>", response[0])


if __name__ == "__main__":
    client = HTTPClient()
    client.start()
