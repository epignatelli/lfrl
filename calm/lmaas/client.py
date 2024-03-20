"""Provides a Large Language Model as a Service (LLMaaS).
Uses sockets to communicate between the server and the clients."""
import uuid
import requests
from simple_term_menu import TerminalMenu


WELCOME_MESSAGE = """
Welcome to the Large Language Model as a Service (LLMaaS)!
This is a simple command line client to interact with the LLMaaS server.
Enter an option and type your prompt and press enter to get a response.
"""


class HTTPClient:
    def __init__(self):
        self.conv_id = None

    def start(self):
        url = "http://localhost:5000/respond"
        params = {"conv_id": self.conv_id} if self.conv_id is not None else {}
        while True:
            prompt = input("> ") or ""
            response = requests.post(url, params=params, data=prompt.encode())
            response = response.json()
            if isinstance(response, list) and len(response) > 0:
                response = response[0]

            print("\n")
            print(response)
            print("\n")

    def start_new(self):
        self.conv_id = str(uuid.uuid4())
        self.start()

    def clear(self):
        self.conv_id = None


def prompt_selection(title="") -> int:
    options = ["New chat", "1-shot prompting"]
    terminal_menu = TerminalMenu(options, title=title)
    menu_entry_index = terminal_menu.show()
    if isinstance(menu_entry_index, int):
        return menu_entry_index
    else:
        raise ValueError("Invalid selection")


if __name__ == "__main__":
    selection = prompt_selection(title=WELCOME_MESSAGE)
    client = HTTPClient()
    if selection == 0:
        client.start_new()
    elif selection == 1:
        client.start()
    elif selection == 2:
        client.clear()
