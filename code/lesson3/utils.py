import json


def read_json_file(file_path: str) -> dict:
    """
    Read a JSON file and return the contents as a dictionary.
    """
    with open(file_path, "r") as file:
        return json.load(file)
