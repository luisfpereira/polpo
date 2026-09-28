import json


def load_json(path):
    with open(path) as file:
        return json.load(file)


def dump_json(data, path, indent=2):
    with open(path, "w") as file:
        json.dump(data, file, indent=indent)
