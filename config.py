import json
def load_config(path="./demo_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)