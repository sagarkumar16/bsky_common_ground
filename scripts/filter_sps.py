import json


with open("starterpacks.jsonl", "r") as f:
    starterpacks = [json.loads(line) for line in f]
    filtered_starterpacks = [sp for sp in starterpacks if sp["description"] != ""]

with open("starterpacks_filtered.jsonl", "w") as f:
    for sp in filtered_starterpacks:
        f.write(json.dumps(sp) + "\n")
