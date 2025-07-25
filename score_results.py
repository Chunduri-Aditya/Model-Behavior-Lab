import json
from analyzers.scoring import score_logic, score_emotion, score_hallucination, score_code

with open("data/results.json") as f:
    data = json.load(f)

for item in data:
    task_type = item["id"].split("-")[0]  # logic, emotion, etc.
    for response in item["responses"]:
        output = response["output"]

        if task_type == "logic":
            score = score_logic(output)
        elif task_type == "emotion":
            score = score_emotion(output)
        elif task_type == "hallucination":
            score = score_hallucination(output)
        elif task_type == "coding":
            score = score_code(output)
        else:
            score = 0

        response["score"] = score

with open("data/results.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Scores added to data/results.json")