import json
import pandas as pd

with open("data/results.json") as f:
    data = json.load(f)

rows = []
for item in data:
    for response in item["responses"]:
        rows.append({
            "prompt_id": item["id"],
            "task": item["task"],
            "model": response["model"],
            "output": response["output"],
            "score": response["score"]
        })

df = pd.DataFrame(rows)
df.to_csv("data/results.csv", index=False)
print("✅ Results saved to data/results.csv")