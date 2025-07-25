import json
from models.ollama_runner import run_model
from analyzers.apply_scoring import apply_scoring
import os
import time

# Load prompts
with open("prompts/sample_set.json") as f:
    prompts = json.load(f)

# Model list
models = ["phi3:3.8b", "mistral:7b", "samantha-mistral:7b"]

# Results will be stored here
results = []

# Loop through each prompt
for prompt in prompts:
    print(f"\n🔍 Running prompt: {prompt['id']}")
    entry = {
        "id": prompt["id"],
        "task": prompt["task"],
        "responses": []
    }

    for model in models:
        print(f" → Running model: {model}")
        try:
            # Run model and apply score
            output = run_model(prompt["task"], model)
            score = apply_scoring({
                "id": prompt["id"],
                "task": prompt["task"],
                "output": output
            })
        except Exception as e:
            print(f"    ❌ Error from {model}: {str(e)}")
            output = f"[Error: {str(e)}]"
            score = 0

        entry["responses"].append({
            "model": model,
            "output": output,
            "score": score
        })

        time.sleep(1)  # Prevent rate-limit issues or overheating

    results.append(entry)
    print("✔ Completed prompt.\n")

# Make sure output folder exists
os.makedirs("data", exist_ok=True)

# Save results
with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ All prompts complete. Results saved to data/results.json.")