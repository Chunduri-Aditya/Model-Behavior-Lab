import subprocess

def run_model(prompt, model_name):
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[Runtime Error: {str(e)}]"