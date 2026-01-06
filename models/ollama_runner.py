import subprocess
import time
import os


def _build_ollama_cmd(model_name, sampling):
    """
    Build an ollama CLI command with best-effort sampling flags.
    """
    cmd = ["ollama", "run", model_name]
    if not sampling:
        return cmd, {}

    flags = []
    # Map sampling parameters to Ollama CLI flags (best-effort)
    if "temperature" in sampling:
        flags.extend(["--temperature", str(sampling["temperature"])])
    if "top_p" in sampling:
        flags.extend(["--top-p", str(sampling["top_p"])])
    if "top_k" in sampling:
        flags.extend(["--top-k", str(sampling["top_k"])])
    if "seed" in sampling:
        flags.extend(["--seed", str(sampling["seed"])])
    if "max_tokens" in sampling:
        flags.extend(["--num-predict", str(sampling["max_tokens"])])

    return cmd + flags, sampling


def run_model(prompt, model_name, sampling=None, timeout_s=90):
    """
    Run a model with Ollama CLI, capturing output, latency, and errors.

    Args:
        prompt: Input prompt string
        model_name: Ollama model name (e.g., "phi3:3.8b")
        sampling: Dict with temperature, top_p, top_k, seed, max_tokens
        timeout_s: Timeout in seconds

    Returns:
        Dict:
        {
          "output": str,
          "latency_ms": int,
          "exit_code": int,
          "error": str | None,
          "used_sampling": dict,
          "sampling_supported": bool
        }
    """
    start_time = time.time()
    env = os.environ.copy()

    # First attempt: with sampling flags (if provided)
    cmd_with_flags, used_sampling = _build_ollama_cmd(model_name, sampling)
    sampling_supported = True if sampling else False
    output = ""
    error = None
    exit_code = 0

    def _run(cmd):
        return subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )

    try:
        result = _run(cmd_with_flags)
        exit_code = result.returncode
        if result.returncode != 0 and sampling:
            # If flags are not supported, fall back to basic command
            stderr_lower = (result.stderr or "").lower()
            if "unknown flag" in stderr_lower or "flag provided but not defined" in stderr_lower:
                sampling_supported = False
                # Retry without any sampling flags
                base_cmd, _ = _build_ollama_cmd(model_name, None)
                result = _run(base_cmd)
                exit_code = result.returncode
                used_sampling = {}

        if result.returncode != 0:
            error = f"Ollama returned exit code {result.returncode}: {result.stderr}"
            output = (result.stdout or "").strip()
        else:
            output = (result.stdout or "").strip()

    except subprocess.TimeoutExpired:
        exit_code = -1
        error = f"Timeout after {timeout_s} seconds"
        output = ""
    except Exception as e:
        exit_code = -1
        error = f"Runtime error: {str(e)}"
        output = ""

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "output": output,
        "latency_ms": latency_ms,
        "exit_code": exit_code,
        "error": error,
        "used_sampling": used_sampling or {},
        "sampling_supported": bool(sampling) and sampling_supported,
    }
