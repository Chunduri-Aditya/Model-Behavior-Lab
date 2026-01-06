import re
import json
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional
from models.ollama_runner import run_model


def normalize_text(text):
    """Normalize text for comparison: lowercase, strip, remove extra whitespace."""
    if not isinstance(text, (str, bytes)):
        text = str(text)
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def evaluate_exact_match(model_output, expected, params):
    """Evaluate using exact match with optional normalization."""
    normalize = params.get("normalize", True)
    case_sensitive = params.get("case_sensitive", False)
    
    if normalize:
        model_norm = normalize_text(model_output)
        expected_norm = normalize_text(expected)
        return 1.0 if model_norm == expected_norm else 0.0
    else:
        if not case_sensitive:
            return 1.0 if model_output.lower().strip() == expected.lower().strip() else 0.0
        else:
            return 1.0 if model_output.strip() == expected.strip() else 0.0


def evaluate_numeric_tolerance(model_output, expected, params):
    """Evaluate numeric output within tolerance."""
    tolerance = params.get("tolerance", 0.01)
    
    # Extract numbers from output
    numbers = re.findall(r'-?\d+\.?\d*', model_output)
    if not numbers:
        return 0.0
    
    try:
        # Try to parse the first number found
        model_value = float(numbers[0])
        expected_value = float(expected)
        
        if abs(model_value - expected_value) <= tolerance:
            return 1.0
        else:
            # Partial credit for close answers
            diff = abs(model_value - expected_value)
            max_diff = max(abs(expected_value), 1.0)
            partial = max(0.0, 1.0 - (diff / max_diff))
            return partial
    except (ValueError, TypeError):
        return 0.0


def evaluate_contains(model_output, expected, params):
    """Evaluate if output contains required terms and doesn't contain forbidden terms."""
    required_terms = params.get("required_terms", [])
    forbidden_terms = params.get("forbidden_terms", [])
    
    model_lower = model_output.lower()
    
    # Check required terms
    score = 1.0
    for term in required_terms:
        if term.lower() not in model_lower:
            score = 0.0
            break
    
    # Check forbidden terms
    for term in forbidden_terms:
        if term.lower() in model_lower:
            score = 0.0
            break
    
    return score


def evaluate_llm_judge(model_output, test_case, judge_runner, config):
    """Use an LLM judge to evaluate the response."""
    judge_model = config.get("judge_model", "mistral:7b")
    judge_sampling = config.get("judge_sampling", {"temperature": 0.0, "max_tokens": 512})
    judge_prompt_template = test_case["eval"]["params"].get("judge_prompt", "")
    
    # judge_runner should be a function that takes (prompt, model, sampling, timeout_s)
    # and returns a dict with "output" key
    
    # Build judge prompt
    judge_prompt = f"""You are evaluating a model's response to a test case.

Test Category: {test_case['category']}
Test Prompt: {test_case['prompt']}
Expected Behavior: {json.dumps(test_case.get('expected', {}), indent=2)}
Rubric: {json.dumps(test_case.get('rubric', {}), indent=2)}

Model's Response:
{model_output}

Evaluation Criteria:
{judge_prompt_template}

Please provide a score from 0.0 to 1.0, where:
- 1.0 = Perfect response
- 0.5 = Partially correct or acceptable
- 0.0 = Incorrect or inappropriate response

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    try:
        result = judge_runner(judge_prompt, judge_model, judge_sampling, timeout_s=30)
        judge_output = result["output"]
        
        # Extract numeric score
        numbers = re.findall(r'0?\.\d+|1\.0|\d+\.\d+', judge_output)
        if numbers:
            score = float(numbers[0])
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            return score
        else:
            # Fallback: check for keywords
            judge_lower = judge_output.lower()
            if "correct" in judge_lower or "perfect" in judge_lower or "1.0" in judge_output:
                return 1.0
            elif "partial" in judge_lower or "0.5" in judge_output:
                return 0.5
            else:
                return 0.0
    except Exception as e:
        # Fallback to heuristic if judge fails
        return 0.5


def extract_python_code(text):
    """Extract Python code blocks from markdown or plain text."""
    # Try to extract from markdown code blocks
    code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Try to extract from indented blocks
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('import ') or line.strip().startswith('class '):
            in_code = True
        if in_code:
            code_lines.append(line)
        if in_code and line.strip() and not (line.startswith(' ') or line.startswith('\t') or line.strip().startswith('#')):
            if 'def ' not in line and 'import ' not in line:
                break
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Return full text if no clear code block found
    return text.strip()


def evaluate_python_exec(model_output, test_case, params):
    """Execute Python code and run tests."""
    timeout_seconds = params.get("timeout_seconds", 10)
    expected = test_case.get("expected", {})
    entrypoint = expected.get("entrypoint")
    tests = expected.get("tests", [])
    
    if not entrypoint or not tests:
        return 0.0
    
    # Extract code
    code = extract_python_code(model_output)
    if not code:
        return 0.0
    
    # Build test script
    test_script = f"""{code}

# Test cases
test_results = []
"""
    
    for i, test in enumerate(tests):
        test_input = test.get("input")
        test_output = test.get("output")
        
        # Build call expression for entrypoint
        # Supports:
        # - Single positional arg: input -> fn(input)
        # - Dict with "args"/"kwargs": {"args": [...], "kwargs": {...}} -> fn(*args, **kwargs)
        if isinstance(test_input, dict) and ("args" in test_input or "kwargs" in test_input):
            args_list = test_input.get("args", [])
            kwargs_dict = test_input.get("kwargs", {})
            args_src = ", ".join(repr(a) for a in args_list)
            kwargs_src = ", ".join(f"{k}={repr(v)}" for k, v in kwargs_dict.items())
            call_args = ", ".join([s for s in (args_src, kwargs_src) if s])
            call_expr = f"{entrypoint}({call_args})"
        else:
            # Treat the entire value as a single positional argument
            if isinstance(test_input, str):
                arg_src = repr(test_input)
            else:
                arg_src = repr(test_input)
            call_expr = f"{entrypoint}({arg_src})"
        
        # Handle different output types
        if isinstance(test_output, bool):
            expected_str = "True" if test_output else "False"
        elif isinstance(test_output, str):
            expected_str = f'"{test_output}"'
        elif test_output is None:
            expected_str = "None"
        else:
            expected_str = str(test_output)
        
        test_script += f"""
try:
    result_{i} = {call_expr}
    if result_{i} == {expected_str}:
        test_results.append(True)
    else:
        test_results.append(False)
        print(f"Test {i} failed: expected {expected_str}, got {{result_{i}}}")
except Exception as e:
    test_results.append(False)
    print(f"Test {i} error: {{e}}")
"""
    
    test_script += """
# Output results
passed = sum(test_results)
total = len(test_results)
print(f"PASSED: {passed}/{total}")
"""
    
    # Execute in subprocess
    try:
        result = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tempfile.gettempdir()
        )
        
        if result.returncode == 0:
            # Parse output to get pass count
            output = result.stdout
            match = re.search(r'PASSED: (\d+)/(\d+)', output)
            if match:
                passed = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    return passed / total
            # If all tests passed (no error output), assume success
            if "error" not in output.lower() and "failed" not in output.lower():
                return 1.0
            return 0.0
        else:
            return 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:
        return 0.0


def evaluate(test_case: Dict[str, Any], model_output: str, judge_runner, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main evaluation function that routes to appropriate evaluation method.
    
    Returns:
        Dict with keys: score_detail, final_score, failure_tags
    """
    eval_method = test_case["eval"]["method"]
    eval_params = test_case["eval"].get("params", {})
    expected = test_case.get("expected")
    
    score_detail = {}
    final_score = 0.0
    failure_tags = []
    
    try:
        if eval_method == "exact_match":
            final_score = evaluate_exact_match(model_output, expected, eval_params)
            if final_score < 1.0:
                failure_tags.append("exact_match_failed")
        
        elif eval_method == "numeric_tolerance":
            final_score = evaluate_numeric_tolerance(model_output, expected, eval_params)
            if final_score < 1.0:
                failure_tags.append("numeric_mismatch")
        
        elif eval_method == "contains":
            final_score = evaluate_contains(model_output, expected, eval_params)
            if final_score < 1.0:
                failure_tags.append("missing_required_terms")
        
        elif eval_method == "llm_judge":
            final_score = evaluate_llm_judge(model_output, test_case, judge_runner, config)
            score_detail["judge_output"] = "evaluated"
            if final_score < 0.5:
                failure_tags.append("low_judge_score")
            if final_score < 1.0:
                failure_tags.append("imperfect_response")
        
        elif eval_method == "python_exec":
            final_score = evaluate_python_exec(model_output, test_case, eval_params)
            if final_score < 1.0:
                if final_score == 0.0:
                    failure_tags.append("code_execution_failed")
                else:
                    failure_tags.append("partial_test_pass")
            if final_score == 0.0:
                failure_tags.append("syntax_error")
                failure_tags.append("runtime_error")
        
        else:
            failure_tags.append("unknown_eval_method")
            final_score = 0.0
        
        score_detail["method"] = eval_method
        score_detail["raw_score"] = final_score
        
    except Exception as e:
        failure_tags.append("evaluation_error")
        failure_tags.append(f"error_{type(e).__name__}")
        final_score = 0.0
        score_detail["error"] = str(e)
    
    return {
        "score_detail": score_detail,
        "final_score": final_score,
        "failure_tags": failure_tags
    }
