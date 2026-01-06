"""
Tests for evaluator module.
"""

import pytest
from analyzers.evaluator import (
    evaluate_exact_match,
    evaluate_numeric_tolerance,
    evaluate_contains,
    extract_python_code,
    normalize_text
)


def test_normalize_text():
    """Test text normalization."""
    assert normalize_text("Hello World!") == "hello world"
    assert normalize_text("  Test  \n  String  ") == "test string"
    assert normalize_text("A, B, C.") == "a b c"


def test_evaluate_exact_match():
    """Test exact match evaluation."""
    params = {"normalize": True, "case_sensitive": False}
    
    # Exact match
    assert evaluate_exact_match("yes", "yes", params) == 1.0
    assert evaluate_exact_match("YES", "yes", params) == 1.0
    assert evaluate_exact_match("  Yes  ", "yes", params) == 1.0
    
    # Mismatch
    assert evaluate_exact_match("no", "yes", params) == 0.0
    
    # Without normalization
    params_no_norm = {"normalize": False, "case_sensitive": False}
    assert evaluate_exact_match("Yes", "yes", params_no_norm) == 1.0
    assert evaluate_exact_match("Yes", "Yes", params_no_norm) == 1.0


def test_evaluate_numeric_tolerance():
    """Test numeric tolerance evaluation."""
    params = {"tolerance": 0.01}
    
    # Exact match
    assert evaluate_numeric_tolerance("42", 42, params) == 1.0
    assert evaluate_numeric_tolerance("42.0", 42, params) == 1.0
    
    # Within tolerance
    assert evaluate_numeric_tolerance("42.005", 42, params) == 1.0
    
    # Outside tolerance
    result = evaluate_numeric_tolerance("43", 42, params)
    assert result < 1.0
    assert result > 0.0  # Partial credit
    
    # No number in output
    assert evaluate_numeric_tolerance("no number here", 42, params) == 0.0


def test_evaluate_contains():
    """Test contains evaluation."""
    params = {
        "required_terms": ["hello", "world"],
        "forbidden_terms": ["error"]
    }
    
    # All required, no forbidden
    assert evaluate_contains("hello world", None, params) == 1.0
    
    # Missing required
    assert evaluate_contains("hello", None, params) == 0.0
    
    # Has forbidden
    assert evaluate_contains("hello world error", None, params) == 0.0
    
    # Case insensitive
    assert evaluate_contains("HELLO WORLD", None, params) == 1.0


def test_extract_python_code():
    """Test Python code extraction."""
    # Markdown code block
    markdown = "```python\ndef test():\n    return 42\n```"
    code = extract_python_code(markdown)
    assert "def test" in code
    assert "return 42" in code
    
    # Plain code
    plain = "def test():\n    return 42"
    code = extract_python_code(plain)
    assert "def test" in code
    
    # No code block
    text = "This is just text"
    code = extract_python_code(text)
    assert code == text.strip()


def test_evaluate_python_exec_simple():
    """Test Python execution evaluation with a simple case."""
    from analyzers.evaluator import evaluate_python_exec
    
    test_case = {
        "expected": {
            "entrypoint": "add",
            "tests": [
                {"input": {"args": [1, 2]}, "output": 3},
                {"input": {"args": [5, 7]}, "output": 12}
            ]
        }
    }
    
    # Correct code
    correct_code = "def add(a, b):\n    return a + b"
    score = evaluate_python_exec(correct_code, test_case, {"timeout_seconds": 5})
    assert score == 1.0
    
    # Incorrect code
    incorrect_code = "def add(a, b):\n    return a - b"
    score = evaluate_python_exec(incorrect_code, test_case, {"timeout_seconds": 5})
    assert score == 0.0


def test_evaluate_python_exec_with_markdown():
    """Test Python execution with markdown-wrapped code."""
    from analyzers.evaluator import evaluate_python_exec
    
    test_case = {
        "expected": {
            "entrypoint": "multiply",
            "tests": [
                {"input": {"args": [2, 3]}, "output": 6}
            ]
        }
    }
    
    markdown_code = "```python\ndef multiply(a, b):\n    return a * b\n```"
    score = evaluate_python_exec(markdown_code, test_case, {"timeout_seconds": 5})
    assert score == 1.0
