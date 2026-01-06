"""
Tests for configuration and test suite loading.
"""

import json
import pytest
import os
from pathlib import Path


def test_run_config_schema():
    """Test that run_config.json has required fields."""
    config_path = Path("configs/run_config.json")
    assert config_path.exists(), "run_config.json should exist"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Required fields
    assert "models" in config
    assert "suite_path" in config
    assert "sampling" in config
    assert "repeats" in config
    
    # Sampling should have temperature and seed for reproducibility
    assert "temperature" in config["sampling"]
    assert "seed" in config["sampling"]


def test_core_suite_schema():
    """Test that core_suite.json follows the schema."""
    suite_path = Path("prompts/suites/core_suite.json")
    assert suite_path.exists(), "core_suite.json should exist"
    
    with open(suite_path) as f:
        suite = json.load(f)
    
    assert isinstance(suite, list), "Suite should be a list"
    assert len(suite) >= 40, "Suite should have at least 40 tests"
    
    # Check first test has required fields
    test = suite[0]
    required_fields = ["id", "category", "prompt", "expected", "eval", "meta"]
    for field in required_fields:
        assert field in test, f"Test should have '{field}' field"
    
    # Check categories
    categories = set(t["category"] for t in suite)
    expected_categories = {"reasoning", "hallucination", "emotion", "code"}
    assert categories == expected_categories, f"Categories should be {expected_categories}"
    
    # Count tests per category
    category_counts = {}
    for test in suite:
        cat = test["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat in expected_categories:
        assert category_counts.get(cat, 0) >= 10, f"Should have at least 10 {cat} tests"


def test_eval_methods():
    """Test that eval methods are valid."""
    suite_path = Path("prompts/suites/core_suite.json")
    with open(suite_path) as f:
        suite = json.load(f)
    
    valid_methods = {"exact_match", "numeric_tolerance", "contains", "llm_judge", "python_exec"}
    
    for test in suite:
        method = test["eval"]["method"]
        assert method in valid_methods, f"Invalid eval method: {method}"


def test_emotion_variant_groups():
    """Test that emotion tests have variant groups for consistency analysis."""
    suite_path = Path("prompts/suites/core_suite.json")
    with open(suite_path) as f:
        suite = json.load(f)
    
    emotion_tests = [t for t in suite if t["category"] == "emotion"]
    assert len(emotion_tests) >= 10, "Should have at least 10 emotion tests"
    
    # Check that emotion tests have variant groups
    variant_groups = set()
    for test in emotion_tests:
        variant_group = test.get("meta", {}).get("prompt_variant_group")
        if variant_group:
            variant_groups.add(variant_group)
    
    assert len(variant_groups) > 0, "Emotion tests should have variant groups for consistency analysis"
