"""
Tests for consistency analysis.
"""

import pytest
from analyzers.consistency import (
    calculate_std,
    compute_consistency_score,
    analyze_consistency
)


def test_calculate_std():
    """Test standard deviation calculation."""
    # Perfect consistency
    assert calculate_std([1.0, 1.0, 1.0]) == 0.0
    
    # Some variance
    std = calculate_std([0.5, 1.0, 1.5])
    assert std > 0.0
    assert std < 1.0
    
    # Single value
    assert calculate_std([1.0]) == 0.0
    
    # Empty list
    assert calculate_std([]) == 0.0


def test_compute_consistency_score():
    """Test consistency score computation."""
    # Perfect consistency (std = 0)
    assert compute_consistency_score(0.0) == 1.0
    
    # Some variance
    score = compute_consistency_score(0.5)
    assert 0.0 < score < 1.0
    
    # High variance
    score = compute_consistency_score(2.0)
    assert score < 0.5


def test_analyze_consistency():
    """Test consistency analysis on sample results."""
    results = [
        {
            "model": "model1",
            "category": "reasoning",
            "test_id": "test1",
            "final_score": 1.0,
            "variant_group": None
        },
        {
            "model": "model1",
            "category": "reasoning",
            "test_id": "test1",
            "final_score": 1.0,
            "variant_group": None
        },
        {
            "model": "model1",
            "category": "reasoning",
            "test_id": "test1",
            "final_score": 0.8,
            "variant_group": None
        },
        {
            "model": "model1",
            "category": "emotion",
            "test_id": "emotion-001",
            "final_score": 0.9,
            "variant_group": "emotion-set-01"
        },
        {
            "model": "model1",
            "category": "emotion",
            "test_id": "emotion-002",
            "final_score": 0.85,
            "variant_group": "emotion-set-01"
        }
    ]
    
    consistency = analyze_consistency(results)
    
    # Should have per_model_category
    assert "per_model_category" in consistency
    
    # Should have per_variant_group for emotion tests
    assert "per_variant_group" in consistency
    
    # Should have overall_consistency
    assert "overall_consistency" in consistency
    assert "model1" in consistency["overall_consistency"]
