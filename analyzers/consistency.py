"""
Consistency analysis: variance, standard deviation, and consistency scoring.
"""

import json
from typing import Dict, List, Any
from collections import defaultdict


def calculate_std(values):
    """Calculate standard deviation."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def compute_consistency_score(std_dev):
    """Compute normalized consistency score from standard deviation.
    
    Consistency score = 1 / (1 + std_dev)
    Higher score = more consistent (lower variance)
    """
    if std_dev == 0:
        return 1.0
    return 1.0 / (1.0 + std_dev)


def analyze_consistency(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze consistency across repeats and variant groups.
    
    Returns:
        Dict with consistency metrics per model, category, and variant group
    """
    consistency_metrics = {
        "per_model_category": {},
        "per_variant_group": {},
        "overall_consistency": {}
    }
    
    # Group results by model + category + test_id
    model_category_test = defaultdict(list)
    for result in results:
        key = (result["model"], result["category"], result["test_id"])
        model_category_test[key].append(result["final_score"])
    
    # Compute consistency per model + category
    for (model, category, test_id), scores in model_category_test.items():
        if len(scores) < 2:
            continue
        
        mean_score = sum(scores) / len(scores)
        std_score = calculate_std(scores)
        consistency = compute_consistency_score(std_score)
        
        model_cat_key = f"{model}:{category}"
        if model_cat_key not in consistency_metrics["per_model_category"]:
            consistency_metrics["per_model_category"][model_cat_key] = {
                "mean_std": [],
                "mean_consistency": [],
                "test_count": 0
            }
        
        consistency_metrics["per_model_category"][model_cat_key]["mean_std"].append(std_score)
        consistency_metrics["per_model_category"][model_cat_key]["mean_consistency"].append(consistency)
        consistency_metrics["per_model_category"][model_cat_key]["test_count"] += 1
    
    # Aggregate per model + category
    for key, data in consistency_metrics["per_model_category"].items():
        if data["mean_std"]:
            data["avg_std"] = sum(data["mean_std"]) / len(data["mean_std"])
            data["avg_consistency"] = sum(data["mean_consistency"]) / len(data["mean_consistency"])
        else:
            data["avg_std"] = 0.0
            data["avg_consistency"] = 1.0
    
    # Analyze variant groups (for emotion tests)
    # Keyed by (model, variant_group) -> list of scores
    variant_groups = defaultdict(list)
    for result in results:
        variant_group = result.get("variant_group")
        if variant_group:
            key = (result["model"], variant_group)
            variant_groups[key].append(result["final_score"])
    
    for (model, variant_group), scores in variant_groups.items():
        if len(scores) < 2:
            continue
        
        mean_score = sum(scores) / len(scores)
        std_score = calculate_std(scores)
        consistency = compute_consistency_score(std_score)
        
        group_key = f"{model}:{variant_group}"
        consistency_metrics["per_variant_group"][group_key] = {
            "mean_score": mean_score,
            "std_score": std_score,
            "consistency_score": consistency,
            "sample_count": len(scores)
        }
    
    # Overall consistency per model
    model_scores = defaultdict(list)
    for result in results:
        model_scores[result["model"]].append(result["final_score"])
    
    for model, scores in model_scores.items():
        if len(scores) < 2:
            continue
        std_score = calculate_std(scores)
        consistency = compute_consistency_score(std_score)
        consistency_metrics["overall_consistency"][model] = {
            "std_score": std_score,
            "consistency_score": consistency,
            "sample_count": len(scores)
        }
    
    return consistency_metrics


def add_consistency_to_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add consistency metrics to existing summary."""
    consistency_metrics = analyze_consistency(results)
    summary["consistency"] = consistency_metrics
    return summary
