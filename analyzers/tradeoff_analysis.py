"""
Behavioral tradeoff analysis across evaluation axes.
"""

from typing import Dict, List, Any
from collections import defaultdict
import math


def calculate_correlation(x_values, y_values):
    """Calculate Pearson correlation coefficient."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    
    numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def analyze_tradeoffs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze tradeoffs between different evaluation axes.
    
    Returns:
        Dict with tradeoff metrics and correlations
    """
    tradeoff_analysis = {
        "per_model_tradeoffs": {},
        "correlations": {},
        "strengths_weaknesses": {}
    }
    
    # Group results by model
    model_results = defaultdict(list)
    for result in results:
        model_results[result["model"]].append(result)
    
    # Analyze tradeoffs per model
    for model, model_data in model_results.items():
        # Calculate mean scores per category
        category_scores = defaultdict(list)
        for result in model_data:
            category_scores[result["category"]].append(result["final_score"])
        
        category_means = {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_scores.items()
        }
        
        tradeoff_analysis["per_model_tradeoffs"][model] = category_means
        
        # Identify strengths and weaknesses
        sorted_categories = sorted(category_means.items(), key=lambda x: x[1], reverse=True)
        strengths = [cat for cat, score in sorted_categories[:2] if score > 0.7]
        weaknesses = [cat for cat, score in sorted_categories[-2:] if score < 0.5]
        
        tradeoff_analysis["strengths_weaknesses"][model] = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "scores": category_means
        }
    
    # Calculate cross-axis correlations (across all models)
    categories = ["reasoning", "hallucination", "emotion", "code"]
    
    # Get per-model averages for each category
    model_category_avgs = defaultdict(lambda: defaultdict(float))
    for model, model_data in model_results.items():
        category_scores = defaultdict(list)
        for result in model_data:
            category_scores[result["category"]].append(result["final_score"])
        
        for cat in categories:
            scores = category_scores.get(cat, [])
            if scores:
                model_category_avgs[cat][model] = sum(scores) / len(scores)
    
    # Calculate correlations between categories
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            # Get scores for both categories across all models
            cat1_scores = []
            cat2_scores = []
            
            for model in set(r["model"] for r in results):
                if model in model_category_avgs[cat1] and model in model_category_avgs[cat2]:
                    cat1_scores.append(model_category_avgs[cat1][model])
                    cat2_scores.append(model_category_avgs[cat2][model])
            
            if len(cat1_scores) >= 2:
                correlation = calculate_correlation(cat1_scores, cat2_scores)
                key = f"{cat1}_vs_{cat2}"
                tradeoff_analysis["correlations"][key] = {
                    "correlation": correlation,
                    "interpretation": interpret_correlation(correlation)
                }
    
    # Specific tradeoff analyses
    tradeoff_analysis["specific_tradeoffs"] = {}
    
    # Reasoning vs Hallucination
    if "reasoning_vs_hallucination" not in tradeoff_analysis["correlations"]:
        reasoning_scores = [model_category_avgs["reasoning"].get(m, 0) for m in set(r["model"] for r in results)]
        hallucination_scores = [model_category_avgs["hallucination"].get(m, 0) for m in set(r["model"] for r in results)]
        if len(reasoning_scores) >= 2:
            corr = calculate_correlation(reasoning_scores, hallucination_scores)
            tradeoff_analysis["specific_tradeoffs"]["reasoning_vs_hallucination"] = {
                "correlation": corr,
                "interpretation": interpret_correlation(corr)
            }
    
    # Code vs Hallucination
    code_scores = [model_category_avgs["code"].get(m, 0) for m in set(r["model"] for r in results)]
    hallucination_scores = [model_category_avgs["hallucination"].get(m, 0) for m in set(r["model"] for r in results)]
    if len(code_scores) >= 2:
        corr = calculate_correlation(code_scores, hallucination_scores)
        tradeoff_analysis["specific_tradeoffs"]["code_vs_hallucination"] = {
            "correlation": corr,
            "interpretation": interpret_correlation(corr)
        }
    
    # Emotion vs Reasoning
    emotion_scores = [model_category_avgs["emotion"].get(m, 0) for m in set(r["model"] for r in results)]
    reasoning_scores = [model_category_avgs["reasoning"].get(m, 0) for m in set(r["model"] for r in results)]
    if len(emotion_scores) >= 2:
        corr = calculate_correlation(emotion_scores, reasoning_scores)
        tradeoff_analysis["specific_tradeoffs"]["emotion_vs_reasoning"] = {
            "correlation": corr,
            "interpretation": interpret_correlation(corr)
        }
    
    return tradeoff_analysis


def interpret_correlation(corr):
    """Interpret correlation coefficient."""
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        return "no relationship"
    elif abs_corr < 0.3:
        return "weak relationship"
    elif abs_corr < 0.5:
        return "moderate relationship"
    elif abs_corr < 0.7:
        return "strong relationship"
    else:
        return "very strong relationship"


def add_tradeoffs_to_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add tradeoff analysis to existing summary."""
    tradeoff_analysis = analyze_tradeoffs(results)
    summary["tradeoffs"] = tradeoff_analysis
    return summary
