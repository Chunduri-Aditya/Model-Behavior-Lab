#!/usr/bin/env python3
"""
Model selection and recommendation system based on evaluation results.
"""

import json
import argparse
import os
from typing import Dict, List, Any


def load_summary(run_id, results_dir=None):
    """Load summary.json for a given run."""
    if results_dir is None:
        results_dir = os.path.join("data", "runs", run_id)
    
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def rank_models(summary, weights=None):
    """
    Rank models by weighted average score.
    
    Args:
        summary: Summary dict from experiment
        weights: Dict mapping category to weight (default: equal weights)
    
    Returns:
        List of (model, weighted_score) tuples, sorted descending
    """
    if weights is None:
        weights = {
            "reasoning": 1.0,
            "hallucination": 1.0,
            "emotion": 1.0,
            "code": 1.0
        }
    
    model_scores = {}
    
    if "models" not in summary:
        return []
    
    for model, data in summary["models"].items():
        weighted_score = 0.0
        total_weight = 0.0
        
        if "categories" in data:
            for category, cat_data in data["categories"].items():
                weight = weights.get(category, 1.0)
                mean_score = cat_data.get("mean_score", 0.0)
                weighted_score += mean_score * weight
                total_weight += weight
        
        if total_weight > 0:
            model_scores[model] = weighted_score / total_weight
        else:
            # Fallback to overall mean
            model_scores[model] = data.get("mean_score", 0.0)
    
    # Sort by score descending
    ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def recommend_best_model_per_category(summary):
    """Recommend the best model for each category."""
    recommendations = {}
    
    if "models" not in summary:
        return recommendations
    
    categories = ["reasoning", "hallucination", "emotion", "code"]
    
    for category in categories:
        best_model = None
        best_score = -1.0
        
        for model, data in summary["models"].items():
            if "categories" in data and category in data["categories"]:
                score = data["categories"][category].get("mean_score", 0.0)
                if score > best_score:
                    best_score = score
                    best_model = model
        
        if best_model:
            recommendations[category] = {
                "model": best_model,
                "score": best_score
            }
    
    return recommendations


def generate_recommendations(summary, weights=None):
    """Generate comprehensive model recommendations."""
    recommendations = {
        "overall_ranking": [],
        "best_per_category": {},
        "deployment_recommendations": [],
        "use_cases": {}
    }
    
    # Overall ranking
    ranked = rank_models(summary, weights)
    recommendations["overall_ranking"] = [
        {"model": model, "weighted_score": score}
        for model, score in ranked
    ]
    
    # Best per category
    recommendations["best_per_category"] = recommend_best_model_per_category(summary)
    
    # Deployment-oriented recommendations
    if "tradeoffs" in summary and "strengths_weaknesses" in summary["tradeoffs"]:
        strengths_weaknesses = summary["tradeoffs"]["strengths_weaknesses"]
        
        for model, data in strengths_weaknesses.items():
            strengths = data.get("strengths", [])
            weaknesses = data.get("weaknesses", [])
            scores = data.get("scores", {})
            
            # Generate use case recommendations
            use_cases = []
            
            if "code" in strengths and scores.get("code", 0) > 0.7:
                use_cases.append({
                    "use_case": "Code generation and programming assistants",
                    "rationale": f"Strong code correctness score ({scores.get('code', 0):.2f})"
                })
            
            if "emotion" in strengths and scores.get("emotion", 0) > 0.7:
                use_cases.append({
                    "use_case": "Empathetic chatbots and emotional support",
                    "rationale": f"High emotional alignment score ({scores.get('emotion', 0):.2f})"
                })
            
            if "reasoning" in strengths and scores.get("reasoning", 0) > 0.7:
                use_cases.append({
                    "use_case": "Analytical tasks and problem-solving",
                    "rationale": f"Strong reasoning accuracy ({scores.get('reasoning', 0):.2f})"
                })
            
            if "hallucination" in strengths and scores.get("hallucination", 0) > 0.7:
                use_cases.append({
                    "use_case": "Factual information retrieval (low hallucination risk)",
                    "rationale": f"Low hallucination propensity ({scores.get('hallucination', 0):.2f})"
                })
            
            if use_cases:
                recommendations["use_cases"][model] = use_cases
            
            # Generate deployment recommendation text
            if strengths:
                rec_text = f"**{model}**: Best for "
                rec_text += ", ".join(strengths)
                if weaknesses:
                    rec_text += f". Avoid for: {', '.join(weaknesses)}"
                recommendations["deployment_recommendations"].append(rec_text)
    
    # Add consistency-based recommendations
    if "consistency" in summary and "overall_consistency" in summary["consistency"]:
        consistency_scores = {}
        for model, data in summary["consistency"]["overall_consistency"].items():
            consistency_scores[model] = data.get("consistency_score", 0.0)
        
        if consistency_scores:
            most_consistent = max(consistency_scores.items(), key=lambda x: x[1])
            recommendations["deployment_recommendations"].append(
                f"**{most_consistent[0]}**: Most consistent across runs (consistency score: {most_consistent[1]:.3f})"
            )
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Generate model selection recommendations"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID to analyze"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Results directory (default: data/runs/<run_id>)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="JSON string with category weights, e.g., '{\"reasoning\": 2.0, \"hallucination\": 1.5}'"
    )
    
    args = parser.parse_args()
    
    # Load summary
    summary = load_summary(args.run_id, args.results_dir)
    
    # Parse weights if provided
    weights = None
    if args.weights:
        weights = json.loads(args.weights)
    
    # Generate recommendations
    recommendations = generate_recommendations(summary, weights)
    
    # Save recommendations
    results_dir = args.results_dir or os.path.join("data", "runs", args.run_id)
    os.makedirs(results_dir, exist_ok=True)
    recommendations_path = os.path.join(results_dir, "recommendations.json")
    
    with open(recommendations_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Model Selection Recommendations")
    print(f"{'='*60}\n")
    
    print("Overall Ranking (by weighted score):")
    for i, item in enumerate(recommendations["overall_ranking"], 1):
        print(f"  {i}. {item['model']}: {item['weighted_score']:.3f}")
    
    print("\nBest Model per Category:")
    for category, rec in recommendations["best_per_category"].items():
        print(f"  {category}: {rec['model']} (score: {rec['score']:.3f})")
    
    print("\nDeployment Recommendations:")
    for rec in recommendations["deployment_recommendations"]:
        print(f"  • {rec}")
    
    print(f"\n✅ Recommendations saved to: {recommendations_path}")


if __name__ == "__main__":
    main()
