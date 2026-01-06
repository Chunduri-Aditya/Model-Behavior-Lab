"""
Failure mode detection and categorization.
"""

from typing import Dict, List, Any
from collections import defaultdict, Counter


def analyze_failure_modes(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze failure modes across models and categories.
    
    Returns:
        Dict with failure mode statistics
    """
    failure_analysis = {
        "per_model": {},
        "per_category": {},
        "systematic_failures": {},
        "sporadic_failures": {},
        "top_failure_tags": {},
        "worst_tests": {}
    }
    
    # Collect all failure tags
    model_failures = defaultdict(lambda: defaultdict(list))
    category_failures = defaultdict(lambda: defaultdict(list))
    test_failures = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        model = result["model"]
        category = result["category"]
        test_id = result["test_id"]
        failure_tags = result.get("failure_tags", [])
        score = result.get("final_score", 0.0)
        
        # Track failures per model
        for tag in failure_tags:
            model_failures[model][tag].append(score)
            category_failures[category][tag].append(score)
            test_failures[test_id][tag].append(score)
    
    # Analyze per model
    for model, tag_scores in model_failures.items():
        failure_analysis["per_model"][model] = {
            "failure_tags": {},
            "total_failures": 0,
            "failure_rate": 0.0
        }
        
        model_results = [r for r in results if r["model"] == model]
        total_runs = len(model_results)
        failed_runs = sum(1 for r in model_results if r.get("final_score", 1.0) < 1.0)
        
        failure_analysis["per_model"][model]["total_failures"] = failed_runs
        failure_analysis["per_model"][model]["failure_rate"] = failed_runs / total_runs if total_runs > 0 else 0.0
        
        # Count failure tags
        tag_counts = Counter()
        for tag, scores in tag_scores.items():
            tag_counts[tag] = len(scores)
            failure_analysis["per_model"][model]["failure_tags"][tag] = {
                "count": len(scores),
                "frequency": len(scores) / total_runs if total_runs > 0 else 0.0,
                "avg_score_when_present": sum(scores) / len(scores) if scores else 0.0
            }
        
        # Top failure tags for this model
        failure_analysis["top_failure_tags"][model] = [
            {"tag": tag, "count": count, "frequency": count / total_runs if total_runs > 0 else 0.0}
            for tag, count in tag_counts.most_common(5)
        ]
    
    # Analyze per category
    for category, tag_scores in category_failures.items():
        category_results = [r for r in results if r["category"] == category]
        total_runs = len(category_results)
        failed_runs = sum(1 for r in category_results if r.get("final_score", 1.0) < 1.0)
        
        failure_analysis["per_category"][category] = {
            "failure_tags": {},
            "total_failures": failed_runs,
            "failure_rate": failed_runs / total_runs if total_runs > 0 else 0.0
        }
        
        tag_counts = Counter()
        for tag, scores in tag_scores.items():
            tag_counts[tag] = len(scores)
            failure_analysis["per_category"][category]["failure_tags"][tag] = {
                "count": len(scores),
                "frequency": len(scores) / total_runs if total_runs > 0 else 0.0
            }
    
    # Identify systematic failures (same failure in >30% of tests in a category)
    SYSTEMATIC_THRESHOLD = 0.3
    for model in set(r["model"] for r in results):
        for category in set(r["category"] for r in results):
            category_model_results = [r for r in results if r["model"] == model and r["category"] == category]
            if not category_model_results:
                continue
            
            total_tests = len(category_model_results)
            tag_frequencies = defaultdict(int)
            
            for result in category_model_results:
                for tag in result.get("failure_tags", []):
                    tag_frequencies[tag] += 1
            
            systematic = {}
            for tag, count in tag_frequencies.items():
                frequency = count / total_tests
                if frequency >= SYSTEMATIC_THRESHOLD:
                    systematic[tag] = {
                        "frequency": frequency,
                        "count": count,
                        "total_tests": total_tests
                    }
            
            if systematic:
                key = f"{model}:{category}"
                failure_analysis["systematic_failures"][key] = systematic
    
    # Identify sporadic failures (low frequency, scattered)
    SPORADIC_THRESHOLD = 0.1
    for model in set(r["model"] for r in results):
        for category in set(r["category"] for r in results):
            category_model_results = [r for r in results if r["model"] == model and r["category"] == category]
            if not category_model_results:
                continue
            
            total_tests = len(category_model_results)
            tag_frequencies = defaultdict(int)
            
            for result in category_model_results:
                for tag in result.get("failure_tags", []):
                    tag_frequencies[tag] += 1
            
            sporadic = {}
            for tag, count in tag_frequencies.items():
                frequency = count / total_tests
                if 0 < frequency < SPORADIC_THRESHOLD:
                    sporadic[tag] = {
                        "frequency": frequency,
                        "count": count
                    }
            
            if sporadic:
                key = f"{model}:{category}"
                failure_analysis["sporadic_failures"][key] = sporadic
    
    # Worst tests per model (lowest average scores)
    for model in set(r["model"] for r in results):
        test_scores = defaultdict(list)
        for result in results:
            if result["model"] == model:
                test_scores[result["test_id"]].append(result["final_score"])
        
        worst_tests = []
        for test_id, scores in test_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            worst_tests.append({
                "test_id": test_id,
                "avg_score": avg_score,
                "runs": len(scores)
            })
        
        worst_tests.sort(key=lambda x: x["avg_score"])
        failure_analysis["worst_tests"][model] = worst_tests[:5]  # Top 5 worst
    
    return failure_analysis


def add_failure_modes_to_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add failure mode analysis to existing summary."""
    failure_analysis = analyze_failure_modes(results)
    summary["failure_modes"] = failure_analysis
    return summary
