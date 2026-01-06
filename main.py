#!/usr/bin/env python3
"""
Model Behavior Lab - Automated Evaluation Framework
Runs reproducible experiments across multiple LLMs with comprehensive evaluation.
"""

import json
import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import subprocess
import hashlib

from models.ollama_runner import run_model
from analyzers.evaluator import evaluate
from analyzers.consistency import add_consistency_to_summary
from analyzers.failure_modes import add_failure_modes_to_summary
from analyzers.tradeoff_analysis import add_tradeoffs_to_summary
from analyze_results import export_to_csv, generate_aggregated_tables


def get_git_commit():
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def generate_run_id(timestamp=None):
    """Generate a unique run ID based on timestamp."""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


def load_config(config_path):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_test_suite(suite_path):
    """Load test suite from JSON file."""
    with open(suite_path, 'r') as f:
        suite = json.load(f)
    return suite


def get_cache_key(model_name, test_id, repeat_idx, sampling):
    """Generate cache key for output caching."""
    key_str = f"{model_name}:{test_id}:{repeat_idx}:{json.dumps(sampling, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cached_output(cache_dir, cache_key):
    """Load cached output if it exists."""
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


def save_cached_output(cache_dir, cache_key, result):
    """Save output to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)


def run_experiment(config, run_id, out_dir):
    """Run the full experiment."""
    # Create output directories
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load test suite
    suite_path = config["suite_path"]
    test_suite = load_test_suite(suite_path)
    
    # Get metadata
    git_commit = get_git_commit()
    timestamp = datetime.now().isoformat()
    
    # Initialize results
    results_jsonl_path = os.path.join(out_dir, "results.jsonl")
    results = []
    
    # Judge runner function (wrapper for LLM judge calls)
    def judge_runner(prompt, model, sampling, timeout_s):
        return run_model(prompt, model, sampling, timeout_s)
    
    # Experiment loop
    total_tests = len(test_suite) * len(config["models"]) * config["repeats"]
    current_test = 0
    
    print(f"\n{'='*60}")
    print(f"Starting Experiment Run: {run_id}")
    print(f"{'='*60}")
    print(f"Models: {', '.join(config['models'])}")
    print(f"Test Suite: {suite_path} ({len(test_suite)} tests)")
    print(f"Repeats per test: {config['repeats']}")
    print(f"Total runs: {total_tests}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}\n")
    
    for test_case in test_suite:
        test_id = test_case["id"]
        category = test_case["category"]
        variant_group = test_case.get("meta", {}).get("prompt_variant_group")
        
        print(f"\n🔍 Test: {test_id} ({category})")
        
        for model_name in config["models"]:
            print(f"  → Model: {model_name}")
            
            for repeat_idx in range(config["repeats"]):
                current_test += 1
                cache_key = get_cache_key(model_name, test_id, repeat_idx, config["sampling"])
                
                # Check cache
                cached_result = None
                if config.get("cache_outputs", False):
                    cached_result = load_cached_output(cache_dir, cache_key)
                
                if cached_result:
                    print(f"    [Cached] Repeat {repeat_idx + 1}/{config['repeats']}")
                    result = cached_result
                else:
                    print(f"    [Running] Repeat {repeat_idx + 1}/{config['repeats']} ({current_test}/{total_tests})")
                    
                    # Run model
                    model_result = run_model(
                        test_case["prompt"],
                        model_name,
                        config["sampling"],
                        config.get("timeout_s", 90)
                    )
                    
                    # Evaluate
                    eval_result = evaluate(
                        test_case,
                        model_result["output"],
                        judge_runner,
                        {
                            "judge_model": config.get("judge_model", "mistral:7b"),
                            "judge_sampling": config.get("judge_sampling", {"temperature": 0.0})
                        }
                    )
                    
                    # Build result record
                    result = {
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "git_commit": git_commit,
                        "model": model_name,
                        "test_id": test_id,
                        "category": category,
                        "variant_group": variant_group,
                        "repeat_idx": repeat_idx,
                        "prompt": test_case["prompt"],
                        "raw_output": model_result["output"],
                        "score_detail": eval_result["score_detail"],
                        "final_score": eval_result["final_score"],
                        "failure_tags": eval_result["failure_tags"],
                        "latency_ms": model_result["latency_ms"],
                        "error": model_result.get("error")
                    }
                    
                    # Cache result
                    if config.get("cache_outputs", False):
                        save_cached_output(cache_dir, cache_key, result)
                
                # Append to results
                results.append(result)
                
                # Write incrementally to JSONL
                with open(results_jsonl_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                
                # Small delay to prevent rate limiting
                time.sleep(0.5)
    
    # Generate summary
    print(f"\n{'='*60}")
    print("Generating summary...")
    print(f"{'='*60}\n")
    
    summary = generate_summary(results, config)
    
    # Add advanced analyses
    print("  → Computing consistency metrics...")
    summary = add_consistency_to_summary(summary, results)
    
    print("  → Analyzing failure modes...")
    summary = add_failure_modes_to_summary(summary, results)
    
    print("  → Computing tradeoffs...")
    summary = add_tradeoffs_to_summary(summary, results)
    
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Also save full results as JSON for backward compatibility
    results_json_path = os.path.join(out_dir, "results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate CSV and aggregated tables for this run
    csv_path = os.path.join(out_dir, "results.csv")
    export_to_csv(results, csv_path)
    generate_aggregated_tables(results, out_dir)

    print(f"✅ Experiment complete!")
    print(f"   Results: {results_jsonl_path}")
    print(f"   Summary: {summary_path}")
    print(f"   CSV:     {csv_path}")
    print(f"   Total runs: {len(results)}")

    return results, summary


def generate_summary(results, config):
    """Generate summary statistics from results."""
    summary = {
        "run_id": results[0]["run_id"] if results else None,
        "timestamp": results[0]["timestamp"] if results else None,
        "git_commit": results[0]["git_commit"] if results else None,
        "config": config,
        "total_runs": len(results),
        "models": {},
        "categories": {},
        "overall": {}
    }
    
    # Per-model statistics
    for model in config["models"]:
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            continue
        
        scores = [r["final_score"] for r in model_results if r.get("final_score") is not None]
        latencies = [r["latency_ms"] for r in model_results if r.get("latency_ms") is not None]
        
        summary["models"][model] = {
            "total_runs": len(model_results),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "std_score": calculate_std(scores) if scores else 0.0,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "categories": {}
        }
        
        # Per-category for this model
        for category in ["reasoning", "hallucination", "emotion", "code"]:
            cat_results = [r for r in model_results if r["category"] == category]
            if cat_results:
                cat_scores = [r["final_score"] for r in cat_results if r.get("final_score") is not None]
                summary["models"][model]["categories"][category] = {
                    "mean_score": sum(cat_scores) / len(cat_scores) if cat_scores else 0.0,
                    "std_score": calculate_std(cat_scores) if cat_scores else 0.0,
                    "total_runs": len(cat_results)
                }
    
    # Per-category statistics (across all models)
    for category in ["reasoning", "hallucination", "emotion", "code"]:
        cat_results = [r for r in results if r["category"] == category]
        if cat_results:
            cat_scores = [r["final_score"] for r in cat_results if r.get("final_score") is not None]
            summary["categories"][category] = {
                "mean_score": sum(cat_scores) / len(cat_scores) if cat_scores else 0.0,
                "std_score": calculate_std(cat_scores) if cat_scores else 0.0,
                "total_runs": len(cat_results)
            }
    
    # Overall statistics
    all_scores = [r["final_score"] for r in results if r.get("final_score") is not None]
    if all_scores:
        summary["overall"] = {
            "mean_score": sum(all_scores) / len(all_scores),
            "std_score": calculate_std(all_scores),
            "total_runs": len(results)
        }
    
    return summary


def calculate_std(values):
    """Calculate standard deviation."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Run automated LLM evaluation experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run_config.json",
        help="Path to experiment configuration JSON file"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID (default: timestamp-based)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: data/runs/<run_id>)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to override config.models"
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Override suite path (e.g., prompts/suites/core_suite.json)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)

    # Apply optional overrides
    if args.models:
        override_models = [m.strip() for m in args.models.split(",") if m.strip()]
        if override_models:
            config["models"] = override_models
    if args.suite:
        config["suite_path"] = args.suite
    
    # Generate run ID
    run_id = args.run_id or generate_run_id()
    
    # Set output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join("data", "runs", run_id)
    
    # Run experiment
    results, summary = run_experiment(config, run_id, out_dir)
    
    print(f"\n✅ All done! Run ID: {run_id}")


if __name__ == "__main__":
    main()
