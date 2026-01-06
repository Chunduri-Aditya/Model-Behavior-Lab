#!/usr/bin/env python3
"""
Analyze experiment results and export to CSV.
"""

import json
import argparse
import os
import pandas as pd
from pathlib import Path


def load_results_jsonl(jsonl_path):
    """Load results from JSONL file."""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_results_json(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def export_to_csv(results, output_path):
    """Export results to CSV with comprehensive columns."""
    rows = []
    for result in results:
        raw_output = result.get("raw_output", "") or ""
        row = {
            "run_id": result.get("run_id", ""),
            "timestamp": result.get("timestamp", ""),
            "git_commit": result.get("git_commit", ""),
            "model": result.get("model", ""),
            "test_id": result.get("test_id", ""),
            "category": result.get("category", ""),
            "variant_group": result.get("variant_group", ""),
            "repeat_idx": result.get("repeat_idx", 0),
            "prompt": result.get("prompt", ""),
            "output_preview": raw_output[:200],
            "final_score": result.get("final_score", 0.0),
            "latency_ms": result.get("latency_ms", 0),
            "error": result.get("error", ""),
            "failure_tags": ",".join(result.get("failure_tags", [])),
        }
        
        # Flatten score_detail if present
        score_detail = result.get("score_detail", {})
        if score_detail:
            row["eval_method"] = score_detail.get("method", "")
            row["raw_eval_score"] = score_detail.get("raw_score", "")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def generate_aggregated_tables(results, output_dir):
    """Generate aggregated summary tables."""
    df = pd.DataFrame(results)
    
    # Per-model, per-category aggregation
    if not df.empty:
        # Model + Category aggregation
        model_cat_agg = df.groupby(["model", "category"]).agg({
            "final_score": ["mean", "std", "count"],
            "latency_ms": "mean"
        }).reset_index()
        model_cat_agg.columns = ["model", "category", "mean_score", "std_score", "count", "mean_latency_ms"]
        model_cat_agg.to_csv(os.path.join(output_dir, "aggregated_model_category.csv"), index=False)
        
        # Per-model aggregation
        model_agg = df.groupby("model").agg({
            "final_score": ["mean", "std", "count"],
            "latency_ms": "mean"
        }).reset_index()
        model_agg.columns = ["model", "mean_score", "std_score", "count", "mean_latency_ms"]
        model_agg.to_csv(os.path.join(output_dir, "aggregated_model.csv"), index=False)
        
        # Per-category aggregation
        cat_agg = df.groupby("category").agg({
            "final_score": ["mean", "std", "count"]
        }).reset_index()
        cat_agg.columns = ["category", "mean_score", "std_score", "count"]
        cat_agg.to_csv(os.path.join(output_dir, "aggregated_category.csv"), index=False)
        
        print(f"✅ Generated aggregated tables:")
        print(f"   - aggregated_model_category.csv")
        print(f"   - aggregated_model.csv")
        print(f"   - aggregated_category.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and export to CSV"
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
    
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join("data", "runs", args.run_id)
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Try to load JSONL first, fall back to JSON
    jsonl_path = os.path.join(results_dir, "results.jsonl")
    json_path = os.path.join(results_dir, "results.json")
    
    if os.path.exists(jsonl_path):
        print(f"📊 Loading results from: {jsonl_path}")
        results = load_results_jsonl(jsonl_path)
    elif os.path.exists(json_path):
        print(f"📊 Loading results from: {json_path}")
        results = load_results_json(json_path)
    else:
        print(f"❌ No results file found in {results_dir}")
        return
    
    print(f"   Loaded {len(results)} result records")
    
    # Export to CSV
    csv_path = os.path.join(results_dir, "results.csv")
    print(f"\n📝 Exporting to CSV: {csv_path}")
    df = export_to_csv(results, csv_path)
    print(f"   Exported {len(df)} rows to CSV")
    
    # Generate aggregated tables
    print(f"\n📊 Generating aggregated tables...")
    generate_aggregated_tables(results, results_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"   CSV: {csv_path}")


if __name__ == "__main__":
    main()
