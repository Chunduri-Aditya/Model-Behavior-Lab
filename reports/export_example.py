#!/usr/bin/env python3
"""
Export a small example run snapshot for documentation.

Usage:
  python reports/export_example.py --run_id <run_id> --test_id <test_id> --out_dir examples/sample_run
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def load_results_jsonl(path):
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def export_example(run_id, test_id, out_dir, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join("data", "runs", run_id)

    results_dir_path = Path(results_dir)
    if not results_dir_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Copy/trim summary.json
    summary_src = results_dir_path / "summary.json"
    if summary_src.exists():
        with open(summary_src, "r") as f:
            summary = json.load(f)
        summary_dest = out_path / "summary.json"
        with open(summary_dest, "w") as f:
            json.dump(summary, f, indent=2)

    # Load and filter results.jsonl by test_id
    results_jsonl = results_dir_path / "results.jsonl"
    if not results_jsonl.exists():
        raise FileNotFoundError(f"results.jsonl not found in {results_dir}")

    all_results = load_results_jsonl(results_jsonl)
    filtered = [r for r in all_results if r.get("test_id") == test_id]

    if not filtered:
        raise ValueError(f"No results found for test_id={test_id} in run_id={run_id}")

    # Build compact CSV with truncated outputs
    rows = []
    for r in filtered:
        raw_output = r.get("raw_output", "") or ""
        rows.append(
            {
                "run_id": r.get("run_id", ""),
                "model": r.get("model", ""),
                "test_id": r.get("test_id", ""),
                "category": r.get("category", ""),
                "variant_group": r.get("variant_group", ""),
                "repeat_idx": r.get("repeat_idx", 0),
                "final_score": r.get("final_score", 0.0),
                "latency_ms": r.get("latency_ms", 0),
                "failure_tags": ",".join(r.get("failure_tags", [])),
                "output_preview": raw_output[:200],
            }
        )

    df = pd.DataFrame(rows)
    example_csv = out_path / "example_results.csv"
    df.to_csv(example_csv, index=False)

    # Copy recommendations.json if present
    rec_src = results_dir_path / "recommendations.json"
    if rec_src.exists():
        rec_dest = out_path / "recommendations.json"
        with open(rec_src, "r") as f:
            rec = json.load(f)
        with open(rec_dest, "w") as f:
            json.dump(rec, f, indent=2)

    print(f"✅ Exported example snapshot for run_id={run_id}, test_id={test_id} to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export small example snapshot from a run")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID to export from")
    parser.add_argument("--test_id", type=str, required=True, help="Test ID to keep (e.g., reasoning-001)")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for example snapshot (e.g., examples/sample_run)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override results directory (default: data/runs/<run_id>)",
    )

    args = parser.parse_args()
    export_example(args.run_id, args.test_id, args.out_dir, args.results_dir)


if __name__ == "__main__":
    main()

