#!/usr/bin/env python3
"""
Generate human-readable report from experiment results.
"""

import json
import argparse
import os
from datetime import datetime
from pathlib import Path


def load_data(run_id, results_dir=None):
    """Load summary and recommendations."""
    if results_dir is None:
        results_dir = os.path.join("data", "runs", run_id)
    
    summary_path = os.path.join(results_dir, "summary.json")
    recommendations_path = os.path.join(results_dir, "recommendations.json")
    
    summary = None
    recommendations = None
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    if os.path.exists(recommendations_path):
        with open(recommendations_path, 'r') as f:
            recommendations = json.load(f)
    
    return summary, recommendations


def generate_report(summary, recommendations):
    """Generate markdown report."""
    report = []
    
    # Header
    report.append("# Model Behavior Lab - Evaluation Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if summary:
        report.append(f"**Run ID:** {summary.get('run_id', 'N/A')}\n")
        if summary.get('git_commit'):
            report.append(f"**Git Commit:** {summary['git_commit'][:8]}\n")
        report.append(f"**Total Runs:** {summary.get('total_runs', 0)}\n")
    
    report.append("\n---\n")
    
    # Topline Metrics
    report.append("## Topline Metrics\n")
    
    if summary and "overall" in summary:
        overall = summary["overall"]
        report.append(f"- **Overall Mean Score:** {overall.get('mean_score', 0.0):.3f}")
        report.append(f"- **Overall Std Dev:** {overall.get('std_score', 0.0):.3f}")
        report.append(f"- **Total Runs:** {overall.get('total_runs', 0)}\n")
    
    if summary and "models" in summary:
        report.append("\n### Per-Model Performance\n")
        report.append("| Model | Mean Score | Std Dev | Mean Latency (ms) |\n")
        report.append("|-------|------------|---------|-------------------|\n")
        
        for model, data in summary["models"].items():
            report.append(
                f"| {model} | {data.get('mean_score', 0.0):.3f} | "
                f"{data.get('std_score', 0.0):.3f} | {data.get('mean_latency_ms', 0):.0f} |\n"
            )
    
    if summary and "categories" in summary:
        report.append("\n### Per-Category Performance\n")
        report.append("| Category | Mean Score | Std Dev |\n")
        report.append("|----------|------------|---------|\n")
        
        for category, data in summary["categories"].items():
            report.append(
                f"| {category} | {data.get('mean_score', 0.0):.3f} | "
                f"{data.get('std_score', 0.0):.3f} |\n"
            )
    
    report.append("\n---\n")
    
    # Tradeoffs Narrative
    report.append("## Behavioral Tradeoffs\n")
    
    if summary and "tradeoffs" in summary:
        tradeoffs = summary["tradeoffs"]
        
        if "strengths_weaknesses" in tradeoffs:
            report.append("### Model Strengths and Weaknesses\n")
            for model, data in tradeoffs["strengths_weaknesses"].items():
                strengths = data.get("strengths", [])
                weaknesses = data.get("weaknesses", [])
                scores = data.get("scores", {})
                
                report.append(f"**{model}:**\n")
                if strengths:
                    report.append(f"- Strengths: {', '.join(strengths)}")
                    for strength in strengths:
                        score = scores.get(strength, 0.0)
                        report.append(f"  - {strength}: {score:.3f}")
                if weaknesses:
                    report.append(f"- Weaknesses: {', '.join(weaknesses)}")
                    for weakness in weaknesses:
                        score = scores.get(weakness, 0.0)
                        report.append(f"  - {weakness}: {score:.3f}")
                report.append("")
        
        if "specific_tradeoffs" in tradeoffs:
            report.append("### Cross-Axis Correlations\n")
            for key, data in tradeoffs["specific_tradeoffs"].items():
                report.append(
                    f"- **{key.replace('_', ' ').title()}**: "
                    f"Correlation = {data['correlation']:.3f} "
                    f"({data['interpretation']})\n"
                )
    
    report.append("\n---\n")
    
    # Failure Mode Insights
    report.append("## Failure Mode Analysis\n")
    
    if summary and "failure_modes" in summary:
        failure_modes = summary["failure_modes"]
        
        if "per_model" in failure_modes:
            report.append("### Failure Rates by Model\n")
            report.append("| Model | Failure Rate | Total Failures |\n")
            report.append("|-------|--------------|----------------|\n")
            
            for model, data in failure_modes["per_model"].items():
                report.append(
                    f"| {model} | {data.get('failure_rate', 0.0):.1%} | "
                    f"{data.get('total_failures', 0)} |\n"
                )
        
        if "top_failure_tags" in failure_modes:
            report.append("\n### Top Failure Tags\n")
            for model, tags in failure_modes["top_failure_tags"].items():
                if tags:
                    report.append(f"**{model}:**\n")
                    for tag in tags[:5]:
                        report.append(
                            f"- {tag['tag']}: {tag['count']} occurrences "
                            f"({tag['frequency']:.1%})\n"
                        )
                    report.append("")
        
        if "systematic_failures" in failure_modes:
            report.append("\n### Systematic Failures\n")
            report.append("Failures occurring in >30% of tests:\n")
            for key, failures in failure_modes["systematic_failures"].items():
                report.append(f"**{key}:**\n")
                for tag, data in failures.items():
                    report.append(
                        f"- {tag}: {data['frequency']:.1%} "
                        f"({data['count']}/{data['total_tests']} tests)\n"
                    )
                report.append("")
    
    report.append("\n---\n")
    
    # Consistency Gaps
    report.append("## Consistency Analysis\n")
    
    if summary and "consistency" in summary:
        consistency = summary["consistency"]
        
        if "overall_consistency" in consistency:
            report.append("### Overall Consistency by Model\n")
            report.append("| Model | Consistency Score | Std Dev |\n")
            report.append("|-------|-------------------|---------|\n")
            
            for model, data in consistency["overall_consistency"].items():
                report.append(
                    f"| {model} | {data.get('consistency_score', 0.0):.3f} | "
                    f"{data.get('std_score', 0.0):.3f} |\n"
                )
        
        if "per_variant_group" in consistency:
            report.append("\n### Emotion Variant Group Consistency\n")
            report.append("Consistency across prompt paraphrases:\n")
            for key, data in consistency["per_variant_group"].items():
                report.append(
                    f"- **{key}**: Consistency = {data.get('consistency_score', 0.0):.3f}, "
                    f"Std Dev = {data.get('std_score', 0.0):.3f}\n"
                )
    
    report.append("\n---\n")
    
    # Deployment-Oriented Recommendations
    report.append("## Deployment-Oriented Recommendations\n")
    
    if recommendations:
        if "overall_ranking" in recommendations:
            report.append("### Overall Model Ranking\n")
            for i, item in enumerate(recommendations["overall_ranking"], 1):
                report.append(f"{i}. **{item['model']}** (weighted score: {item['weighted_score']:.3f})\n")
            report.append("")
        
        if "best_per_category" in recommendations:
            report.append("### Best Model per Category\n")
            for category, rec in recommendations["best_per_category"].items():
                report.append(
                    f"- **{category}**: {rec['model']} (score: {rec['score']:.3f})\n"
                )
            report.append("")
        
        if "use_cases" in recommendations:
            report.append("### Recommended Use Cases\n")
            for model, use_cases in recommendations["use_cases"].items():
                report.append(f"**{model}:**\n")
                for uc in use_cases:
                    report.append(f"- {uc['use_case']}: {uc['rationale']}\n")
                report.append("")
        
        if "deployment_recommendations" in recommendations:
            report.append("### Deployment Recommendations\n")
            for rec in recommendations["deployment_recommendations"]:
                report.append(f"- {rec}\n")
            report.append("")
    
    report.append("\n---\n")
    
    # Prompt Strategies
    report.append("## Prompt Strategy Recommendations\n")
    
    if summary and "failure_modes" in summary:
        failure_modes = summary["failure_modes"]
        
        # Analyze failures to suggest prompt strategies
        strategies = []
        
        # Check for hallucination failures
        if "per_category" in failure_modes:
            halluc_failures = failure_modes["per_category"].get("hallucination", {})
            if halluc_failures.get("failure_rate", 0) > 0.3:
                strategies.append({
                    "issue": "High hallucination rate",
                    "strategy": "Add explicit instruction: 'If you are unsure or don't know the answer, say so explicitly. Do not invent facts or names.'",
                    "template_name": "Hallucination Prevention"
                })
        
        # Check for code execution failures
        code_failures = failure_modes["per_category"].get("code", {})
        if code_failures.get("failure_rate", 0) > 0.3:
            strategies.append({
                "issue": "High code execution failure rate",
                "strategy": "Add instruction: 'Before outputting code, mentally trace through the logic. Ensure your function handles edge cases.'",
                "template_name": "Code Quality"
            })
        
        # Check for low consistency
        if "consistency" in summary and "overall_consistency" in summary["consistency"]:
            low_consistency_models = []
            for model, data in summary["consistency"]["overall_consistency"].items():
                if data.get("consistency_score", 1.0) < 0.7:
                    low_consistency_models.append(model)
            
            if low_consistency_models:
                strategies.append({
                    "issue": f"Low consistency in {', '.join(low_consistency_models)}",
                    "strategy": "Use temperature=0.0 and set seed for deterministic outputs. Consider prompt engineering to reduce variance.",
                    "template_name": "Consistency Improvement"
                })
        
        if strategies:
            for strategy in strategies:
                report.append(f"### {strategy['issue']}\n")
                report.append(f"**Recommended Strategy:** {strategy['strategy']}\n")
                report.append(f"See README.md 'Prompt Strategy Templates' section for the {strategy['template_name']} template.\n\n")
        else:
            report.append("No specific prompt strategy recommendations based on current failure patterns.\n")
    
    report.append("\n---\n")
    report.append("## Conclusion\n")
    report.append("\nThis report provides a comprehensive analysis of model behavior across multiple evaluation axes. ")
    report.append("Use the recommendations above to inform model selection and prompt engineering for your specific use case.\n")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation report"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID to generate report for"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Results directory (default: data/runs/<run_id>)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/runs/<run_id>/REPORT.md)"
    )
    
    args = parser.parse_args()
    
    # Load data
    summary, recommendations = load_data(args.run_id, args.results_dir)
    
    if not summary:
        print(f"❌ Summary not found for run: {args.run_id}")
        return
    
    # Generate report
    report = generate_report(summary, recommendations)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        results_dir = args.results_dir or os.path.join("data", "runs", args.run_id)
        output_path = os.path.join(results_dir, "REPORT.md")
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report generated: {output_path}")


if __name__ == "__main__":
    main()
