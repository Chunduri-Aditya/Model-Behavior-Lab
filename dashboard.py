#!/usr/bin/env python3
"""
Interactive visualization dashboard for Model Behavior Lab results.
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


@st.cache_data
def load_results(run_id):
    """Load results for a given run ID."""
    results_dir = os.path.join("data", "runs", run_id)
    jsonl_path = os.path.join(results_dir, "results.jsonl")
    json_path = os.path.join(results_dir, "results.json")
    summary_path = os.path.join(results_dir, "summary.json")
    
    results = []
    summary = None
    
    # Load results
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    
    # Load summary
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    return results, summary


def get_available_runs():
    """Get list of available run IDs."""
    runs_dir = Path("data/runs")
    if not runs_dir.exists():
        return []
    
    runs = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                runs.append(run_dir.name)
    
    return sorted(runs, reverse=True)  # Most recent first


def main():
    st.set_page_config(
        page_title="Model Behavior Lab Dashboard",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Model Behavior Lab Dashboard")
    st.markdown("Interactive visualization and analysis of LLM evaluation results")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Run selection
    available_runs = get_available_runs()
    if not available_runs:
        st.error("No experiment runs found. Please run an experiment first using `python main.py`")
        return
    
    selected_run = st.sidebar.selectbox(
        "Select Run ID",
        available_runs,
        index=0
    )
    
    # Load data
    results, summary = load_results(selected_run)
    
    if not results:
        st.error(f"No results found for run: {selected_run}")
        return
    
    df = pd.DataFrame(results)
    
    # Model and category filters
    st.sidebar.subheader("Filters")
    available_models = sorted(df["model"].unique())
    selected_models = st.sidebar.multiselect(
        "Models",
        available_models,
        default=available_models
    )
    
    available_categories = sorted(df["category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Categories",
        available_categories,
        default=available_categories
    )
    
    # Filter data
    df_filtered = df[
        (df["model"].isin(selected_models)) &
        (df["category"].isin(selected_categories))
    ]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Heatmap",
        "Tradeoffs",
        "Failure Modes",
        "Consistency"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Overview")
        
        if summary:
            st.subheader("Run Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Runs", summary.get("total_runs", 0))
            with col2:
                st.metric("Models", len(summary.get("models", {})))
            with col3:
                if summary.get("git_commit"):
                    st.metric("Git Commit", summary["git_commit"][:8])
        
        st.subheader("Per-Model Averages")
        if summary and "models" in summary:
            model_data = []
            for model, data in summary["models"].items():
                model_data.append({
                    "Model": model,
                    "Mean Score": data.get("mean_score", 0.0),
                    "Std Dev": data.get("std_score", 0.0),
                    "Mean Latency (ms)": data.get("mean_latency_ms", 0.0)
                })
            
            model_df = pd.DataFrame(model_data)
            st.dataframe(model_df, use_container_width=True)
            
            # Bar chart
            fig = px.bar(
                model_df,
                x="Model",
                y="Mean Score",
                error_y="Std Dev",
                title="Average Score by Model",
                color="Mean Score",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Per-Category Averages")
        if summary and "categories" in summary:
            cat_data = []
            for category, data in summary["categories"].items():
                cat_data.append({
                    "Category": category,
                    "Mean Score": data.get("mean_score", 0.0),
                    "Std Dev": data.get("std_score", 0.0)
                })
            
            cat_df = pd.DataFrame(cat_data)
            st.dataframe(cat_df, use_container_width=True)
            
            fig = px.bar(
                cat_df,
                x="Category",
                y="Mean Score",
                error_y="Std Dev",
                title="Average Score by Category",
                color="Mean Score",
                color_continuous_scale="Plasma"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Heatmap
    with tab2:
        st.header("Score Heatmap")
        
        # Create pivot table: model x test_id
        pivot_data = df_filtered.groupby(["model", "test_id"])["final_score"].mean().reset_index()
        pivot_table = pivot_data.pivot(index="test_id", columns="model", values="final_score")
        
        if not pivot_table.empty:
            fig = px.imshow(
                pivot_table,
                labels=dict(x="Model", y="Test ID", color="Score"),
                title="Average Score Heatmap (Model × Test)",
                color_continuous_scale="RdYlGn",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Data Table")
            st.dataframe(pivot_table, use_container_width=True)
        else:
            st.info("No data available for selected filters")
    
    # Tab 3: Tradeoffs
    with tab3:
        st.header("Behavioral Tradeoffs")
        
        if summary and "tradeoffs" in summary:
            tradeoffs = summary["tradeoffs"]
            
            # Per-model tradeoffs
            if "per_model_tradeoffs" in tradeoffs:
                st.subheader("Per-Model Category Scores")
                tradeoff_data = []
                for model, scores in tradeoffs["per_model_tradeoffs"].items():
                    for category, score in scores.items():
                        tradeoff_data.append({
                            "Model": model,
                            "Category": category,
                            "Score": score
                        })
                
                if tradeoff_data:
                    tradeoff_df = pd.DataFrame(tradeoff_data)
                    
                    # Radar-style visualization (using bar chart grouped)
                    fig = px.bar(
                        tradeoff_df,
                        x="Category",
                        y="Score",
                        color="Model",
                        barmode="group",
                        title="Category Scores by Model (Tradeoff View)",
                        labels={"Score": "Average Score"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plots
            st.subheader("Cross-Axis Correlations")
            
            # Reasoning vs Hallucination
            if "specific_tradeoffs" in tradeoffs:
                spec_tradeoffs = tradeoffs["specific_tradeoffs"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Reasoning vs Hallucination**")
                    if "reasoning_vs_hallucination" in spec_tradeoffs:
                        corr_data = spec_tradeoffs["reasoning_vs_hallucination"]
                        st.metric("Correlation", f"{corr_data['correlation']:.3f}")
                        st.caption(f"Interpretation: {corr_data['interpretation']}")
                
                with col2:
                    st.write("**Code vs Hallucination**")
                    if "code_vs_hallucination" in spec_tradeoffs:
                        corr_data = spec_tradeoffs["code_vs_hallucination"]
                        st.metric("Correlation", f"{corr_data['correlation']:.3f}")
                        st.caption(f"Interpretation: {corr_data['interpretation']}")
            
            # Scatter plot: Reasoning vs Hallucination
            st.subheader("Scatter Plot: Reasoning vs Hallucination")
            model_scores = {}
            for model in selected_models:
                model_df = df_filtered[df_filtered["model"] == model]
                reasoning = model_df[model_df["category"] == "reasoning"]["final_score"].mean()
                hallucination = model_df[model_df["category"] == "hallucination"]["final_score"].mean()
                if not pd.isna(reasoning) and not pd.isna(hallucination):
                    model_scores[model] = {"reasoning": reasoning, "hallucination": hallucination}
            
            if model_scores:
                scatter_data = pd.DataFrame(model_scores).T.reset_index()
                scatter_data.columns = ["Model", "Reasoning", "Hallucination"]
                
                fig = px.scatter(
                    scatter_data,
                    x="Reasoning",
                    y="Hallucination",
                    text="Model",
                    title="Reasoning vs Hallucination Tradeoff",
                    labels={"Reasoning": "Reasoning Score", "Hallucination": "Hallucination Score"}
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tradeoff analysis not available in summary")
    
    # Tab 4: Failure Modes
    with tab4:
        st.header("Failure Mode Analysis")
        
        if summary and "failure_modes" in summary:
            failure_modes = summary["failure_modes"]
            
            # Top failure tags
            if "top_failure_tags" in failure_modes:
                st.subheader("Top Failure Tags by Model")
                for model, tags in failure_modes["top_failure_tags"].items():
                    if model in selected_models and tags:
                        st.write(f"**{model}**")
                        tag_df = pd.DataFrame(tags)
                        st.dataframe(tag_df, use_container_width=True)
            
            # Worst tests
            if "worst_tests" in failure_modes:
                st.subheader("Worst Performing Tests by Model")
                for model, tests in failure_modes["worst_tests"].items():
                    if model in selected_models and tests:
                        st.write(f"**{model}**")
                        test_df = pd.DataFrame(tests)
                        st.dataframe(test_df, use_container_width=True)
            
            # Failure rates
            if "per_model" in failure_modes:
                st.subheader("Failure Rates by Model")
                failure_data = []
                for model, data in failure_modes["per_model"].items():
                    if model in selected_models:
                        failure_data.append({
                            "Model": model,
                            "Failure Rate": data.get("failure_rate", 0.0),
                            "Total Failures": data.get("total_failures", 0)
                        })
                
                if failure_data:
                    failure_df = pd.DataFrame(failure_data)
                    st.dataframe(failure_df, use_container_width=True)
                    
                    fig = px.bar(
                        failure_df,
                        x="Model",
                        y="Failure Rate",
                        title="Failure Rate by Model",
                        color="Failure Rate",
                        color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Failure mode analysis not available in summary")
    
    # Tab 5: Consistency
    with tab5:
        st.header("Consistency Analysis")
        
        if summary and "consistency" in summary:
            consistency = summary["consistency"]
            
            # Per-model consistency
            if "overall_consistency" in consistency:
                st.subheader("Overall Consistency by Model")
                consistency_data = []
                for model, data in consistency["overall_consistency"].items():
                    if model in selected_models:
                        consistency_data.append({
                            "Model": model,
                            "Consistency Score": data.get("consistency_score", 0.0),
                            "Std Dev": data.get("std_score", 0.0)
                        })
                
                if consistency_data:
                    consistency_df = pd.DataFrame(consistency_data)
                    st.dataframe(consistency_df, use_container_width=True)
                    
                    fig = px.bar(
                        consistency_df,
                        x="Model",
                        y="Consistency Score",
                        title="Consistency Score by Model (higher = more consistent)",
                        color="Consistency Score",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Per-variant group consistency (for emotion tests)
            if "per_variant_group" in consistency:
                st.subheader("Emotion Variant Group Consistency")
                variant_data = []
                for key, data in consistency["per_variant_group"].items():
                    model = key.split(":")[0]
                    if model in selected_models:
                        variant_data.append({
                            "Model": model,
                            "Variant Group": key.split(":")[1],
                            "Consistency Score": data.get("consistency_score", 0.0),
                            "Std Dev": data.get("std_score", 0.0)
                        })
                
                if variant_data:
                    variant_df = pd.DataFrame(variant_data)
                    st.dataframe(variant_df, use_container_width=True)
        else:
            st.info("Consistency analysis not available in summary")
        
        # Show variance across repeats for selected tests
        st.subheader("Variance Across Repeats")
        if not df_filtered.empty:
            variance_data = df_filtered.groupby(["model", "test_id"])["final_score"].agg(["mean", "std"]).reset_index()
            variance_data = variance_data[variance_data["std"].notna()]
            
            if not variance_data.empty:
                fig = px.scatter(
                    variance_data,
                    x="mean",
                    y="std",
                    color="model",
                    hover_data=["test_id"],
                    title="Score Variance (Mean vs Std Dev)",
                    labels={"mean": "Mean Score", "std": "Standard Deviation"}
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
