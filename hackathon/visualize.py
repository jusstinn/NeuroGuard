#!/usr/bin/env python3
"""
Visualize Authority Bias Benchmark Results
==========================================
Creates charts comparing failure rates across models and difficulty levels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_results(filepath: str = "results.csv") -> pd.DataFrame:
    """Load results from CSV."""
    return pd.read_csv(filepath)


def plot_failure_by_model(df: pd.DataFrame, output: str = "benchmark_chart.png"):
    """Create bar chart of failure rates by model."""
    
    # Set style
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors
    colors = {
        "low": "#10B981",
        "medium": "#F59E0B", 
        "high": "#EF4444",
        "critical": "#DC2626",
    }
    
    # Plot 1: Failure rate by model
    ax1 = axes[0]
    
    model_stats = df.groupby("model_name").agg({
        "capitulated": ["sum", "count"]
    }).reset_index()
    model_stats.columns = ["model", "failures", "total"]
    model_stats["failure_rate"] = model_stats["failures"] / model_stats["total"]
    
    bars = ax1.bar(
        range(len(model_stats)),
        model_stats["failure_rate"] * 100,
        color=["#FF006E" if r > 0.3 else "#00D9FF" for r in model_stats["failure_rate"]],
        edgecolor="white",
        linewidth=0.5,
    )
    
    ax1.set_xticks(range(len(model_stats)))
    ax1.set_xticklabels(model_stats["model"], rotation=45, ha="right")
    ax1.set_ylabel("Capitulation Rate (%)", fontsize=12)
    ax1.set_title("🏆 Authority Bias Benchmark\nCapitulation Rate by Model", fontsize=14, pad=20)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar, rate in zip(bars, model_stats["failure_rate"]):
        height = bar.get_height()
        ax1.annotate(
            f'{rate:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10,
            fontweight='bold',
        )
    
    # Add threshold line
    ax1.axhline(y=15, color="#F59E0B", linestyle="--", alpha=0.7, label="Medium Risk Threshold")
    ax1.axhline(y=30, color="#EF4444", linestyle="--", alpha=0.7, label="High Risk Threshold")
    ax1.legend(loc="upper right", fontsize=9)
    
    # Plot 2: Failure rate by difficulty
    ax2 = axes[1]
    
    difficulty_order = ["Common Knowledge", "Undergrad", "PhD Niche"]
    diff_stats = df.groupby("difficulty").agg({
        "capitulated": ["sum", "count"]
    }).reset_index()
    diff_stats.columns = ["difficulty", "failures", "total"]
    diff_stats["failure_rate"] = diff_stats["failures"] / diff_stats["total"]
    diff_stats["order"] = diff_stats["difficulty"].map({d: i for i, d in enumerate(difficulty_order)})
    diff_stats = diff_stats.sort_values("order")
    
    bar_colors = ["#00D9FF", "#8B5CF6", "#FF006E"]
    
    bars2 = ax2.bar(
        range(len(diff_stats)),
        diff_stats["failure_rate"] * 100,
        color=bar_colors[:len(diff_stats)],
        edgecolor="white",
        linewidth=0.5,
    )
    
    ax2.set_xticks(range(len(diff_stats)))
    ax2.set_xticklabels(diff_stats["difficulty"], rotation=45, ha="right")
    ax2.set_ylabel("Capitulation Rate (%)", fontsize=12)
    ax2.set_title("Capitulation Rate by Difficulty Level", fontsize=14, pad=20)
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bar, rate in zip(bars2, diff_stats["failure_rate"]):
        height = bar.get_height()
        ax2.annotate(
            f'{rate:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10,
            fontweight='bold',
        )
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    print(f"📊 Chart saved to {output}")
    
    return fig


def plot_detailed_breakdown(df: pd.DataFrame, output: str = "detailed_chart.png"):
    """Create detailed breakdown chart."""
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table
    pivot = df.pivot_table(
        values="capitulated",
        index="difficulty",
        columns="model_name",
        aggfunc="mean"
    ) * 100
    
    # Reorder difficulties
    difficulty_order = ["Common Knowledge", "Undergrad", "PhD Niche"]
    pivot = pivot.reindex([d for d in difficulty_order if d in pivot.index])
    
    # Plot grouped bar chart
    x = range(len(pivot.index))
    width = 0.8 / len(pivot.columns)
    
    colors = plt.cm.viridis(range(0, 256, 256 // len(pivot.columns)))
    
    for i, col in enumerate(pivot.columns):
        offset = (i - len(pivot.columns) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            pivot[col],
            width,
            label=col,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Capitulation Rate (%)", fontsize=12)
    ax.set_xlabel("Difficulty Level", fontsize=12)
    ax.set_title("Authority Bias: Model × Difficulty Breakdown", fontsize=14, pad=20)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    print(f"📊 Detailed chart saved to {output}")
    
    return fig


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("📊 AUTHORITY BIAS BENCHMARK - SUMMARY")
    print("=" * 60)
    
    # Overall stats
    total = len(df)
    failures = df["capitulated"].sum()
    rate = failures / total
    
    print(f"\nTotal Tests: {total}")
    print(f"Total Capitulations: {int(failures)}")
    print(f"Overall Capitulation Rate: {rate:.1%}")
    
    # By model
    print("\nBy Model:")
    for model, group in df.groupby("model_name"):
        model_rate = group["capitulated"].mean()
        print(f"   {model}: {model_rate:.1%}")
    
    # By difficulty
    print("\nBy Difficulty:")
    for diff in ["Common Knowledge", "Undergrad", "PhD Niche"]:
        if diff in df["difficulty"].values:
            diff_rate = df[df["difficulty"] == diff]["capitulated"].mean()
            print(f"   {diff}: {diff_rate:.1%}")
    
    # Worst performing topics
    print("\nMost Vulnerable Topics:")
    topic_rates = df.groupby("topic")["capitulated"].mean().sort_values(ascending=False)
    for topic, rate in topic_rates.head(5).items():
        print(f"   {topic}: {rate:.1%}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", default="results.csv", help="Input CSV file")
    parser.add_argument("--output", default="benchmark_chart.png", help="Output chart file")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ Results file not found: {args.input}")
        print("   Run benchmark.py first to generate results.")
        return
    
    df = load_results(args.input)
    
    print(f"📂 Loaded {len(df)} results from {args.input}")
    
    plot_failure_by_model(df, args.output)
    
    if len(df["model_name"].unique()) > 1:
        plot_detailed_breakdown(df, "detailed_chart.png")
    
    print_summary(df)


if __name__ == "__main__":
    main()
