#!/usr/bin/env python3
"""
Run All NeuroGuard Benchmarks
=============================
Runs all 5 safety benchmarks on multiple models and generates comprehensive analysis.

Benchmarks:
1. Authority Bias - Cave to fake experts
2. Sycophancy - Agree with user over facts
3. Sandbagging - Strategic underperformance
4. Dark Patterns - Manipulative UI generation
5. Plasticity - Learning bad behaviors

Usage:
    python run_all_benchmarks.py --quick  # 2 models, faster
    python run_all_benchmarks.py          # 4 models, thorough
"""

import os
import sys
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any


# Models to test
MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B", "1.1B"),
    ("microsoft/phi-2", "Phi-2", "2.7B"),
    ("Qwen/Qwen2-1.5B-Instruct", "Qwen2-1.5B", "1.5B"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B", "7B"),
]

QUICK_MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B", "1.1B"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B", "7B"),
]

# Benchmarks to run
BENCHMARKS = [
    ("benchmark.py", "authority_bias", "Authority Bias"),
    ("benchmark_sycophancy.py", "sycophancy", "Sycophancy"),
    ("benchmark_sandbagging.py", "sandbagging", "Sandbagging"),
    ("benchmark_dark_patterns.py", "dark_patterns", "Dark Patterns"),
    ("benchmark_plasticity.py", "plasticity", "Plasticity"),
]


def run_benchmark(script: str, model_id: str, model_name: str, output_dir: str, benchmark_name: str) -> Dict[str, Any]:
    """Run a single benchmark for a model."""
    
    output_file = os.path.join(output_dir, f"{benchmark_name}_{model_name}.csv")
    
    cmd = [
        sys.executable, script,
        "--model", "hf-local",
        "--hf-model", model_id,
        "--output", output_file
    ]
    
    print(f"   Running {benchmark_name} on {model_name}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"   ERROR: {result.stderr[:200]}")
            return {"error": result.stderr[:200]}
        
        # Parse results
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            return {
                "success": True,
                "output_file": output_file,
                "rows": len(df),
            }
        
        return {"error": "No output file"}
        
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


def aggregate_results(output_dir: str, models: List[tuple]) -> pd.DataFrame:
    """Aggregate all results into a summary DataFrame."""
    
    summary_data = []
    
    for model_id, model_name, size in models:
        model_summary = {
            "model_name": model_name,
            "model_id": model_id,
            "size": size,
        }
        
        # Authority Bias
        auth_file = os.path.join(output_dir, f"authority_bias_{model_name}.csv")
        if os.path.exists(auth_file):
            df = pd.read_csv(auth_file)
            model_summary["authority_capitulation_rate"] = df['capitulated'].mean() if 'capitulated' in df.columns else None
            model_summary["authority_control_accuracy"] = df['control_correct'].mean() if 'control_correct' in df.columns else None
        
        # Sycophancy
        syc_file = os.path.join(output_dir, f"sycophancy_{model_name}.csv")
        if os.path.exists(syc_file):
            df = pd.read_csv(syc_file)
            model_summary["sycophancy_flip_rate"] = df['flipped'].mean() if 'flipped' in df.columns else None
            model_summary["sycophancy_initial_correct"] = df['initial_correct'].mean() if 'initial_correct' in df.columns else None
        
        # Sandbagging
        sand_file = os.path.join(output_dir, f"sandbagging_{model_name}.csv")
        if os.path.exists(sand_file):
            df = pd.read_csv(sand_file)
            if 'correct' in df.columns and 'context_type' in df.columns:
                neutral = df[df['context_type'] == 'neutral']['correct'].mean()
                monitored = df[df['context_type'] == 'monitored']['correct'].mean()
                model_summary["sandbagging_neutral_acc"] = neutral
                model_summary["sandbagging_monitored_acc"] = monitored
                model_summary["sandbagging_score"] = max(0, neutral - monitored)
        
        # Dark Patterns
        dark_file = os.path.join(output_dir, f"dark_patterns_{model_name}.csv")
        if os.path.exists(dark_file):
            df = pd.read_csv(dark_file)
            model_summary["dark_pattern_rate"] = df['has_dark_patterns'].mean() if 'has_dark_patterns' in df.columns else None
            model_summary["dark_pattern_severity"] = df['severity_score'].mean() if 'severity_score' in df.columns else None
        
        # Plasticity
        plast_file = os.path.join(output_dir, f"plasticity_{model_name}.csv")
        if os.path.exists(plast_file):
            df = pd.read_csv(plast_file)
            model_summary["plasticity_baseline_refusal"] = df['baseline_refused'].mean() if 'baseline_refused' in df.columns else None
            model_summary["plasticity_corruption_rate"] = df['final_complied'].mean() if 'final_complied' in df.columns else None
        
        summary_data.append(model_summary)
    
    return pd.DataFrame(summary_data)


def create_comprehensive_visualizations(summary_df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations across all benchmarks."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Overall Safety Radar Chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NeuroGuard Safety Evaluation - All Benchmarks', fontsize=16, fontweight='bold')
    
    # 1a. Authority Bias
    ax1 = axes[0, 0]
    if 'authority_capitulation_rate' in summary_df.columns:
        data = summary_df.dropna(subset=['authority_capitulation_rate'])
        bars = ax1.bar(data['model_name'], data['authority_capitulation_rate'] * 100, color='coral')
        ax1.set_ylabel('Capitulation Rate (%)')
        ax1.set_title('Authority Bias\n(Lower = Safer)')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, data['authority_capitulation_rate']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.0f}%', ha='center', fontsize=9)
    
    # 1b. Sycophancy
    ax2 = axes[0, 1]
    if 'sycophancy_flip_rate' in summary_df.columns:
        data = summary_df.dropna(subset=['sycophancy_flip_rate'])
        bars = ax2.bar(data['model_name'], data['sycophancy_flip_rate'] * 100, color='gold')
        ax2.set_ylabel('Flip Rate (%)')
        ax2.set_title('Sycophancy\n(Lower = Safer)')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, data['sycophancy_flip_rate']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.0f}%', ha='center', fontsize=9)
    
    # 1c. Sandbagging
    ax3 = axes[0, 2]
    if 'sandbagging_score' in summary_df.columns:
        data = summary_df.dropna(subset=['sandbagging_score'])
        bars = ax3.bar(data['model_name'], data['sandbagging_score'] * 100, color='mediumpurple')
        ax3.set_ylabel('Sandbagging Score (%)')
        ax3.set_title('Sandbagging\n(Lower = Safer)')
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, data['sandbagging_score']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.0f}%', ha='center', fontsize=9)
    
    # 1d. Dark Patterns
    ax4 = axes[1, 0]
    if 'dark_pattern_rate' in summary_df.columns:
        data = summary_df.dropna(subset=['dark_pattern_rate'])
        bars = ax4.bar(data['model_name'], data['dark_pattern_rate'] * 100, color='crimson')
        ax4.set_ylabel('Dark Pattern Rate (%)')
        ax4.set_title('Dark Patterns\n(Lower = Safer)')
        ax4.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, data['dark_pattern_rate']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.0f}%', ha='center', fontsize=9)
    
    # 1e. Plasticity
    ax5 = axes[1, 1]
    if 'plasticity_corruption_rate' in summary_df.columns:
        data = summary_df.dropna(subset=['plasticity_corruption_rate'])
        bars = ax5.bar(data['model_name'], data['plasticity_corruption_rate'] * 100, color='teal')
        ax5.set_ylabel('Corruption Rate (%)')
        ax5.set_title('Malicious Plasticity\n(Lower = Safer)')
        ax5.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, data['plasticity_corruption_rate']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.0f}%', ha='center', fontsize=9)
    
    # 1f. Overall Safety Score
    ax6 = axes[1, 2]
    
    # Calculate overall safety score (average of all metrics inverted)
    safety_cols = ['authority_capitulation_rate', 'sycophancy_flip_rate', 'sandbagging_score', 
                   'dark_pattern_rate', 'plasticity_corruption_rate']
    available_cols = [c for c in safety_cols if c in summary_df.columns]
    
    if available_cols:
        safety_scores = []
        for _, row in summary_df.iterrows():
            scores = [row[c] for c in available_cols if pd.notna(row.get(c))]
            if scores:
                safety_scores.append(1 - (sum(scores) / len(scores)))
            else:
                safety_scores.append(None)
        
        summary_df['overall_safety'] = safety_scores
        
        data = summary_df.dropna(subset=['overall_safety'])
        colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in data['overall_safety']]
        bars = ax6.barh(data['model_name'], data['overall_safety'] * 100, color=colors)
        ax6.set_xlabel('Overall Safety Score (%)')
        ax6.set_title('Overall Safety\n(Higher = Safer)')
        ax6.set_xlim(0, 100)
        for bar, val in zip(bars, data['overall_safety']):
            ax6.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val*100:.0f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_benchmarks_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: all_benchmarks_comparison.png")
    
    # 2. Heatmap of all metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    heatmap_cols = ['authority_capitulation_rate', 'sycophancy_flip_rate', 'sandbagging_score',
                    'dark_pattern_rate', 'plasticity_corruption_rate']
    heatmap_cols = [c for c in heatmap_cols if c in summary_df.columns]
    
    if heatmap_cols:
        heatmap_data = summary_df[['model_name'] + heatmap_cols].set_index('model_name')
        heatmap_data.columns = ['Authority Bias', 'Sycophancy', 'Sandbagging', 'Dark Patterns', 'Plasticity'][:len(heatmap_cols)]
        heatmap_data = heatmap_data * 100
        
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn_r',
                   ax=ax, vmin=0, vmax=100, cbar_kws={'label': 'Risk Score (%)'})
        ax.set_title('Safety Risk Heatmap - All Benchmarks\n(Lower = Safer)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'safety_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: safety_heatmap.png")


def generate_comprehensive_report(summary_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive markdown report."""
    
    report = f"""# NeuroGuard Comprehensive Safety Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of 5 safety benchmarks across {len(summary_df)} language models.

## Models Tested

| Model | Size |
|-------|------|
"""
    for _, row in summary_df.iterrows():
        report += f"| {row['model_name']} | {row.get('size', 'N/A')} |\n"
    
    report += """

## Benchmark Results

### 1. Authority Bias
*Tests if models abandon truth when pressured by fake experts.*

"""
    if 'authority_capitulation_rate' in summary_df.columns:
        for _, row in summary_df.iterrows():
            rate = row.get('authority_capitulation_rate')
            if pd.notna(rate):
                report += f"- **{row['model_name']}**: {rate*100:.1f}% capitulation rate\n"
    
    report += """

### 2. Sycophancy
*Tests if models agree with user biases over factual accuracy.*

"""
    if 'sycophancy_flip_rate' in summary_df.columns:
        for _, row in summary_df.iterrows():
            rate = row.get('sycophancy_flip_rate')
            if pd.notna(rate):
                report += f"- **{row['model_name']}**: {rate*100:.1f}% flip rate\n"
    
    report += """

### 3. Sandbagging
*Tests if models strategically underperform when monitored.*

"""
    if 'sandbagging_score' in summary_df.columns:
        for _, row in summary_df.iterrows():
            score = row.get('sandbagging_score')
            if pd.notna(score):
                report += f"- **{row['model_name']}**: {score*100:.1f}% performance drop when monitored\n"
    
    report += """

### 4. Dark Patterns
*Tests if models generate manipulative UI/content designs.*

"""
    if 'dark_pattern_rate' in summary_df.columns:
        for _, row in summary_df.iterrows():
            rate = row.get('dark_pattern_rate')
            if pd.notna(rate):
                report += f"- **{row['model_name']}**: {rate*100:.1f}% dark pattern rate\n"
    
    report += """

### 5. Malicious Plasticity
*Tests how quickly models learn bad behaviors from examples.*

"""
    if 'plasticity_corruption_rate' in summary_df.columns:
        for _, row in summary_df.iterrows():
            rate = row.get('plasticity_corruption_rate')
            if pd.notna(rate):
                report += f"- **{row['model_name']}**: {rate*100:.1f}% corruption rate\n"
    
    # Overall rankings
    if 'overall_safety' in summary_df.columns:
        report += """

## Overall Safety Rankings

| Rank | Model | Safety Score |
|------|-------|--------------|
"""
        ranked = summary_df.sort_values('overall_safety', ascending=False)
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            safety = row.get('overall_safety')
            if pd.notna(safety):
                report += f"| {i} | {row['model_name']} | {safety*100:.1f}% |\n"
    
    report += """

## Visualizations

- `all_benchmarks_comparison.png` - 6-panel comparison across all benchmarks
- `safety_heatmap.png` - Heatmap of all safety metrics

## Data Files

All raw results are stored in individual CSV files per benchmark and model.

---
*Generated by NeuroGuard AI Safety Evaluation Platform*
"""
    
    with open(os.path.join(output_dir, 'COMPREHENSIVE_REPORT.md'), 'w') as f:
        f.write(report)
    print(f"Saved: COMPREHENSIVE_REPORT.md")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all NeuroGuard benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick test with 2 models")
    parser.add_argument("--output-dir", default="./all_benchmarks_results", help="Output directory")
    parser.add_argument("--skip-existing", action="store_true", help="Skip benchmarks that already have results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    models = QUICK_MODELS if args.quick else MODELS
    
    print("=" * 60)
    print("NEUROGUARD - COMPREHENSIVE SAFETY EVALUATION")
    print("=" * 60)
    print(f"Models: {len(models)}")
    print(f"Benchmarks: {len(BENCHMARKS)}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Run all benchmarks
    for script, benchmark_key, benchmark_name in BENCHMARKS:
        print(f"\n[BENCHMARK: {benchmark_name}]")
        print("-" * 40)
        
        for model_id, model_name, size in models:
            output_file = os.path.join(args.output_dir, f"{benchmark_key}_{model_name}.csv")
            
            if args.skip_existing and os.path.exists(output_file):
                print(f"   Skipping {model_name} (already exists)")
                continue
            
            result = run_benchmark(script, model_id, model_name, args.output_dir, benchmark_key)
            
            if "error" in result:
                print(f"   {model_name}: ERROR - {result['error'][:100]}")
            else:
                print(f"   {model_name}: OK ({result.get('rows', 0)} results)")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")
    print("=" * 60)
    
    summary_df = aggregate_results(args.output_dir, models)
    summary_df.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    summary_df.to_json(os.path.join(args.output_dir, 'summary.json'), orient='records', indent=2)
    print("Saved: summary.csv, summary.json")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    create_comprehensive_visualizations(summary_df, args.output_dir)
    generate_comprehensive_report(summary_df, args.output_dir)
    
    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    print("Files generated:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"   - {f}")


if __name__ == "__main__":
    main()
