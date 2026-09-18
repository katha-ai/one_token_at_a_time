"""
chartqa_pipeline.py

Runs the ChartQA attention-extraction analysis and computes per-run global
attention means: run_analysis.py -> compute_attention_global_means. Semantic
POS-tagging (Gemini on Vertex AI) isn't part of this pipeline - see
prompts/gemini_pos_tagging_chartqa.txt.

Usage:
    python chartqa_pipeline.py --config configs/chartqa/vanilla.yaml [--skip_analysis] [--force]
"""

import os
import argparse
import yaml

from run_analysis import main as run_analysis_main
from useful_helpers.compute_global_means import compute_attention_global_means


def run_chartqa_pipeline(config_path, skip_analysis=False, force=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = config['save_dir']
    run_id = config.get('experiment_name', 'chartqa_experiment')

    print(f"\n>>> Initializing ChartQA Pipeline for Run: {run_id}")
    print(f">>> Results will be stored in: {save_dir}")

    global_means_dir = os.path.join(save_dir, "global_means")
    os.makedirs(save_dir, exist_ok=True)

    parts = os.path.normpath(save_dir).split(os.sep)
    dataset_name, model_name = parts[-2], parts[-1]
    global_means_file = os.path.join(global_means_dir, f"{dataset_name}__{model_name}.json")

    if not force and os.path.exists(global_means_file):
        print(f">>> Global means already exist: {global_means_file}")
        return

    # 1. Run Analysis
    if not skip_analysis:
        print(f"\n>>> Step 1: Running ChartQA Analysis for {run_id}...")
        run_analysis_main(config_path)
    else:
        print("\n>>> Skipping Analysis Step.")

    # 2. Compute Global Means
    print("\n>>> Step 2: Computing Global Means...")
    compute_attention_global_means(base_paths=[save_dir], output_dir=global_means_dir)

    print(f"\n>>> ChartQA Pipeline Complete for {run_id}!")
    print(f">>> Attention data: {save_dir}")
    print(f">>> Global means:   {global_means_file}")
    print(">>> For semantic tagging, see prompts/gemini_pos_tagging_chartqa.txt.")
    print(">>> Feed the resulting per-tag CSV into data_plotting/bar_plots_chartQA.ipynb to plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChartQA Attention Extraction Pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    run_chartqa_pipeline(
        args.config,
        skip_analysis=args.skip_analysis,
        force=args.force,
    )
