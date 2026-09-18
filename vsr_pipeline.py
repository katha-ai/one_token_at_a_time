"""
vsr_pipeline.py

Runs the VSR attention-extraction analysis and computes per-run global attention
means. This is the reproducible, self-contained part of the VSR workflow.

Chains:
  1. run_analysis.py                  -> per-layer attention .npz under save_dir
  2. compute_attention_global_means   -> global_means/<dataset>__<model>.json

What this does NOT do: the paper's semantic POS-tagging and accuracy grading of
generated text were done via Gemini on Vertex AI. That orchestration (GCS upload,
batch-job submission/polling, result parsing) is cluster/GCP-specific plumbing
we don't ship here — reproduce it yourself against your own GCP project using
the exact prompts we used:
  - prompts/gemini_pos_tagging_vsr.txt      (semantic tagging of generated tokens)
  - prompts/gemini_evaluation_vsr_accuracy.txt  (accuracy/alignment grading)

Usage:
    python vsr_pipeline.py --config configs/vsr/vanilla.yaml [--skip_analysis] [--force]
"""

import os
import yaml
import argparse

from run_analysis import main as run_analysis_main
from useful_helpers.compute_global_means import compute_attention_global_means


def run_vsr_pipeline(config_path, skip_analysis=False, force=False):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = config['save_dir']
    run_id = config.get('experiment_name', 'vsr_experiment')

    print(f"\n>>> Initializing VSR Pipeline for Run: {run_id}")
    print(f">>> Results will be stored in: {save_dir}")

    global_means_dir = os.path.join(save_dir, "global_means")
    os.makedirs(save_dir, exist_ok=True)

    parts = os.path.normpath(save_dir).split(os.sep)
    dataset_name, model_name = parts[-2], parts[-1]
    global_means_file = os.path.join(global_means_dir, f"{dataset_name}__{model_name}.json")

    if not force and os.path.exists(global_means_file):
        print(f">>> Global means already exist: {global_means_file}")
        return

    # 2. Run Analysis
    if not skip_analysis:
        print(f"\n>>> Step 1: Running VSR Analysis for {run_id}...")
        run_analysis_main(config_path)
    else:
        print("\n>>> Skipping Analysis Step.")

    # 3. Compute Global Means
    print("\n>>> Step 2: Computing Global Means...")
    compute_attention_global_means(base_paths=[save_dir], output_dir=global_means_dir)

    print(f"\n>>> VSR Pipeline Complete for {run_id}!")
    print(f">>> Attention data: {save_dir}")
    print(f">>> Global means:   {global_means_file}")
    print(">>> For semantic tagging + accuracy grading, see prompts/gemini_pos_tagging_vsr.txt")
    print(">>> and prompts/gemini_evaluation_vsr_accuracy.txt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSR Attention Extraction Pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    run_vsr_pipeline(
        args.config,
        skip_analysis=args.skip_analysis,
        force=args.force,
    )
