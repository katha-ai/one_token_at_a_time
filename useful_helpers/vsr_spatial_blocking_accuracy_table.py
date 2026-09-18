"""
vsr_spatial_blocking_accuracy_table.py

Builds the V(isual) / T(ext) / VT accuracy table for the total-image-blocking-at-
IMG_SPATIAL_RELATION and total-image-blocking-at-TXT_SPATIAL_RELATION VSR experiments
(LOV-7B, QVL-7B, Gemma-4-12B), from each run's vsr_pipeline.py final_results.csv
(Visual_Correct / Text_Correct columns - same convention as
data_plotting/beta_ablation_gemma4_vsr.py's load_ids_and_vt_accuracy).

Usage:
    python useful_helpers/vsr_spatial_blocking_accuracy_table.py
    python useful_helpers/vsr_spatial_blocking_accuracy_table.py --out_csv path/to/table.csv
"""

import argparse
import csv
import os

# Root under which run_analysis.py's `save_dir` outputs (per-layer attention_progression/*.npz) live.
# Override with the OTAT_RUN_ROOT env var, or pass --base directly, to point at your run outputs.
_RUN_ROOT = os.environ.get("OTAT_RUN_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))
BASE = os.path.join(_RUN_ROOT, "otat/vsr_blocking")

# (tag_dir, model, prefix) -> final_results.csv is {BASE}/{tag_dir}/{model}/{prefix}_final_results.csv
RUNS = [
    ("img_spatial_relation", "lov_7b",     "vsr_img_block_lov7b"),
    ("img_spatial_relation", "qvl_7b",     "vsr_img_block_qvl7b"),
    ("img_spatial_relation", "gemma4_12b", "vsr_img_block_gemma4_12b"),
    ("txt_spatial_relation", "lov_7b",     "vsr_txt_block_lov7b"),
    ("txt_spatial_relation", "qvl_7b",     "vsr_txt_block_qvl7b"),
    ("txt_spatial_relation", "gemma4_12b", "vsr_txt_block_gemma4_12b"),
]

DISPLAY_TAG = {
    "img_spatial_relation": "Block@IMG_SPATIAL_RELATION",
    "txt_spatial_relation": "Block@TXT_SPATIAL_RELATION",
}
DISPLAY_MODEL = {
    "lov_7b": "LOV-7B",
    "qvl_7b": "QVL-7B",
    "gemma4_12b": "Gemma-4-12B",
}


def load_vtv_accuracy(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    v_correct = sum(1 for r in rows if r.get("Visual_Correct") == "True")
    t_correct = sum(1 for r in rows if r.get("Text_Correct") == "True")
    vt_correct = sum(1 for r in rows if r.get("Visual_Correct") == "True" and r.get("Text_Correct") == "True")
    return n, 100.0 * v_correct / n, 100.0 * t_correct / n, 100.0 * vt_correct / n


def main():
    parser = argparse.ArgumentParser(description="VSR spatial-relation blocking V/T/VT accuracy table")
    parser.add_argument("--base", default=BASE, help="Base dir containing <tag_dir>/<model>/ run outputs")
    parser.add_argument("--out_csv", default=None, help="Optional path to also write the table as CSV")
    parser.add_argument("--prefix_suffix", default="", help="Appended to each run's prefix, e.g. '_fullspan'")
    args = parser.parse_args()

    rows_out = []
    print(f"{'Blocking':<28} {'Model':<14} {'n':>5} {'V-Acc':>8} {'T-Acc':>8} {'VT-Acc':>8}")
    print("-" * 76)
    for tag_dir, model, prefix in RUNS:
        prefix = f"{prefix}{args.prefix_suffix}"
        path = os.path.join(args.base, tag_dir, model, f"{prefix}_final_results.csv")
        if not os.path.exists(path):
            print(f"{DISPLAY_TAG[tag_dir]:<28} {DISPLAY_MODEL[model]:<14} {'MISSING: ' + path}")
            continue
        n, v_acc, t_acc, vt_acc = load_vtv_accuracy(path)
        print(f"{DISPLAY_TAG[tag_dir]:<28} {DISPLAY_MODEL[model]:<14} {n:>5} {v_acc:>7.2f}% {t_acc:>7.2f}% {vt_acc:>7.2f}%")
        rows_out.append({
            "blocking": DISPLAY_TAG[tag_dir], "model": DISPLAY_MODEL[model], "n": n,
            "V_Accuracy": round(v_acc, 2), "T_Accuracy": round(t_acc, 2), "VT_Accuracy": round(vt_acc, 2),
        })

    if args.out_csv and rows_out:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["blocking", "model", "n", "V_Accuracy", "T_Accuracy", "VT_Accuracy"])
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nWrote table -> {args.out_csv}")


if __name__ == "__main__":
    main()
