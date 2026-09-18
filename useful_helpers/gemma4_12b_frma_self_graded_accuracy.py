"""
Self-graded (deterministic, no Gemini) FR / MA / combined accuracy for the
gemma4_12b vanilla fruit_math OTAT run - same grading method as
self_graded_accuracy_comparison.py (closed-vocabulary fruit-name match +
numeric digit match against ground truth, reading straight from raw
generated_tokens). Ground truth is parsed directly from each sample's
filename (fruit-{X}__math-{Y}__id-{Z}).

This is a fast, deterministic first look, not a substitute for Gemini-based
grading (see prompts/gemini_evaluation_fruit_math_error_profiling.txt) - it
will not credit synonyms/word-forms Gemini might reasonably accept, and
combined accuracy in particular is a strict AND of both, so treat this as a
lower-bound sanity check, not a final number.

Output:
  data/csv/gemma4_12b_frma_self_graded_accuracy.csv (per-sample)
  data/csv/gemma4_12b_frma_self_graded_accuracy_barplot.png
"""

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Root under which run_analysis.py's `save_dir` outputs (per-layer attention_progression/*.npz) live.
# Override with the OTAT_RUN_ROOT env var to point at wherever you configured `save_dir` to write.
RUN_ROOT = os.environ.get("OTAT_RUN_ROOT", os.path.join(REPO_ROOT, "outputs"))

NPZ_DIR = os.path.join(RUN_ROOT, "otat/gemma4_12b/FrMaSc/0/attention_progression")
OUT_CSV = os.path.join(REPO_ROOT, "data/csv/gemma4_12b_frma_self_graded_accuracy.csv")
OUT_PLOT = os.path.join(REPO_ROOT, "data/csv/gemma4_12b_frma_self_graded_accuracy_barplot.png")

DENIAL_PHRASES = ["no fruit", "not applicable", "not specified", "cannot determine", "not present", "not identifiable"]
NUMBER_RE = re.compile(r"-?\$?\s?(\d[\d,]*\.?\d*)")
FILENAME_RE = re.compile(r"fruit-([^_]+)__math-([^_]+)__id-(\d+)")

FRUIT_SURFACE_FORMS = {
    "apple": {"apple", "apples"},
    "banana": {"banana", "bananas"},
    "coconut": {"coconut", "coconuts"},
    "grape": {"grape", "grapes"},
    "grapefruit": {"grapefruit", "grapefruits"},
    "mango": {"mango", "mangos", "mangoes"},
    "peach": {"peach", "peaches"},
    "pear": {"pear", "pears"},
    "pineapple": {"pineapple", "pineapples"},
    "strawberry": {"strawberry", "strawberries"},
    "watermelon": {"watermelon", "watermelons"},
}


def grade_fruit(text, gt_fruit):
    clause = text.split(".")[0].strip().lower()
    if any(phrase in clause for phrase in DENIAL_PHRASES):
        return False
    accepted = FRUIT_SURFACE_FORMS.get(gt_fruit.lower().strip(), {gt_fruit.lower().strip()})
    words = set(re.findall(r"[a-zA-Z]+", clause))
    return bool(accepted & words)


def grade_math(text, gt_math):
    parts = text.split(".", 1)
    remainder = parts[1] if len(parts) > 1 else ""
    match = NUMBER_RE.search(remainder)
    if not match:
        return False
    try:
        return float(match.group(1).replace(",", "")) == float(str(gt_math).replace(",", ""))
    except ValueError:
        return False


def main():
    files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))
    rows = []
    for fname in files:
        m = FILENAME_RE.search(fname)
        if not m:
            print(f"[WARN] Could not parse GT from filename, skipping: {fname}")
            continue
        gt_fruit, gt_math, sample_id = m.group(1), m.group(2), m.group(3)

        npz = np.load(os.path.join(NPZ_DIR, fname), allow_pickle=True)
        text = "".join(str(t) for t in npz["generated_tokens"])

        fruit_ok = grade_fruit(text, gt_fruit)
        math_ok = grade_math(text, gt_math)
        rows.append({
            "filename": fname,
            "id": sample_id,
            "gt_fruit": gt_fruit,
            "gt_math": gt_math,
            "generated_text": text,
            "fruit_correct": fruit_ok,
            "math_correct": math_ok,
            "combined_correct": fruit_ok and math_ok,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    n = len(df)
    fr_acc = df["fruit_correct"].mean()
    ma_acc = df["math_correct"].mean()
    combined_acc = df["combined_correct"].mean()

    print(f"n={n}")
    print(f"FR accuracy (fruit):        {fr_acc:.4f}")
    print(f"MA accuracy (math):         {ma_acc:.4f}")
    print(f"Combined accuracy (both):   {combined_acc:.4f}")
    print(f"\nWrote per-sample grading -> {OUT_CSV}")

    # --- bar plot ---
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })
    labels = ["FR (fruit)", "MA (math)", "Combined"]
    values = [fr_acc, ma_acc, combined_acc]
    colors = ["#FFA500", "#267E59", "#3B5B92"]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.1%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Gemma-4 (12B-it) vanilla fruit_math\nself-graded accuracy (n={n})", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote bar plot -> {OUT_PLOT}")


if __name__ == "__main__":
    main()
