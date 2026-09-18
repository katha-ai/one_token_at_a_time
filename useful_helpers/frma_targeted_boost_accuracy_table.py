"""
Deterministic (no-Gemini) Fruit / Math / Combined accuracy table for the
fruit_math targeted-boosting experiment, across LOV-7B, QVL-7B, and
Gemma-4-12B.

Grades every sample directly from the raw `generated_tokens` array saved in
each run's .npz files (layer 0 only - generated_tokens is identical across
layers within a run, see CLAUDE.md's fruit_math filename-provenance notes for
why we do NOT trust any existing Gemini `final_evaluation.csv` for "vanilla"
without checking provenance first). This sidesteps that whole class of bug:
every number below is derived straight from ground-truth CSV + raw generation
output on disk, nothing pre-graded. This script is used to provenance-check
the Gemini eval files, not as the final source of truth for the table -- see
frma_targeted_boost_gemini_accuracy_table.py for that.

Definitions used throughout (fruit_math asks the model to identify the fruit
in the image and solve the math puzzle stated in the text):
  - Fruit Accuracy    = fraction of samples with the correct fruit named
  - Math Accuracy     = fraction of samples with the correct math answer
  - Combined Accuracy = fraction of samples correct on BOTH (joint accuracy)

Two accuracy numbers are reported per model per condition:
  - full vanilla N (927, the whole CSV) - the unconditional vanilla baseline
  - paired-subset N (= targeted run's N) - vanilla and targeted computed on
    the *identical* sample set, for an apples-to-apples targeted-boost delta.
    Targeted runs are a strict subset of vanilla because `style: precomputed`
    boosting only fires on samples where the POS tagger actually found both
    FRUIT_CONCEPT and MATH_ANSWER spans; samples missing either tag can't be
    boosted and are dropped from the run.

Output: per-model per-sample grading CSVs under
data/csv/frma_targeted_boost_accuracy/{model}_{condition}_self_graded.csv,
plus a summary table printed to stdout and written as markdown to
data/csv/frma_targeted_boost_accuracy/summary_table.md.
"""

import os
import re

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Root under which run_analysis.py's `save_dir` outputs (per-layer attention_progression/*.npz) live.
# Override with the OTAT_RUN_ROOT env var to point at wherever you configured `save_dir` to write.
RUN_ROOT = os.environ.get("OTAT_RUN_ROOT", os.path.join(REPO_ROOT, "outputs"))
SOURCE_CSV = os.path.join(REPO_ROOT, "data/csv/fruit_math_oid_scaled1K.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data/csv/frma_targeted_boost_accuracy")

MODELS = {
    "lov_7b": {
        "vanilla_dir": os.path.join(RUN_ROOT, "otat/FrMaSc/lov_7b/0/attention_progression"),
        "targeted_dir": os.path.join(RUN_ROOT, "multiplicative_boost_final/fruit_math_targeted/lov_7b_image_and_text_boost_50beta/0/attention_progression"),
        "boost_factor": 50.0,
    },
    "qvl_7b": {
        "vanilla_dir": os.path.join(RUN_ROOT, "otat/FrMaSc/qvl_7b/0/attention_progression"),
        "targeted_dir": os.path.join(RUN_ROOT, "multiplicative_boost_final/fruit_math_targeted/qvl_7b_image_and_text_boost_200beta/0/attention_progression"),
        "boost_factor": 200.0,
    },
    "gemma4_12b": {
        "vanilla_dir": os.path.join(RUN_ROOT, "otat/gemma4_12b/FrMaSc/0/attention_progression"),
        "targeted_dir": os.path.join(RUN_ROOT, "multiplicative_boost_final/fruit_math_targeted/gemma4_12b_image_and_text_boost_50beta/0/attention_progression"),
        "boost_factor": 50.0,
    },
}

TOKEN_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")
DENIAL_PHRASES = ["no fruit", "not applicable", "not specified", "cannot determine", "not present", "not identifiable"]
NUMBER_RE = re.compile(r"-?\$?\s?(\d[\d,]*\.?\d*)")

# Closed vocabulary (data/csv/fruit_math_oid_scaled1K.csv has exactly these
# 11 fruit_category values) - hardcoded accepted surface forms instead of a
# generic English stemmer, which mishandles "grape"+"s"="grapes" (not the
# "-es" plural pattern) vs "peach"+"es"="peaches" (which is) ambiguously.
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
        return False, clause
    gt_key = gt_fruit.lower().strip()
    accepted = FRUIT_SURFACE_FORMS.get(gt_key, {gt_key})
    words = set(re.findall(r"[a-zA-Z]+", clause))
    return bool(accepted & words), clause


def grade_math(text, gt_math):
    parts = text.split(".", 1)
    remainder = parts[1] if len(parts) > 1 else ""
    match = NUMBER_RE.search(remainder)
    if not match:
        return False, None
    raw = match.group(1).replace(",", "")
    try:
        predicted = float(raw)
        gt = float(str(gt_math).replace(",", ""))
        return predicted == gt, predicted
    except ValueError:
        return False, None


def load_sample_table(npz_dir, source):
    """(filename, gt_fruit, gt_math, generated_text) for every .npz actually
    present in npz_dir, matched to ground truth via the __id-N row-index
    convention utils/generic_utils.get_filename_for_experiment uses."""
    id_re = re.compile(r"__id-(\d+)\.npz$")
    rows = []
    for fname in sorted(os.listdir(npz_dir)):
        if not fname.endswith(".npz"):
            continue
        match = id_re.search(fname)
        if not match:
            continue
        idx = int(match.group(1))
        src_row = source.iloc[idx]
        npz = np.load(os.path.join(npz_dir, fname), allow_pickle=True)
        text = "".join(str(t) for t in npz["generated_tokens"])
        rows.append(
            {
                "filename": fname,
                "gt_fruit": src_row["fruit_category"],
                "gt_math": src_row["math_answer"],
                "generated_text": text,
            }
        )
    return pd.DataFrame(rows)


def grade_table(df):
    df = df.copy()
    fruit_results = df.apply(lambda r: grade_fruit(r["generated_text"], r["gt_fruit"]), axis=1)
    math_results = df.apply(lambda r: grade_math(r["generated_text"], r["gt_math"]), axis=1)
    df["fruit_clause"] = [r[1] for r in fruit_results]
    df["self_fruit_correct"] = [r[0] for r in fruit_results]
    df["self_predicted_math"] = [r[1] for r in math_results]
    df["self_math_correct"] = [r[0] for r in math_results]
    df["self_joint_correct"] = df["self_fruit_correct"] & df["self_math_correct"]
    return df


def accuracy_row(df):
    n = len(df)
    return {
        "n": n,
        "Fruit Accuracy": df["self_fruit_correct"].mean() if n else float("nan"),
        "Math Accuracy": df["self_math_correct"].mean() if n else float("nan"),
        "Combined Accuracy": df["self_joint_correct"].mean() if n else float("nan"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    source = pd.read_csv(SOURCE_CSV)

    summary_rows = []
    for model, paths in MODELS.items():
        vanilla_df = grade_table(load_sample_table(paths["vanilla_dir"], source))
        targeted_df = grade_table(load_sample_table(paths["targeted_dir"], source))

        vanilla_df.to_csv(os.path.join(OUT_DIR, f"{model}_vanilla_self_graded.csv"), index=False)
        targeted_df.to_csv(os.path.join(OUT_DIR, f"{model}_targeted_self_graded.csv"), index=False)

        paired_filenames = set(vanilla_df["filename"]) & set(targeted_df["filename"])
        assert paired_filenames == set(targeted_df["filename"]), (
            f"{model}: targeted run has samples absent from vanilla run - unexpected, check provenance"
        )
        vanilla_paired_df = vanilla_df[vanilla_df["filename"].isin(paired_filenames)]

        full_vanilla_acc = accuracy_row(vanilla_df)
        paired_vanilla_acc = accuracy_row(vanilla_paired_df)
        targeted_acc = accuracy_row(targeted_df)

        summary_rows.append(
            {
                "model": model,
                "boost_factor": paths["boost_factor"],
                "condition": "vanilla (full N)",
                **full_vanilla_acc,
            }
        )
        summary_rows.append(
            {
                "model": model,
                "boost_factor": paths["boost_factor"],
                "condition": "vanilla (paired subset)",
                **paired_vanilla_acc,
            }
        )
        summary_rows.append(
            {
                "model": model,
                "boost_factor": paths["boost_factor"],
                "condition": "targeted V-T boost",
                **targeted_acc,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    md_lines = ["| Model | Boost factor | Condition | N | Fruit Accuracy | Math Accuracy | Combined Accuracy |"]
    md_lines.append("|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        md_lines.append(
            f"| {r['model']} | {r['boost_factor']:g} | {r['condition']} | {r['n']} | "
            f"{r['Fruit Accuracy']:.4f} | {r['Math Accuracy']:.4f} | {r['Combined Accuracy']:.4f} |"
        )
    md_path = os.path.join(OUT_DIR, "summary_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nWrote per-model per-sample grading -> {OUT_DIR}/*_self_graded.csv")
    print(f"Wrote summary table -> {md_path}")


if __name__ == "__main__":
    main()
