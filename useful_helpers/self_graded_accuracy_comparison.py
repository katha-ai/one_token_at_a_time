"""
Accuracy comparison for vanilla / targeted / controlled-random LOV-7B
fruit_math runs, grading ourselves directly from the raw generated text and
ground truth - no Gemini involved at all.

The vanilla sample list (filename, gt_fruit, gt_math) is derived directly
from data/csv/fruit_math_oid_scaled1K.csv + the OTAT directory listing, not
from data/automated_gemini_acc_evals/FrMaSc/vanilla/lov_7b/final_evaluation.csv
- that file was deleted (2026-08-09): it turned out to be a Gemini grading of
a *different*, unidentified generation run that happened to share the same
sample-naming convention as the OTAT directory, not an actual grading of the
OTAT text (see CLAUDE.md's "Known issue" section). gt_fruit/gt_math were
always filename-derived rather than Gemini-derived, so this loses nothing -
it just removes any risk of also pulling that file's now-untrustworthy
predicted_fruit/predicted_math/fruit_correct columns.

Fruit grading: extract the clause before the first "." (or full text if no
period), split into words, normalize each for simple English pluralization
(cats->cat, berries->berry, boxes->box), and check whether the normalized
ground-truth fruit name appears among them. A clause containing an explicit
denial phrase ("no fruit", "not applicable", "not specified", "cannot
determine") is graded as unanswered/incorrect regardless.

Math grading: extract the first numeric token (handling "$", ",", and
decimals) from the remainder of the text after the first ".", and compare it
numerically to the ground-truth math answer.

This is intentionally simple and literal - it will not catch every synonym
or word-form Gemini's grading might reasonably credit (e.g. "sixty" for
"60"), but it is fully deterministic and reproducible, which is the point:
a stable baseline to sanity-check the Gemini-graded numbers against.

Output: prints a comparison table; also writes per-sample grading detail to
data/csv/frma_random_boost_samples/lov_7b_self_graded_accuracy.csv
"""

import json
import os
import re

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Root under which run_analysis.py's `save_dir` outputs (per-layer attention_progression/*.npz) live.
# Override with the OTAT_RUN_ROOT env var to point at wherever you configured `save_dir` to write.
RUN_ROOT = os.environ.get("OTAT_RUN_ROOT", os.path.join(REPO_ROOT, "outputs"))

SOURCE_CSV = os.path.join(REPO_ROOT, "data/csv/fruit_math_oid_scaled1K.csv")
VANILLA_NPZ_DIR = os.path.join(RUN_ROOT, "otat/FrMaSc/lov_7b/0/attention_progression")

CONTROLLED_EVAL_CSV = os.path.join(
    REPO_ROOT, "data/automated_gemini_acc_evals/lov_7b_50_30_controlled_random/lov_7b_control_random/final_evaluation.csv"
)
CONTROLLED_NPZ_DIR = os.path.join(
    RUN_ROOT,
    "multiplicative_boost_final/fruit_math_targeted/controlled_random/"
    "lov_7b_intro_first50_controlled_random_50beta/0/attention_progression",
)

TARGETED_REQUEST_JSONL = os.path.join(REPO_ROOT, "data/automated_gemini_acc_evals/frma_boost_lov/request.jsonl")
TARGETED_EVAL_CSV = os.path.join(REPO_ROOT, "data/automated_gemini_acc_evals/frma_boost_lov/final_evaluation.csv")

OUT_CSV = os.path.join(REPO_ROOT, "data/csv/frma_random_boost_samples/lov_7b_self_graded_accuracy.csv")

TOKEN_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")
DENIAL_PHRASES = ["no fruit", "not applicable", "not specified", "cannot determine", "not present", "not identifiable"]
NUMBER_RE = re.compile(r"-?\$?\s?(\d[\d,]*\.?\d*)")


def load_npz_text(npz_dir, filename):
    path = os.path.join(npz_dir, filename)
    if not os.path.exists(path):
        return None
    npz = np.load(path, allow_pickle=True)
    return "".join(str(t) for t in npz["generated_tokens"])


def parse_targeted_texts():
    result = {}
    with open(TARGETED_REQUEST_JSONL) as f:
        for line in f:
            req = json.loads(line)
            filename = req["filename"]
            text = req["request"]["contents"][0]["parts"][0]["text"]
            start = text.index("MODEL RESPONSE ---") + len("MODEL RESPONSE ---")
            end = text.index("--- NORMALIZATION")
            block = text[start:end].strip()
            tokens = TOKEN_RE.findall(block)
            result[filename] = "".join(tokens)
    return result


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


def load_vanilla_sample_list():
    """(filename, gt_fruit, gt_math) triples for every sample in the OTAT
    directory, derived from the source CSV by the same __id-N row-index
    convention run_analysis.py uses (see utils/generic_utils.
    get_filename_for_experiment) - no Gemini file needed."""
    source = pd.read_csv(SOURCE_CSV)
    id_re = re.compile(r"__id-(\d+)\.npz$")
    rows = []
    for fname in sorted(os.listdir(VANILLA_NPZ_DIR)):
        if not fname.endswith(".npz"):
            continue
        match = id_re.search(fname)
        if not match:
            continue
        idx = int(match.group(1))
        src_row = source.iloc[idx]
        rows.append({"filename": fname, "gt_fruit": src_row["fruit_category"], "gt_math": src_row["math_answer"]})
    return pd.DataFrame(rows)


def evaluate_run(name, eval_df, text_lookup):
    rows = []
    for _, row in eval_df.iterrows():
        filename = row["filename"]
        text = text_lookup(filename)
        if text is None:
            continue
        fruit_ok, fruit_clause = grade_fruit(text, row["gt_fruit"])
        math_ok, predicted_math = grade_math(text, row["gt_math"])
        rows.append(
            {
                "filename": filename,
                "gt_fruit": row["gt_fruit"],
                "gt_math": row["gt_math"],
                "fruit_clause": fruit_clause,
                "self_fruit_correct": fruit_ok,
                "self_predicted_math": predicted_math,
                "self_math_correct": math_ok,
            }
        )
    df = pd.DataFrame(rows)
    df["run"] = name
    return df


def main():
    vanilla_eval = load_vanilla_sample_list()
    controlled_eval = pd.read_csv(CONTROLLED_EVAL_CSV)
    targeted_eval = pd.read_csv(TARGETED_EVAL_CSV)
    targeted_texts = parse_targeted_texts()

    vanilla_df = evaluate_run("vanilla", vanilla_eval, lambda fn: load_npz_text(VANILLA_NPZ_DIR, fn))
    controlled_df = evaluate_run("controlled_random", controlled_eval, lambda fn: load_npz_text(CONTROLLED_NPZ_DIR, fn))
    targeted_df = evaluate_run("targeted", targeted_eval, lambda fn: targeted_texts.get(fn))

    all_df = pd.concat([vanilla_df, controlled_df, targeted_df], ignore_index=True)
    all_df.to_csv(OUT_CSV, index=False)

    print("=== Self-graded accuracy (no Gemini) ===\n")
    print(f"{'run':<20}{'n':>6}{'fruit_acc':>12}{'math_acc':>12}")
    for name, df in [("vanilla", vanilla_df), ("targeted", targeted_df), ("controlled_random", controlled_df)]:
        n = len(df)
        fruit_acc = df["self_fruit_correct"].mean()
        math_acc = df["self_math_correct"].mean()
        print(f"{name:<20}{n:>6}{fruit_acc:>12.4f}{math_acc:>12.4f}")

    print(f"\nWrote per-sample grading -> {OUT_CSV}")


if __name__ == "__main__":
    main()
