"""
Human spot-check of Gemini's two automated-judge roles in this pipeline:
  1. Accuracy grading (FrMaSc fruit_math final_evaluation.csv fruit_correct/math_correct)
  2. POS tagging (otat_tagged_outputs word/tag assignments used to drive boosting/blocking)

Scope: gemma4_12b vanilla runs only, for both roles.

Standalone module: sampling persistence, request.jsonl parsing (extracting the
MODEL RESPONSE Gemini was actually shown), and metrics computation. Run directly to
(re)generate the report:

    python useful_helpers/spot_check_metrics.py
"""
import csv
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Accuracy grading source (FrMaSc, gemma4_12b, vanilla) ---
ACC_EVAL_CSV = REPO_ROOT / "data/automated_gemini_acc_evals/FrMaSc_vanilla_otat_gemma4_12b/final_evaluation.csv"
ACC_REQUEST_JSONL = REPO_ROOT / "data/automated_gemini_acc_evals/FrMaSc_vanilla_otat_gemma4_12b/request.jsonl"
FRUIT_MATH_META_CSV = REPO_ROOT / "data/csv/fruit_math_oid_scaled_1K_withId.csv"
FRUIT_MATH_IMAGE_BASE = REPO_ROOT / "data"  # + meta 'image_path' (already 'images/...')

# --- POS tagging sources: mixed across three tasks (2026-08-28 decision - VSR's 11-tag
# discrepancy scheme was judged too complex to be the sole POS-tag spot check, so the pool
# is now weighted toward fruit_math's simpler 6-tag scheme, with VSR and ChartQA as smaller
# supplementary samples). No gemma4_12b tagging exists for fruit_math/ChartQA, so those two
# use whatever models have tagged output; ChartQA mixes across all four available models. ---
FRMA_POS_DIR = REPO_ROOT / "data/tagged_outputs/otat_tagged_outputs/frma_sc/qvl35_vanilla/qwen_35_vl/fruit_math"
VSR_POS_DIR = REPO_ROOT / "data/tagged_outputs/otat_tagged_outputs/vsr/vanilla/gemma4_12b/vsr"
CHARTQA_MODELS = ["qvl3b", "lov05b", "qvl7b", "lov7b"]
CHARTQA_POS_DIRS = {
    m: REPO_ROOT / f"data/tagged_outputs/otat_tagged_outputs/chartQA/{m}" for m in CHARTQA_MODELS
}

VSR_META_CSV = REPO_ROOT / "data/csv/vsr_filtered_task_gemini.csv"
VSR_IMAGE_BASE = REPO_ROOT / "data/images"  # + meta 'image_path' (already 'vsr_images/...')
CHARTQA_META_CSV = REPO_ROOT / "data/csv/chartqa_300.csv"
CHARTQA_IMAGE_BASE = REPO_ROOT / "data/images/ChartQA"  # + meta 'image_name'

POS_TASK_MIX = [("fruit_math", 30), ("vsr", 10), ("chartqa", 10)]

# --- Where spot-check state/results live (gitignored under data/**) ---
RESULTS_DIR = REPO_ROOT / "data/spot_check_results"
ACC_SAMPLE_JSON = RESULTS_DIR / "accuracy_grading_gemma4_12b_vanilla_sample.json"
ACC_RESULTS_CSV = RESULTS_DIR / "accuracy_grading_gemma4_12b_vanilla_results.csv"
POS_SAMPLE_JSON = RESULTS_DIR / "pos_tagging_mixed_sample.json"
POS_RESULTS_CSV = RESULTS_DIR / "pos_tagging_mixed_results.csv"
REPORT_MD = REPO_ROOT / "gemini_spot_check_report.md"

SAMPLE_SIZE = 50
SEED = 42

ACC_RESULT_FIELDS = [
    "filename", "gt_fruit", "gt_math",
    "gemini_predicted_fruit", "gemini_predicted_math",
    "gemini_fruit_correct", "gemini_math_correct",
    "human_fruit_correct", "human_math_correct",
    "human_note", "reviewed_at",
]
POS_RESULT_FIELDS = [
    "sample_id", "task", "model", "num_words", "human_verdict",
    "flagged_word_indices", "flagged_note", "reviewed_at",
]

TOKEN_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")


def get_or_create_sample(pool, sample_json_path, n=SAMPLE_SIZE, seed=SEED):
    """Deterministic one-time sample, persisted to disk so re-running the dashboard
    (or the report script) always reviews/reports on the same 50 examples."""
    if sample_json_path.exists():
        return json.loads(sample_json_path.read_text())
    rng = random.Random(seed)
    pool = list(pool)
    rng.shuffle(pool)
    sample = pool[:n]
    sample_json_path.parent.mkdir(parents=True, exist_ok=True)
    sample_json_path.write_text(json.dumps(sample, indent=2))
    return sample


def load_acc_eval_rows():
    with open(ACC_EVAL_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_request_response_map(jsonl_path=ACC_REQUEST_JSONL):
    """{filename: model_response_text}, reconstructed from the Gemini grading request's
    embedded '--- MODEL RESPONSE ---' block (the exact text Gemini was actually shown -
    see CLAUDE.md's provenance-tracing guidance for why this beats loading a .npz)."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            req = json.loads(line)
            filename = req["filename"]
            text = req["request"]["contents"][0]["parts"][0]["text"]
            start = text.index("MODEL RESPONSE ---") + len("MODEL RESPONSE ---")
            end = text.index("--- NORMALIZATION")
            block = text[start:end].strip()
            tokens = TOKEN_RE.findall(block)
            out[filename] = "".join(tokens)
    return out


def load_fruit_math_meta():
    with open(FRUIT_MATH_META_CSV, newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def load_vsr_meta():
    with open(VSR_META_CSV, newline="") as f:
        return list(csv.DictReader(f))  # row index (0-based) == sample id


def extract_id(filename):
    m = re.search(r"id-(\d+)", filename)
    return m.group(1) if m else None


_FRMA_INDEX = None
_CHARTQA_INDEX = {}


def _frma_index():
    global _FRMA_INDEX
    if _FRMA_INDEX is None:
        _FRMA_INDEX = {extract_id(p.name): p for p in FRMA_POS_DIR.glob("*.json")}
    return _FRMA_INDEX


def _chartqa_index(model):
    if model not in _CHARTQA_INDEX:
        _CHARTQA_INDEX[model] = {extract_id(p.name): p for p in CHARTQA_POS_DIRS[model].glob("*.json")}
    return _CHARTQA_INDEX[model]


def list_pos_pool(task):
    """[(task, model, id), ...] for every tagged sample available for `task`."""
    if task == "fruit_math":
        return [("fruit_math", "qwen_35_vl", i) for i in _frma_index()]
    if task == "vsr":
        return [("vsr", "gemma4_12b", extract_id(p.name)) for p in VSR_POS_DIR.glob("*.json")]
    if task == "chartqa":
        return [("chartqa", m, i) for m in CHARTQA_MODELS for i in _chartqa_index(m)]
    raise ValueError(f"unknown POS task: {task}")


def make_composite_id(task, model, id_):
    return f"{task}|{model}|{id_}"


def parse_composite_id(cid):
    task, model, id_ = cid.split("|")
    return task, model, id_


def get_or_create_pos_sample(mix=POS_TASK_MIX, seed=SEED):
    """Deterministic one-time mixed sample across POS_TASK_MIX, persisted to POS_SAMPLE_JSON."""
    if POS_SAMPLE_JSON.exists():
        return json.loads(POS_SAMPLE_JSON.read_text())
    rng = random.Random(seed)
    sample = []
    for task, n in mix:
        pool = list_pos_pool(task)
        rng.shuffle(pool)
        sample += [make_composite_id(*item) for item in pool[:n]]
    rng.shuffle(sample)
    POS_SAMPLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    POS_SAMPLE_JSON.write_text(json.dumps(sample, indent=2))
    return sample


def _pos_tag_path(task, model, id_):
    if task == "fruit_math":
        return _frma_index()[id_]
    if task == "vsr":
        return VSR_POS_DIR / f"vsr__image-actual__vsr_prompt_discrepancy_updated__id-{id_}_1.json"
    if task == "chartqa":
        return _chartqa_index(model)[id_]
    raise ValueError(f"unknown POS task: {task}")


def load_pos_tag_words(cid):
    task, model, id_ = parse_composite_id(cid)
    return json.loads(_pos_tag_path(task, model, id_).read_text())


def load_chartqa_meta():
    with open(CHARTQA_META_CSV, newline="") as f:
        return list(csv.DictReader(f))  # row index (0-based) == sample id


def get_pos_context(cid):
    """Image path + display fields for one POS-tag sample, dispatched by task."""
    task, model, id_ = parse_composite_id(cid)
    if task == "fruit_math":
        meta = load_fruit_math_meta()[id_]
        return dict(
            image_path=FRUIT_MATH_IMAGE_BASE / meta["image_path"],
            lines=[("Fruit (GT)", meta["fruit_category"]), ("Math question", meta["math_question"]),
                   ("Math answer (GT)", meta["math_answer"])],
        )
    if task == "vsr":
        meta = load_vsr_meta()[int(id_)]
        return dict(
            image_path=VSR_IMAGE_BASE / meta["image_path"],
            lines=[("Relation", meta["relation"]), ("Label", meta["label"]),
                   ("Text shown to model", meta["extended_caption_modify"])],
        )
    if task == "chartqa":
        meta = load_chartqa_meta()[int(id_)]
        return dict(
            image_path=CHARTQA_IMAGE_BASE / meta["image_name"],
            lines=[("Q1", meta["q1"]), ("A1 (GT)", meta["a1"]), ("Q2", meta["q2"]), ("A2 (GT)", meta["a2"])],
        )
    raise ValueError(f"unknown POS task: {task}")


TAG_COLORS = {
    "IMG_INTRO": "#1f6feb22", "IMG_OBJECT_1": "#1f6feb55", "IMG_SPATIAL_RELATION": "#1f6febaa",
    "IMG_OBJECT_2": "#1f6feb55",
    "TXT_INTRO": "#d2992222", "TXT_OBJECT_1": "#d2992255", "TXT_SPATIAL_RELATION": "#d29922aa",
    "TXT_OBJECT_2": "#d2992255",
    "FRUIT_INTRO": "#1f6feb22", "FRUIT_CONCEPT": "#1f6feb55",
    "MATH_INTRO": "#d2992222", "MATH_ANSWER": "#d2992255",
    "Q1_INTRO": "#1f6feb22", "Q1_ANSWER": "#1f6feb55",
    "Q2_INTRO": "#d2992222", "Q2_ANSWER": "#d2992255",
    "HANDOFF": "#8957e555", "FIRST_HANDOFF": "#8957e555", "OTHER_HANDOFF": "#8957e555",
    "SOG": "#30363d33", "POSTAMBLE": "#30363d33",
}


def tag_color(tag):
    return TAG_COLORS.get(tag, "#6e768133")


def read_results_csv(path, fields):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def upsert_result_row(path, fields, key_field, row):
    rows = read_results_csv(path, fields)
    rows = [r for r in rows if r[key_field] != row[key_field]]
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _as_bool(s):
    return str(s).strip().lower() == "true"


def compute_accuracy_grading_metrics(rows):
    """rows: list of dicts from ACC_RESULTS_CSV.

    human_fruit_correct / human_math_correct record an EXTRACTION-FIDELITY check: does
    gemini_predicted_fruit / gemini_predicted_math correctly reflect what the model's raw
    response text actually said, independent of whether that answer matches ground truth
    (ground-truth correctness is gemini_fruit_correct / gemini_math_correct, Gemini's own
    call, not re-judged here). So the right metric is a straight extraction-accuracy rate,
    not an agreement/kappa comparison against gemini_{field}_correct - those two columns
    answer different questions and aren't a rater pair."""
    out = {}
    for field, pred_key in (("fruit_correct", "gemini_predicted_fruit"), ("math_correct", "gemini_predicted_math")):
        h_key = f"human_{field}"
        sub = [r for r in rows if r.get(h_key, "") != ""]
        if not sub:
            out[field] = None
            continue
        n = len(sub)
        correct = sum(1 for r in sub if _as_bool(r[h_key]))
        failures = [(r["filename"], r[pred_key]) for r in sub if not _as_bool(r[h_key])]
        out[field] = dict(n=n, extraction_correct=correct, extraction_rate=correct / n, failures=failures)
    return out


def _flag_count(r):
    return len([x for x in str(r.get("flagged_word_indices", "")).split(";") if x.strip() != ""])


def _pos_tagging_metrics_for(rows):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(1 for r in rows if r.get("human_verdict") == "correct")
    has_errors = sum(1 for r in rows if r.get("human_verdict") == "has_errors")
    flag_counts = [_flag_count(r) for r in rows if r.get("human_verdict") == "has_errors"]

    total_words = sum(int(r["num_words"]) for r in rows)
    total_flagged_words = sum(_flag_count(r) for r in rows)  # 0 for "correct" rows
    per_sample_pct_wrong = sorted(
        ((r["sample_id"], _flag_count(r) / int(r["num_words"])) for r in rows if r.get("human_verdict") == "has_errors"),
        key=lambda x: -x[1],
    )

    return dict(
        n=n, fully_correct=correct, has_errors=has_errors,
        pct_fully_correct=correct / n,
        avg_flagged_words_per_error_sample=(sum(flag_counts) / len(flag_counts)) if flag_counts else 0.0,
        total_words=total_words, total_flagged_words=total_flagged_words,
        token_error_rate=(total_flagged_words / total_words) if total_words else 0.0,
        per_sample_pct_wrong=per_sample_pct_wrong,
    )


def compute_pos_tagging_metrics(rows):
    """Overall + per-task breakdown (fruit_math / vsr / chartqa mix)."""
    overall = _pos_tagging_metrics_for(rows)
    by_task = {
        task: _pos_tagging_metrics_for([r for r in rows if r.get("task") == task])
        for task in sorted({r.get("task") for r in rows})
    }
    return dict(overall=overall, by_task=by_task)


def write_report():
    acc_rows = read_results_csv(ACC_RESULTS_CSV, ACC_RESULT_FIELDS)
    pos_rows = read_results_csv(POS_RESULTS_CSV, POS_RESULT_FIELDS)
    acc_metrics = compute_accuracy_grading_metrics(acc_rows) if acc_rows else None
    pos_metrics = compute_pos_tagging_metrics(pos_rows) if pos_rows else None

    lines = [
        "# Gemini spot-check: human validation of automated grading/tagging",
        "",
        "Scope: accuracy grading on FrMaSc fruit_math, gemma4_12b vanilla, N=50. POS tagging "
        f"mixed across tasks per {POS_TASK_MIX} (fruit_math/qwen_35_vl, vsr/gemma4_12b, "
        "chartqa/mixed across qvl3b+lov05b+qvl7b+lov7b). Both sampled once (seed=42) and "
        "persisted so repeat reviews/reruns target the same items.",
        "",
        "## 1. Extraction fidelity (predicted_fruit / predicted_math vs. what the model actually said)",
        "",
        "Does Gemini's `predicted_fruit` / `predicted_math` correctly reflect the model's raw response "
        "text? This is independent of whether that answer is correct vs. ground truth (that's Gemini's "
        "own `fruit_correct` / `math_correct` call, not re-judged here).",
        "",
    ]
    if acc_metrics is None:
        lines.append(f"_No reviews recorded yet in `{ACC_RESULTS_CSV.relative_to(REPO_ROOT)}`._")
    else:
        for field, m in acc_metrics.items():
            if m is None:
                lines.append(f"- **{field}**: no reviewed rows yet.")
                continue
            lines.append(
                f"- **{field.replace('_correct', '')}** (n={m['n']}): extraction correct = "
                f"**{m['extraction_rate']:.1%}** ({m['extraction_correct']}/{m['n']})"
            )
            if m["failures"]:
                lines.append("  - flagged extraction failures:")
                for fn, pred in m["failures"]:
                    lines.append(f"    - `{pred}` extracted from `{fn}`")
    lines += ["", "## 2. POS tagging (word/tag assignments)", ""]
    if pos_metrics is None or pos_metrics["overall"] is None:
        lines.append(f"_No reviews recorded yet in `{POS_RESULTS_CSV.relative_to(REPO_ROOT)}`._")
    else:
        o = pos_metrics["overall"]
        lines.append(
            f"- **Overall** (n={o['n']}): fully correct = **{o['pct_fully_correct']:.1%}** "
            f"({o['fully_correct']}/{o['n']}), has errors = {o['has_errors']}/{o['n']}"
        )
        lines.append(
            f"- **Token-level error rate**: **{o['token_error_rate']:.1%}** "
            f"({o['total_flagged_words']} flagged / {o['total_words']} total words reviewed); "
            f"avg flagged words per erroring sample = {o['avg_flagged_words_per_error_sample']:.1f}"
        )
        for task, m in pos_metrics["by_task"].items():
            if m is None:
                continue
            lines.append(
                f"  - **{task}** (n={m['n']}): fully correct = {m['pct_fully_correct']:.1%} "
                f"({m['fully_correct']}/{m['n']}), token error rate = {m['token_error_rate']:.1%} "
                f"({m['total_flagged_words']}/{m['total_words']})"
            )
        if o["per_sample_pct_wrong"]:
            lines.append("- erroring samples, worst first:")
            for sid, pct in o["per_sample_pct_wrong"]:
                lines.append(f"  - `{sid}`: {pct:.0%} of words flagged")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines))
    return REPORT_MD


if __name__ == "__main__":
    path = write_report()
    print(f"Wrote {path}")
    print(path.read_text())
