"""
Scores the CSV produced by qwen35vl_fruit_math_inference.py (or any results CSV
with the same schema: fruit_category, math_answer, generated_text) and reports
separate fruit-identification and math-solving accuracy for the Fruit Math task.

Usage:
    python evaluate_fruit_math_accuracy.py --results_csv qwen35vl_fruit_math_results.csv
"""
import argparse
import re

import pandas as pd

# Matches: "... fruit ... is [a/an] <fruit_name>." (per the fruit_math_prompt instruction format)
FRUIT_PATTERN = re.compile(r"fruit[^.]*?\bis\s+(?:an?\s+)?([A-Za-z][A-Za-z\-]*)", re.IGNORECASE)
# Matches: "... (answer|math puzzle/problem) ... is <number>"
MATH_PATTERN = re.compile(r"(?:math (?:puzzle|problem)|answer)[^.\d\-]*?is\s+(-?\$?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")


def extract_fruit(generated_text):
    match = FRUIT_PATTERN.search(generated_text)
    if not match:
        return None
    return match.group(1).strip().strip(".,").lower()


def extract_math_answer(generated_text):
    match = MATH_PATTERN.search(generated_text)
    raw = match.group(1) if match else None
    if raw is None:
        # Fallback: assume the last standalone number mentioned is the final answer.
        numbers = NUMBER_PATTERN.findall(generated_text)
        raw = numbers[-1] if numbers else None
    if raw is None:
        return None
    raw = raw.replace("$", "").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def is_fruit_correct(generated_text, gold_fruit, predicted_fruit):
    gold_norm = str(gold_fruit).strip().lower()
    if predicted_fruit is not None and predicted_fruit == gold_norm:
        return True
    # Fallback: the model may not have followed the exact response format -
    # credit it if the gold fruit name is mentioned anywhere in the response.
    return gold_norm in generated_text.lower()


def is_math_correct(predicted_answer, gold_answer):
    if predicted_answer is None:
        return False
    try:
        return abs(float(predicted_answer) - float(gold_answer)) < 1e-6
    except (TypeError, ValueError):
        return False


def main(args):
    df = pd.read_csv(args.results_csv)
    generated = df["generated_text"].fillna("")

    df["predicted_fruit"] = generated.apply(extract_fruit)
    df["predicted_math_answer"] = generated.apply(extract_math_answer)

    df["fruit_correct"] = [
        is_fruit_correct(text, gold, pred)
        for text, gold, pred in zip(generated, df["fruit_category"], df["predicted_fruit"])
    ]
    df["math_correct"] = [
        is_math_correct(pred, gold)
        for pred, gold in zip(df["predicted_math_answer"], df["math_answer"])
    ]

    total = len(df)
    fruit_acc = df["fruit_correct"].mean()
    math_acc = df["math_correct"].mean()
    both_acc = (df["fruit_correct"] & df["math_correct"]).mean()

    fruit_no_answer = df["predicted_fruit"].isna()
    math_no_answer = df["predicted_math_answer"].isna()
    both_no_answer = (fruit_no_answer & math_no_answer).sum()

    print(f"Samples evaluated: {total}")
    print(f"Fruit accuracy:    {fruit_acc:.2%}  (unparseable: {fruit_no_answer.sum()})")
    print(f"Math accuracy:     {math_acc:.2%}  (unparseable: {math_no_answer.sum()})")
    print(f"Both correct:      {both_acc:.2%}")
    print(f"No fruit answer:   {fruit_no_answer.sum()} ({fruit_no_answer.mean():.2%})")
    print(f"No math answer:    {math_no_answer.sum()} ({math_no_answer.mean():.2%})")
    print(f"No answer at all:  {both_no_answer} ({both_no_answer / total:.2%})")

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Per-sample scored results written to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute fruit/math accuracy for Fruit Math task inference results.")
    parser.add_argument("--results_csv", type=str, default="qwen35vl_fruit_math_results.csv")
    parser.add_argument("--output_csv", type=str, default="qwen35vl_fruit_math_scored.csv")
    args = parser.parse_args()
    main(args)
