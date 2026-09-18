import os
import json
import numpy as np
from collections import defaultdict
from natsort import natsorted
import csv

def compute_attention_global_means(
    base_paths,
    output_dir,
    skip_dirs=('logs',),
):
    """
    Args:
        base_paths (list[str]): list of base experiment directories
        output_dir (str): directory where jsons will be written
        skip_dirs (tuple[str]): directory names to skip (default: ('logs',))
    """

    os.makedirs(output_dir, exist_ok=True)

    for base_path in base_paths:
        sums = defaultdict(float)
        counts = defaultdict(int)

        for layer_idx in natsorted(os.listdir(base_path)):
            if layer_idx in skip_dirs:
                continue

            layer_dir = os.path.join(base_path, layer_idx, 'attention_progression')
            if not os.path.isdir(layer_dir):
                continue

            for sample_file in natsorted(os.listdir(layer_dir)):
                file_path = os.path.join(layer_dir, sample_file)

                try:
                    data = np.load(file_path, allow_pickle=True)
                except Exception:
                    continue

                for key, value in data.items():
                    if key == 'generated_tokens':
                        continue

                    arr = np.array(value, dtype=object)
                    arr = np.where(arr == None, np.nan, arr)

                    try:
                        float_arr = arr.astype(float)
                    except Exception:
                        float_arr = np.array(
                            [np.nan if x is None else float(x) for x in arr],
                            dtype=float
                        )

                    sample_mean = np.nanmean(float_arr)
                    if np.isnan(sample_mean):
                        continue

                    sums[key] += float(sample_mean)
                    counts[key] += 1

        global_mean = {
            k: sums[k] / counts[k]
            for k in sums
            if counts[k] > 0
        }

        # ---- filename logic ----
        parts = os.path.normpath(base_path).split(os.sep)
        if len(parts) < 2:
            raise ValueError(f"Path too shallow to infer name: {base_path}")

        dataset_name = parts[-2]
        model_name = parts[-1]
        
        json_name = f"{dataset_name}__{model_name}.json"
        json_out_path = os.path.join(output_dir, json_name)
        with open(json_out_path, 'w') as f:
            json.dump(global_mean, f, indent=2)

        csv_name = f"{dataset_name}__{model_name}.csv"
        csv_out_path = os.path.join(output_dir, csv_name)
        with open(csv_out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Mean'])  # header
            for k, v in sorted(global_mean.items()):
                writer.writerow([k, v])

        print(f"[OK] Saved: {json_out_path} and {csv_out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute per-chunk global attention means across one or more run_analysis.py output directories.")
    parser.add_argument("base_paths", nargs="+", help="One or more run save_dir paths (each containing per-layer attention_progression/*.npz).")
    parser.add_argument("--output_dir", required=True, help="Where to write the {dataset}__{model}.json / .csv global-mean files.")
    args = parser.parse_args()

    compute_attention_global_means(
        base_paths=args.base_paths,
        output_dir=args.output_dir
    )