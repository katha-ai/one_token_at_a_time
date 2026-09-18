"""
mafr_flip_pos_tag_bar_plot.py

Parametrized version of gemma4_12b_pos_tag_bar_plot.py: builds the per-tag attention CSV
(same schema/method as data/bar_plot_csv/FrMaSc/*.csv, via
data_plotting/make_bar_plot_csv_oidScaled.ipynb's extract_pos_tag_stats_to_csv, copied
verbatim) and renders the grouped normalized-attention bar plot (via
data_plotting/bar_plots_scaledOID.ipynb's plot_grouped_normalized_attention, copied
verbatim) for one Ma-Fr ("flip", math_fruit_prompt) fruit_math run, against the global
means produced by fruit_math_pipeline.py.

Usage:
    python useful_helpers/mafr_flip_pos_tag_bar_plot.py --model qvl_7b \
        --npz_root $OTAT_RUN_ROOT/otat/FrMaSc/qvl_7b_flip \
        --json_root $OTAT_RUN_ROOT/otat/FrMaSc/qvl_7b_flip/qvl_7b_mafr_flip_tagging_results/split_jsons

Writes:
    data/bar_plot_csv/MaFrSc_Flip/<model>.csv
    data/plots/bar_plots/MaFrSc_Flip/<model>.png (+ .pdf)
(override with --out_csv / --out_png / --out_pdf)
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_TO_PROCESS = [
    "currently_generating_token__attends_to__image",
    "currently_generating_token__attends_to__text",
    "currently_generating_token__attends_to__instruction",
    "currently_generating_token__attends_to__previously_generating_tokens",
]


# ─── verbatim from data_plotting/make_bar_plot_csv_oidScaled.ipynb ──────────

def _process_sample_padded(sample_basename, layer_folders, npz_root, json_root, chunks_to_process):
    out = {}
    json_path = os.path.join(json_root, f"{sample_basename}.json")
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            tags_json = json.load(f)
    except Exception:
        return {}
    if not isinstance(tags_json, list) or len(tags_json) == 0:
        return {}
    token_tags = [tok.get('tag', 'OTHER') for tok in tags_json]

    for chunk in chunks_to_process:
        layer_arrays = []
        for layer in layer_folders:
            npz_path = os.path.join(npz_root, layer, "attention_progression", f"{sample_basename}.npz")
            if not os.path.exists(npz_path):
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception:
                continue
            if chunk not in data:
                continue
            arr = np.array(data[chunk], dtype=object)
            arr = np.where(arr == None, np.nan, arr).astype(float)
            layer_arrays.append(arr)

        if not layer_arrays:
            continue

        max_len = max(a.shape[0] for a in layer_arrays)
        if max_len == 0:
            continue

        padded = np.full((len(layer_arrays), max_len), np.nan, dtype=float)
        for i, a in enumerate(layer_arrays):
            l = a.shape[0]
            padded[i, :l] = a

        with np.errstate(all='ignore'):
            per_token = np.nanmean(padded, axis=0)

        eff_seq_len = min(len(token_tags), per_token.shape[0])
        if eff_seq_len == 0:
            continue

        tags_slice = token_tags[:eff_seq_len]
        tag_to_inds = {}
        for idx, t in enumerate(tags_slice):
            tag_to_inds.setdefault(t, []).append(idx)

        sample_tag_means = {}
        for tag, inds in tag_to_inds.items():
            vals = per_token[inds]
            if np.all(np.isnan(vals)):
                continue
            sample_tag_means[tag] = float(np.nanmean(vals))

        if sample_tag_means:
            out[chunk] = sample_tag_means

    return out


def extract_pos_tag_stats_to_csv(npz_root, json_root, chunks_to_process, output_csv_path,
                                  tags_to_include=None, layer_folders=None, sample_list=None,
                                  max_samples=None, max_workers=None, baseline_json_path=None,
                                  subtract_baseline=False, verbose=False):
    baseline = {}
    if baseline_json_path is not None:
        if not os.path.exists(baseline_json_path):
            raise FileNotFoundError(f"Baseline JSON not found: {baseline_json_path}")
        with open(baseline_json_path, 'r') as f:
            baseline = json.load(f)
        baseline = {k: float(v) for k, v in baseline.items()}

    if layer_folders is None:
        cand = [d for d in os.listdir(npz_root) if os.path.isdir(os.path.join(npz_root, d))]
        layer_folders = natsorted(cand)
    else:
        layer_folders = list(layer_folders)
    if not layer_folders:
        raise ValueError("No layer folders found in npz_root.")

    if sample_list is None:
        sample_list = []
        for layer in layer_folders:
            att_dir = os.path.join(npz_root, layer, "attention_progression")
            if os.path.isdir(att_dir):
                files = [f for f in os.listdir(att_dir) if f.endswith('.npz')]
                sample_list = natsorted([os.path.splitext(f)[0] for f in files])
                break
    if max_samples is not None:
        sample_list = sample_list[:max_samples]
    if not sample_list:
        raise ValueError("No samples found.")

    if max_workers is None:
        try:
            import multiprocessing
            max_workers = min(8, multiprocessing.cpu_count() or 4)
        except Exception:
            max_workers = 4

    worker = partial(_process_sample_padded, layer_folders=layer_folders, npz_root=npz_root,
                      json_root=json_root, chunks_to_process=chunks_to_process)

    collector = {chunk: defaultdict(list) for chunk in chunks_to_process}

    if verbose:
        print(f"Processing {len(sample_list)} samples with {max_workers} workers...")
        print(f"Target file: {output_csv_path}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, sb): sb for sb in sample_list}
        for fut in as_completed(futures):
            sb = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                if verbose:
                    print(f"[error] sample {sb} -> {e}")
                continue
            if not res:
                continue
            for chunk, tagmap in res.items():
                for tag, val in tagmap.items():
                    if np.isnan(val):
                        continue
                    collector[chunk][tag].append(float(val))

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows_written = 0
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['chunk', 'tag', 'mean', 'std', 'median', 'count']
        if subtract_baseline:
            header.append('mean_baseline_subtracted')
        writer.writerow(header)

        for chunk in chunks_to_process:
            available_tags = ["SOG", "FRUIT_INTRO", "FRUIT_CONCEPT", "FIRST_HANDOFF", "OTHER_HANDOFF",
                               "MATH_INTRO", "MATH_ANSWER", "POSTAMBLE"]
            for tag in available_tags:
                if tags_to_include is not None and tag not in tags_to_include:
                    continue
                vals = collector[chunk][tag]
                arr = np.array(vals, dtype=float)
                if arr.size == 0:
                    continue
                mean_val = float(np.nanmean(arr))
                std_val = float(np.nanstd(arr)) if arr.size > 1 else 0.0
                median_val = float(np.nanmedian(arr))
                count_val = int(np.count_nonzero(~np.isnan(arr)))
                row = [chunk, tag, mean_val, std_val, median_val, count_val]
                if subtract_baseline:
                    adjusted_mean = mean_val - baseline[tag] if tag in baseline else mean_val
                    row.append(adjusted_mean)
                writer.writerow(row)
                rows_written += 1

    if verbose:
        print(f"Done. Wrote {rows_written} rows to {output_csv_path}")

    return collector


# ─── verbatim from data_plotting/bar_plots_scaledOID.ipynb ──────────────────

def plot_grouped_normalized_attention(
    per_tag_csv, global_csv, tag_order=None, tag_rename=None, colors=None, chunk_to_stack=None,
    xlabel="", ylabel="Mean attention − global mean", title="", figsize=(6.5, 3.5), bar_width=0.18,
    linewidth=0.6, tick_fontsize=13, label_fontsize=12, title_fontsize=10, plot_std=False,
    show_spines=False, legend_loc="upper right", legend_fontsize=12, legend_ncol=2, legend_frame=False,
    show_top_xticks=False, show_trend_lines=False, merge_tags=None,
):
    if chunk_to_stack is None:
        chunk_to_stack = {
            "currently_generating_token__attends_to__image": "Image",
            "currently_generating_token__attends_to__text": "Text",
            "currently_generating_token__attends_to__instruction": "Instruction",
            "currently_generating_token__attends_to__previously_generating_tokens": "Previous",
        }

    STACKS = list(dict.fromkeys(chunk_to_stack.values()))

    if colors is None:
        raise ValueError("You must provide `colors` explicitly.")
    missing_colors = set(STACKS) - set(colors)
    if missing_colors:
        raise ValueError(f"Missing colors for stacks: {missing_colors}")

    df = pd.read_csv(per_tag_csv)
    global_df = pd.read_csv(global_csv)

    df = df[df["chunk"].isin(chunk_to_stack)]
    df["stack"] = df["chunk"].map(chunk_to_stack)

    mean_pivot = df.pivot_table(index="tag", columns="stack", values="mean", aggfunc="mean").reset_index()
    std_pivot = df.pivot_table(index="tag", columns="stack", values="std", aggfunc="mean").reset_index()

    global_mean = dict(zip(global_df["Metric"], global_df["Mean"]))
    for chunk, stack in chunk_to_stack.items():
        if chunk not in global_mean:
            raise ValueError(f"Missing global mean for chunk: {chunk}")
        mean_pivot[stack] -= global_mean[chunk]

    if merge_tags:
        for keep_tag, remove_tag in merge_tags:
            if (keep_tag in mean_pivot['tag'].values) and (remove_tag in mean_pivot['tag'].values):
                keep_vals = mean_pivot.loc[mean_pivot['tag'] == keep_tag, STACKS].values
                remove_vals = mean_pivot.loc[mean_pivot['tag'] == remove_tag, STACKS].values
                mean_pivot.loc[mean_pivot['tag'] == keep_tag, STACKS] = (keep_vals + remove_vals) / 2.0

                keep_std = std_pivot.loc[std_pivot['tag'] == keep_tag, STACKS].values
                remove_std = std_pivot.loc[std_pivot['tag'] == remove_tag, STACKS].values
                std_pivot.loc[std_pivot['tag'] == keep_tag, STACKS] = (keep_std + remove_std) / 2.0

                mean_pivot = mean_pivot[mean_pivot['tag'] != remove_tag]
                std_pivot = std_pivot[std_pivot['tag'] != remove_tag]
            else:
                print(f"Warning: Could not merge {remove_tag} into {keep_tag}.")

    if tag_order is not None:
        available_tags = set(mean_pivot["tag"].unique())
        filtered_order = [t for t in tag_order if t in available_tags]
        mean_pivot["tag"] = pd.Categorical(mean_pivot["tag"], filtered_order, ordered=True)
        std_pivot["tag"] = pd.Categorical(std_pivot["tag"], filtered_order, ordered=True)
        mean_pivot = mean_pivot.sort_values("tag")
        std_pivot = std_pivot.sort_values("tag")
        mean_pivot = mean_pivot.dropna(subset=["tag"])
        std_pivot = std_pivot.dropna(subset=["tag"])

    x = np.arange(len(mean_pivot))
    fig, ax = plt.subplots(figsize=figsize)

    for i, stack in enumerate(STACKS):
        y = mean_pivot[stack].values
        yerr = std_pivot[stack].values if plot_std else None
        ax.bar(
            x + i * bar_width, y, bar_width, color=colors[stack], linewidth=linewidth, label=stack,
            yerr=yerr,
            error_kw=dict(elinewidth=0.6, capsize=1.5, capthick=0.6) if plot_std else None,
        )

    if show_trend_lines:
        for i, stack in enumerate(STACKS):
            y = mean_pivot[stack].values
            x_centers = x + i * bar_width
            ax.plot(x_centers, y, color=colors[stack], linewidth=0.1, alpha=0.9, zorder=3)

    ax.axhline(0, linewidth=0.6)

    raw_xticks = mean_pivot["tag"].astype(str).tolist()
    final_xticks = []
    for t in raw_xticks:
        label = tag_rename.get(t, t) if tag_rename else t
        label = label.replace(" ", "\n")
        final_xticks.append(label)

    xtick_pos = x + bar_width * (len(STACKS) - 1) / 2
    ax.set_xticks(xtick_pos, final_xticks, rotation=0, ha="center", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    ax_top = None
    if show_top_xticks:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(xtick_pos)
        ax_top.set_xticklabels(final_xticks, rotation=0, ha="center", fontsize=tick_fontsize)
        ax_top.tick_params(axis="x", length=0)

    ax.legend(loc=legend_loc, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frame,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)

    if not show_spines:
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
            if ax_top is not None:
                ax_top.spines[spine].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        if ax_top is not None:
            ax_top.tick_params(axis="x", length=0)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Ma-Fr (flip) per-tag attention bar plot")
    parser.add_argument("--model", required=True, help="Model key used in output filenames, e.g. qvl_7b")
    parser.add_argument("--npz_root", required=True, help="save_dir of the flip run (layer subfolders live here)")
    parser.add_argument("--json_root", required=True, help="Directory of per-sample POS-tag JSONs (tagging split_jsons)")
    parser.add_argument("--global_csv", default=None, help="Global means CSV; default: <npz_root>/global_means/FrMaSc__<model>_flip.csv")
    parser.add_argument("--out_csv", default=None, help="Default: data/bar_plot_csv/MaFrSc_Flip/<model>.csv")
    parser.add_argument("--out_png", default=None, help="Default: data/plots/bar_plots/MaFrSc_Flip/<model>.png")
    parser.add_argument("--out_pdf", default=None, help="Default: data/plots/bar_plots/MaFrSc_Flip/<model>.pdf")
    args = parser.parse_args()

    npz_root = os.path.normpath(args.npz_root)
    parts = npz_root.split(os.sep)
    dataset_name, model_dir_name = parts[-2], parts[-1]

    global_csv = args.global_csv or os.path.join(npz_root, "global_means", f"{dataset_name}__{model_dir_name}.csv")
    out_csv = args.out_csv or os.path.join(REPO_ROOT, "data/bar_plot_csv/MaFrSc_Flip", f"{args.model}.csv")
    out_png = args.out_png or os.path.join(REPO_ROOT, "data/plots/bar_plots/MaFrSc_Flip", f"{args.model}.png")
    out_pdf = args.out_pdf or os.path.join(REPO_ROOT, "data/plots/bar_plots/MaFrSc_Flip", f"{args.model}.pdf")

    if not os.path.exists(global_csv):
        raise FileNotFoundError(
            f"Global means CSV not found: {global_csv}. Run fruit_math_pipeline.py first "
            "(it calls compute_attention_global_means, which writes this file)."
        )

    print(f"Building per-tag CSV -> {out_csv}")
    extract_pos_tag_stats_to_csv(
        npz_root, args.json_root, CHUNKS_TO_PROCESS, out_csv,
        tags_to_include=None, verbose=True,
    )

    colors = {
        "Image": "#FFA500",
        "Text": "#267E59",
        "Instruction": "#D25B5B",
        "Previous": "#C01BA7",
    }
    tag_rename = {
        "SOG": "SOG",
        "FRUIT_INTRO": "Fruit Intro",
        "FRUIT_CONCEPT": "Fruit Concept",
        "FIRST_HANDOFF": "First-HO",
        "OTHER_HANDOFF": "Handoff",
        "MATH_INTRO": "Math Intro",
        "MATH_ANSWER": "Math Answer",
        "POSTAMBLE": "EOG",
    }
    # Ma-Fr order (math answered first, fruit named second) - unlike the vanilla
    # Fr-Ma script this is based on, MATH_INTRO/MATH_ANSWER come before
    # FRUIT_INTRO/FRUIT_CONCEPT here. merge_tags below is unchanged: FIRST_HANDOFF
    # is still (per the tagging prompt) "first token after FRUIT_CONCEPT", so for
    # this reversed order it's the tail token right before POSTAMBLE, not a token
    # between MATH_ANSWER and FRUIT_INTRO - confirmed against actual per-sample tag
    # JSONs (e.g. qvl_7b_flip id-493: ...MATH_ANSWER, OTHER_HANDOFF, FRUIT_INTRO,
    # FRUIT_CONCEPT, FIRST_HANDOFF) before relying on this.
    tag_order = ["SOG", "MATH_INTRO", "MATH_ANSWER", "FIRST_HANDOFF", "OTHER_HANDOFF",
                 "FRUIT_INTRO", "FRUIT_CONCEPT", "POSTAMBLE"]

    fig = plot_grouped_normalized_attention(
        per_tag_csv=out_csv,
        global_csv=global_csv,
        tag_order=tag_order,
        tag_rename=tag_rename,
        ylabel="Normalized Attention",
        title="",
        legend_loc="upper left",
        legend_ncol=4,
        plot_std=True,
        show_spines=False,
        colors=colors,
        merge_tags=[("FRUIT_CONCEPT", "FIRST_HANDOFF")],
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Wrote plot -> {out_png}")
    print(f"Wrote plot -> {out_pdf}")


if __name__ == "__main__":
    main()
