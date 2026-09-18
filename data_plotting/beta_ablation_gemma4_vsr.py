"""
Line plot of VSR VT-Accuracy vs. multiplicative boost factor (beta) for gemma-4-12b,
in the style of data/plots/beta_analysis.png (the equivalent LOV plot).

Data sources (targeted image+text boosting on IMG_SPATIAL_RELATION/TXT_SPATIAL_RELATION,
same config shape as configs/boost_factor_ablate/lov/lov_beta_*.yaml):
  boost_factor_{30,50,100,200,300,400} ->
    <RUN_ROOT>/multiplicative_boost_final_ablate/vsr_boost/gemma4_12b/
    ablate_snesitivity/gemma4/boost_factor_{beta}.0/gem_vsr_beta{beta}_final_results.csv
  vanilla (no boosting) baseline -> fixed at 72.1 (VT-Accuracy)
  (671 samples; the beta-ablation runs use a fixed 102-sample subset of the same
  pool, except beta=50 which only has 38 graded samples -- a subset of the 102.)

gamma(beta) = beta / z_i,  z_i = (1 - p_i) + beta * p_i,  p_i = 0.1250 for gemma-4-12b.
"""
import csv

import matplotlib.pyplot as plt

P_I = 0.1250
BETAS = [30.0, 50.0, 100.0, 200.0, 300.0, 400.0]

ABLATE_TEMPLATE = (
    "<RUN_ROOT>/multiplicative_boost_final_ablate/vsr_boost/gemma4_12b/"
    "ablate_snesitivity/gemma4/boost_factor_{b}.0/gem_vsr_beta{b_int}_final_results.csv"
)
BASELINE_ACC = 72.1

OUT_DIR = "<REPO_ROOT>/data/plots"

LINE_COLOR = "#6b9e78"
BASELINE_COLOR = "#808080"


def load_ids_and_vt_accuracy(path, id_filter=None):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if id_filter is not None:
        rows = [r for r in rows if r["Sample_ID"] in id_filter]
    n = len(rows)
    vt_correct = sum(
        1 for r in rows if r["Visual_Correct"] == "True" and r["Text_Correct"] == "True"
    )
    ids = {r["Sample_ID"] for r in rows}
    return ids, n, 100.0 * vt_correct / n


def gamma_for_beta(beta, p_i=P_I):
    z_i = (1 - p_i) + beta * p_i
    return beta / z_i


def main():
    accuracies = []
    all_ids = None
    for beta in BETAS:
        b_int = int(beta)
        path = ABLATE_TEMPLATE.format(b=b_int, b_int=b_int)
        ids, n, acc = load_ids_and_vt_accuracy(path)
        accuracies.append(acc)
        print(f"beta={b_int:>4}  n={n:>4}  VT-Acc={acc:.2f}%  gamma={gamma_for_beta(beta):.2f}")
        if b_int != 50:  # beta=50 is a smaller (38-sample) subset of the shared 102-sample pool
            all_ids = ids if all_ids is None else (all_ids & ids)

    baseline_acc = BASELINE_ACC
    print(f"vanilla (no boosting), fixed baseline: VT-Acc={baseline_acc:.2f}%")

    gammas = [gamma_for_beta(b) for b in BETAS]

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(baseline_acc, color=BASELINE_COLOR, linestyle="--", linewidth=2.5,
               label=r"no boosting ($\beta=1$)", zorder=2)
    ax.plot(BETAS, accuracies, color=LINE_COLOR, marker="s", markersize=10,
            linewidth=3, label="boosting", zorder=3)

    for beta, acc, gamma in zip(BETAS, accuracies, gammas):
        ax.annotate(rf"$\gamma={gamma:.1f}$", xy=(beta, acc), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=13, color="black")

    ax.set_xlabel(r"Boost Factor ($\beta$)", fontsize=15)
    ax.set_ylabel("VT-Accuracy", fontsize=15)
    ax.set_title("Gemma-4", fontsize=20, color=LINE_COLOR, fontweight="normal")

    ax.set_xticks(BETAS)
    ax.tick_params(axis="both", labelsize=12)

    ymin = min(baseline_acc, min(accuracies)) - 8
    ymax = max(accuracies) + 8
    ax.set_ylim(ymin, ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="center right", fontsize=13, frameon=False)

    fig.tight_layout()

    pdf_path = f"{OUT_DIR}/beta_analysis_gemma4.pdf"
    png_path = f"{OUT_DIR}/beta_analysis_gemma4.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
