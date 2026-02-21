import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ============ Style Configuration ============
STYLE_CFG = {
    "figsize": (6, 6),
    "dpi": 800,
    "font.size": 16,
    "line_width": 2,
    "colors": ["#8A2BE2", "#ff2121"],  # colors for micro and macro averages
}

def setup_plot_style():
    """Set global matplotlib parameters for high-quality output."""
    plt.rcParams.update({
        'font.size': STYLE_CFG["font.size"],
        'axes.labelsize': STYLE_CFG["font.size"],
        'axes.titlesize': STYLE_CFG["font.size"],
        'xtick.labelsize': STYLE_CFG["font.size"] - 1,
        'ytick.labelsize': STYLE_CFG["font.size"] - 1,
        'legend.fontsize': STYLE_CFG["font.size"] - 1,
    })

def main(input_path, output_dir, classes):
    """
    Main logic to generate and save ROC/PR curves.
    """
    # Initialization
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found. Please check the path.")
        return

    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()
    curve_label_map = {"micro": "Micro-avg", "macro": "Macro-avg"}

    # --- Data Loading ---
    df = pd.read_csv(input_path)
    y_true = df["GT"].values
    # Column names must match your CSV header
    y_score = df[["prob_class1", "prob_class2", "prob_class3"]].values

    # Binarize labels for multi-class metric calculation
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = y_bin.shape[1]

    # ============ ROC Calculation ============
    fpr, tpr, roc_auc = {}, {}, {}

    # Calculate micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Calculate macro-average ROC
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], y_score[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        f, t, _ = roc_curve(y_bin[:, i], y_score[:, i])
        mean_tpr += np.interp(all_fpr, f, t)

    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # --- Draw ROC Plot ---
    fig_roc, ax_roc = plt.subplots(figsize=STYLE_CFG["figsize"], dpi=STYLE_CFG["dpi"])
    for curve, color in zip(["micro", "macro"], STYLE_CFG["colors"]):
        ax_roc.plot(fpr[curve], tpr[curve],
                    label=f"{curve_label_map[curve]} (AUC={roc_auc[curve]:.3f})",
                    color=color, linestyle="-", linewidth=STYLE_CFG["line_width"])

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(True, linestyle="--", alpha=0.4)
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_analysis.svg"), format="svg", bbox_inches="tight")
    plt.close()

    # ============ PR Calculation ============
    prec, rec, ap = {}, {}, {}

    # Calculate micro-average Precision-Recall
    prec["micro"], rec["micro"], _ = precision_recall_curve(y_bin.ravel(), y_score.ravel())
    ap["micro"] = average_precision_score(y_bin, y_score, average="micro")

    # Calculate macro-average Precision-Recall
    all_rec = np.linspace(0, 1, 100)
    mean_prec = np.zeros_like(all_rec)
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        mean_prec += np.interp(all_rec, r[::-1], p[::-1])

    mean_prec /= n_classes
    rec["macro"], prec["macro"] = all_rec, mean_prec
    ap["macro"] = average_precision_score(y_bin, y_score, average="macro")

    # --- Draw PR Plot ---
    fig_pr, ax_pr = plt.subplots(figsize=STYLE_CFG["figsize"], dpi=STYLE_CFG["dpi"])
    for curve, color in zip(["micro", "macro"], STYLE_CFG["colors"]):
        ax_pr.plot(rec[curve], prec[curve],
                   label=f"{curve_label_map[curve]} (AP={ap[curve]:.3f})",
                   color=color, linestyle="-", linewidth=STYLE_CFG["line_width"])

    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, linestyle="--", alpha=0.4)
    ax_pr.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_analysis.svg"), format="svg", bbox_inches="tight")
    plt.close()

    print(f"Success! Curves saved in the '{output_dir}' directory.")


if __name__ == "__main__":
    # ============ SETTINGS ============
    # Update these values as needed
    INPUT_FILE = "avg.csv"
    OUTPUT_FOLDER = "plots"
    CLASS_LABELS = [1, 2, 3]

    # Run the main process
    main(
        input_path=INPUT_FILE,
        output_dir=OUTPUT_FOLDER,
        classes=CLASS_LABELS
    )