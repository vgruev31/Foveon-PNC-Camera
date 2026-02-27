"""ROC curve dialog — Sensitivity vs. Specificity for UV images."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QDialog, QVBoxLayout


def _compute_roc(labels: np.ndarray, scores: np.ndarray):
    """Compute ROC from per-file scores.

    Parameters
    ----------
    labels : ndarray, shape (N,)
        Ground truth: 1 = cancerous, 0 = non-cancerous.
    scores : ndarray, shape (N,)
        Per-file score: max threshold (0-100) at which the file is
        classified positive.  At threshold *t*, predict positive
        if ``score >= t``.

    Returns None if there are not both positive and negative samples.
    """
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    thresholds = np.round(np.arange(0, 100.1, 0.1), 1)  # 0.0 … 100.0

    # Vectorised: at threshold t, predict positive if score >= t
    preds = (scores[:, None] >= thresholds[None, :]).astype(int)  # (N, T)
    pos_mask = labels == 1
    neg_mask = labels == 0

    tp_arr = preds[pos_mask].sum(axis=0)
    fp_arr = preds[neg_mask].sum(axis=0)
    tn_arr = n_neg - fp_arr
    fn_arr = n_pos - tp_arr
    tpr = tp_arr / n_pos
    fpr = fp_arr / n_neg

    sort_idx = np.argsort(fpr)
    _trapz = getattr(np, "trapezoid", np.trapz)
    auc = float(abs(_trapz(tpr[sort_idx], fpr[sort_idx])))

    return fpr, tpr, thresholds, auc, tp_arr, tn_arr, fp_arr, fn_arr, n_pos, n_neg


def _print_roc_details(
    group_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    filenames: list[str],
    fpr, tpr, thresholds, auc, tp_arr, tn_arr, fp_arr, fn_arr, n_pos, n_neg,
):
    """Print per-file scores, TP/TN/FP/FN table, and misclassified samples."""
    print(f"\n{'=' * 60}")
    print(f"  ROC — {group_name}  ({n_pos} positive, {n_neg} negative)")
    print(f"{'=' * 60}")

    # Optimal threshold (Youden's J)
    spec_arr = tn_arr / n_neg
    j_scores = tpr + spec_arr - 1.0
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])

    print(f"\nAUC = {auc:.4f}")
    print(f"Optimal threshold = {best_thresh:.1f} / 100  "
          f"(Sens={tpr[best_idx]:.3f}, Spec={spec_arr[best_idx]:.3f})")

    # Per-file score and classification at optimal threshold
    print(f"\n{'Filename':<50s}  {'Label':>5s}  {'Score':>6s}  {'Pred':>5s}  {'Result':>8s}")
    for i in range(len(labels)):
        label_tag = "POS" if labels[i] else "NEG"
        pred = 1 if scores[i] >= best_thresh else 0
        pred_tag = "POS" if pred else "NEG"
        if labels[i] == pred:
            result = "OK"
        elif labels[i] == 1:
            result = "FN"
        else:
            result = "FP"
        print(f"{filenames[i]:<50s}  {label_tag:>5s}  {scores[i]:6.1f}  {pred_tag:>5s}  {result:>8s}")

    # TP / TN / FP / FN table (~20 rows)
    step = max(1, len(thresholds) // 20)
    print(f"\n{'Threshold':>10s}  {'TP':>4s}  {'TN':>4s}  {'FP':>4s}  {'FN':>4s}  {'Sens':>6s}  {'Spec':>6s}")
    for i in range(0, len(thresholds), step):
        t = float(thresholds[i])
        sens = tpr[i]
        spec = spec_arr[i]
        print(f"{t:10.1f}  {tp_arr[i]:4d}  {tn_arr[i]:4d}  {fp_arr[i]:4d}  {fn_arr[i]:4d}  {sens:6.3f}  {spec:6.3f}")

    # Misclassified samples at optimal threshold
    preds_at_opt = (scores >= best_thresh).astype(int)
    n_mis = int((preds_at_opt != labels).sum())
    if n_mis == 0:
        print("\nAll samples correctly classified at optimal threshold.")
    else:
        print(f"\nMisclassified samples ({n_mis}) at threshold {best_thresh:.1f}:")
        for i in range(len(labels)):
            if preds_at_opt[i] == 1 and labels[i] == 0:
                print(f"  FP: {filenames[i]}  label=0, score={scores[i]:.1f}")
            elif preds_at_opt[i] == 0 and labels[i] == 1:
                print(f"  FN: {filenames[i]}  label=1, score={scores[i]:.1f}")


class ROCDialog(QDialog):
    """Display ROC curves (ALL, TUMOR, LN) on one plot."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROC Curve — UV Images")
        self.setMinimumSize(650, 600)

        self._fig = Figure(figsize=(6.5, 6), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self._canvas)
        self.setLayout(layout)

    def plot_roc(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        filenames: list[str],
        tissue_types: list[str],
    ) -> float | None:
        """Compute and plot three ROC curves from per-file scores.

        For each subset (ALL, TUMOR, LN), plots the empirical ROC curve
        and marks the optimal operating point (Youden's J).

        Parameters
        ----------
        labels : ndarray, shape (N,)
            Ground truth: 1 = cancerous, 0 = non-cancerous.
        scores : ndarray, shape (N,)
            Per-file score: max threshold (0-100) at which the file
            is classified positive.
        filenames : list[str]
            Filenames corresponding to each sample.
        tissue_types : list[str]
            Tissue type for each sample ("LN" or "TUMOR").

        Returns
        -------
        float or None
            Optimal threshold value (0-100) from the ALL curve
            (Youden's J), or None if the curve could not be computed.
        """
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)
        tissue_arr = np.array(tissue_types)

        ax = self._ax
        ax.clear()

        subsets = [
            ("ALL",   np.ones(len(labels), dtype=bool), "blue"),
            ("TUMOR", tissue_arr == "TUMOR",             "red"),
            ("LN",    tissue_arr == "LN",                "green"),
        ]

        any_plotted = False
        all_optimal_thresh = None

        for name, mask, color in subsets:
            sub_labels = labels[mask]
            sub_scores = scores[mask]
            sub_files = [f for f, m in zip(filenames, mask) if m]

            n = int(mask.sum())
            n_pos = int(sub_labels.sum())
            n_neg = n - n_pos

            if n_pos == 0 or n_neg == 0:
                print(f"\n[ROC] Skipping {name}: {n_pos} positive, {n_neg} negative — need both.")
                continue

            result = _compute_roc(sub_labels, sub_scores)
            if result is None:
                continue

            fpr, tpr, thresholds, auc, tp_arr, tn_arr, fp_arr, fn_arr, rn_pos, rn_neg = result

            _print_roc_details(
                name, sub_labels, sub_scores, sub_files,
                fpr, tpr, thresholds, auc,
                tp_arr, tn_arr, fp_arr, fn_arr, rn_pos, rn_neg,
            )

            # Plot empirical ROC
            ax.plot(
                fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f}, n={n})",
            )

            # Optimal point (Youden's J) from empirical data
            specificity_arr = tn_arr / rn_neg
            j_scores = tpr + specificity_arr - 1.0
            best_idx = int(np.argmax(j_scores))
            ax.plot(
                fpr[best_idx], tpr[best_idx], "o", color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=1,
            )

            if name == "ALL":
                all_optimal_thresh = float(thresholds[best_idx])

            any_plotted = True

        if not any_plotted:
            ax.text(
                0.5, 0.5,
                "Cannot compute ROC:\nneed both positive and negative samples",
                ha="center", va="center", fontsize=12, transform=ax.transAxes,
            )
            self._canvas.draw()
            return None

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")

        ax.set_xlabel("1 − Specificity (False Positive Rate)")
        ax.set_ylabel("Sensitivity (True Positive Rate)")
        ax.set_title("ROC Curves — ALL / TUMOR / LN")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        self._fig.tight_layout()
        self._canvas.draw()

        return all_optimal_thresh
