"""ROC curve dialog — Sensitivity vs. Specificity for UV images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
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


def _unique_operating_points(fpr, tpr):
    """Return deduplicated (fpr, tpr) keeping only points where values change."""
    pts = np.column_stack([fpr, tpr])
    mask = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    return fpr[mask], tpr[mask]


def _plot_roc_curve(ax, fpr, tpr, sub_labels, sub_scores, color, label, mode):
    """Plot a single ROC curve using the selected display mode."""
    if mode == "Linear":
        fpr_u, tpr_u = _unique_operating_points(fpr, tpr)
        ax.plot(fpr_u, tpr_u, color=color, linewidth=2, label=label)

    elif mode == "Smooth (Spline)":
        from scipy.interpolate import PchipInterpolator

        unique_fpr, inv = np.unique(fpr, return_inverse=True)
        mean_tpr = np.array(
            [tpr[inv == i].mean() for i in range(len(unique_fpr))]
        )
        if len(unique_fpr) >= 2:
            pchip = PchipInterpolator(unique_fpr, mean_tpr)
            fpr_fine = np.linspace(0, 1, 300)
            tpr_fine = np.clip(pchip(fpr_fine), 0.0, 1.0)
            ax.plot(fpr_fine, tpr_fine, color=color, linewidth=2, label=label)
        else:
            ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    elif mode == "Smooth (KDE)":
        from scipy.stats import gaussian_kde

        pos_scores = sub_scores[sub_labels == 1].astype(float)
        neg_scores = sub_scores[sub_labels == 0].astype(float)

        if len(pos_scores) >= 2 and len(neg_scores) >= 2:
            pos_kde = gaussian_kde(pos_scores)
            neg_kde = gaussian_kde(neg_scores)
            t_range = np.linspace(-5, 105, 500)
            tpr_kde = np.array(
                [pos_kde.integrate_box_1d(t, 200.0) for t in t_range]
            )
            fpr_kde = np.array(
                [neg_kde.integrate_box_1d(t, 200.0) for t in t_range]
            )
            ax.plot(fpr_kde, tpr_kde, color=color, linewidth=2, label=label)
        else:
            ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    elif mode == "LOWESS":
        from statsmodels.nonparametric.smoothers_lowess import lowess

        fpr_u, tpr_u = _unique_operating_points(fpr, tpr)
        if len(fpr_u) >= 4:
            # frac controls smoothness: smaller = less smooth
            smoothed = lowess(tpr_u, fpr_u, frac=0.3, return_sorted=True)
            ax.plot(smoothed[:, 0], np.clip(smoothed[:, 1], 0, 1),
                    color=color, linewidth=2, label=label)
        else:
            ax.plot(fpr_u, tpr_u, color=color, linewidth=2, label=label)

    elif mode == "Savitzky-Golay":
        from scipy.signal import savgol_filter

        fpr_u, tpr_u = _unique_operating_points(fpr, tpr)
        n = len(fpr_u)
        if n >= 7:
            # Window must be odd and <= n; polyorder < window
            win = min(n | 1, max(7, (n // 3) | 1))  # odd, ~1/3 of points
            tpr_sg = savgol_filter(tpr_u, window_length=win, polyorder=3)
            ax.plot(fpr_u, np.clip(tpr_sg, 0, 1),
                    color=color, linewidth=2, label=label)
        else:
            ax.plot(fpr_u, tpr_u, color=color, linewidth=2, label=label)

    elif mode == "Bootstrap Average":
        rng = np.random.default_rng(42)
        n_boot = 200
        n_samples = len(sub_labels)
        fpr_grid = np.linspace(0, 1, 200)
        tpr_boot = np.zeros((n_boot, len(fpr_grid)))

        for b in range(n_boot):
            idx = rng.choice(n_samples, n_samples, replace=True)
            bl = sub_labels[idx]
            bs = sub_scores[idx]
            result = _compute_roc(bl, bs)
            if result is None:
                tpr_boot[b] = np.nan
                continue
            b_fpr, b_tpr = result[0], result[1]
            # Interpolate onto common FPR grid
            sort_idx = np.argsort(b_fpr)
            tpr_boot[b] = np.interp(fpr_grid, b_fpr[sort_idx], b_tpr[sort_idx])

        mean_tpr = np.nanmean(tpr_boot, axis=0)
        lo_tpr = np.nanpercentile(tpr_boot, 2.5, axis=0)
        hi_tpr = np.nanpercentile(tpr_boot, 97.5, axis=0)

        ax.fill_between(fpr_grid, lo_tpr, hi_tpr, color=color, alpha=0.15)
        ax.plot(fpr_grid, mean_tpr, color=color, linewidth=2, label=label)

    elif mode == "Bezier":
        from scipy.interpolate import make_interp_spline

        fpr_u, tpr_u = _unique_operating_points(fpr, tpr)
        if len(fpr_u) >= 4:
            # Cubic B-spline through the operating points
            sort_idx = np.argsort(fpr_u)
            fpr_s, tpr_s = fpr_u[sort_idx], tpr_u[sort_idx]
            k = min(3, len(fpr_s) - 1)
            spl = make_interp_spline(fpr_s, tpr_s, k=k)
            fpr_fine = np.linspace(fpr_s[0], fpr_s[-1], 300)
            tpr_fine = np.clip(spl(fpr_fine), 0.0, 1.0)
            ax.plot(fpr_fine, tpr_fine, color=color, linewidth=2, label=label)
        else:
            ax.plot(fpr_u, tpr_u, color=color, linewidth=2, label=label)

    else:  # "Empirical" or unknown
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)


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


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    More accurate than the normal approximation, especially for small n
    or proportions near 0/1.  Returns (lower, upper) bounds.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _auc_ci(auc: float, n_pos: int, n_neg: int, z: float = 1.96) -> tuple[float, float]:
    """Hanley-McNeil 95% confidence interval for AUC.

    Uses the Hanley & McNeil (1982) standard error estimate, which models
    the AUC as a Mann-Whitney U statistic.  Returns (lower, upper) bounds.
    """
    if n_pos == 0 or n_neg == 0:
        return (0.0, 0.0)
    q1 = auc / (2.0 - auc)
    q2 = (2.0 * auc * auc) / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


def _compute_loocv(labels: np.ndarray, scores: np.ndarray):
    """Leave-One-Out Cross-Validation for ROC threshold selection.

    For each sample i, holds it out, computes the optimal threshold
    (Youden's J) on the remaining N-1 samples, then classifies sample i
    using that threshold.

    Returns (loo_sens, loo_spec, loo_preds, loo_thresholds) or None.
    """
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0 or n < 3:
        return None

    loo_preds = np.zeros(n, dtype=int)
    loo_thresholds = np.zeros(n, dtype=float)

    for i in range(n):
        # Hold out sample i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_labels = labels[mask]
        train_scores = scores[mask]

        result = _compute_roc(train_labels, train_scores)
        if result is None:
            # Can't compute ROC without both classes — predict majority
            loo_preds[i] = 1 if train_labels.sum() > len(train_labels) / 2 else 0
            loo_thresholds[i] = 50.0
            continue

        fpr, tpr, thresholds, auc, tp_arr, tn_arr, fp_arr, fn_arr, rp, rn = result
        spec_arr = tn_arr / rn
        j_scores = tpr + spec_arr - 1.0
        best_idx = int(np.argmax(j_scores))
        opt_thresh = float(thresholds[best_idx])

        loo_thresholds[i] = opt_thresh
        loo_preds[i] = 1 if scores[i] >= opt_thresh else 0

    # Compute LOO confusion matrix
    tp = int(((loo_preds == 1) & (labels == 1)).sum())
    fp = int(((loo_preds == 1) & (labels == 0)).sum())
    tn = int(((loo_preds == 0) & (labels == 0)).sum())
    fn = int(((loo_preds == 0) & (labels == 1)).sum())

    sens = tp / n_pos if n_pos > 0 else 0.0
    spec = tn / n_neg if n_neg > 0 else 0.0

    return sens, spec, loo_preds, loo_thresholds, tp, fp, tn, fn


def _print_loocv_details(
    group_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    filenames: list[str],
    loo_preds: np.ndarray,
    loo_thresholds: np.ndarray,
    sens: float,
    spec: float,
    tp: int, fp: int, tn: int, fn: int,
):
    """Print LOO cross-validation details to the terminal."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"\n{'=' * 60}")
    print(f"  LOO Cross-Validation — {group_name}  ({n_pos} positive, {n_neg} negative)")
    print(f"{'=' * 60}")
    print(f"\nLOO Sensitivity = {sens:.3f}  ({tp}/{tp + fn})")
    print(f"LOO Specificity = {spec:.3f}  ({tn}/{tn + fp})")
    print(f"LOO Accuracy    = {(tp + tn) / len(labels):.3f}  ({tp + tn}/{len(labels)})")
    print(f"\nConfusion Matrix:  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    print(f"\n{'Filename':<50s}  {'Label':>5s}  {'Score':>6s}  {'LOO-T':>6s}  {'Pred':>5s}  {'Result':>8s}")
    for i in range(len(labels)):
        label_tag = "POS" if labels[i] else "NEG"
        pred_tag = "POS" if loo_preds[i] else "NEG"
        if labels[i] == loo_preds[i]:
            result = "OK"
        elif labels[i] == 1:
            result = "FN"
        else:
            result = "FP"
        print(f"{filenames[i]:<50s}  {label_tag:>5s}  {scores[i]:6.1f}  "
              f"{loo_thresholds[i]:6.1f}  {pred_tag:>5s}  {result:>8s}")


def _write_roc_excel(
    excel_path: Path,
    group_results: list[dict],
    mode: str,
) -> None:
    """Write ROC analysis results to an Excel workbook.

    Creates one Summary sheet plus per-group sheets containing per-file
    scores/predictions and the full threshold sweep (TP/TN/FP/FN/Sens/Spec).

    Parameters
    ----------
    excel_path : Path
        Destination .xlsx file (will be overwritten).
    group_results : list of dict
        One entry per group (ALL, LN, ...) with keys:
        name, labels, scores, filenames, fpr, tpr, thresholds, auc,
        tp_arr, tn_arr, fp_arr, fn_arr, n_pos, n_neg, opt_thresh,
        opt_sens, opt_spec, loo (optional dict with sens/spec/preds/thresholds/tp/fp/tn/fn)
    mode : str
        ROC display mode (recorded in the Summary sheet).
    """
    wb = openpyxl.Workbook()

    # Remove the default sheet — we'll create our own
    default_ws = wb.active
    wb.remove(default_ws)

    bold = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    centre = Alignment(horizontal="center")

    # ---- Summary sheet ----
    summary = wb.create_sheet("Summary")
    summary.append(["ROC Analysis Summary"])
    summary["A1"].font = Font(bold=True, size=14)
    summary.append([f"Mode: {mode}"])
    summary.append([])

    summary_headers = [
        "Group", "N", "N Positive", "N Negative", "AUC",
        "Optimal Threshold", "Sensitivity @ Opt", "Specificity @ Opt",
    ]
    if any("loo" in g for g in group_results):
        summary_headers += ["LOO Sensitivity", "LOO Specificity", "LOO Accuracy"]
    summary.append(summary_headers)
    header_row = summary.max_row
    for col in range(1, len(summary_headers) + 1):
        cell = summary.cell(row=header_row, column=col)
        cell.font = bold
        cell.fill = header_fill
        cell.alignment = centre

    for g in group_results:
        n = g["n_pos"] + g["n_neg"]
        row = [
            g["name"], n, g["n_pos"], g["n_neg"],
            round(g["auc"], 4),
            round(g["opt_thresh"], 2),
            round(g["opt_sens"], 4),
            round(g["opt_spec"], 4),
        ]
        if "loo" in g:
            loo = g["loo"]
            acc = (loo["tp"] + loo["tn"]) / n if n > 0 else 0.0
            row += [
                round(loo["sens"], 4),
                round(loo["spec"], 4),
                round(acc, 4),
            ]
        elif any("loo" in gr for gr in group_results):
            row += ["", "", ""]
        summary.append(row)

    for col in summary.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        summary.column_dimensions[col[0].column_letter].width = max(max_len + 2, 10)

    # ---- Per-group sheets ----
    for g in group_results:
        # Per-file sheet
        ws = wb.create_sheet(f"{g['name']} - Per File")
        per_file_headers = ["Filename", "Label", "Score",
                            f"Pred @ {g['opt_thresh']:.1f}", "Result"]
        if "loo" in g:
            per_file_headers += ["LOO Threshold", "LOO Pred", "LOO Result"]
        ws.append(per_file_headers)
        for col in range(1, len(per_file_headers) + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = bold
            cell.fill = header_fill
            cell.alignment = centre

        labels = g["labels"]
        scores = g["scores"]
        files = g["filenames"]
        opt_thresh = g["opt_thresh"]

        loo_preds = g["loo"]["preds"] if "loo" in g else None
        loo_thresholds = g["loo"]["thresholds"] if "loo" in g else None

        for i in range(len(labels)):
            label_tag = "POS" if labels[i] else "NEG"
            pred = 1 if scores[i] >= opt_thresh else 0
            pred_tag = "POS" if pred else "NEG"
            if labels[i] == pred:
                result = "OK"
            elif labels[i] == 1:
                result = "FN"
            else:
                result = "FP"

            row = [files[i], label_tag, round(float(scores[i]), 2),
                   pred_tag, result]
            if loo_preds is not None:
                lpred_tag = "POS" if loo_preds[i] else "NEG"
                if labels[i] == loo_preds[i]:
                    lresult = "OK"
                elif labels[i] == 1:
                    lresult = "FN"
                else:
                    lresult = "FP"
                row += [round(float(loo_thresholds[i]), 2), lpred_tag, lresult]
            ws.append(row)

        for col in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col[0].column_letter].width = max(max_len + 2, 10)

        # Threshold sweep sheet
        sweep = wb.create_sheet(f"{g['name']} - Threshold Sweep")
        sweep_headers = ["Threshold", "TP", "TN", "FP", "FN",
                         "Sensitivity", "Specificity", "1 - Specificity"]
        sweep.append(sweep_headers)
        for col in range(1, len(sweep_headers) + 1):
            cell = sweep.cell(row=1, column=col)
            cell.font = bold
            cell.fill = header_fill
            cell.alignment = centre

        thresholds = g["thresholds"]
        tp_arr = g["tp_arr"]
        tn_arr = g["tn_arr"]
        fp_arr = g["fp_arr"]
        fn_arr = g["fn_arr"]
        sens_arr = g["tpr"]
        spec_arr = tn_arr / g["n_neg"]

        for i in range(len(thresholds)):
            sweep.append([
                round(float(thresholds[i]), 2),
                int(tp_arr[i]),
                int(tn_arr[i]),
                int(fp_arr[i]),
                int(fn_arr[i]),
                round(float(sens_arr[i]), 4),
                round(float(spec_arr[i]), 4),
                round(float(1.0 - spec_arr[i]), 4),
            ])

        for col in sweep.columns:
            sweep.column_dimensions[col[0].column_letter].width = 14

        # ROC Curve (XY) sheet — clean (1-Spec, Sens) coordinates suitable
        # for direct replotting in Excel/Origin/Prism/MATLAB.  Sorted by
        # FPR ascending and deduplicated so each operating point appears
        # once.  Includes (0, 0) and (1, 1) endpoints.
        xy = wb.create_sheet(f"{g['name']} - ROC Curve (XY)")
        xy.append(["Notes:"])
        xy["A1"].font = bold
        xy.append([
            "X = 1 - Specificity (False Positive Rate),  "
            "Y = Sensitivity (True Positive Rate)"
        ])
        xy.append([
            f"AUC = {g['auc']:.4f},   "
            f"Optimal threshold = {g['opt_thresh']:.2f},   "
            f"Sens@Opt = {g['opt_sens']:.4f},   "
            f"Spec@Opt = {g['opt_spec']:.4f}"
        ])
        xy.append([])

        xy_headers = [
            "1 - Specificity (X)",
            "Sensitivity (Y)",
            "Threshold",
            "TP", "FN", "FP", "TN",
        ]
        xy.append(xy_headers)
        header_row = xy.max_row
        for col in range(1, len(xy_headers) + 1):
            cell = xy.cell(row=header_row, column=col)
            cell.font = bold
            cell.fill = header_fill
            cell.alignment = centre

        # Build (fpr, tpr, threshold, tp, fn, fp, tn) tuples and sort/dedup
        fpr_arr = 1.0 - spec_arr
        points = []
        for i in range(len(thresholds)):
            points.append((
                float(fpr_arr[i]),
                float(sens_arr[i]),
                float(thresholds[i]),
                int(tp_arr[i]),
                int(fn_arr[i]),
                int(fp_arr[i]),
                int(tn_arr[i]),
            ))
        # Sort by FPR ascending, then TPR ascending
        points.sort(key=lambda p: (p[0], p[1]))
        # Deduplicate adjacent identical (FPR, TPR) pairs (keep first)
        dedup = []
        for p in points:
            if not dedup or (round(dedup[-1][0], 6) != round(p[0], 6)
                             or round(dedup[-1][1], 6) != round(p[1], 6)):
                dedup.append(p)

        # Ensure the curve starts at (0, 0) and ends at (1, 1)
        if not dedup or (dedup[0][0] > 1e-9 or dedup[0][1] > 1e-9):
            dedup.insert(0, (0.0, 0.0, 100.0, 0, g["n_pos"], 0, g["n_neg"]))
        if dedup[-1][0] < 1.0 - 1e-9 or dedup[-1][1] < 1.0 - 1e-9:
            dedup.append((1.0, 1.0, 0.0, g["n_pos"], 0, g["n_neg"], 0))

        for x, y, t, tp_v, fn_v, fp_v, tn_v in dedup:
            xy.append([
                round(x, 6),
                round(y, 6),
                round(t, 2),
                tp_v, fn_v, fp_v, tn_v,
            ])

        for col in xy.columns:
            xy.column_dimensions[col[0].column_letter].width = 18

    wb.save(excel_path)


class ROCDialog(QDialog):
    """Display the LN-only ROC curve."""

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
        mode: str = "Empirical",
        excel_path: Path | None = None,
    ) -> float | None:
        """Compute and plot three ROC curves from per-file scores.

        For each subset (ALL, TUMOR, LN), plots the ROC curve using the
        selected display mode and marks the optimal operating point
        (Youden's J).

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
        mode : str
            Display mode: "Empirical", "Linear", "Smooth (Spline)",
            or "Smooth (KDE)".

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

        # Only LN samples are used for ROC — TUMOR samples are all positive
        # (no negatives) so ROC is undefined; ALL mixes tissues and isn't the
        # diagnostic question.  The real question is: can we separate
        # cancerous from benign lymph nodes?
        subsets = [
            ("LN", tissue_arr == "LN", "green"),
        ]

        is_loo = mode == "Leave-One-Out"
        any_plotted = False
        all_optimal_thresh = None
        group_results: list[dict] = []  # for Excel export

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

            # Plot the empirical ROC curve as background for LOO,
            # or the selected mode otherwise
            plot_mode = "Linear" if is_loo else mode
            label_str = f"{name} (AUC={auc:.3f}, n={n})"
            _plot_roc_curve(ax, fpr, tpr, sub_labels, sub_scores, color,
                            label_str, plot_mode)

            # Optimal point (Youden's J) from empirical data
            specificity_arr = tn_arr / rn_neg
            j_scores = tpr + specificity_arr - 1.0
            best_idx = int(np.argmax(j_scores))

            # Collect data for Excel export
            group_data = {
                "name": name,
                "labels": sub_labels,
                "scores": sub_scores,
                "filenames": sub_files,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": auc,
                "tp_arr": tp_arr,
                "tn_arr": tn_arr,
                "fp_arr": fp_arr,
                "fn_arr": fn_arr,
                "n_pos": rn_pos,
                "n_neg": rn_neg,
                "opt_thresh": float(thresholds[best_idx]),
                "opt_sens": float(tpr[best_idx]),
                "opt_spec": float(specificity_arr[best_idx]),
            }

            if is_loo:
                # Compute Leave-One-Out cross-validated operating point
                loo_result = _compute_loocv(sub_labels, sub_scores)
                if loo_result is not None:
                    loo_sens, loo_spec, loo_preds, loo_thresholds, tp, fp, tn, fn = loo_result
                    loo_fpr = 1.0 - loo_spec

                    _print_loocv_details(
                        name, sub_labels, sub_scores, sub_files,
                        loo_preds, loo_thresholds, loo_sens, loo_spec,
                        tp, fp, tn, fn,
                    )

                    group_data["loo"] = {
                        "sens": loo_sens,
                        "spec": loo_spec,
                        "preds": loo_preds,
                        "thresholds": loo_thresholds,
                        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    }

                    # Plot LOO operating point as a star
                    ax.plot(
                        loo_fpr, loo_sens, "*", color=color, markersize=14,
                        markeredgecolor="black", markeredgewidth=1,
                        label=f"{name} LOO (Sens={loo_sens:.2f}, Spec={loo_spec:.2f})",
                    )

                # Also plot the standard Youden's J point (circle) for comparison
                ax.plot(
                    fpr[best_idx], tpr[best_idx], "o", color=color, markersize=8,
                    markeredgecolor="black", markeredgewidth=1, alpha=0.5,
                )
            else:
                ax.plot(
                    fpr[best_idx], tpr[best_idx], "o", color=color, markersize=8,
                    markeredgecolor="black", markeredgewidth=1,
                )

            group_results.append(group_data)

            if name == "LN":
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

        # Title: optimal sensitivity / specificity / AUC with 95% CIs
        # (from the LN group at Youden's J optimal point)
        ln_group = next((g for g in group_results if g["name"] == "LN"), None)
        if ln_group is not None:
            n_pos = ln_group["n_pos"]
            n_neg = ln_group["n_neg"]
            sens = ln_group["opt_sens"]
            spec = ln_group["opt_spec"]
            auc = ln_group["auc"]

            tp = int(round(sens * n_pos))
            tn = int(round(spec * n_neg))
            sens_lo, sens_hi = _wilson_ci(tp, n_pos)
            spec_lo, spec_hi = _wilson_ci(tn, n_neg)
            auc_lo, auc_hi = _auc_ci(auc, n_pos, n_neg)

            title = (
                f"Sens = {sens:.3f}  [{sens_lo:.3f}, {sens_hi:.3f}]    "
                f"Spec = {spec:.3f}  [{spec_lo:.3f}, {spec_hi:.3f}]\n"
                f"AUC = {auc:.3f}  [{auc_lo:.3f}, {auc_hi:.3f}]    "
                f"(95% CI, n={n_pos + n_neg}: {n_pos} pos, {n_neg} neg)"
            )
            ax.set_title(title, fontsize=10)
        else:
            ax.set_title("ROC Curve — no data", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        self._fig.tight_layout()
        self._canvas.draw()

        # Write ROC results to Excel (ALL and LN groups)
        if excel_path is not None:
            export_groups = [g for g in group_results if g["name"] == "LN"]
            if export_groups:
                try:
                    _write_roc_excel(excel_path, export_groups, mode)
                except Exception as exc:
                    print(f"[ROC] Failed to write Excel: {exc}")

        return all_optimal_thresh
