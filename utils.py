
"""
Utility functions for simple model evaluation

Functions
----------
- positive_scores(estimator, X): 
    Returns continuous scores for the positive class (class 1) from an estimator. Supports both `predict_proba` and `decision_function`.

- auc_report(y_true, y_score, name="model", plot=True): 
    Prints and returns ROC AUC, PR AUC, prevalence, and lift. Optionally plots ROC and PR curves.

- tradeoff_table(y_true, y_score, thresholds=None): 
    Computes a DataFrame of precision, recall, TP, FP, TN, FN, alerts per 1000, and true positives per 1000 for a range of thresholds.

- pick_threshold_cost(y_true, y_score, C_FP, C_FN, thresholds=None): 
    Selects thresholds to minimize expected cost, using both theoretical (Bayes formula) and empirical minima. Returns summary and table.

- pick_threshold_recall_floor(y_true, y_score, recall_floor, thresholds=None): 
    Finds the threshold that maximizes precision subject to a minimum recall constraint. Returns summary and table.

- pick_threshold_workload(y_true, y_score, alerts_per_1000_max, thresholds=None): 
    Finds the threshold that maximizes true positives per 1000 under an alerts-per-1000 budget. Returns summary and table.

- summary_at_threshold(y_true, y_score, threshold): 
    Returns a DataFrame with precision, recall, TP, FP, TN, FN, alerts per 1000, and true positives per 1000 at a specific threshold.

- plot_recall_floor_curves(y_true, y_score, recall_floor, chosen_threshold): 
    Plots precision and recall versus threshold, with lines for recall floor and chosen threshold, and labels at the chosen threshold.

- plot_cumulative_recall_at_threshold(y_true, y_score, chosen_threshold): 
    Plots cumulative recall versus number of alerts, with a vertical line and label at the number of alerts implied by the chosen threshold.

- plot_topk_at_threshold(y_true, y_score, chosen_threshold, top_k=30): 
    Plots a bar chart of the top-k highest risk cases, coloring true positives and false positives, with a line for the chosen threshold.
"""
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
)

__all__ = [
    "positive_scores",
    "auc_report",
    "tradeoff_table",
    "pick_threshold_cost",
    "pick_threshold_recall_floor",
    "pick_threshold_workload",
    "summary_at_threshold",
    "plot_recall_floor_curves",
    "plot_cumulative_recall_at_threshold",
    "plot_topk_at_threshold",
    "make_thresholded_estimator",
]


def positive_scores(estimator, X) -> np.ndarray:
    """
    Return continuous scores for the positive class (class 1) from an estimator.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            classes = list(getattr(estimator, "classes_", [0, 1]))
            pos_idx = classes.index(1) if 1 in classes else 1
            return proba[:, pos_idx].ravel()
        return proba.ravel()
    if hasattr(estimator, "decision_function"):
        df = estimator.decision_function(X)
        df = np.asarray(df)
        if df.ndim == 2 and df.shape[1] >= 2:
            classes = list(getattr(estimator, "classes_", [0, 1]))
            pos_idx = classes.index(1) if 1 in classes else 1
            return df[:, pos_idx].ravel()
        return df.ravel()
    raise AttributeError(
        "Estimator must implement predict_proba or decision_function")


def auc_report(y_true, y_score, name: str = "model", plot: bool = True) -> Dict[str, float]:
    """
    Print and return ROC AUC, PR AUC, prevalence, and lift. Optionally plot ROC and PR curves.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    pr_auc = float(average_precision_score(y_true, y_score))
    roc = float(roc_auc_score(y_true, y_score))
    prevalence = float(np.mean(y_true))
    lift = float(pr_auc / prevalence) if prevalence > 0 else float("inf")

    print(f"{name}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"ROC AUC: {roc:.3f}")
    print(
        f"Prevalence p = {prevalence:.3f}  |  PR AUC lift = {lift:.2f}× over baseline")

    if plot:
        RocCurveDisplay.from_predictions(y_true, y_score)
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.title(f"ROC curve {name} (AUC = {roc:.3f})")
        plt.show()

        PrecisionRecallDisplay.from_predictions(y_true, y_score)
        plt.hlines(prevalence, 0, 1, colors="gray", linestyles="dotted")
        plt.title(f"Precision recall curve {name} (PR AUC = {pr_auc:.3f})")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    return {"name": name, "roc_auc": roc, "pr_auc": pr_auc, "prevalence": prevalence, "lift": lift}


def tradeoff_table(y_true, y_score, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Compute precision, recall, confusion matrix, and alert rates for a range of thresholds.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.asarray(y_score).ravel()

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    thresholds = np.asarray(thresholds)

    rows = []
    n = len(y_true)
    for t in thresholds:
        y_hat = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        alerts_per_1000 = 1000.0 * float(np.mean(y_hat))
        true_pos_per_1000 = 1000.0 * float(tp) / n
        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "alerts_per_1000": float(alerts_per_1000),
            "true_pos_per_1000": float(true_pos_per_1000),
        })
    return pd.DataFrame(rows)


def pick_threshold_cost(y_true, y_score, C_FP: float, C_FN: float, thresholds: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Select thresholds to minimize expected cost using both theoretical and empirical minima.
    """
    tbl = tradeoff_table(y_true, y_score, thresholds)
    denom = C_FP + C_FN
    t_formula = float(C_FP / denom) if denom > 0 else 1.0

    threshold_values = np.asarray(tbl["threshold"].values, dtype=float)
    idx_near = int(np.argmin(np.abs(threshold_values - t_formula)))
    row_formula = tbl.iloc[idx_near].copy()

    exp_cost = C_FP * tbl["FP"] + C_FN * tbl["FN"]
    idx_emp = int(exp_cost.values.argmin())
    row_emp = tbl.iloc[idx_emp].copy()

    summary = pd.DataFrame([
        {"rule": "Bayes formula", **row_formula.to_dict(),
         "expected_cost": float(exp_cost.iloc[idx_near])},
        {"rule": "Empirical min cost", **
            row_emp.to_dict(), "expected_cost": float(exp_cost.iloc[idx_emp])},
    ])

    return {
        "threshold_formula": float(row_formula["threshold"]),
        "threshold_empirical": float(row_emp["threshold"]),
        "summary": summary,
        "table": tbl,
    }


def pick_threshold_recall_floor(y_true, y_score, recall_floor: float, thresholds: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Find threshold maximizing precision subject to a minimum recall constraint.
    """
    tbl = tradeoff_table(y_true, y_score, thresholds)
    feasible = tbl[tbl["recall"] >= recall_floor]
    if len(feasible) == 0:
        idx = int(np.argmax(tbl["recall"].to_numpy()))
        chosen = tbl.iloc[idx]
        rule = "Max recall fallback"
    else:
        max_prec = feasible["precision"].max()
        candidates = feasible[feasible["precision"] == max_prec]
        idx = int(candidates["threshold"].values.argmax())
        chosen = candidates.iloc[idx]
        rule = "Recall floor then max precision"
    summary = pd.DataFrame([{"rule": rule, **chosen.to_dict()}])
    return {
        "threshold": float(chosen["threshold"]),
        "summary": summary,
        "table": tbl,
    }


def pick_threshold_workload(y_true, y_score, alerts_per_1000_max: float, thresholds: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Find threshold maximizing true positives per 1000 under an alerts-per-1000 budget.
    """
    tbl = tradeoff_table(y_true, y_score, thresholds)
    feasible = tbl[tbl["alerts_per_1000"] <= alerts_per_1000_max + 1e-9]
    if len(feasible) == 0:
        idx = int((tbl["alerts_per_1000"] -
                  alerts_per_1000_max).abs().values.argmin())
        chosen = tbl.iloc[idx]
        rule = "Closest to alerts budget fallback"
    else:
        best_tp = feasible["true_pos_per_1000"].max()
        candidates = feasible[feasible["true_pos_per_1000"] == best_tp]
        best_prec = candidates["precision"].max()
        candidates = candidates[candidates["precision"] == best_prec]
        idx = int(candidates["threshold"].values.argmax())
        chosen = candidates.iloc[idx]
        rule = "Max TP per 1000 under budget"
    summary = pd.DataFrame([{"rule": rule, **chosen.to_dict()}])
    return {
        "threshold": float(chosen["threshold"]),
        "summary": summary,
        "table": tbl,
    }


def summary_at_threshold(y_true, y_score, threshold):
    """
    Return precision, recall, confusion matrix, and alert rates at a specific threshold.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    y_hat = (y_score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec = recall_score(y_true, y_hat, zero_division=0)
    n = len(y_true)
    row = pd.DataFrame([{
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "alerts_per_1000": 1000.0 * float(np.mean(y_hat)),
        "true_pos_per_1000": 1000.0 * float(tp) / n
    }])
    return row


def plot_recall_floor_curves(y_true, y_score, recall_floor, chosen_threshold):
    """
    Plot precision and recall vs threshold, with lines for recall floor and chosen threshold.
    """
    # Use local function defined in this module (no package-relative import)
    tbl = tradeoff_table(y_true, y_score)
    chosen = summary_at_threshold(y_true, y_score, chosen_threshold).iloc[0]

    plt.figure()
    plt.plot(tbl["threshold"], tbl["recall"], label="Recall")
    plt.plot(tbl["threshold"], tbl["precision"], label="Precision")
    plt.axhline(float(recall_floor), linestyle="--", color="red",
                label=f"Recall floor = {float(recall_floor):.2f}")
    plt.axvline(float(chosen_threshold), linestyle=":", color="black",
                label=f"Chosen threshold = {float(chosen_threshold):.2f}")

    plt.scatter(float(chosen_threshold),
                chosen["recall"], color="blue", zorder=5)
    plt.text(float(chosen_threshold) + 0.01,
             chosen["recall"], f"Recall={chosen['recall']:.2f}", va="center")
    plt.scatter(float(chosen_threshold),
                chosen["precision"], color="orange", zorder=5)
    plt.text(float(chosen_threshold) + 0.01,
             chosen["precision"], f"Prec={chosen['precision']:.2f}", va="center")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Recall floor then maximize precision")
    plt.legend()
    plt.xlim(0, 0.55)
    plt.show()


def plot_cumulative_recall_at_threshold(y_true, y_score, chosen_threshold):
    """
    Plot cumulative recall vs number of alerts, with a vertical line at the chosen threshold.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    cum_tp = np.cumsum(y_sorted)
    total_pos = int(cum_tp[-1]) if cum_tp.size else 0
    alerts = np.arange(1, len(y_sorted) + 1)
    recall_curve = cum_tp / \
        total_pos if total_pos > 0 else np.zeros_like(cum_tp, dtype=float)

    # alerts implied by threshold
    y_hat = (y_score >= float(chosen_threshold)).astype(int)
    n_alerts = int(y_hat.sum())
    rec_at_thr = float(recall_curve[n_alerts - 1]
                       ) if 0 < n_alerts <= len(y_sorted) else 0.0

    plt.figure()
    plt.plot(alerts, recall_curve, label="Cumulative recall")
    plt.axvline(n_alerts, linestyle="--", color="red",
                label=f"Alerts = {n_alerts}")
    plt.scatter(n_alerts, rec_at_thr, color="black", zorder=5)
    plt.text(n_alerts + max(2, len(y_sorted)//100), rec_at_thr,
             f"Recall = {rec_at_thr:.2f}", va="center")
    plt.xlabel("Number of alerts")
    plt.ylabel("Recall")
    plt.title("Cumulative capture of true cases vs alerts")
    plt.legend()
    plt.show()


def plot_topk_at_threshold(y_true, y_score, chosen_threshold, top_k=30):
    """
    Plot a bar chart of the top-k highest risk cases, coloring true and false positives.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()

    order = np.argsort(-y_score)
    top_idx = order[:int(top_k)]
    top_scores = y_score[top_idx]
    top_true = y_true[top_idx]

    tp_idx = np.where(top_true == 1)[0]
    fp_idx = np.where(top_true == 0)[0]

    plt.figure(figsize=(10, 4))
    plt.bar(tp_idx, top_scores[tp_idx],
            label="True addicted TP", color="tab:red")
    plt.bar(fp_idx, top_scores[fp_idx],
            label="Not addicted FP", color="tab:gray")
    plt.axhline(float(chosen_threshold), linestyle="--", color="black",
                label=f"Threshold = {float(chosen_threshold):.2f}")
    plt.xlabel("Patients ranked by predicted risk")
    plt.ylabel("Predicted risk")
    plt.title(f"Top {int(top_k)} highest-risk patients on validation")
    plt.legend()
    plt.tight_layout()
    plt.show()


class ThresholdedEstimator(BaseEstimator, ClassifierMixin):
    """
    Wrap a probabilistic classifier so .predict applies a custom threshold to P(y = positive_label)
    Keeps predict_proba intact for RAI/SHAP/Fairlearn compatibility
    """

    def __init__(self, base, threshold: float = 0.5, positive_label=1):
        self.base = base
        self.threshold = float(threshold)
        self.positive_label = positive_label

    # scikit-learn API

    def fit(self, X, y=None, **fit_params):
        # Train base if possible, otherwise act as a pure wrapper
        if hasattr(self.base, "fit") and y is not None:
            self.base.fit(X, y, **fit_params)
        # Mirror common fitted attrs so downstream tooling can introspect
        self.classes_ = getattr(self.base, "classes_", None)
        self.n_features_in_ = getattr(self.base, "n_features_in_", None)
        self.feature_names_in_ = getattr(self.base, "feature_names_in_", None)
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        # RAI classification assumes predict_proba is available
        if not hasattr(self.base, "predict_proba"):
            raise AttributeError(
                f"{type(self.base).__name__} does not implement predict_proba, required for RAI explanations and fairness"
            )
        proba = self.base.predict_proba(X)
        return self._ensure_2d_proba(proba)

    def predict(self, X):
        p_pos = self._positive_proba(self.predict_proba(X))
        if getattr(self, "classes_", None) is not None and len(self.classes_) == 2:
            neg_label = self._negative_label()
            return np.where(p_pos >= self.threshold, self.positive_label, neg_label)
        # Fallback to {0,1}
        return (p_pos >= self.threshold).astype(int)

    def decision_function(self, X):
        # Prefer base margin if available
        if hasattr(self.base, "decision_function"):
            return self.base.decision_function(X)
        # Otherwise return centered probability margin
        return self._positive_proba(self.predict_proba(X)) - self.threshold

    def predict_log_proba(self, X):
        if hasattr(self.base, "predict_log_proba"):
            logp = self.base.predict_log_proba(X)
            return self._ensure_2d_proba(logp)
        proba = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)
        return np.log(proba)

    # Convenience

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)
        return self

    # Utilities

    def _ensure_2d_proba(self, arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            # Normalize 1D binary proba to 2D [P(neg), P(pos)]
            arr = np.c_[1 - arr, arr]
        return arr

    def _positive_index(self, proba_2d: np.ndarray) -> int:
        if getattr(self, "classes_", None) is not None:
            classes = list(self.classes_)
            if self.positive_label in classes:
                return classes.index(self.positive_label)
        # Conventional second column is positive
        if proba_2d.ndim == 2 and proba_2d.shape[1] >= 2:
            return 1
        return 0

    def _negative_label(self):
        classes = list(self.classes_)
        for c in classes:
            if c != self.positive_label:
                return c
        return 0

    def _positive_proba(self, proba):
        proba = np.asarray(proba)
        if proba.ndim == 1:
            return proba
        return proba[:, self._positive_index(proba)]

    def __getattr__(self, name):
        # Delegate unknown attributes to the base estimator
        return getattr(self.base, name)


def make_thresholded_estimator(base_estimator, threshold: float = 0.5, positive_label=1):
    return ThresholdedEstimator(base_estimator, threshold=threshold, positive_label=positive_label)
