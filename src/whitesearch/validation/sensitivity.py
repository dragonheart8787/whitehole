"""Sensitivity curves, false positive rate (FPR), and ROC analysis.

Implements
----------
- Sensitivity curve: detection fraction vs. injected signal parameter
- False positive rate (FPR): trigger rate on null-data windows
- ROC curve: TPR vs FPR as the detection threshold varies
- Trial-factor-corrected p-value
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from ..inference.bilby_runner import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Results of a sensitivity analysis campaign.

    Attributes
    ----------
    param_bins : ndarray — parameter bin centres
    recovery_fraction : ndarray — fraction of events recovered in each bin
    fpr : float — false positive rate on null windows
    fpr_err : float — binomial uncertainty on FPR
    roc_fpr : ndarray — false positive rate axis for ROC
    roc_tpr : ndarray — true positive rate axis for ROC
    auc : float — area under the ROC curve
    threshold_50pct : float — parameter value for 50% recovery
    threshold_90pct : float — parameter value for 90% recovery
    """

    param_bins: np.ndarray
    recovery_fraction: np.ndarray
    fpr: float
    fpr_err: float
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    auc: float
    threshold_50pct: float = float("nan")
    threshold_90pct: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)

    def plot_sensitivity(self, save_path: str | None = None, xlabel: str = "Parameter") -> Any:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.param_bins, self.recovery_fraction, "o-", color="steelblue", lw=2)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.7, label="50% recovery")
        ax.axhline(0.9, color="orange", ls="--", alpha=0.7, label="90% recovery")
        if np.isfinite(self.threshold_50pct):
            ax.axvline(self.threshold_50pct, color="gray", ls=":", alpha=0.7)
        if np.isfinite(self.threshold_90pct):
            ax.axvline(self.threshold_90pct, color="orange", ls=":", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Recovery fraction")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_roc(self, save_path: str | None = None) -> Any:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.roc_fpr, self.roc_tpr, "b-", lw=2, label=f"AUC = {self.auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


class SensitivityAnalyzer:
    """Compute sensitivity curves, FPR, and ROC from injection/recovery results.

    Parameters
    ----------
    detection_threshold : float
        Minimum ln Z (log evidence) to count as a detection.
    n_time_shifts : int
        Number of time-shifted null windows for FPR estimation.
    rng_seed : int
    """

    def __init__(
        self,
        detection_threshold: float = 3.0,
        n_time_shifts: int = 100,
        rng_seed: int = 42,
    ) -> None:
        self.threshold = detection_threshold
        self.n_shifts = n_time_shifts
        self.rng_seed = rng_seed

    def compute_sensitivity(
        self,
        theta_true: list[dict[str, float]],
        evidences: list[float],
        param_name: str,
        n_bins: int = 20,
    ) -> dict[str, np.ndarray]:
        """Compute detection fraction vs. injected parameter value."""
        param_vals = np.array([t.get(param_name, np.nan) for t in theta_true])
        evs = np.array(evidences)

        finite = np.isfinite(param_vals)
        param_vals = param_vals[finite]
        evs = evs[finite]

        if len(param_vals) == 0:
            return {
                "param_bins": np.array([]),
                "recovery_fraction": np.array([]),
                "n_per_bin": np.array([]),
            }

        edges = np.linspace(param_vals.min(), param_vals.max(), n_bins + 1)
        recovery = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            in_bin = (param_vals >= edges[i]) & (param_vals < edges[i + 1])
            counts[i] = int(np.sum(in_bin))
            if counts[i] > 0:
                recovery[i] = float(np.sum(evs[in_bin] > self.threshold)) / counts[i]

        return {
            "param_bins": 0.5 * (edges[:-1] + edges[1:]),
            "recovery_fraction": recovery,
            "n_per_bin": counts,
        }

    def compute_fpr(
        self,
        null_evidences: list[float],
    ) -> tuple[float, float]:
        """Compute false positive rate from null-window evidences.

        Returns (fpr, fpr_binomial_uncertainty).
        """
        n = len(null_evidences)
        if n == 0:
            return 0.0, 0.0
        n_false = sum(ev > self.threshold for ev in null_evidences)
        fpr = n_false / n
        fpr_err = float(np.sqrt(fpr * (1.0 - fpr) / max(n, 1)))
        return float(fpr), fpr_err

    def compute_roc(
        self,
        signal_evidences: list[float],
        null_evidences: list[float],
        n_thresholds: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute ROC curve and AUC.

        Returns (fpr_axis, tpr_axis, auc).
        """
        sig = np.array(signal_evidences)
        nul = np.array(null_evidences)

        thresholds = np.linspace(
            min(sig.min(), nul.min()) - 1,
            max(sig.max(), nul.max()) + 1,
            n_thresholds,
        )[::-1]

        fpr_arr = np.array([np.mean(nul >= t) for t in thresholds])
        tpr_arr = np.array([np.mean(sig >= t) for t in thresholds])

        auc = float(np.trapezoid(tpr_arr, fpr_arr))
        if auc < 0:
            auc = -auc  # trapz sign depends on axis direction

        return fpr_arr, tpr_arr, auc

    def find_threshold_recovery(
        self,
        param_bins: np.ndarray,
        recovery_fraction: np.ndarray,
        target: float = 0.9,
    ) -> float:
        """Find the parameter value at which recovery equals the target fraction."""
        if len(param_bins) == 0:
            return float("nan")
        for i in range(len(param_bins) - 1):
            if recovery_fraction[i] <= target <= recovery_fraction[i + 1]:
                # Linear interpolation
                t = (target - recovery_fraction[i]) / (recovery_fraction[i + 1] - recovery_fraction[i] + 1e-30)
                return float(param_bins[i] + t * (param_bins[i + 1] - param_bins[i]))
        above = param_bins[recovery_fraction >= target]
        return float(above[0]) if len(above) > 0 else float("nan")

    def run_full_analysis(
        self,
        theta_true: list[dict[str, float]],
        signal_evidences: list[float],
        null_evidences: list[float],
        param_name: str,
        n_bins: int = 20,
    ) -> SensitivityResult:
        """Full sensitivity analysis combining all metrics."""
        sens = self.compute_sensitivity(theta_true, signal_evidences, param_name, n_bins)
        fpr, fpr_err = self.compute_fpr(null_evidences)
        fpr_arr, tpr_arr, auc = self.compute_roc(signal_evidences, null_evidences)

        bins = sens["param_bins"]
        recovery = sens["recovery_fraction"]

        t50 = self.find_threshold_recovery(bins, recovery, 0.5)
        t90 = self.find_threshold_recovery(bins, recovery, 0.9)

        return SensitivityResult(
            param_bins=bins,
            recovery_fraction=recovery,
            fpr=fpr,
            fpr_err=fpr_err,
            roc_fpr=fpr_arr,
            roc_tpr=tpr_arr,
            auc=auc,
            threshold_50pct=t50,
            threshold_90pct=t90,
            metadata={
                "detection_threshold": self.threshold,
                "n_signal": len(signal_evidences),
                "n_null": len(null_evidences),
                "param_name": param_name,
            },
        )


def trial_factor_correction(
    raw_pvalue: float,
    n_trials: int,
    method: str = "bonferroni",
) -> float:
    """Apply trial factor correction to a p-value.

    Parameters
    ----------
    raw_pvalue : float — raw p-value before correction
    n_trials : int — number of independent trials
    method : 'bonferroni' or 'sidak'

    Returns
    -------
    float — corrected p-value
    """
    if method == "bonferroni":
        return min(1.0, raw_pvalue * n_trials)
    elif method == "sidak":
        return 1.0 - (1.0 - raw_pvalue) ** n_trials
    else:
        raise ValueError(f"Unknown method {method!r}")
