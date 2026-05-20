"""Posterior Predictive Checks (PPC).

For each posterior sample θ^(i), generate replicated data y^rep ~ p(y|θ^(i)).
Compare the replicated data to observed data on key summary statistics.
A mismatch indicates model misspecification or systematic errors.

Reference: Gelman et al., "Bayesian Data Analysis", Ch. 6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..simulators.base import BaseSimulator, SimData
from ..inference.bilby_runner import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class PPCResult:
    """Results of posterior predictive checks.

    Attributes
    ----------
    observed_stats : dict[str, float]
        Summary statistics of the observed data.
    predicted_stats : dict[str, list[float]]
        Summary statistics over posterior predictive samples.
    pvalues : dict[str, float]
        Bayesian p-values: fraction of replicates exceeding observed value.
    """

    observed_stats: dict[str, float]
    predicted_stats: dict[str, list[float]]
    pvalues: dict[str, float] = field(default_factory=dict)
    n_replicates: int = 0

    def __post_init__(self) -> None:
        self._compute_pvalues()

    def _compute_pvalues(self) -> None:
        for stat, obs in self.observed_stats.items():
            pred = self.predicted_stats.get(stat, [])
            if len(pred) == 0:
                self.pvalues[stat] = np.nan
                continue
            pval = float(np.mean(np.array(pred) >= obs))
            self.pvalues[stat] = pval

    def summary(self) -> pd.DataFrame:
        rows = []
        for stat, obs in self.observed_stats.items():
            pred = self.predicted_stats.get(stat, [])
            pred_arr = np.array(pred)
            rows.append(
                {
                    "statistic": stat,
                    "observed": obs,
                    "pred_mean": float(np.mean(pred_arr)) if len(pred_arr) > 0 else np.nan,
                    "pred_std": float(np.std(pred_arr)) if len(pred_arr) > 0 else np.nan,
                    "pvalue": self.pvalues.get(stat, np.nan),
                    "suspicious": not (0.05 < self.pvalues.get(stat, 0.5) < 0.95),
                }
            )
        return pd.DataFrame(rows)

    def plot(self, save_path: str | None = None) -> Any:
        """Plot histograms of predictive distributions with observed overlay."""
        stats = list(self.observed_stats.keys())
        n = len(stats)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, stat in enumerate(stats):
            ax = axes[idx // ncols][idx % ncols]
            pred = self.predicted_stats.get(stat, [])
            obs = self.observed_stats[stat]

            if len(pred) > 0:
                ax.hist(pred, bins=30, density=True, color="steelblue", alpha=0.7, label="Predicted")
            ax.axvline(obs, color="red", ls="--", lw=2, label="Observed")
            pval = self.pvalues.get(stat, np.nan)
            ax.set_title(f"{stat}\np-value={pval:.3f}")
            ax.set_xlabel(stat)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        for idx in range(len(stats), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("PPC plot saved to %s", save_path)
        return fig


class PosteriorPredictiveCheck:
    """Run posterior predictive checks for a given inference result.

    Parameters
    ----------
    simulator : BaseSimulator
        Forward simulator used to generate predictive replicates.
    summary_fns : dict[str, Callable]
        Dictionary of {name: fn} where fn(SimData) → float computes a
        summary statistic from a simulated dataset.
    n_replicates : int
        Number of posterior samples to use for predictive replicates.
    rng_seed : int
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        summary_fns: dict[str, Callable] | None = None,
        n_replicates: int = 200,
        rng_seed: int = 42,
    ) -> None:
        self.simulator = simulator
        self.summary_fns = summary_fns or self._default_summary_fns(simulator.channel)
        self.n_replicates = n_replicates
        self.rng_seed = rng_seed

    def run(
        self,
        observed_data: SimData,
        inference_result: InferenceResult,
        context: dict[str, Any],
    ) -> PPCResult:
        """Run PPC.

        Parameters
        ----------
        observed_data : SimData — the real/injected data
        inference_result : InferenceResult — posterior from inference
        context : dict — instrument configuration
        """
        # Compute observed summary statistics
        obs_stats = {name: fn(observed_data) for name, fn in self.summary_fns.items()}

        # Sample posterior parameters
        posterior = inference_result.posterior
        n_avail = len(posterior)
        n_reps = min(self.n_replicates, n_avail)
        rng = np.random.default_rng(self.rng_seed)
        idx = rng.choice(n_avail, size=n_reps, replace=False)

        pred_stats: dict[str, list[float]] = {name: [] for name in self.summary_fns}

        for i, row_idx in enumerate(idx):
            theta_i = posterior.iloc[row_idx].to_dict()
            ctx_i = dict(context)
            ctx_i["rng_seed"] = self.rng_seed + i
            try:
                rep = self.simulator.simulate(theta_i, ctx_i, rng=np.random.default_rng(self.rng_seed + i))
                for name, fn in self.summary_fns.items():
                    try:
                        pred_stats[name].append(float(fn(rep)))
                    except Exception:
                        pred_stats[name].append(np.nan)
            except Exception as exc:
                logger.warning("PPC replicate %d failed: %s", i, exc)

        return PPCResult(
            observed_stats=obs_stats,
            predicted_stats=pred_stats,
            n_replicates=n_reps,
        )

    @staticmethod
    def _default_summary_fns(channel: str) -> dict[str, Callable]:
        """Return default summary statistics for each channel."""
        if channel == "gw":
            return {
                "peak_strain": lambda d: float(np.max(np.abs(d.data))),
                "rms_strain": lambda d: float(np.std(d.data)),
                "snr_proxy": lambda d: float(
                    np.max(np.abs(d.data)) / (np.std(d.data) + 1e-30)
                ),
            }
        elif channel == "radio":
            return {
                "peak_flux_jy": lambda d: float(np.nanmax(np.mean(d.data, axis=0))),
                "rms_noise_jy": lambda d: float(np.nanstd(d.data)),
                "band_avg_max": lambda d: float(np.nanmax(np.nanmean(d.data, axis=0))),
            }
        elif channel == "xray":
            return {
                "max_counts": lambda d: float(np.max(d.data)),
                "total_counts": lambda d: float(np.sum(d.data)),
                "peak_snr": lambda d: float(
                    (np.max(d.data) - np.median(d.data)) / (np.std(d.data) + 1e-30)
                ),
            }
        elif channel == "image":
            return {
                "vis_amp_mean": lambda d: float(np.mean(np.abs(d.data))),
                "vis_amp_max": lambda d: float(np.max(np.abs(d.data))),
                "closure_phase_rms": lambda d: float(
                    np.std(d.metadata.get("closure_phases", [0.0]))
                ),
            }
        return {
            "data_max": lambda d: float(np.nanmax(d.data)),
            "data_rms": lambda d: float(np.nanstd(d.data)),
        }
