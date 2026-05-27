"""Simulation-Based Calibration (SBC).

SBC verifies that the posterior inference is self-consistent:
given a correct prior and simulator, the rank of the true parameter
value among posterior samples should be uniformly distributed.

Reference: Talts et al. (2018), arXiv:1804.06788
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ..models.base import BaseModel
from ..simulators.base import BaseSimulator
from ..likelihoods.base import BaseLikelihood
from ..inference.bilby_runner import BilbyRunner
from ..utils.math_utils import compute_sbc_rank

logger = logging.getLogger(__name__)


@dataclass
class SBCResult:
    """Results of an SBC run.

    Attributes
    ----------
    ranks : dict[param_name → list[int]]
        Rank of true value among L posterior samples for each simulation.
    n_posterior_samples : int
        Number of posterior samples L used in each rank computation.
    n_simulations : int
        Number of prior-predictive simulations.
    uniformity_pvalues : dict[str, float]
        Kolmogorov-Smirnov p-values for rank uniformity (per parameter).
    calibrated : dict[str, bool]
        True if KS p-value > 0.05 (passes SBC at 5% level).
    """

    ranks: dict[str, list[int]]
    n_posterior_samples: int
    n_simulations: int
    uniformity_pvalues: dict[str, float] = field(default_factory=dict)
    calibrated: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._compute_uniformity()

    def _compute_uniformity(self) -> None:
        for param, rank_list in self.ranks.items():
            if len(rank_list) < 10:
                self.uniformity_pvalues[param] = np.nan
                self.calibrated[param] = False
                continue
            ranks_arr = np.array(rank_list, dtype=float)
            # KS test against Uniform(0, L)
            uniform_samples = np.random.uniform(0, self.n_posterior_samples, len(ranks_arr))
            ks_stat, pval = stats.ks_2samp(ranks_arr, uniform_samples)
            self.uniformity_pvalues[param] = float(pval)
            self.calibrated[param] = pval > 0.05

    def summary(self) -> pd.DataFrame:
        rows = []
        for p in self.ranks:
            ranks_arr = np.array(self.ranks[p])
            rows.append(
                {
                    "parameter": p,
                    "n_simulations": len(ranks_arr),
                    "rank_mean": float(np.mean(ranks_arr)) if len(ranks_arr) > 0 else np.nan,
                    "rank_std": float(np.std(ranks_arr)) if len(ranks_arr) > 0 else np.nan,
                    "expected_mean": self.n_posterior_samples / 2.0,
                    "ks_pvalue": self.uniformity_pvalues.get(p, np.nan),
                    "calibrated": self.calibrated.get(p, False),
                }
            )
        return pd.DataFrame(rows)

    def plot(self, save_path: str | None = None) -> Any:
        """Plot SBC rank histograms for all parameters."""
        import matplotlib.pyplot as plt

        n_params = len(self.ranks)
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for idx, (param, rank_list) in enumerate(self.ranks.items()):
            ax = axes[idx // ncols][idx % ncols]
            if len(rank_list) == 0:
                ax.set_title(f"{param} (no data)")
                continue
            ax.hist(
                rank_list,
                bins=min(20, self.n_posterior_samples),
                range=(0, self.n_posterior_samples),
                density=True,
                color="steelblue",
                alpha=0.7,
            )
            ax.axhline(
                1.0 / self.n_posterior_samples,
                color="red",
                ls="--",
                label="uniform",
            )
            pval = self.uniformity_pvalues.get(param, np.nan)
            ax.set_title(f"{param}\nKS p={pval:.3f}")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(len(self.ranks), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SBC plot saved to %s", save_path)
        return fig

    def plot_parameter(self, param: str, save_path: str | Path) -> Any:
        """Plot rank histogram for a single parameter."""
        import matplotlib.pyplot as plt

        rank_list = self.ranks.get(param, [])
        fig, ax = plt.subplots(figsize=(5, 4))
        if len(rank_list) == 0:
            ax.set_title(f"{param} (no data)")
        else:
            ax.hist(
                rank_list,
                bins=min(20, self.n_posterior_samples),
                range=(0, self.n_posterior_samples),
                density=True,
                color="steelblue",
                alpha=0.7,
            )
            ax.axhline(
                1.0 / self.n_posterior_samples,
                color="red",
                ls="--",
                label="uniform",
            )
            pval = self.uniformity_pvalues.get(param, np.nan)
            ax.set_title(f"{param}\nKS p={pval:.3f}")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    def plot_all(self, outdir: str | Path) -> list[Path]:
        """Write ``rank_hist_<param>.png`` for each parameter."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for param in self.ranks:
            p = outdir / f"rank_hist_{param}.png"
            self.plot_parameter(param, p)
            paths.append(p)
        return paths


class SBCRunner:
    """Run simulation-based calibration to validate Bayesian computation.

    Parameters
    ----------
    n_simulations : int
        Number of prior-predictive simulations (recommended: ≥ 1000).
    n_posterior_samples : int
        Number of posterior samples per simulation (rank denominator L).
    rng_seed : int
        Reproducibility seed.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        n_posterior_samples: int = 1000,
        rng_seed: int = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self.n_posterior_samples = n_posterior_samples
        self.rng_seed = rng_seed

    def run(
        self,
        model: BaseModel,
        simulator: BaseSimulator,
        likelihood: BaseLikelihood,
        runner: BilbyRunner,
        context: dict[str, Any],
    ) -> SBCResult:
        """Run the full SBC pipeline.

        For each simulation:
          1. θ ~ p(θ)
          2. d ~ p(d|θ)  [from simulator]
          3. Run inference → posterior samples
          4. rank(θ_k) = #{posterior samples < θ_k}
        """
        param_names = model.parameter_names
        ranks: dict[str, list[int]] = {p: [] for p in param_names}

        logger.info(
            "Starting SBC: model=%s, N_sim=%d, L=%d",
            model.name,
            self.n_simulations,
            self.n_posterior_samples,
        )

        for i in tqdm(range(self.n_simulations), desc="SBC", unit="sim"):
            seed_i = self.rng_seed + i
            rng_i = np.random.default_rng(seed_i)

            theta = model.sample_prior(rng_i)
            ctx_i = dict(context)
            ctx_i["rng_seed"] = seed_i

            sim_data = simulator.simulate(theta, ctx_i, rng=rng_i)

            try:
                result = runner.run(
                    likelihood, sim_data, context, model,
                    label=f"sbc_{i:04d}",
                )
                posterior = result.posterior
            except Exception as exc:
                logger.warning("SBC simulation %d failed: %s", i, exc)
                continue

            # Thin posterior to n_posterior_samples
            n_avail = len(posterior)
            if n_avail == 0:
                continue
            idx = np.random.default_rng(seed_i).choice(
                n_avail,
                size=min(self.n_posterior_samples, n_avail),
                replace=False,
            )
            post_thin = posterior.iloc[idx]

            for p in param_names:
                true_val = theta.get(p)
                if true_val is None or p not in post_thin.columns:
                    continue
                rank = compute_sbc_rank(true_val, post_thin[p].values)
                ranks[p].append(rank)

        return SBCResult(
            ranks=ranks,
            n_posterior_samples=self.n_posterior_samples,
            n_simulations=self.n_simulations,
        )
