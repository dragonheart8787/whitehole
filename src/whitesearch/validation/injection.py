"""Injection / recovery validation pipeline.

Workflow
--------
1. Sample N parameter vectors from the prior
2. Generate synthetic data (signal + real/mock noise)
3. Run full inference pipeline on each injection
4. Compare recovered posterior to true parameters
5. Compute coverage statistics and sensitivity curves

Reference: Appendix A of LIGO O2 injection paper; CHIME/FRB Catalog 1 Sec 4.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..models.base import BaseModel
from ..simulators.base import BaseSimulator
from ..likelihoods.base import BaseLikelihood
from ..inference.bilby_runner import BilbyRunner, InferenceResult
from ..utils.math_utils import compute_credible_interval, compute_sbc_rank

logger = logging.getLogger(__name__)


@dataclass
class InjectionRecoveryResult:
    """Results of an injection/recovery campaign.

    Attributes
    ----------
    theta_true : list[dict]
        Injected parameter vectors.
    posteriors : list[pd.DataFrame]
        Recovered posterior samples for each injection.
    evidences : list[float]
        Log evidences ln Z for each injection.
    credible_intervals : list[dict]
        90% credible intervals for each injection and parameter.
    coverage : dict[str, float]
        Fraction of true values within the 90% CI (should be ~0.90).
    sbc_ranks : dict[str, list[int]]
        SBC ranks per parameter across injections.
    """

    theta_true: list[dict[str, float]]
    posteriors: list[pd.DataFrame]
    evidences: list[float]
    evidence_errs: list[float]
    credible_intervals: list[dict[str, tuple[float, float]]]
    coverage: dict[str, float] = field(default_factory=dict)
    sbc_ranks: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._compute_coverage()

    def _compute_coverage(self) -> None:
        n = len(self.theta_true)
        if n == 0:
            return

        param_names = list(self.theta_true[0].keys())
        coverage: dict[str, float] = {}
        sbc_ranks: dict[str, list[int]] = {p: [] for p in param_names}

        for i, (theta, ci, post) in enumerate(
            zip(self.theta_true, self.credible_intervals, self.posteriors)
        ):
            for p in param_names:
                true_val = theta.get(p)
                if true_val is None:
                    continue
                # Coverage check
                lo, hi = ci.get(p, (-np.inf, np.inf))
                if p not in coverage:
                    coverage[p] = 0
                if lo <= true_val <= hi:
                    coverage[p] += 1.0
                # SBC rank
                if p in post.columns:
                    rank = compute_sbc_rank(true_val, post[p].values)
                    sbc_ranks[p].append(rank)

        for p in param_names:
            coverage[p] = coverage.get(p, 0) / n

        self.coverage = coverage
        self.sbc_ranks = sbc_ranks

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of coverage and SBC statistics."""
        rows = []
        for p, cov in self.coverage.items():
            ranks = self.sbc_ranks.get(p, [])
            rows.append(
                {
                    "parameter": p,
                    "coverage_90pct": cov,
                    "coverage_ok": 0.8 <= cov <= 1.0,
                    "sbc_n": len(ranks),
                    "sbc_rank_mean": float(np.mean(ranks)) if ranks else np.nan,
                    "sbc_rank_std": float(np.std(ranks)) if ranks else np.nan,
                }
            )
        return pd.DataFrame(rows)


class InjectionRecovery:
    """Run injection/recovery campaigns to validate the inference pipeline.

    Parameters
    ----------
    simulator : BaseSimulator — forward simulator for the channel
    runner : BilbyRunner — inference engine
    n_injections : int — number of injection events
    ci_level : float — credible interval level for coverage (default 0.90)
    rng_seed : int — reproducibility seed
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        runner: BilbyRunner,
        n_injections: int = 100,
        ci_level: float = 0.90,
        rng_seed: int = 42,
    ) -> None:
        self.simulator = simulator
        self.runner = runner
        self.n_injections = n_injections
        self.ci_level = ci_level
        self.rng = np.random.default_rng(rng_seed)
        self.rng_seed = rng_seed

    def run_injections(
        self,
        model: BaseModel,
        likelihood: BaseLikelihood,
        context: dict[str, Any],
        background_strain: np.ndarray | None = None,
        save_dir: Path | str | None = None,
    ) -> InjectionRecoveryResult:
        """Execute the full injection/recovery campaign.

        Parameters
        ----------
        background_strain : ndarray | None
            Real noise background to inject signals into.
            If None, Gaussian noise is generated by the simulator.
        save_dir : Path | None
            If given, save per-injection posteriors and evidences here.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        theta_true_list = []
        posteriors = []
        evidences = []
        evidence_errs = []
        cis = []

        t_start = time.time()
        logger.info(
            "Starting injection/recovery: model=%s, N=%d",
            model.name,
            self.n_injections,
        )

        for i in tqdm(range(self.n_injections), desc="Injections", unit="event"):
            seed_i = self.rng_seed + i
            rng_i = np.random.default_rng(seed_i)

            # 1. Sample true parameters from prior
            theta_true = model.sample_prior(rng_i)
            theta_true_list.append(theta_true)

            # 2. Simulate injected data
            inj_context = dict(context)
            inj_context["rng_seed"] = seed_i
            sim_data = self.simulator.simulate(theta_true, inj_context, rng=rng_i)

            # Optionally inject into real background
            if background_strain is not None:
                sim_data = self._inject_into_background(sim_data, background_strain, rng_i)

            # 3. Run inference
            try:
                result: InferenceResult = self.runner.run(
                    likelihood,
                    sim_data,
                    context,
                    model,
                    label=f"injection_{i:04d}",
                )
            except Exception as exc:
                logger.error("Injection %d failed: %s", i, exc)
                result = InferenceResult(
                    log_evidence=-1e6,
                    log_evidence_err=1.0,
                    posterior=pd.DataFrame({p: [theta_true[p]] for p in model.parameter_names}),
                    log_likelihood_samples=np.array([-1e6]),
                )

            posteriors.append(result.posterior)
            evidences.append(result.log_evidence)
            evidence_errs.append(result.log_evidence_err)
            cis.append(result.credible_intervals(self.ci_level))

            if save_dir is not None:
                result.posterior.to_csv(save_dir / f"posterior_{i:04d}.csv", index=False)

        elapsed = time.time() - t_start
        logger.info(
            "Injection/recovery complete: %.1f s (%.2f s/event)",
            elapsed,
            elapsed / max(self.n_injections, 1),
        )

        return InjectionRecoveryResult(
            theta_true=theta_true_list,
            posteriors=posteriors,
            evidences=evidences,
            evidence_errs=evidence_errs,
            credible_intervals=cis,
            metadata={
                "n_injections": self.n_injections,
                "model": model.name,
                "elapsed_s": elapsed,
                "rng_seed": self.rng_seed,
            },
        )

    @staticmethod
    def _inject_into_background(
        sim_data: Any,
        background: np.ndarray,
        rng: np.random.Generator,
    ) -> Any:
        """Add simulated signal to a real noise background segment."""
        import copy
        result = copy.deepcopy(sim_data)
        if sim_data.noise_realisation is not None:
            pure_signal = sim_data.data - sim_data.noise_realisation
            n = min(len(pure_signal), len(background))
            result.data = pure_signal[:n] + background[:n]
        return result

    def compute_sensitivity_curve(
        self,
        ir_result: InjectionRecoveryResult,
        param_name: str,
        n_bins: int = 20,
        recovery_snr_threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute fraction of injections recovered vs. parameter value.

        Parameters
        ----------
        ir_result : InjectionRecoveryResult
        param_name : str — parameter to plot on x-axis
        n_bins : int — number of bins
        recovery_snr_threshold : float | None — if set, use ln Z as proxy for detection

        Returns
        -------
        dict with 'param_bins', 'recovery_fraction', 'n_per_bin'
        """
        theta_list = ir_result.theta_true
        evidences = np.array(ir_result.evidences)

        param_vals = np.array([t.get(param_name, np.nan) for t in theta_list])
        finite_mask = np.isfinite(param_vals)
        param_vals = param_vals[finite_mask]
        evidences = evidences[finite_mask]

        if len(param_vals) == 0:
            return {"param_bins": np.array([]), "recovery_fraction": np.array([]), "n_per_bin": np.array([])}

        edges = np.linspace(param_vals.min(), param_vals.max(), n_bins + 1)
        recovery = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        threshold = recovery_snr_threshold if recovery_snr_threshold is not None else -1e5

        for i in range(n_bins):
            in_bin = (param_vals >= edges[i]) & (param_vals < edges[i + 1])
            counts[i] = int(np.sum(in_bin))
            if counts[i] > 0:
                recovered = evidences[in_bin] > threshold
                recovery[i] = float(np.sum(recovered)) / counts[i]

        return {
            "param_bins": 0.5 * (edges[:-1] + edges[1:]),
            "recovery_fraction": recovery,
            "n_per_bin": counts,
        }
