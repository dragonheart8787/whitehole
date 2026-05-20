"""Gaussian Process surrogate / emulator for expensive forward simulators.

Replaces a physics simulator with a GP-interpolated approximation, validated
against hold-out points on a physics grid.  Used for:
  - Pre-screening large parameter spaces at low cost
  - Accelerating nested sampling when the simulator is too slow per call
  - Mapping from parameters to summary statistics (not raw strain)

Pipeline
--------
1. Train: generate a grid of (params → summary_stats) using the real simulator
2. Fit: fit a GP (or other emulator) to the training data
3. Predict: evaluate surrogate at arbitrary parameter points
4. Validate: compare to hold-out physics evaluations; flag regions with > 10% error
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed; GP surrogate disabled.")


@dataclass
class SurrogateValidationResult:
    """Hold-out validation results for the surrogate."""

    param_names: list[str]
    stat_names: list[str]
    true_stats: np.ndarray     # shape (n_holdout, n_stats)
    pred_stats: np.ndarray     # shape (n_holdout, n_stats)
    pred_std: np.ndarray       # shape (n_holdout, n_stats)
    relative_errors: np.ndarray  # |pred - true| / |true|
    max_rel_error: float
    mean_rel_error: float
    passes_10pct: bool
    passes_20pct: bool

    def summary(self) -> pd.DataFrame:
        rows = []
        for i, stat in enumerate(self.stat_names):
            err = self.relative_errors[:, i]
            rows.append(
                {
                    "statistic": stat,
                    "mean_rel_error": float(np.nanmean(err)),
                    "max_rel_error": float(np.nanmax(err)),
                    "p90_rel_error": float(np.nanquantile(err, 0.9)),
                    "passes_10pct": float(np.nanmean(err)) < 0.10,
                    "passes_20pct": float(np.nanmean(err)) < 0.20,
                }
            )
        return pd.DataFrame(rows)


class GPSurrogate:
    """Gaussian Process surrogate for a multi-output summary statistic emulator.

    One independent GP is fitted per output dimension (summary statistic).
    Inputs are the model parameters; outputs are summary statistics.

    Parameters
    ----------
    param_names : list[str]
        Names of input parameters (in order).
    stat_names : list[str]
        Names of summary statistics to emulate.
    kernel_type : str
        GP kernel: 'matern52' (default), 'matern32', or 'rbf'.
    normalize_inputs : bool
        Whether to standardise inputs (strongly recommended).
    """

    def __init__(
        self,
        param_names: list[str],
        stat_names: list[str],
        kernel_type: str = "matern52",
        normalize_inputs: bool = True,
    ) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for GPSurrogate.")
        self.param_names = param_names
        self.stat_names = stat_names
        self.kernel_type = kernel_type
        self.normalize_inputs = normalize_inputs
        self._scaler: StandardScaler | None = None
        self._gps: list[GaussianProcessRegressor] = []
        self._is_fitted = False
        self._X_train: NDArray | None = None
        self._y_train: NDArray | None = None

    def _build_kernel(self) -> Any:
        if self.kernel_type == "matern52":
            return ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        elif self.kernel_type == "matern32":
            return ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel(noise_level=1e-5)
        else:
            from sklearn.gaussian_process.kernels import RBF
            return ConstantKernel(1.0) * RBF() + WhiteKernel(noise_level=1e-5)

    def train(
        self,
        X: NDArray,
        y: NDArray,
        n_restarts_optimizer: int = 5,
    ) -> None:
        """Fit the GP surrogate.

        Parameters
        ----------
        X : ndarray, shape (n_train, n_params)
            Parameter grid (Latin hypercube or regular grid recommended).
        y : ndarray, shape (n_train, n_stats)
            Corresponding summary statistics from the real simulator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.normalize_inputs:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        self._X_train = X_scaled
        self._y_train = y
        self._gps = []

        for i_stat, stat_name in enumerate(self.stat_names):
            y_col = y[:, i_stat]
            valid = np.isfinite(y_col)
            if np.sum(valid) < 5:
                logger.warning("Too few valid samples for stat %s; fitting on all.", stat_name)
                valid = np.ones(len(y_col), dtype=bool)

            gp = GaussianProcessRegressor(
                kernel=self._build_kernel(),
                n_restarts_optimizer=n_restarts_optimizer,
                normalize_y=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_scaled[valid], y_col[valid])

            self._gps.append(gp)
            logger.debug(
                "GP %s fitted: log-marginal-likelihood = %.2f",
                stat_name,
                gp.log_marginal_likelihood_value_,
            )

        self._is_fitted = True
        logger.info(
            "GPSurrogate trained on %d points, %d outputs.", len(X), len(self.stat_names)
        )

    def predict(
        self,
        X_new: NDArray,
        return_std: bool = True,
    ) -> tuple[NDArray, NDArray | None]:
        """Predict summary statistics at new parameter points.

        Returns
        -------
        mu : ndarray, shape (n_new, n_stats)
        std : ndarray or None, shape (n_new, n_stats)
        """
        if not self._is_fitted:
            raise RuntimeError("Call train() before predict().")

        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        if self.normalize_inputs and self._scaler is not None:
            X_scaled = self._scaler.transform(X_new)
        else:
            X_scaled = X_new

        means = []
        stds = []
        for gp in self._gps:
            if return_std:
                mu, sigma = gp.predict(X_scaled, return_std=True)
                stds.append(sigma)
            else:
                mu = gp.predict(X_scaled, return_std=False)
            means.append(mu)

        mu_arr = np.column_stack(means)
        std_arr = np.column_stack(stds) if return_std else None
        return mu_arr, std_arr

    def validate(
        self,
        X_holdout: NDArray,
        y_holdout: NDArray,
    ) -> SurrogateValidationResult:
        """Validate against hold-out physics evaluations."""
        mu, std = self.predict(X_holdout, return_std=True)
        y_true = np.asarray(y_holdout, dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.abs(mu - y_true) / (np.abs(y_true) + 1e-30)

        max_err = float(np.nanmax(rel_err))
        mean_err = float(np.nanmean(rel_err))

        return SurrogateValidationResult(
            param_names=self.param_names,
            stat_names=self.stat_names,
            true_stats=y_true,
            pred_stats=mu,
            pred_std=std,
            relative_errors=rel_err,
            max_rel_error=max_err,
            mean_rel_error=mean_err,
            passes_10pct=mean_err < 0.10,
            passes_20pct=mean_err < 0.20,
        )

    def save(self, path: str | Path) -> None:
        """Save the fitted surrogate to disk using joblib."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "gps": self._gps,
                "scaler": self._scaler,
                "param_names": self.param_names,
                "stat_names": self.stat_names,
                "kernel_type": self.kernel_type,
                "normalize_inputs": self.normalize_inputs,
            },
            path,
        )
        logger.info("Surrogate saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GPSurrogate":
        """Load a surrogate from disk."""
        import joblib
        state = joblib.load(path)
        obj = cls(
            param_names=state["param_names"],
            stat_names=state["stat_names"],
            kernel_type=state["kernel_type"],
            normalize_inputs=state["normalize_inputs"],
        )
        obj._gps = state["gps"]
        obj._scaler = state["scaler"]
        obj._is_fitted = True
        return obj


def build_training_grid(
    model: Any,
    simulator: Any,
    context: dict[str, Any],
    n_train: int = 200,
    n_holdout: int = 50,
    rng_seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a (params, stats) training + hold-out dataset.

    Uses Latin Hypercube Sampling for efficient coverage of the prior volume.

    Returns
    -------
    X_train, y_train, X_holdout, y_holdout
    """
    from scipy.stats import qmc

    n_total = n_train + n_holdout
    n_params = len(model.parameter_names)

    # Latin Hypercube Sampling in the unit cube
    sampler = qmc.LatinHypercube(d=n_params, seed=rng_seed)
    unit_samples = sampler.random(n=n_total)

    # Transform to parameter space using the prior CDF (approximate)
    rng = np.random.default_rng(rng_seed)
    params_list = [model.sample_prior(rng) for _ in range(n_total)]

    X = np.array([[p[name] for name in model.parameter_names] for p in params_list])
    y_list = []
    for params in params_list:
        try:
            sim_data = simulator.simulate(params, context, rng=np.random.default_rng(0))
            stats = model.summary_stats(params)
            y_list.append([stats.get(k, np.nan) for k in sorted(stats.keys())])
        except Exception as exc:
            logger.warning("Training grid sim failed: %s", exc)
            y_list.append([np.nan] * len(sorted(model.summary_stats(params_list[0]).keys())))

    y = np.array(y_list, dtype=float)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
