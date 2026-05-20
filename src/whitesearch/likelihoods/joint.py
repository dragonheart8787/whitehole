"""Joint multi-channel likelihood.

log L_joint(θ) = log L_GW(θ) + log L_radio(θ) + log L_xray(θ)

Channels are assumed conditionally independent given the model parameters θ.
Each channel may have missing data (None) and is silently skipped.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import BaseLikelihood
from .gw_likelihood import GWLikelihood
from .em_likelihood import RadioBurstLikelihood, XRayBurstLikelihood
from .visibility import VisibilityLikelihood

logger = logging.getLogger(__name__)


class JointLikelihood(BaseLikelihood):
    """Sum of active channel likelihoods.

    Parameters
    ----------
    channels : list[str]
        Ordered list of active channels: subset of ['gw', 'radio', 'xray', 'image'].
    model_name : str
        Which white hole model is being fit.  Controls which sub-likelihood
        parameters are requested.
    channel_weights : dict[str, float] | None
        Optional per-channel weight (default 1.0 for each).  Primarily useful
        for ablation studies.
    """

    CHANNEL_LIKELIHOODS: dict[str, type[BaseLikelihood]] = {
        "gw": GWLikelihood,
        "radio": RadioBurstLikelihood,
        "xray": XRayBurstLikelihood,
        "image": VisibilityLikelihood,
    }

    def __init__(
        self,
        channels: list[str] | None = None,
        model_name: str = "pbh_tunneling",
        channel_weights: dict[str, float] | None = None,
    ) -> None:
        self.channels = channels or ["gw", "radio"]
        self.model_name = model_name
        self.weights = channel_weights or {ch: 1.0 for ch in self.channels}

        self._likelihoods: dict[str, BaseLikelihood] = {}
        for ch in self.channels:
            cls = self.CHANNEL_LIKELIHOODS.get(ch)
            if cls is None:
                raise ValueError(f"Unknown channel {ch!r}. Available: {list(self.CHANNEL_LIKELIHOODS)}")
            kwargs: dict = {}
            if ch in ("gw", "radio", "xray"):
                kwargs["model_name"] = model_name
            self._likelihoods[ch] = cls(**kwargs)

    @property
    def parameter_names(self) -> list[str]:
        seen: list[str] = []
        for ll in self._likelihoods.values():
            for p in ll.parameter_names:
                if p not in seen:
                    seen.append(p)
        return seen

    def loglike(
        self,
        theta: dict[str, float],
        data: dict[str, Any],
        context: dict[str, Any],
    ) -> float:
        """Compute joint log-likelihood.

        Parameters
        ----------
        theta : dict[str, float]
        data : dict[channel → data_object]
            e.g. {'gw': gw_sim_data, 'radio': radio_sim_data}
        context : dict[channel → context_dict] or single context dict
        """
        ll_total = 0.0
        per_channel: dict[str, float] = {}

        for ch in self.channels:
            ch_data = data.get(ch, None) if isinstance(data, dict) else data
            if ch_data is None:
                logger.debug("Channel %s: no data provided, skipping.", ch)
                continue

            ch_context = (
                context.get(ch, context) if isinstance(context, dict) and ch in context
                else context
            )

            try:
                ll_ch = self._likelihoods[ch].loglike(theta, ch_data, ch_context)
            except Exception as exc:
                logger.error("Channel %s likelihood failed: %s", ch, exc)
                ll_ch = -np.inf

            w = self.weights.get(ch, 1.0)
            per_channel[ch] = ll_ch
            ll_total += w * ll_ch

        logger.debug("Joint log-likelihood: %.3f | per channel: %s", ll_total, per_channel)
        return ll_total

    def channel_log_likelihoods(
        self,
        theta: dict[str, float],
        data: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Return log-likelihood broken down by channel (for diagnostics)."""
        per_channel: dict[str, float] = {}
        for ch in self.channels:
            ch_data = data.get(ch, None) if isinstance(data, dict) else data
            if ch_data is None:
                per_channel[ch] = 0.0
                continue
            ch_context = (
                context.get(ch, context) if isinstance(context, dict) and ch in context
                else context
            )
            try:
                per_channel[ch] = self._likelihoods[ch].loglike(theta, ch_data, ch_context)
            except Exception as exc:
                logger.error("Channel %s: %s", ch, exc)
                per_channel[ch] = -np.inf
        return per_channel

    def ablation_study(
        self,
        theta: dict[str, float],
        data: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Compute joint log-likelihoods with each channel removed (leave-one-out)."""
        results = {"all": self.loglike(theta, data, context)}
        for ch_exclude in self.channels:
            sub = JointLikelihood(
                channels=[c for c in self.channels if c != ch_exclude],
                model_name=self.model_name,
            )
            results[f"no_{ch_exclude}"] = sub.loglike(theta, data, context)
        return results
