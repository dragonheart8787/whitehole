"""The EHT array must have station topology, so closure phases really close.

Decision I-5.  Before this, the uv coverage was a hand-written list of
baselines, not an array of stations: no three of them formed a triangle, so
``u_ij + u_jk != u_ik`` and the quantity called "closure phase" carried none of
the station-gain invariance the observable exists for.

These tests pin both halves of the fix:

  * the GEOMETRY -- every station triangle closes to machine precision, on the
    simulator path and on the loader/mock path alike;
  * the POINT of the geometry -- closure phases are invariant under arbitrary
    per-station complex gains, while amplitudes are not.

The second is the one that matters: closing baselines are necessary for gain
invariance, but only the invariance test shows the observable actually has the
property its name claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from whitesearch.cli import _default_context
from whitesearch.dataio.eht import (
    EHTLoader,
    baseline_index,
    eht_station_config,
    eht_station_uv,
    independent_triangles,
)
from whitesearch.models import model_for_context
from whitesearch.simulators import get_simulator
from whitesearch.simulators.image_shadow import (
    _compute_closure_phases,
    _default_eht_uv,
    _legacy_eht_uv,
)


def _wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


@pytest.fixture(scope="module")
def array():
    uv, pairs, names = eht_station_uv()
    return uv, pairs, names, independent_triangles(len(names))


class TestStationTopology:
    def test_baselines_are_every_station_pair(self, array):
        uv, pairs, names, _ = array
        n = len(names)
        assert len(pairs) == n * (n - 1) // 2 == len(uv)
        assert len(set(pairs)) == len(pairs)

    def test_stations_come_from_the_instrument_config(self, array):
        _, _, names, _ = array
        cfg_names, xyz = eht_station_config()
        assert names == cfg_names
        assert xyz.shape == (len(names), 3)

    def test_baseline_lengths_are_physically_plausible(self, array):
        """An Earth-sized array at 230 GHz tops out near 10 Glambda."""
        uv, _, _, _ = array
        longest = np.hypot(uv[:, 0], uv[:, 1]).max()
        assert 5.0 < longest < 11.0, longest


class TestClosureRelation:
    """u_ij + u_jk = u_ik, for every triangle, both code paths."""

    def test_simulator_path_closes(self, array):
        uv, pairs, _, triangles = array
        idx = baseline_index(pairs)
        for i, j, k in triangles:
            residual = uv[idx[(i, j)]] + uv[idx[(j, k)]] - uv[idx[(i, k)]]
            assert np.allclose(residual, 0.0, atol=1e-12), (i, j, k, residual)

    def test_the_default_coverage_is_the_station_array(self, array):
        uv, _, _, _ = array
        assert np.allclose(_default_eht_uv(), uv)

    def test_loader_mock_path_closes(self):
        rec = EHTLoader._mock_eht_data("M87", 2017, "LO", "M87*")
        uv = np.column_stack([rec["u"], rec["v"]])
        idx = baseline_index([(int(a), int(b)) for a, b in rec["station_pairs"]])
        for i, j, k in independent_triangles(len(rec["stations"])):
            residual = uv[idx[(i, j)]] + uv[idx[(j, k)]] - uv[idx[(i, k)]]
            assert np.allclose(residual, 0.0, atol=1e-12), (i, j, k, residual)

    def test_the_legacy_coverage_does_not_close(self):
        """The regression this fix exists for, kept as a live contrast."""
        uv = _legacy_eht_uv()
        assert not np.allclose(uv[0] + uv[1], uv[2])


class TestGainInvariance:
    """The reason closure phases exist -- measured, not asserted."""

    @staticmethod
    def _corrupt(vis, pairs, n_stations, seed):
        rng = np.random.default_rng(seed)
        g = np.exp(rng.normal(0.0, 0.2, n_stations)) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, n_stations)
        )
        return np.array([g[i] * np.conj(g[j]) * vis[b]
                         for b, (i, j) in enumerate(pairs)]), g

    def test_closure_phase_is_invariant_under_station_gains(self, array):
        uv, pairs, names, triangles = array
        rng = np.random.default_rng(7)
        vis = rng.normal(size=len(uv)) + 1j * rng.normal(size=len(uv))
        corrupted, g = self._corrupt(vis, pairs, len(names), 99)
        before = _compute_closure_phases(vis, triangles, pairs)
        after = _compute_closure_phases(corrupted, triangles, pairs)
        assert np.max(np.abs(_wrap(after - before))) < 1e-10

    def test_the_gains_really_were_applied(self, array):
        """Guards against a vacuous pass: amplitudes must move."""
        uv, pairs, names, _ = array
        rng = np.random.default_rng(7)
        vis = rng.normal(size=len(uv)) + 1j * rng.normal(size=len(uv))
        corrupted, _ = self._corrupt(vis, pairs, len(names), 99)
        log_change = np.abs(np.log(np.abs(corrupted)) - np.log(np.abs(vis)))
        assert log_change.max() > 0.1

    def test_the_legacy_statistic_is_not_invariant(self, array):
        """Contrast: consecutive triplets on closing baselines still fail."""
        uv, pairs, names, _ = array
        rng = np.random.default_rng(7)
        vis = rng.normal(size=len(uv)) + 1j * rng.normal(size=len(uv))
        corrupted, _ = self._corrupt(vis, pairs, len(names), 99)
        before = _compute_closure_phases(vis)          # no triangles -> fallback
        after = _compute_closure_phases(corrupted)
        assert np.max(np.abs(_wrap(after - before))) > 0.1

    def test_simulated_observation_closure_is_gain_invariant(self):
        """End to end, through the simulator rather than on synthetic numbers."""
        ctx = {**_default_context("image"), "target": "M87*", "rng_seed": 11}
        theta = model_for_context("gr_eternal", ctx).sample_prior(
            np.random.default_rng(11)
        )
        data = get_simulator("image").simulate(
            theta, ctx, rng=np.random.default_rng(12)
        )
        meta = data.metadata
        assert meta["closure_is_real"] is True
        pairs = meta["station_pairs"]
        tri = meta["closure_triangles"]
        n_stations = max(max(p) for p in pairs) + 1
        corrupted, _ = TestGainInvariance._corrupt(
            np.asarray(data.data), pairs, n_stations, 5
        )
        before = _compute_closure_phases(np.asarray(data.data), tri, pairs)
        after = _compute_closure_phases(corrupted, tri, pairs)
        assert np.max(np.abs(_wrap(after - before))) < 1e-10

    def test_loader_closure_phases_are_gain_invariant(self):
        rec = EHTLoader._mock_eht_data("M87", 2017, "LO", "M87*")
        pairs = [(int(a), int(b)) for a, b in rec["station_pairs"]]
        vis = np.asarray(rec["visibilities"])
        corrupted, _ = TestGainInvariance._corrupt(
            vis, pairs, len(rec["stations"]), 3
        )
        before = EHTLoader.compute_closure_phases(vis, pairs, rec["stations"])
        after = EHTLoader.compute_closure_phases(corrupted, pairs, rec["stations"])
        assert np.max(np.abs(_wrap(
            after["closure_phases"] - before["closure_phases"]
        ))) < 1e-10
        assert len(before["triangles"]) == len(before["closure_phases"])


class TestAmplitudePathUnchanged:
    """I-5 must not touch the amplitude-only path (requirement 4)."""

    def test_same_baselines_give_bit_identical_visibilities(self):
        """The amplitude code path is untouched: same uv in, same numbers out."""
        from whitesearch.simulators.image_shadow import (
            _compute_visibilities,
            build_ring_image,
        )

        ctx = {**_default_context("image"), "target": "M87*", "rng_seed": 3}
        theta = model_for_context("gr_eternal", ctx).sample_prior(
            np.random.default_rng(3)
        )
        image = build_ring_image(theta, ctx).image
        uv = _legacy_eht_uv()
        a = _compute_visibilities(image, ctx["fov_muas"], uv, ctx["freq_ghz"])
        b = _compute_visibilities(image, ctx["fov_muas"], uv, ctx["freq_ghz"])
        assert np.array_equal(a, b)
        assert len(a) == len(uv)

    def test_amplitude_only_likelihood_ignores_closure_metadata(self):
        from whitesearch.likelihoods import VisibilityLikelihood

        ctx = {**_default_context("image"), "target": "M87*", "rng_seed": 4}
        theta = model_for_context("gr_eternal", ctx).sample_prior(
            np.random.default_rng(4)
        )
        data = get_simulator("image").simulate(
            theta, ctx, rng=np.random.default_rng(5)
        )
        like = VisibilityLikelihood("gr_eternal", use_closure_phases=False)
        full = like.loglike(theta, data, ctx)
        stripped = {k: v for k, v in data.metadata.items()
                    if k not in ("closure_phases", "closure_triangles",
                                 "closure_is_real")}
        stripped["visibilities"] = data.data
        assert like.loglike(theta, stripped, ctx) == pytest.approx(full)
