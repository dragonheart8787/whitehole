"""image channel: baseline units, visibility sampling, and the Kerr shadow size.

Locks in audit X.13.  Every check here is against something derived
independently of this codebase -- an analytic Fourier transform, the Bardeen
critical-curve integral, or a published measurement -- because the bug this
file exists to prevent was invisible to forward-model consistency checks: the
simulator and the mock data generator carried the *same* wrong Gλ -> rad^-1
conversion, so they agreed with each other perfectly while both being wrong.

Reference facts used, none of them produced by this package:

* A baseline quoted in Gλ is an angular frequency -- fringe cycles per radian.
  A point source offset by x0 therefore produces a visibility phase
  ``-2 pi u x0`` with ``u`` in λ, which is the crispest possible test of the
  conversion: reinstating the wavelength factor changes it by 767x at 230 GHz.
* A thin circular ring of radius r0 and Gaussian thickness w has the Hankel
  transform ``V(u) = F J0(2 pi r0 u) exp(-2 pi^2 w^2 u^2)``, so its first
  visibility minimum sits at ``u = 2.405 / (2 pi r0)``.  For M87*'s 42 μas ring
  that is 3.76 Gλ, and EHT 2019 (ApJL 875, L4) reports the observed minimum at
  ~3.4 Gλ.
* The Kerr shadow's areal radius follows from the Bardeen (1973) spherical
  photon orbits; ``TestKerrShadowRadius`` recomputes it here rather than
  trusting the fit in ``utils.math_utils``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0

from whitesearch.dataio import EHTLoader
from whitesearch.models import get_model
from whitesearch.simulators.image_shadow import (
    ImageShadowSimulator,
    _compute_closure_phases,
    _compute_visibilities,
    _default_eht_uv,
    _gaussian_ring_image,
    _shadow_radius_muas,
)
from whitesearch.utils.constants import C, MUAS_RAD
from whitesearch.utils.math_utils import kerr_shadow_radius_rg

FOV, NPIX = 50.0, 128
DX = 2.0 * FOV / NPIX

#: M87*-like ring: 42 μas diameter, 10% fractional thickness, 0.6 Jy total.
R0_MUAS, W_MUAS, FLUX_JY = 21.0, 2.1, 0.6


def _ring_image(fov=FOV, n_pix=NPIX, r0=R0_MUAS, w=W_MUAS, flux=FLUX_JY):
    """Axisymmetric Gaussian ring normalised to `flux` Jy in total."""
    coords = np.linspace(-fov, fov, n_pix)
    xx, yy = np.meshgrid(coords, coords)
    img = np.exp(-0.5 * ((np.hypot(xx, yy) - r0) / w) ** 2)
    return img * flux / (img.sum() * (2.0 * fov / n_pix) ** 2)


def _analytic(u_glam, r0=R0_MUAS, w=W_MUAS, flux=FLUX_JY):
    """F J0(2 pi r0 u) exp(-2 pi^2 w^2 u^2) -- the ring's Hankel transform."""
    u = np.asarray(u_glam, dtype=float) * 1e9
    return flux * j0(2.0 * np.pi * r0 * MUAS_RAD * u) * np.exp(
        -2.0 * np.pi**2 * (w * MUAS_RAD) ** 2 * u**2
    )


def _radial_uv(u_glam):
    return np.column_stack([np.asarray(u_glam, dtype=float), np.zeros(len(u_glam))])


class TestBaselineUnitsAreAngularFrequency:
    """The I-4 fix itself: Gλ -> λ, with no wavelength factor."""

    def test_point_source_fringe_rate(self):
        """A source offset by x0 gives phase -2 pi u x0, with u in lambda.

        This is where the old code was wrong by exactly ``wavelength_m``:
        at 230 GHz the measured phase would come out 767x too small.
        """
        x0_muas = 12.0
        img = np.zeros((NPIX, NPIX))
        coords = np.linspace(-FOV, FOV, NPIX)
        ix = int(np.argmin(np.abs(coords - x0_muas)))
        img[NPIX // 2, ix] = 1.0 / DX**2  # one pixel, 1 Jy

        u_glam = np.array([0.5, 1.0, 2.0, 4.0])
        vis = _compute_visibilities(img, FOV, _radial_uv(u_glam), 230.0)

        expected = -2.0 * np.pi * (u_glam * 1e9) * (coords[ix] * MUAS_RAD)
        measured = np.angle(vis)
        assert np.allclose(
            np.angle(np.exp(1j * (measured - expected))), 0.0, atol=1e-9
        )
        assert np.allclose(np.abs(vis), 1.0, rtol=1e-9)

        wrong = -2.0 * np.pi * (u_glam * 1e9 * C / 230e9) * (coords[ix] * MUAS_RAD)
        assert np.abs(expected - wrong).max() > 1.0, "the old form differs by ~767x"

    def test_zero_spacing_is_the_total_flux(self):
        """V(0, 0) = total flux is the one baseline where the old code was right."""
        img = _ring_image()
        vis = _compute_visibilities(img, FOV, np.array([[0.0, 0.0]]), 230.0)
        assert float(np.abs(vis[0])) == pytest.approx(FLUX_JY, rel=1e-6)

    def test_frequency_argument_no_longer_changes_anything(self):
        """The sampler works in Gλ, so the observing frequency is irrelevant."""
        img = _ring_image()
        uv = _default_eht_uv()
        a = _compute_visibilities(img, FOV, uv, 230.0)
        b = _compute_visibilities(img, FOV, uv, 345.0)
        assert np.allclose(a, b)


class TestAgainstTheAnalyticRingTransform:
    """Independent physical check, not a comparison with a previous version."""

    U = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 3.7, 4.5, 6.0, 8.0])

    def test_matches_the_hankel_transform(self):
        vis = _compute_visibilities(_ring_image(), FOV, _radial_uv(self.U), 230.0)
        ana = np.abs(_analytic(self.U))
        dev = np.abs(np.abs(vis) - ana).max() / ana.max()
        assert dev < 0.02, f"max deviation {dev:.3%} of peak (measured 0.76%)"

    def test_the_first_minimum_lands_where_theory_puts_it(self):
        """u = 2.405 / (2 pi r0) = 3.76 Gλ for a 42 μas ring."""
        u = np.linspace(2.5, 5.0, 251)
        vis = np.abs(_compute_visibilities(_ring_image(), FOV, _radial_uv(u), 230.0))
        predicted = 2.405 / (2.0 * np.pi * R0_MUAS * MUAS_RAD) / 1e9
        assert predicted == pytest.approx(3.76, abs=0.02)
        assert u[int(np.argmin(vis))] == pytest.approx(predicted, abs=0.1)

    def test_the_field_of_view_does_not_set_the_uv_accuracy(self):
        """The old FFT-and-interpolate coupled them; a direct DFT does not.

        The interpolation grid's spacing is 1 / (2 fov), so at the 50 μas field
        this project images it was 2.06 Gλ -- coarser than the structure being
        sampled, and wrong by up to 9.0% of the peak.
        """
        uv = _radial_uv(self.U)
        near = _compute_visibilities(_ring_image(fov=FOV), FOV, uv, 230.0)
        wide = _compute_visibilities(_ring_image(fov=200.0, n_pix=512), 200.0, uv, 230.0)
        assert np.abs(np.abs(near) - np.abs(wide)).max() / np.abs(wide).max() < 0.01

    def test_correlated_flux_falls_to_the_right_order_of_magnitude(self):
        """0.6 Jy zero-spacing -> tens of mJy on the longest EHT baselines."""
        vis = np.abs(_compute_visibilities(_ring_image(), FOV, _radial_uv(self.U), 230.0))
        assert vis[0] == pytest.approx(FLUX_JY, rel=0.02)
        assert 0.01 < vis[self.U == 8.0][0] < 0.2


class TestVisibilitiesCarryImageStructure:
    """Requirement 5: a known asymmetric ring must show up in the visibilities."""

    @staticmethod
    def _asymmetric_ring(pos_angle_rad=0.0):
        return _gaussian_ring_image(
            FOV, NPIX, R0_MUAS, W_MUAS, 1.0,
            axial_ratio=0.65, pos_angle_rad=pos_angle_rad, asym_amp=2.0,
        )

    def test_amplitude_varies_with_baseline_length(self):
        """The old sampler returned the total flux on every baseline."""
        uv = _default_eht_uv()
        img = self._asymmetric_ring()
        vis = np.abs(_compute_visibilities(img, FOV, uv, 230.0))
        flux = img.sum() * DX**2

        # The old sampler gave exactly 1.000 here, on every baseline.
        assert vis.max() / vis.min() > 2.5
        # No baseline may exceed the zero-spacing flux, and the longest must
        # resolve the ring well below it.
        assert vis.max() < flux
        length = np.hypot(uv[:, 0], uv[:, 1])
        assert vis[length > 4.0].mean() < 0.5 * vis[length < 1.5].mean()
        assert vis[length > 4.0].mean() < 0.5 * flux

    def test_rotating_the_ring_changes_amplitudes_and_phases(self):
        uv = _default_eht_uv()
        a = _compute_visibilities(self._asymmetric_ring(0.0), FOV, uv, 230.0)
        b = _compute_visibilities(self._asymmetric_ring(1.2), FOV, uv, 230.0)
        peak = np.abs(a).max()
        assert np.abs(b - a).max() / peak > 0.2
        assert np.abs(np.abs(b) - np.abs(a)).max() / peak > 0.05
        assert np.abs(np.angle(b / a)).max() > 0.5

    def test_a_symmetric_ring_is_the_control(self):
        """Rotating an axisymmetric ring is exactly a no-op, as it must be."""
        uv = _default_eht_uv()
        a = _compute_visibilities(_ring_image(), FOV, uv, 230.0)
        rotated = _gaussian_ring_image(FOV, NPIX, R0_MUAS, W_MUAS, 1.0,
                                       axial_ratio=1.0, pos_angle_rad=1.2)
        plain = _gaussian_ring_image(FOV, NPIX, R0_MUAS, W_MUAS, 1.0,
                                     axial_ratio=1.0, pos_angle_rad=0.0)
        b = _compute_visibilities(rotated, FOV, uv, 230.0)
        c = _compute_visibilities(plain, FOV, uv, 230.0)
        assert np.abs(b - c).max() / np.abs(c).max() < 1e-12
        assert np.abs(a).max() > 0.0

    def test_the_simulator_inherits_all_of_this(self):
        ctx = {"target": "M87*", "fov_muas": FOV, "n_pixels": NPIX,
               "freq_ghz": 230.0, "thermal_noise_jy": 0.0}
        truth = {"M": 6.5e9, "a_star": 0.5, "i": 0.9, "position_angle": 0.7,
                 "log10_mdot_edd": -3.0, "jet_power_frac": 0.5}
        sim = ImageShadowSimulator()
        d = sim.simulate(truth, ctx, rng=np.random.default_rng(0))
        vis = np.abs(d.metadata["vis_signal"])
        flux = d.metadata["image"].sum() * DX**2
        assert vis.max() / vis.min() > 2.0
        assert vis.max() < flux, "no baseline may exceed the zero-spacing flux"


class TestMockEhtDataUsesTheSameUnits:
    """The second copy of the bug: dataio.eht._mock_eht_data."""

    def test_mock_visibilities_are_not_flat(self):
        record = EHTLoader(cache_dir=None)._mock_eht_data("M87", 2017, "LO")
        amp = np.abs(record["visibilities"])
        assert amp.max() / amp.min() > 3.0

    def test_mock_follows_the_ring_transform_it_claims(self):
        """42 μas ring for M87*, 51.8 μas for Sgr A* -- J0, not a pair of sincs."""
        for source, diameter in (("M87", 42.0), ("SgrA", 51.8)):
            record = EHTLoader(cache_dir=None)._mock_eht_data(source, 2017, "LO")
            uv = np.asarray(record["uv_coverage"], dtype=float)
            radial = np.hypot(uv[:, 0], uv[:, 1]) * 1e9
            expected = np.abs(j0(2.0 * np.pi * 0.5 * diameter * MUAS_RAD * radial))
            measured = np.abs(record["visibilities"])
            # sigma is drawn in [0.02, 0.1] Jy and added as complex noise.
            assert np.abs(measured - expected).max() < 0.35, source
            assert np.corrcoef(measured, expected)[0, 1] > 0.9, source


class TestClosurePhasePath:
    """Requirement 4: is the closure-phase path affected by the scale error?"""

    def test_wrap_convention_matches_the_loader(self):
        """A zero phase closure must map to 0, not to -pi."""
        unit = np.ones(3, dtype=complex)
        assert _compute_closure_phases(unit)[0] == pytest.approx(0.0)

    def test_closure_phases_now_respond_to_the_image(self):
        """They used to be pinned at +/-pi whatever the image was.

        Every visibility equalled the real, positive zero-spacing flux, so every
        phase was 0 and every triplet combination was 0 -- which the old wrap
        then mapped to -pi.  The term carried no information at all.
        """
        uv = _default_eht_uv()
        a = _compute_visibilities(
            TestVisibilitiesCarryImageStructure._asymmetric_ring(0.0), FOV, uv, 230.0)
        b = _compute_visibilities(
            TestVisibilitiesCarryImageStructure._asymmetric_ring(1.2), FOV, uv, 230.0)
        cp_a, cp_b = _compute_closure_phases(a), _compute_closure_phases(b)
        assert np.abs(np.angle(np.exp(1j * (cp_b - cp_a)))).max() > 0.1
        assert cp_a.std() > 0.1, "not a constant any more"

    def test_the_shipped_triplets_do_not_close(self):
        """KNOWN LIMITATION, recorded rather than silently tolerated.

        ``_compute_closure_phases`` takes consecutive array entries as a
        triangle, but ``_default_eht_uv()`` is a list of baselines rather than a
        station array, so no triplet satisfies u_ij + u_jk = u_ik.  The quantity
        is therefore a phase combination, not a gain-invariant closure phase.
        Both the simulator and the likelihood use this same function, so it is a
        self-consistent statistic and does not bias inference -- it simply is
        not the robust observable the name implies.  Fixing it means deriving
        the uv coverage from the station positions in
        configs/instruments/eht.yaml.  See audit X.13.4.

        If this test starts failing, real triangles were introduced and the
        limitation note should be removed.
        """
        uv = _default_eht_uv()
        closes = [
            np.allclose(uv[3 * k] + uv[3 * k + 1], uv[3 * k + 2])
            for k in range(len(uv) // 3)
        ]
        assert not any(closes)


class TestKerrShadowRadius:
    """順手三: one formula, checked against the exact critical curve."""

    @staticmethod
    def _exact_areal_radius(a, inc_deg):
        """sqrt(Area/pi) of the Bardeen critical curve, recomputed here.

        Deliberately an independent implementation rather than a call into
        whitesearch: this is the reference the shipped fit is judged against.
        """
        if a < 1e-12:
            return 3.0 * np.sqrt(3.0)
        r1 = 2.0 * (1 + np.cos((2 / 3) * np.arccos(-a)))
        r2 = 2.0 * (1 + np.cos((2 / 3) * np.arccos(a)))
        r = np.linspace(r1, r2, 200001)
        xi = ((r**2 - a**2) - r * (r**2 - 2 * r + a**2)) / (a * (r - 1.0))
        eta = r**3 * (4.0 * a**2 - r * (r - 3.0) ** 2) / (a**2 * (r - 1.0) ** 2)
        if inc_deg < 1e-9:
            from scipy.optimize import brentq

            r0 = brentq(
                lambda rr: ((rr**2 - a**2) - rr * (rr**2 - 2 * rr + a**2)),
                r1 + 1e-12, r2 - 1e-12,
            )
            eta0 = r0**3 * (4.0 * a**2 - r0 * (r0 - 3.0) ** 2) / (a**2 * (r0 - 1.0) ** 2)
            return float(np.sqrt(eta0 + a**2))
        th = np.deg2rad(inc_deg)
        disc = eta + a**2 * np.cos(th) ** 2 - xi**2 * (np.cos(th) / np.sin(th)) ** 2
        ok = disc >= 0
        alpha, beta = -xi[ok] / np.sin(th), np.sqrt(disc[ok])
        return float(np.sqrt(abs(2.0 * np.trapezoid(beta, alpha)) / np.pi))

    def test_schwarzschild_is_exact(self):
        assert kerr_shadow_radius_rg(0.0) == pytest.approx(3.0 * np.sqrt(3.0))

    @pytest.mark.parametrize("a", [0.0, 0.3, 0.5, 0.7, 0.9, 0.998])
    @pytest.mark.parametrize("inc_deg", [0.0, 30.0, 90.0])
    def test_fit_tracks_the_exact_critical_curve(self, a, inc_deg):
        exact = self._exact_areal_radius(a, inc_deg)
        fit = kerr_shadow_radius_rg(a, np.deg2rad(inc_deg))
        assert fit == pytest.approx(exact, rel=0.005)

    def test_matches_a_published_value(self):
        """Near-extremal Kerr, face-on: ~4.83 rg in the shadow literature."""
        assert self._exact_areal_radius(0.998, 0.0) == pytest.approx(4.83, abs=0.01)
        assert kerr_shadow_radius_rg(0.998) == pytest.approx(4.83, abs=0.02)

    def test_the_shadow_shrinks_monotonically_with_spin(self):
        """The old model-side formula made it GROW up to a* ~ 0.4."""
        spins = np.linspace(0.0, 0.998, 40)
        r = np.array([kerr_shadow_radius_rg(a) for a in spins])
        assert np.all(np.diff(r) < 0.0)
        assert r[-1] / r[0] == pytest.approx(0.928, abs=0.005), "7% over the range"

    def test_the_old_model_side_formula_was_the_wrong_one(self):
        """Both ends of the disagreement, quantified."""
        a = 0.998
        exact = self._exact_areal_radius(a, 0.0)
        old_model = 3.0 + np.sqrt(9.0 - 8.0 * a**2)          # photon_ring_radius_m
        old_sim = 3 * np.sqrt(3.0) * (1 - 0.0136 * a + 0.0038 * a**2)
        assert abs(old_model / exact - 1) > 0.15
        assert abs(old_sim / exact - 1) > 0.05
        assert abs(kerr_shadow_radius_rg(a) / exact - 1) < 0.005
        # the two in-code formulas disagreed with each other by 28%
        assert abs(old_model / old_sim - 1) == pytest.approx(0.2196, abs=0.005)

    def test_model_and_simulator_now_agree_exactly(self):
        model = get_model("gr_eternal", target="M87*")
        for a in (0.0, 0.5, 0.9, 0.998):
            for inc in (0.0, 1.2):
                half = model.shadow_angular_diameter_muas(6.5e9, a, 16.8, inc) / 2.0
                assert half == _shadow_radius_muas(6.5e9, a, 16.8, inc)


class TestFluxParameterisation:
    """X.16: gr_eternal samples the integrated flux, not the peak brightness.

    The regression this locks in: the image's actual total flux must equal
    ``10 ** log10_total_flux_jy`` exactly.  Under the old parameterisation the
    sampled quantity was the peak surface brightness and the flux was derived
    through the geometry factor ``G``, which spans 2.264 dex across the prior --
    so "this source emits about 1 Jy" could not be written as a prior at all.
    """

    CTX = {"target": "M87*", "fov_muas": FOV, "n_pixels": NPIX,
           "freq_ghz": 230.0, "thermal_noise_jy": 0.05}
    TRUTH = {"M": 6.5e9, "a_star": 0.5, "i": 0.9, "position_angle": 0.7,
             "ring_width_frac": 0.12, "log10_total_flux_jy": 0.0}

    @pytest.mark.parametrize("log10_flux", [-1.3, -0.7, 0.0, 0.6, 1.08])
    def test_imaged_flux_equals_the_sampled_flux(self, log10_flux):
        theta = {**self.TRUTH, "log10_total_flux_jy": log10_flux}
        data = ImageShadowSimulator().simulate(
            theta, self.CTX, rng=np.random.default_rng(0)
        )
        imaged = data.metadata["image"].sum() * (2.0 * FOV / NPIX) ** 2
        assert imaged == pytest.approx(10.0**log10_flux, rel=1e-12)
        assert data.metadata["total_flux_jy"] == pytest.approx(imaged, rel=1e-12)

    @pytest.mark.parametrize(
        "name,values",
        [("M", [3.5e9, 9.5e9]), ("i", [0.2, 1.3]),
         ("ring_width_frac", [0.02, 0.45]), ("a_star", [0.0, 0.95])],
    )
    def test_geometry_no_longer_leaks_into_the_flux(self, name, values):
        """The whole point: changing geometry must not change the flux."""
        fluxes, brightnesses = [], []
        for v in values:
            d = ImageShadowSimulator().simulate(
                {**self.TRUTH, name: v}, self.CTX, rng=np.random.default_rng(0)
            )
            fluxes.append(d.metadata["total_flux_jy"])
            brightnesses.append(d.metadata["brightness_jy_per_muas2"])
        assert fluxes[0] == pytest.approx(fluxes[1], rel=1e-12), name
        # ...and the derived peak brightness absorbs it instead.
        assert brightnesses[0] != pytest.approx(brightnesses[1], rel=1e-6), name

    def test_brightness_is_the_flux_divided_by_the_geometry_factor(self):
        d = ImageShadowSimulator().simulate(
            self.TRUTH, self.CTX, rng=np.random.default_rng(0)
        )
        assert d.metadata["brightness_jy_per_muas2"] == pytest.approx(
            d.metadata["total_flux_jy"] / d.metadata["geometry_factor_muas2"]
        )
        assert d.metadata["normalisation"] == "total_flux"

    def test_bh_accretion_still_normalises_on_peak_brightness(self):
        """Its rate drives I0 directly; that coupling is the hypothesis."""
        accretion = {"M": 6.5e9, "a_star": 0.5, "i": 0.9, "position_angle": 0.7,
                     "log10_mdot_edd": -3.0, "jet_power_frac": 0.3}
        d = ImageShadowSimulator().simulate(
            accretion, self.CTX, rng=np.random.default_rng(0)
        )
        assert d.metadata["normalisation"] == "peak_brightness"
        assert d.metadata["brightness_jy_per_muas2"] == pytest.approx(1e-3)

    def test_prior_is_built_from_the_measured_flux_not_the_old_bounds(self):
        from whitesearch.utils.targets import EHT_TARGETS

        for name, tgt in EHT_TARGETS.items():
            spec = next(
                s for s in get_model("gr_eternal", target=name).parameters()
                if s.name == "log10_total_flux_jy"
            )
            lo, hi = spec.prior_kwargs["low"], spec.prior_kwargs["high"]
            # one decade of margin either side of the published range
            assert lo == pytest.approx(np.log10(tgt.flux_low_jy) - 1.0, abs=1e-9)
            assert hi == pytest.approx(np.log10(tgt.flux_high_jy) + 1.0, abs=1e-9)
            assert lo < np.log10(tgt.flux_low_jy) < np.log10(tgt.flux_high_jy) < hi
            assert tgt.flux_source.strip()
        # the two targets differ in flux, so the priors are not the same
        m87 = get_model("gr_eternal", target="M87*").parameters()
        sgra = get_model("gr_eternal", target="SgrA*").parameters()
        f = lambda specs: next(s.prior_kwargs for s in specs
                               if s.name == "log10_total_flux_jy")
        assert f(m87) != f(sgra)

    def test_every_parameter_still_moves_the_visibilities_and_the_likelihood(self):
        """No new dead parameter was created by the reparameterisation."""
        from whitesearch.likelihoods import VisibilityLikelihood

        like = VisibilityLikelihood("gr_eternal", use_closure_phases=False)
        base = ImageShadowSimulator().simulate(
            self.TRUTH, self.CTX, rng=np.random.default_rng(3)
        )
        ll0 = like.loglike(self.TRUTH, base, self.CTX)
        base_vis = np.abs(base.metadata["vis_signal"])
        perturbed = {"M": 6.5e9 * 1.08, "a_star": 0.95, "i": 1.3,
                     "position_angle": 1.9, "ring_width_frac": 0.30,
                     "log10_total_flux_jy": 0.35}
        assert sorted(perturbed) == sorted(
            VisibilityLikelihood("gr_eternal").parameter_names
        )
        for name, value in perturbed.items():
            theta = {**self.TRUTH, name: value}
            moved = np.abs(
                ImageShadowSimulator()
                .simulate(theta, self.CTX, rng=np.random.default_rng(3))
                .metadata["vis_signal"]
            )
            rel = np.abs(moved - base_vis).max() / base_vis.max()
            assert rel > 1e-3, f"{name} does not move the visibilities"
            assert abs(like.loglike(theta, base, self.CTX) - ll0) > 1.0, name


class TestUnrepresentableRingIsRejectedNotFatal:
    """A geometry the grid cannot image must not abort a sampling run."""

    CTX = TestFluxParameterisation.CTX
    TRUTH = TestFluxParameterisation.TRUTH

    def test_build_ring_image_raises_for_a_degenerate_ring(self, monkeypatch):
        from whitesearch.simulators import image_shadow

        monkeypatch.setattr(
            image_shadow, "_gaussian_ring_image",
            lambda *a, **k: np.zeros((NPIX, NPIX)),
        )
        with pytest.raises(image_shadow.UnrepresentableRingError, match="between pixels"):
            image_shadow.build_ring_image(self.TRUTH, self.CTX)

    def test_likelihood_returns_minus_inf_and_counts_it(self, monkeypatch):
        from whitesearch.likelihoods import VisibilityLikelihood
        from whitesearch.likelihoods import visibility as vis_mod
        from whitesearch.simulators.image_shadow import UnrepresentableRingError

        data = ImageShadowSimulator().simulate(
            self.TRUTH, self.CTX, rng=np.random.default_rng(0)
        )
        like = VisibilityLikelihood("gr_eternal", use_closure_phases=False)
        assert np.isfinite(like.loglike(self.TRUTH, data, self.CTX))
        assert like.n_unrepresentable == 0

        def boom(*a, **k):
            raise UnrepresentableRingError("falls between pixels")

        monkeypatch.setattr(vis_mod, "build_ring_image", boom)
        assert like.loglike(self.TRUTH, data, self.CTX) == float("-inf")
        assert like.n_unrepresentable == 1
        assert "between pixels" in like.last_unrepresentable
