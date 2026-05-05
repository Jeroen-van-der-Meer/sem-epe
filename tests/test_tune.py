import math

import numpy as np
import pytest

import sem_epe as epe

RNG = np.random.default_rng(0)


def _target(layout: epe.Layout) -> epe.SEMImage:
    return epe.SEMImage(layout.render().copy(), nm_per_pixel=1.0)


# ---------------------------------------------------------------------------
# Single line: recover CD
# ---------------------------------------------------------------------------

def test_recover_line_cd():
    """Optimizer recovers a line's thickness from a 0.8 px CD over-estimate."""
    line_ref = epe.Line(epe.Orientation.HORIZONTAL, thickness=10.0, position=50.0)
    layer = epe.Layer("l", gray_value=0.7, z_order=1)
    layer.add_feature(line_ref)
    ref = epe.Layout(100, 100, background=0.1)
    ref.add_layer(layer)

    line_fit = epe.Line(epe.Orientation.HORIZONTAL, thickness=10.8, position=50.0)
    layer_fit = epe.Layer("l", gray_value=0.7, z_order=1)
    layer_fit.add_feature(line_fit)
    fit_layout = epe.Layout(100, 100, background=0.1)
    fit_layout.add_layer(layer_fit)

    params = [epe.Parameter(line_fit, "thickness")]
    epe.fit(_target(ref), fit_layout, params)

    assert abs(line_fit.thickness - 10.0) < 0.05


# ---------------------------------------------------------------------------
# Multiple parallel lines: recover positions simultaneously
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("orientation,attr", [
    (epe.Orientation.HORIZONTAL, "position"),
    (epe.Orientation.VERTICAL,   "position"),
], ids=["horizontal", "vertical"])
def test_recover_parallel_line_positions(orientation, attr):
    """Optimizer recovers positions of three non-overlapping lines in one pass."""
    TRUE_POS = [25.0, 50.0, 75.0]
    DELTAS   = [ -0.7,  +0.5,  +0.3]

    layer_ref = epe.Layer("l", gray_value=1, z_order=1)
    for p in TRUE_POS:
        layer_ref.add_feature(epe.Line(orientation, thickness=8.0, position=p))
    ref_layout = epe.Layout(100, 100, background=0)
    ref_layout.add_layer(layer_ref)

    layer_fit = epe.Layer("l", gray_value=1, z_order=1)
    lines_fit = [
        epe.Line(orientation, thickness=8.0, position=p + d)
        for p, d in zip(TRUE_POS, DELTAS)
    ]
    for f in lines_fit:
        layer_fit.add_feature(f)
    fit_layout = epe.Layout(100, 100, background=0)
    fit_layout.add_layer(layer_fit)

    params = [epe.Parameter(f, attr) for f in lines_fit]
    epe.fit(_target(ref_layout), fit_layout, params)

    for f, p_true in zip(lines_fit, TRUE_POS):
        assert abs(getattr(f, attr) - p_true) < 0.05


# ---------------------------------------------------------------------------
# Complex two-layer layout: lines + pillars, all positions and CDs
# ---------------------------------------------------------------------------

def test_recover_multilayer_layout():
    """
    Optimizer recovers perturbed positions and CDs across two layers containing
    vertical lines, a horizontal line, and pillars (one obscuring the other).
    """
    # Reference
    m1 = epe.Layer("m1", gray_value=0.50, z_order=0)
    v1 = epe.Line(epe.Orientation.VERTICAL, thickness=12.0, position=35.0)
    v2 = epe.Line(epe.Orientation.VERTICAL, thickness=12.0, position=75.0)
    p1 = epe.Pillar(x=55.0, y=55.0, diameter=18.0)
    for f in (v1, v2, p1):
        m1.add_feature(f)

    m2 = epe.Layer("m2", gray_value=0.85, z_order=1)
    h1 = epe.Line(epe.Orientation.HORIZONTAL, thickness=10.0, position=55.0)
    p2 = epe.Pillar(x=55.0, y=55.0, diameter=10.0)   # fixed; sits on top of p1
    for f in (h1, p2):
        m2.add_feature(f)

    ref = epe.Layout(110, 110, background=0.05)
    ref.add_layer(m1)
    ref.add_layer(m2)

    # Perturbed starting point: 9 free parameters
    PERT = [0.6, -0.5, 0.4, -0.3, 0.5, -0.6, 0.3, -0.4, 0.5]

    m1f = epe.Layer("m1", gray_value=0.50, z_order=0)
    v1f = epe.Line(epe.Orientation.VERTICAL, thickness=12.0 + PERT[0], position=35.0 + PERT[1])
    v2f = epe.Line(epe.Orientation.VERTICAL, thickness=12.0 + PERT[2], position=75.0 + PERT[3])
    p1f = epe.Pillar(x=55.0 + PERT[4], y=55.0 + PERT[5], diameter=18.0 + PERT[6])
    for f in (v1f, v2f, p1f):
        m1f.add_feature(f)

    m2f = epe.Layer("m2", gray_value=0.85, z_order=1)
    h1f = epe.Line(epe.Orientation.HORIZONTAL, thickness=10.0 + PERT[7], position=55.0 + PERT[8])
    p2f = epe.Pillar(x=55.0, y=55.0, diameter=10.0)   # not tuned
    for f in (h1f, p2f):
        m2f.add_feature(f)

    fit_layout = epe.Layout(110, 110, background=0.05)
    fit_layout.add_layer(m1f)
    fit_layout.add_layer(m2f)

    params = [
        epe.Parameter(v1f, "thickness"), epe.Parameter(v1f, "position"),
        epe.Parameter(v2f, "thickness"), epe.Parameter(v2f, "position"),
        epe.Parameter(p1f, "x"),         epe.Parameter(p1f, "y"),
        epe.Parameter(p1f, "diameter" ),
        epe.Parameter(h1f, "thickness"), epe.Parameter(h1f, "position"),
    ]
    epe.fit(_target(ref), fit_layout, params)
    
    TOL = 0.01
    assert abs(v1f.thickness - 12.0) < TOL
    assert abs(v1f.position  - 35.0) < TOL
    assert abs(v2f.thickness - 12.0) < TOL
    assert abs(v2f.position  - 75.0) < TOL
    assert abs(p1f.x         - 55.0) < TOL
    assert abs(p1f.y         - 55.0) < TOL
    assert abs(p1f.diameter  - 18.0) < TOL
    assert abs(h1f.thickness - 10.0) < TOL
    assert abs(h1f.position  - 55.0) < TOL


# ---------------------------------------------------------------------------
# Recovery under Gaussian noise
# ---------------------------------------------------------------------------

def test_recover_under_noise():
    """
    Optimizer recovers a line's position from a noisy target image.
    Noise sigma = 0.01 (normalised gray); tolerance is relaxed to 0.3 px.
    """
    NOISE_SIGMA = 0.01
    TRUE_POS    = 50.0
    DELTA       = 0.6

    line_ref = epe.Line(epe.Orientation.HORIZONTAL, thickness=10.0, position=TRUE_POS)
    layer = epe.Layer("l", gray_value=0.7, z_order=1)
    layer.add_feature(line_ref)
    ref = epe.Layout(100, 100, background=0.1)
    ref.add_layer(layer)
    clean = ref.render().copy()
    noisy = np.clip(
        clean + RNG.normal(scale=NOISE_SIGMA, size=clean.shape).astype(np.float32),
        0.0, 1.0,
    ).astype(np.float32)
    target = epe.SEMImage(noisy, nm_per_pixel=1.0)

    line_fit = epe.Line(epe.Orientation.HORIZONTAL, thickness=10.0, position=TRUE_POS + DELTA)
    layer_fit = epe.Layer("l", gray_value=0.7, z_order=1)
    layer_fit.add_feature(line_fit)
    fit_layout = epe.Layout(100, 100, background=0.1)
    fit_layout.add_layer(layer_fit)

    params = [epe.Parameter(line_fit, "position")]
    epe.fit(target, fit_layout, params)

    assert abs(line_fit.position - TRUE_POS) < 0.3


# ---------------------------------------------------------------------------
# Benchmark regression: 512×512 three-layer layout, Gaussian noise
# ---------------------------------------------------------------------------

def test_benchmark_accuracy():
    """
    Regression: per-feature recovery accuracy on the full benchmark layout.

    Thresholds are the observed values from the reference run, rounded up
    to the nearest 0.2 px.  Marked slow (~4 s); run with ``-m slow``.
    """
    rng = np.random.default_rng(0)

    IMSIZE      = 512
    PITCH       = 32
    PERTURB_POS = 1.0
    PERTURB_CD  = 0.5
    NOISE_SIGMA = 0.2

    sem_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.05)
    fit_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.05)

    sem_M1 = epe.Layer("M1", gray_value=0.3, z_order=1)
    sem_V1 = epe.Layer("V1", gray_value=0.8, z_order=2)
    sem_M2 = epe.Layer("M2", gray_value=0.2, z_order=3)
    fit_M1 = epe.Layer("M1", gray_value=0.3, z_order=1)
    fit_V1 = epe.Layer("V1", gray_value=0.8, z_order=2)
    fit_M2 = epe.Layer("M2", gray_value=0.2, z_order=3)

    sem_features_M1, fit_features_M1, delta_pos_M1, delta_cd_M1 = [], [], [], []
    for col in range(PITCH // 2, IMSIZE, 2 * PITCH):
        dp, dc = rng.normal(scale=PERTURB_POS), rng.normal(scale=PERTURB_CD)
        sem_M1.add_feature(epe.Line(epe.Orientation.VERTICAL, thickness=PITCH // 2 + dc, position=col + dp))
        fit_f = epe.Line(epe.Orientation.VERTICAL, thickness=PITCH // 2, position=col)
        fit_M1.add_feature(fit_f)
        sem_features_M1.append(sem_M1.features[-1])
        fit_features_M1.append(fit_f)
        delta_pos_M1.append(dp); delta_cd_M1.append(dc)
    delta_pos_M1 = np.array(delta_pos_M1)
    delta_cd_M1  = np.array(delta_cd_M1)

    sem_features_V1, fit_features_V1, delta_x_V1, delta_y_V1, delta_cd_V1 = [], [], [], [], []
    for col in range(PITCH // 2, IMSIZE, 2 * PITCH):
        for row in range(PITCH // 2, IMSIZE, PITCH):
            dx, dy, dc = rng.normal(scale=PERTURB_POS), rng.normal(scale=PERTURB_POS), rng.normal(scale=PERTURB_CD)
            sem_V1.add_feature(epe.Pillar(x=col + dx, y=row + dy, diameter=PITCH / math.sqrt(2) + dc))
            fit_f = epe.Pillar(x=col, y=row, diameter=PITCH / math.sqrt(2))
            fit_V1.add_feature(fit_f)
            sem_features_V1.append(sem_V1.features[-1])
            fit_features_V1.append(fit_f)
            delta_x_V1.append(dx); delta_y_V1.append(dy); delta_cd_V1.append(dc)
    delta_x_V1  = np.array(delta_x_V1)
    delta_y_V1  = np.array(delta_y_V1)
    delta_cd_V1 = np.array(delta_cd_V1)

    sem_features_M2, fit_features_M2, delta_pos_M2, delta_cd_M2 = [], [], [], []
    for row in range(PITCH // 2, IMSIZE, PITCH):
        dp, dc = rng.normal(scale=PERTURB_POS), rng.normal(scale=PERTURB_CD)
        sem_M2.add_feature(epe.Line(epe.Orientation.HORIZONTAL, thickness=PITCH // 2 + dc, position=row + dp))
        fit_f = epe.Line(epe.Orientation.HORIZONTAL, thickness=PITCH // 2, position=row)
        fit_M2.add_feature(fit_f)
        sem_features_M2.append(sem_M2.features[-1])
        fit_features_M2.append(fit_f)
        delta_pos_M2.append(dp); delta_cd_M2.append(dc)
    delta_pos_M2 = np.array(delta_pos_M2)
    delta_cd_M2  = np.array(delta_cd_M2)

    sem_layout.add_layer(sem_M1); sem_layout.add_layer(sem_V1); sem_layout.add_layer(sem_M2)
    fit_layout.add_layer(fit_M1); fit_layout.add_layer(fit_V1); fit_layout.add_layer(fit_M2)

    clean = sem_layout.render()
    noisy = np.clip(clean + rng.normal(scale=NOISE_SIGMA, size=clean.shape), 0.0, 1.0)
    sem_image = epe.SEMImage(noisy, nm_per_pixel=1.0)

    # Snapshot nominals before the fit mutates features in-place.
    nom_pos_M1 = np.array([f.position  for f in fit_features_M1])
    nom_cd_M1  = np.array([f.thickness for f in fit_features_M1])
    nom_x_V1   = np.array([f.x         for f in fit_features_V1])
    nom_y_V1   = np.array([f.y         for f in fit_features_V1])
    nom_cd_V1  = np.array([f.diameter  for f in fit_features_V1])
    nom_pos_M2 = np.array([f.position  for f in fit_features_M2])
    nom_cd_M2  = np.array([f.thickness for f in fit_features_M2])

    params = (
        [epe.Parameter(f, "position")  for f in fit_features_M1] +
        [epe.Parameter(f, "thickness") for f in fit_features_M1] +
        [epe.Parameter(f, "x")         for f in fit_features_V1] +
        [epe.Parameter(f, "y")         for f in fit_features_V1] +
        [epe.Parameter(f, "diameter")  for f in fit_features_V1] +
        [epe.Parameter(f, "position")  for f in fit_features_M2] +
        [epe.Parameter(f, "thickness") for f in fit_features_M2]
    )
    epe.fit(sem_image, fit_layout, params)

    def _err(true_delta, fit_features, attr, nom):
        rec = np.array([getattr(f, attr) for f in fit_features])
        return np.abs(true_delta - (rec - nom))

    # (mean_tol, rms_tol, max_tol) — current values rounded up to nearest 0.2 px
    cases = [
        ("M1 position",  _err(delta_pos_M1, fit_features_M1, "position",  nom_pos_M1), (0.2, 0.2, 0.2)),
        ("M1 thickness", _err(delta_cd_M1,  fit_features_M1, "thickness", nom_cd_M1),  (0.2, 0.2, 0.2)),
        ("V1 x",         _err(delta_x_V1,   fit_features_V1, "x",         nom_x_V1),   (0.2, 0.4, 0.8)),
        ("V1 y",         _err(delta_y_V1,   fit_features_V1, "y",         nom_y_V1),   (0.2, 0.2, 0.8)),
        ("V1 diameter",  _err(delta_cd_V1,  fit_features_V1, "diameter",  nom_cd_V1),  (0.2, 0.4, 1.2)),
        ("M2 position",  _err(delta_pos_M2, fit_features_M2, "position",  nom_pos_M2), (0.2, 0.2, 0.2)),
        ("M2 thickness", _err(delta_cd_M2,  fit_features_M2, "thickness", nom_cd_M2),  (0.2, 0.2, 0.4)),
    ]
    for name, err, (t_mean, t_rms, t_max) in cases:
        assert np.mean(err)              <= t_mean, f"{name}: mean {np.mean(err):.3f} > {t_mean}"
        assert np.sqrt(np.mean(err**2))  <= t_rms,  f"{name}: rms {np.sqrt(np.mean(err**2)):.3f} > {t_rms}"
        assert np.max(err)               <= t_max,  f"{name}: max {np.max(err):.3f} > {t_max}"
