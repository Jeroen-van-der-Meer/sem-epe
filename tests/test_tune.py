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

    params = [epe.Parameter(line_fit, "thickness", -3.0, 3.0)]
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

    params = [epe.Parameter(f, attr, -3.0, 3.0) for f in lines_fit]
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
        epe.Parameter(v1f, "thickness", -3, 3), epe.Parameter(v1f, "position", -3, 3),
        epe.Parameter(v2f, "thickness", -3, 3), epe.Parameter(v2f, "position", -3, 3),
        epe.Parameter(p1f, "x",         -3, 3), epe.Parameter(p1f, "y",        -3, 3),
        epe.Parameter(p1f, "diameter",  -3, 3),
        epe.Parameter(h1f, "thickness", -3, 3), epe.Parameter(h1f, "position",  -3, 3),
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

    params = [epe.Parameter(line_fit, "position", -3.0, 3.0)]
    epe.fit(target, fit_layout, params)

    assert abs(line_fit.position - TRUE_POS) < 0.3
