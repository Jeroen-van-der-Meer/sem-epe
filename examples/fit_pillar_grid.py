"""
Fit a grid of pillars back to a reference render.

Workflow
--------
1. Build a 4x4 grid of pillars on a single layer and render the reference.
2. Perturb each pillar's (x, y) position by a small random offset.
3. Call fit() to recover the original positions.
4. Report per-pillar position error before and after fitting.
"""

import numpy as np

import sem_epe as epe

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Layout parameters
# ---------------------------------------------------------------------------
CANVAS    = 200       # pixels
N_GRID    = 4         # pillars per side
DIAMETER  = 12.0      # pixels
SPACING   = 40        # pixels, centre-to-centre
OFFSET    = 30        # pixels, position of top-left pillar centre
GRAY      = 0.8
BG        = 0.1
NM_PX     = 2.5       # nm per pixel (metadata only, not used by the solver)
PERTURB   = 2.0       # std-dev of position perturbation in pixels

# ---------------------------------------------------------------------------
# Build reference layout
# ---------------------------------------------------------------------------
layer = epe.Layer("pillars", gray_value=GRAY, z_order=1)
pillars = []

for row in range(N_GRID):
    for col in range(N_GRID):
        x = OFFSET + col * SPACING
        y = OFFSET + row * SPACING
        p = epe.Pillar(x=float(x), y=float(y), diameter=DIAMETER)
        layer.add_feature(p)
        pillars.append(p)

ref_layout = epe.Layout(CANVAS, CANVAS, background=BG)
ref_layout.add_layer(layer)
ref_image = epe.SEMImage(ref_layout.render().copy(), nm_per_pixel=NM_PX)

# ---------------------------------------------------------------------------
# Perturbed starting layout (separate Layout so we don't share feature objects)
# ---------------------------------------------------------------------------
fit_layer = epe.Layer("pillars", gray_value=GRAY, z_order=1)
fit_pillars = []

for p_ref in pillars:
    dx, dy = RNG.normal(scale=PERTURB, size=2)
    p = epe.Pillar(x=p_ref.x + dx, y=p_ref.y + dy, diameter=DIAMETER)
    fit_layer.add_feature(p)
    fit_pillars.append(p)

fit_layout = epe.Layout(CANVAS, CANVAS, background=BG)
fit_layout.add_layer(fit_layer)

params = epe.ParameterSet.for_features(fit_pillars, ["x", "y"],
                                       deviation=3 * PERTURB)

# ---------------------------------------------------------------------------
# Errors before fitting
# ---------------------------------------------------------------------------
def position_errors(fit_ps, ref_ps):
    return np.array([
        np.hypot(fp.x - rp.x, fp.y - rp.y)
        for fp, rp in zip(fit_ps, ref_ps)
    ])

err_before = position_errors(fit_pillars, pillars)
print(f"Position error before fit — mean: {err_before.mean():.3f} px, "
      f"max: {err_before.max():.3f} px")

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
result = epe.fit(
    ref_image,
    fit_layout,
    params,
    ftol=1e-6,
    max_nfev=500,
    verbose=1,
)

# ---------------------------------------------------------------------------
# Errors after fitting
# ---------------------------------------------------------------------------
err_after = position_errors(fit_pillars, pillars)
print(f"Position error after fit  — mean: {err_after.mean():.3f} px, "
      f"max: {err_after.max():.3f} px")
print(f"Converged: {result.success}  |  {result.optimizer_result.message}")
print(f"RMS pixel error: {result.rms_error:.5f}")
