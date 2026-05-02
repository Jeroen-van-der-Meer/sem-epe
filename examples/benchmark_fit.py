"""How long does it take to fit a realistic SEM image? How well do we do?"""

import math
import time

import numpy as np

import sem_epe as epe
from sem_epe import vis

RNG = np.random.default_rng(0)

IMSIZE      = 512
PITCH       = 32
PERTURB_POS = 1.0    # std dev of position / centre perturbation (pixels)
PERTURB_CD  = 0.5    # std dev of CD (thickness / diameter) perturbation (pixels)
DEVIATION   = 6.0    # optimisation bound: nominal ± DEVIATION
NOISE_SIGMA = 0.02   # Gaussian noise added to the reference render

# ---------------------------------------------------------------------------
# Reference layout
# ---------------------------------------------------------------------------

ref_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.05)

ref_m1   = epe.Layer("metal1", gray_value=0.3, z_order=1)
ref_via1 = epe.Layer("via1",   gray_value=0.8, z_order=2)
ref_m2   = epe.Layer("metal2", gray_value=0.2, z_order=3)

ref_m1_feats   = []
for col in range(PITCH // 2, IMSIZE, 2 * PITCH):
    f = epe.Line(epe.Orientation.VERTICAL, thickness=PITCH // 2, position=col)
    ref_m1.add_feature(f)
    ref_m1_feats.append(f)

ref_via1_feats = []
for col in range(PITCH // 2, IMSIZE, 2 * PITCH):
    for row in range(PITCH // 2, IMSIZE, PITCH):
        f = epe.Pillar(x=col, y=row, diameter=PITCH / math.sqrt(2))
        ref_via1.add_feature(f)
        ref_via1_feats.append(f)

ref_m2_feats   = []
for row in range(PITCH // 2, IMSIZE, PITCH):
    f = epe.Line(epe.Orientation.HORIZONTAL, thickness=PITCH // 2, position=row)
    ref_m2.add_feature(f)
    ref_m2_feats.append(f)

ref_layout.add_layer(ref_m1)
ref_layout.add_layer(ref_via1)
ref_layout.add_layer(ref_m2)

clean = ref_layout.render().copy()
noisy = np.clip(
    clean + RNG.normal(scale=NOISE_SIGMA, size=clean.shape).astype(np.float32),
    0.0, 1.0,
).astype(np.float32)
ref_image = epe.SEMImage(noisy, nm_per_pixel=1.0)

# ---------------------------------------------------------------------------
# Perturbed (fit) layout
# ---------------------------------------------------------------------------

fit_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.05)

fit_m1   = epe.Layer("metal1", gray_value=0.3, z_order=1)
fit_via1 = epe.Layer("via1",   gray_value=0.8, z_order=2)
fit_m2   = epe.Layer("metal2", gray_value=0.2, z_order=3)

# programmed_deltas[i] = (delta_pos, delta_cd) applied to fit feature i
m1_deltas_prog   = []
fit_m1_feats = []
for ref in ref_m1_feats:
    dp, dc = RNG.normal(scale=PERTURB_POS), RNG.normal(scale=PERTURB_CD)
    f = epe.Line(
        epe.Orientation.VERTICAL,
        thickness=ref.thickness + dc,
        position =ref.position  + dp,
    )
    fit_m1.add_feature(f)
    fit_m1_feats.append(f)
    m1_deltas_prog.append((dp, dc))

via1_deltas_prog = []
fit_via1_feats = []
for ref in ref_via1_feats:
    dx, dy, dc = RNG.normal(scale=PERTURB_POS), RNG.normal(scale=PERTURB_POS), RNG.normal(scale=PERTURB_CD)
    f = epe.Pillar(
        x       =ref.x        + dx,
        y       =ref.y        + dy,
        diameter=ref.diameter + dc,
    )
    fit_via1.add_feature(f)
    fit_via1_feats.append(f)
    via1_deltas_prog.append((dx, dy, dc))

m2_deltas_prog   = []
fit_m2_feats = []
for ref in ref_m2_feats:
    dp, dc = RNG.normal(scale=PERTURB_POS), RNG.normal(scale=PERTURB_CD)
    f = epe.Line(
        epe.Orientation.HORIZONTAL,
        thickness=ref.thickness + dc,
        position =ref.position  + dp,
    )
    fit_m2.add_feature(f)
    fit_m2_feats.append(f)
    m2_deltas_prog.append((dp, dc))

fit_layout.add_layer(fit_m1)
fit_layout.add_layer(fit_via1)
fit_layout.add_layer(fit_m2)

entries = []
for f in fit_m1_feats:
    entries += [(f, "position",  -DEVIATION, DEVIATION),
                (f, "thickness", -DEVIATION, DEVIATION)]
for f in fit_via1_feats:
    entries += [(f, "x",        -DEVIATION, DEVIATION),
                (f, "y",        -DEVIATION, DEVIATION),
                (f, "diameter", -DEVIATION, DEVIATION)]
for f in fit_m2_feats:
    entries += [(f, "position",  -DEVIATION, DEVIATION),
                (f, "thickness", -DEVIATION, DEVIATION)]
params = epe.ParameterSet(entries)
print(f"Number of tunable parameters: {len(params)}")

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

t0 = time.perf_counter()

result = epe.fit(
    ref_image,
    fit_layout,
    params,
    ftol=1e-5,
    max_nfev=500
)

t1 = time.perf_counter()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print(f"Fit: {(t1 - t0) * 1e3:.0f} ms")
print(f"RMS: {result.rms_error:.5f}")

def _line_errors(fit_fs, ref_fs):
    pos = np.array([abs(f.position  - r.position)  for f, r in zip(fit_fs, ref_fs)])
    cd  = np.array([abs(f.thickness - r.thickness) for f, r in zip(fit_fs, ref_fs)])
    return pos, cd

def _pillar_errors(fit_fs, ref_fs):
    pos = np.array([math.hypot(f.x - r.x, f.y - r.y) for f, r in zip(fit_fs, ref_fs)])
    cd  = np.array([abs(f.diameter - r.diameter)      for f, r in zip(fit_fs, ref_fs)])
    return pos, cd

m1_pos, m1_cd = _line_errors(fit_m1_feats,   ref_m1_feats)
v1_pos, v1_cd = _pillar_errors(fit_via1_feats, ref_via1_feats)
m2_pos, m2_cd = _line_errors(fit_m2_feats,   ref_m2_feats)

print("\nError statistics:")
hdr = f"{'Layer':<10} {'pos mean':>10} {'pos max':>10} {'CD mean':>10} {'CD max':>10}"
print(hdr)
print("-" * len(hdr))
for name, pos, cd in [("metal1", m1_pos, m1_cd),
                       ("via1",   v1_pos, v1_cd),
                       ("metal2", m2_pos, m2_cd)]:
    print(f"{name:<10} {pos.mean():>9.3f}px {pos.max():>9.3f}px "
          f"{cd.mean():>9.3f}px {cd.max():>9.3f}px")

"""
Number of tunable parameters: 432
Fit: 3108 ms
RMS: 0.02388

Error statistics:
Layer        pos mean    pos max    CD mean     CD max
------------------------------------------------------
metal1         0.138px     0.796px     0.261px     1.527px
via1           0.049px     0.227px     0.019px     0.079px
metal2         0.038px     0.230px     0.073px     0.449px

Not exactly great. Gradients may be dominated by visible features at top?
"""

# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt

fig, (ax_pos, ax_cd) = plt.subplots(1, 2, figsize=(12, 5))

def _diag(ax):
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="perfect recovery")
    ax.axhline(0, color="gray", lw=0.6, linestyle=":")

prog  = np.array([dp for dp, _ in m1_deltas_prog])
corr  = prog - np.array([f.position - r.position for f, r in zip(fit_m1_feats, ref_m1_feats)])
ax_pos.scatter(prog, corr, s=20, alpha=0.9, label="M1 position")

for attr, col in [("x", 0), ("y", 1)]:
    prog = np.array([d[col] for d in via1_deltas_prog])
    corr = prog - np.array([getattr(f, attr) - getattr(r, attr)
                             for f, r in zip(fit_via1_feats, ref_via1_feats)])
    ax_pos.scatter(prog, corr, s=20, alpha=0.9, label=f"V1 {attr}")

prog  = np.array([dp for dp, _ in m2_deltas_prog])
corr  = prog - np.array([f.position - r.position for f, r in zip(fit_m2_feats, ref_m2_feats)])
ax_pos.scatter(prog, corr, s=20, alpha=0.9, label="M2 position")

_diag(ax_pos)
ax_pos.set_xlabel("Programmed Δposition (px)")
ax_pos.set_ylabel("Correction applied (px)")
ax_pos.set_title("Position recovery")
ax_pos.legend(fontsize=8)

prog  = np.array([dc for _, dc in m1_deltas_prog])
corr  = prog - np.array([f.thickness - r.thickness for f, r in zip(fit_m1_feats, ref_m1_feats)])
ax_cd.scatter(prog, corr, s=20, alpha=0.9, label="M1 thickness")

prog  = np.array([dc for *_, dc in via1_deltas_prog])
corr  = prog - np.array([f.diameter - r.diameter for f, r in zip(fit_via1_feats, ref_via1_feats)])
ax_cd.scatter(prog, corr, s=20, alpha=0.9, label="via diameter")

prog  = np.array([dc for _, dc in m2_deltas_prog])
corr  = prog - np.array([f.thickness - r.thickness for f, r in zip(fit_m2_feats, ref_m2_feats)])
ax_cd.scatter(prog, corr, s=20, alpha=0.9, label="M2 thickness")

_diag(ax_cd)
ax_cd.set_xlabel("Programmed ΔCD (px)")
ax_cd.set_ylabel("Correction applied (px)")
ax_cd.set_title("CD recovery")
ax_cd.legend(fontsize=8)

fig.tight_layout()
plt.show()

vis.plot_fit(ref_image, result)

