"""How long does it take to fit a realistic SEM image? How well do we do?"""

import math
import time

import numpy as np

import cProfile
import pstats

import sem_epe as epe
from sem_epe import vis

RNG = np.random.default_rng(0)

IMSIZE      = 512
PITCH       = 32
PERTURB_POS = 1.0   # std of position perturbation (pixels)
PERTURB_CD  = 0.5   # std of CD perturbation (pixels)
DEVIATION   = 6.0   # optimisation bound: nominal ± DEVIATION
NOISE_SIGMA = 0.2   # Gaussian noise added to the SEM image

# Generate SEM layout and fit layout. The SEM layout will be used to generate
# the SEM image, while the fit layout will serve as the starting point for the
# optimization algorithm.

sem_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.05)
fit_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.05)

sem_M1 = epe.Layer("M1", gray_value=0.3, z_order=1)
sem_V1 = epe.Layer("V1", gray_value=0.8, z_order=2)
sem_M2 = epe.Layer("M2", gray_value=0.2, z_order=3)

fit_M1 = epe.Layer("M1", gray_value=0.3, z_order=1)
fit_V1 = epe.Layer("V1", gray_value=0.8, z_order=2)
fit_M2 = epe.Layer("M2", gray_value=0.2, z_order=3)

sem_features_M1 = []
fit_features_M1 = []
delta_pos_M1 = []
delta_cd_M1 = []
for col in range(PITCH // 2, IMSIZE, 2 * PITCH):
    dp = RNG.normal(scale=PERTURB_POS)
    dc = RNG.normal(scale=PERTURB_CD)

    sem_feature = epe.Line(epe.Orientation.VERTICAL, thickness=PITCH // 2 + dc, position=col + dp)
    sem_M1.add_feature(sem_feature)

    fit_feature = epe.Line(epe.Orientation.VERTICAL, thickness=PITCH // 2, position=col)
    fit_M1.add_feature(fit_feature)

    sem_features_M1.append(sem_feature)
    fit_features_M1.append(fit_feature)
    delta_pos_M1.append(dp)
    delta_cd_M1.append(dc)
delta_pos_M1 = np.array(delta_pos_M1)
delta_cd_M1  = np.array(delta_cd_M1)

sem_features_V1 = []
fit_features_V1 = []
delta_x_V1 = []
delta_y_V1 = []
delta_cd_V1 = []
for col in range(PITCH // 2, IMSIZE, 2 * PITCH):
    for row in range(PITCH // 2, IMSIZE, PITCH):
        dx = RNG.normal(scale=PERTURB_POS)
        dy = RNG.normal(scale=PERTURB_POS)
        dc = RNG.normal(scale=PERTURB_CD)

        sem_feature = epe.Pillar(x=col + dx, y=row + dy, diameter=PITCH / math.sqrt(2) + dc)
        sem_V1.add_feature(sem_feature)

        fit_feature = epe.Pillar(x=col, y=row, diameter=PITCH / math.sqrt(2))
        fit_V1.add_feature(fit_feature)

        sem_features_V1.append(sem_feature)
        fit_features_V1.append(fit_feature)
        delta_x_V1.append(dx)
        delta_y_V1.append(dy)
        delta_cd_V1.append(dc)
delta_x_V1  = np.array(delta_x_V1)
delta_y_V1  = np.array(delta_y_V1)
delta_cd_V1 = np.array(delta_cd_V1)

sem_features_M2 = []
fit_features_M2 = []
delta_pos_M2 = []
delta_cd_M2 = []
for row in range(PITCH // 2, IMSIZE, PITCH):
    dp = RNG.normal(scale=PERTURB_POS)
    dc = RNG.normal(scale=PERTURB_CD)

    sem_feature = epe.Line(epe.Orientation.HORIZONTAL, thickness=PITCH // 2 + dc, position=row + dp)
    sem_M2.add_feature(sem_feature)

    fit_feature = epe.Line(epe.Orientation.HORIZONTAL, thickness=PITCH // 2, position=row)
    fit_M2.add_feature(fit_feature)

    sem_features_M2.append(sem_feature)
    fit_features_M2.append(fit_feature)
    delta_pos_M2.append(dp)
    delta_cd_M2.append(dc)
delta_pos_M2 = np.array(delta_pos_M2)
delta_cd_M2  = np.array(delta_cd_M2)

sem_layout.add_layer(sem_M1)
sem_layout.add_layer(sem_V1)
sem_layout.add_layer(sem_M2)

fit_layout.add_layer(fit_M1)
fit_layout.add_layer(fit_V1)
fit_layout.add_layer(fit_M2)

# Add noise to SEM layout to generate input SEM image.

clean = sem_layout.render()
noisy = np.clip(
    clean + RNG.normal(scale=NOISE_SIGMA, size=clean.shape).astype(np.float32),
    0.0, 1.0,
).astype(np.float32)
sem_image = epe.SEMImage(noisy, nm_per_pixel=1.0)

# Collect tunable parameters.

entries = []
for f in fit_features_M1:
    entries += [(f, "position",  -DEVIATION, DEVIATION),
                (f, "thickness", -DEVIATION, DEVIATION)]
for f in fit_features_V1:
    entries += [(f, "x",         -DEVIATION, DEVIATION),
                (f, "y",         -DEVIATION, DEVIATION),
                (f, "diameter",  -DEVIATION, DEVIATION)]
for f in fit_features_M2:
    entries += [(f, "position",  -DEVIATION, DEVIATION),
                (f, "thickness", -DEVIATION, DEVIATION)]
params = epe.ParameterSet(entries)
print(f"Number of tunable parameters: {len(params)}")

# Snapshot nominal values before fitting
nom_pos_M1 = np.array([f.position  for f in fit_features_M1])
nom_cd_M1  = np.array([f.thickness for f in fit_features_M1])
nom_x_V1   = np.array([f.x        for f in fit_features_V1])
nom_y_V1   = np.array([f.y        for f in fit_features_V1])
nom_cd_V1  = np.array([f.diameter for f in fit_features_V1])
nom_pos_M2 = np.array([f.position  for f in fit_features_M2])
nom_cd_M2  = np.array([f.thickness for f in fit_features_M2])

t0 = time.perf_counter()

#profiler = cProfile.Profile()
#profiler.enable()

result = epe.fit_per_feature(
    sem_image,
    fit_layout,
    params
)

#profiler.disable()
#stats = pstats.Stats(profiler)
#stats.sort_stats('cumtime').print_stats(20)

t1 = time.perf_counter()

# Report on timing and errors.

print(f"Fit: {(t1 - t0) * 1e3:.0f} ms")
print(f"RMS: {result.rms_error:.5f}")

# Recovered deltas and residual errors after fitting.
rec_pos_M1 = np.array([f.position  for f in fit_features_M1])
rec_cd_M1  = np.array([f.thickness for f in fit_features_M1])
rec_x_V1   = np.array([f.x        for f in fit_features_V1])
rec_y_V1   = np.array([f.y        for f in fit_features_V1])
rec_cd_V1  = np.array([f.diameter for f in fit_features_V1])
rec_pos_M2 = np.array([f.position  for f in fit_features_M2])
rec_cd_M2  = np.array([f.thickness for f in fit_features_M2])

rec_delta_pos_M1 = rec_pos_M1 - nom_pos_M1
rec_delta_cd_M1  = rec_cd_M1  - nom_cd_M1
rec_delta_x_V1   = rec_x_V1   - nom_x_V1
rec_delta_y_V1   = rec_y_V1   - nom_y_V1
rec_delta_cd_V1  = rec_cd_V1  - nom_cd_V1
rec_delta_pos_M2 = rec_pos_M2 - nom_pos_M2
rec_delta_cd_M2  = rec_cd_M2  - nom_cd_M2

err_pos_M1 = delta_pos_M1 - rec_delta_pos_M1
err_cd_M1  = delta_cd_M1  - rec_delta_cd_M1
err_x_V1   = delta_x_V1   - rec_delta_x_V1
err_y_V1   = delta_y_V1   - rec_delta_y_V1
err_cd_V1  = delta_cd_V1  - rec_delta_cd_V1
err_pos_M2 = delta_pos_M2 - rec_delta_pos_M2
err_cd_M2  = delta_cd_M2  - rec_delta_cd_M2

print("\nError statistics:")
hdr = f"{'Layer':<6} {'Variable':<12} {'mean':>12} {'L2':>12} {'max':>12}"
print(hdr)
print("-" * len(hdr))
rows = [
    ("M1", "position",  err_pos_M1),
    ("",   "thickness", err_cd_M1),
    ("V1", "x",         err_x_V1),
    ("",   "y",         err_y_V1),
    ("",   "diameter",  err_cd_V1),
    ("M2", "position",  err_pos_M2),
    ("",   "thickness", err_cd_M2),
]
for layer, var, err in rows:
    m, l2, mx = np.mean(np.abs(err)), np.linalg.norm(err), np.max(np.abs(err))
    print(f"{layer:<6} {var:<12} {m:9.3f} px {l2:9.3f} px {mx:9.3f} px")

# Visualize results. (FIXME: May want to make an image plot that helps highlight
# where the errors were made, rather than using a scatter plot.)

import matplotlib.pyplot as plt

fig, (ax_pos, ax_cd) = plt.subplots(1, 2, figsize=(12, 5))

def _diag(ax):
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)

ax_pos.scatter(delta_x_V1,   rec_delta_x_V1,   s=30, alpha=0.9, label="V1 X")
ax_pos.scatter(delta_y_V1,   rec_delta_y_V1,   s=30, alpha=0.9, label="V1 Y")
ax_pos.scatter(delta_pos_M1, rec_delta_pos_M1, s=30, alpha=0.9, label="M1 position")
ax_pos.scatter(delta_pos_M2, rec_delta_pos_M2, s=30, alpha=0.9, label="M2 position")

_diag(ax_pos)
ax_pos.set_xlabel("True Δposition (px)")
ax_pos.set_ylabel("Recovered Δposition (px)")
ax_pos.set_title("Position recovery")
ax_pos.legend(fontsize=8)

ax_cd.scatter(delta_cd_V1, rec_delta_cd_V1, s=30, alpha=0.9, label="V1 diameter")
ax_cd.scatter(delta_cd_M1, rec_delta_cd_M1, s=30, alpha=0.9, label="M1 thickness")
ax_cd.scatter(delta_cd_M2, rec_delta_cd_M2, s=30, alpha=0.9, label="M2 thickness")

_diag(ax_cd)
ax_cd.set_xlabel("True ΔCD (px)")
ax_cd.set_ylabel("Recovered ΔCD (px)")
ax_cd.set_title("CD recovery")
ax_cd.legend(fontsize=8)

fig.tight_layout()
plt.show()

vis.plot_fit(sem_image, result)

"""
512x512; single per-feature tuning pass:

Number of tunable parameters: 432
Fit: 1795 ms
RMS: 0.16578

Error statistics:
Layer  Variable             mean           L2          max
----------------------------------------------------------
M1     position         0.055 px     0.189 px     0.122 px
       thickness        0.468 px     1.513 px     1.107 px
V1     x                0.181 px     2.635 px     0.710 px
       y                0.095 px     1.475 px     0.535 px
       diameter         0.156 px     2.390 px     0.938 px
M2     position         0.053 px     0.245 px     0.114 px
       thickness        0.167 px     0.772 px     0.365 px
"""