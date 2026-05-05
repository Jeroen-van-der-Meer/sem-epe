"""Two-layer benchmark: horizontal segments with pillars centred on top."""

import time

import matplotlib.pyplot as plt
import numpy as np

import sem_epe as epe
from sem_epe import vis

RNG = np.random.default_rng(0)

IMSIZE      = 500
PERIOD      = 100
SEG_LENGTH  = 72
SEG_WIDTH   = 20
SEG_ROUND   = SEG_WIDTH / 2   # full roundedness → semicircular ends
PIL_CD      = 20.0

PERTURB_POS_LINE   = 1.0
PERTURB_CD_LINE    = 0.5
PERTURB_POS_PILLAR = 2.0
PERTURB_CD_PILLAR  = 5.0
NOISE_SIGMA        = 0.15

# Grid of feature centres: (x=col, y=row)
centers = [
    (PERIOD // 2 + col * PERIOD, PERIOD // 2 + row * PERIOD)
    for row in range(IMSIZE // PERIOD)
    for col in range(IMSIZE // PERIOD)
]

# ---------------------------------------------------------------------------
# Build SEM (ground-truth) and fit (nominal starting point) layouts.
# ---------------------------------------------------------------------------

sem_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.5)
fit_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.5)

sem_seg_layer = epe.Layer("segments", gray_value=0.7, z_order=1)
sem_pil_layer = epe.Layer("pillars",  gray_value=0.9, z_order=2)
fit_seg_layer = epe.Layer("segments", gray_value=0.7, z_order=1)
fit_pil_layer = epe.Layer("pillars",  gray_value=0.9, z_order=2)

sem_segs, fit_segs = [], []
sem_pils, fit_pils = [], []
delta_x_seg, delta_y_seg, delta_len_seg, delta_w_seg = [], [], [], []
delta_x_pil, delta_y_pil, delta_cd_pil               = [], [], []

for x, y in centers:
    # Segment
    dx, dy = RNG.normal(scale=PERTURB_POS_LINE), RNG.normal(scale=PERTURB_POS_LINE)
    dl, dw = RNG.normal(scale=PERTURB_CD_LINE),  RNG.normal(scale=PERTURB_CD_LINE)

    sem_f = epe.Segment(
        epe.Orientation.HORIZONTAL,
        length=SEG_LENGTH + dl, thickness=SEG_WIDTH + dw,
        x=x + dx, y=y + dy,
        roundedness=SEG_ROUND,
    )
    sem_seg_layer.add_feature(sem_f)

    fit_f = epe.Segment(
        epe.Orientation.HORIZONTAL,
        length=SEG_LENGTH, thickness=SEG_WIDTH,
        x=x, y=y,
        roundedness=SEG_ROUND,
    )
    fit_seg_layer.add_feature(fit_f)

    sem_segs.append(sem_f); fit_segs.append(fit_f)
    delta_x_seg.append(dx);   delta_y_seg.append(dy)
    delta_len_seg.append(dl); delta_w_seg.append(dw)

    # Pillar (same centre as segment)
    dx, dy = RNG.normal(scale=PERTURB_POS_PILLAR), RNG.normal(scale=PERTURB_POS_PILLAR)
    dc     = RNG.normal(scale=PERTURB_CD_PILLAR)

    sem_f = epe.Pillar(x=x + dx, y=y + dy, diameter=PIL_CD + dc)
    sem_pil_layer.add_feature(sem_f)

    fit_f = epe.Pillar(x=x, y=y, diameter=PIL_CD)
    fit_pil_layer.add_feature(fit_f)

    sem_pils.append(sem_f); fit_pils.append(fit_f)
    delta_x_pil.append(dx); delta_y_pil.append(dy); delta_cd_pil.append(dc)

delta_x_seg   = np.array(delta_x_seg);   delta_y_seg   = np.array(delta_y_seg)
delta_len_seg = np.array(delta_len_seg); delta_w_seg   = np.array(delta_w_seg)
delta_x_pil   = np.array(delta_x_pil);  delta_y_pil   = np.array(delta_y_pil)
delta_cd_pil  = np.array(delta_cd_pil)

sem_layout.add_layer(sem_seg_layer); sem_layout.add_layer(sem_pil_layer)
fit_layout.add_layer(fit_seg_layer); fit_layout.add_layer(fit_pil_layer)

# ---------------------------------------------------------------------------
# Render noisy SEM image.
# ---------------------------------------------------------------------------

clean = sem_layout.render()
noisy = np.clip(clean + RNG.normal(scale=NOISE_SIGMA, size=clean.shape), 0.0, 1.0)
sem_image = epe.SEMImage(noisy, nm_per_pixel=1.0)

plt.imshow(noisy)
plt.show()

# ---------------------------------------------------------------------------
# Collect tunable parameters and snapshot nominal values.
# ---------------------------------------------------------------------------

nom_x_seg   = np.array([f.x         for f in fit_segs])
nom_y_seg   = np.array([f.y         for f in fit_segs])
nom_len_seg = np.array([f.length    for f in fit_segs])
nom_w_seg   = np.array([f.thickness for f in fit_segs])
nom_x_pil   = np.array([f.x        for f in fit_pils])
nom_y_pil   = np.array([f.y        for f in fit_pils])
nom_cd_pil  = np.array([f.diameter for f in fit_pils])

params = (
    [epe.Parameter(f, "x")         for f in fit_segs] +
    [epe.Parameter(f, "y")         for f in fit_segs] +
    [epe.Parameter(f, "length")    for f in fit_segs] +
    [epe.Parameter(f, "thickness") for f in fit_segs] +
    [epe.Parameter(f, "x")         for f in fit_pils] +
    [epe.Parameter(f, "y")         for f in fit_pils] +
    [epe.Parameter(f, "diameter")  for f in fit_pils]
)
print(f"Number of tunable parameters: {len(params)}")

# ---------------------------------------------------------------------------
# Fit.
# ---------------------------------------------------------------------------

t0 = time.perf_counter()
epe.fit(sem_image, fit_layout, params)
t1 = time.perf_counter()

residual = fit_layout.image - sem_image.image
rms = np.sqrt(np.mean(residual ** 2))
print(f"Fit: {(t1 - t0) * 1e3:.0f} ms")
print(f"RMS: {rms:.5f}")

# ---------------------------------------------------------------------------
# Error statistics.
# ---------------------------------------------------------------------------

rec_x_seg   = np.array([f.x         for f in fit_segs])
rec_y_seg   = np.array([f.y         for f in fit_segs])
rec_len_seg = np.array([f.length    for f in fit_segs])
rec_w_seg   = np.array([f.thickness for f in fit_segs])
rec_x_pil   = np.array([f.x        for f in fit_pils])
rec_y_pil   = np.array([f.y        for f in fit_pils])
rec_cd_pil  = np.array([f.diameter for f in fit_pils])

err_x_seg   = delta_x_seg   - (rec_x_seg   - nom_x_seg)
err_y_seg   = delta_y_seg   - (rec_y_seg   - nom_y_seg)
err_len_seg = delta_len_seg - (rec_len_seg - nom_len_seg)
err_w_seg   = delta_w_seg   - (rec_w_seg   - nom_w_seg)
err_x_pil   = delta_x_pil   - (rec_x_pil   - nom_x_pil)
err_y_pil   = delta_y_pil   - (rec_y_pil   - nom_y_pil)
err_cd_pil  = delta_cd_pil  - (rec_cd_pil  - nom_cd_pil)

print("\nError statistics:")
hdr = f"{'Layer':<10} {'Variable':<12} {'mean':>12} {'rms':>12} {'max':>12}"
print(hdr)
print("-" * len(hdr))
stats = [
    ("segments", "x",         err_x_seg),
    ("",         "y",         err_y_seg),
    ("",         "length",    err_len_seg),
    ("",         "thickness", err_w_seg),
    ("pillars",  "x",         err_x_pil),
    ("",         "y",         err_y_pil),
    ("",         "diameter",  err_cd_pil),
]
for layer, var, err in stats:
    m  = np.mean(np.abs(err))
    l2 = np.sqrt(np.mean(err ** 2))
    mx = np.max(np.abs(err))
    print(f"{layer:<10} {var:<12} {m:9.3f} px {l2:9.3f} px {mx:9.3f} px")

# ---------------------------------------------------------------------------
# Visualise.
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt

fig, (ax_pos, ax_cd) = plt.subplots(1, 2, figsize=(12, 5))


def _diag(ax):
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)


ax_pos.scatter(delta_x_seg, rec_x_seg - nom_x_seg, s=30, alpha=0.9, label="segment X")
ax_pos.scatter(delta_y_seg, rec_y_seg - nom_y_seg, s=30, alpha=0.9, label="segment Y")
ax_pos.scatter(delta_x_pil, rec_x_pil - nom_x_pil, s=30, alpha=0.9, label="pillar X")
ax_pos.scatter(delta_y_pil, rec_y_pil - nom_y_pil, s=30, alpha=0.9, label="pillar Y")
_diag(ax_pos)
ax_pos.set_xlabel("True Δposition (px)")
ax_pos.set_ylabel("Recovered Δposition (px)")
ax_pos.set_title("Position recovery")
ax_pos.legend(fontsize=8)

ax_cd.scatter(delta_len_seg, rec_len_seg - nom_len_seg, s=30, alpha=0.9, label="segment length")
ax_cd.scatter(delta_w_seg,   rec_w_seg   - nom_w_seg,   s=30, alpha=0.9, label="segment thickness")
ax_cd.scatter(delta_cd_pil,  rec_cd_pil  - nom_cd_pil,  s=30, alpha=0.9, label="pillar diameter")
_diag(ax_cd)
ax_cd.set_xlabel("True ΔCD (px)")
ax_cd.set_ylabel("Recovered ΔCD (px)")
ax_cd.set_title("CD recovery")
ax_cd.legend(fontsize=8)

fig.tight_layout()
plt.show()

vis.plot_overlay(sem_image, fit_layout)
