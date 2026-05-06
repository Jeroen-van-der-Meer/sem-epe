"""Two-layer benchmark: horizontal segments with pillars centred on top."""

import time

import matplotlib.pyplot as plt
import numpy as np

import sem_epe as epe
from sem_epe import vis

IMSIZE      = 755
PERIOD      = 164
SEG_LENGTH  = 120
SEG_WIDTH   = 24
SEG_ROUND   = SEG_WIDTH / 2   # full roundedness → semicircular ends
PIL_CD      = 53

# Grid of feature centres: (x=col, y=row)
centers = ([
    (157 + col * PERIOD, 70 + row * PERIOD)
    for row in range(4)
    for col in range(4)
] + [
    (157 - PERIOD // 2 + col * PERIOD, 70 + PERIOD // 2 + row * PERIOD)
    for row in range(4)
    for col in range(4)
])

fit_layout = epe.Layout(height=IMSIZE, width=IMSIZE, background=0.2)

bot = epe.Layer("Bottom", gray_value = 0.8, z_order = 1)
top = epe.Layer("Top", gray_value = 0.5, z_order = 2, invert = True, transparency = 0.5)

for x, y in centers:
    f = epe.Segment(
        epe.Orientation.HORIZONTAL,
        length=SEG_LENGTH, thickness=SEG_WIDTH,
        x=x, y=y,
        roundedness=SEG_ROUND,
    )
    bot.add_feature(f)

    f = epe.Pillar(x=x, y=y, diameter=PIL_CD)
    top.add_feature(f)

fit_layout.add_layer(bot)
fit_layout.add_layer(top)

params = (
    [epe.Parameter(f, "x")         for f in bot.features] +
    [epe.Parameter(f, "y")         for f in bot.features] +
    [epe.Parameter(f, "length")    for f in bot.features] +
    [epe.Parameter(f, "thickness") for f in bot.features] +
    [epe.Parameter(f, "x")         for f in top.features] +
    [epe.Parameter(f, "y")         for f in top.features] +
    [epe.Parameter(f, "diameter")  for f in top.features]
)

plt.imshow(fit_layout.render(), cmap="gray", vmin=0, vmax=1)
plt.show()
