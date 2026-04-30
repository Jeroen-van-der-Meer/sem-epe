"""How long does it take to re-render an image?"""

import time

from sem_epe import Layer, Layout, Line, Pillar, Orientation

nstep = 100

t0 = time.perf_counter()

# --- Setup ---

layout = Layout(height=2048, width=2048, background=0.1)

metal1 = Layer(name="metal1", gray_value=0.5, z_order=0)
metal1.add_feature(Line(Orientation.VERTICAL, thickness=32, position=1000))
layout.add_layer(metal1)

via1 = Layer(name="via1", gray_value=0.3, z_order=2)
via1.add_feature(Pillar(x=1000, y=1000 - nstep // 2, diameter=40))
layout.add_layer(via1)

metal2 = Layer(name="metal2", gray_value=0.9, z_order=3)
metal2.add_feature(Line(Orientation.HORIZONTAL, thickness=32, position=1000))
layout.add_layer(metal2)

t1 = time.perf_counter()

# --- Initial render ---

img_initial = layout.render()
print(f"Layout:  {layout}")
print(f"Image:   shape={img_initial.shape}, "
      f"min={img_initial.min():.2f}, max={img_initial.max():.2f}")

t2 = time.perf_counter()

# --- Incremental re-render ---

pillar = via1.features[0]
for i in range(nstep):
    pillar.y += 1
    img_updated = layout.rerender_feature(pillar)

t3 = time.perf_counter()

print(f"Setup time:              {t1 - t0:.6f} s")
print(f"Initial render time:     {t2 - t1:.6f} s")
print(f"Re-render loop time:     {t3 - t2:.6f} s")
print(f"Re-render time per step: {(t3 - t2)/nstep:.6f} s")

"""
Last run on my T14:

Setup time:              0.000125 s
Initial render time:     0.078246 s
Re-render loop time:     6.325459 s
Re-render time per step: 0.063255 s

That's out of spec by many orders of magnitude
"""