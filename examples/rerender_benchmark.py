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

img_initial = layout.render().copy()
print(f"Layout:  {layout}")
print(f"Image:   shape={img_initial.shape}, "
      f"min={img_initial.min():.2f}, max={img_initial.max():.2f}")

t2 = time.perf_counter()

# --- Incremental re-render of pillar ---

pillar = via1.features[0]
for i in range(nstep):
    pillar.y += 1
    layout.rerender_feature(pillar)

t3 = time.perf_counter()

# --- Incremental re-render of pillar ---

line = metal1.features[0]
for i in range(nstep):
    line.position += 1
    layout.rerender_feature(line)

t4 = time.perf_counter()

print(f"Setup time:                 {t1 - t0:.6f} s")
print(f"Initial render time:        {t2 - t1:.6f} s")
print(f"Re-render pillar loop time: {t3 - t2:.6f} s")
print(f"Re-render line loop time:   {t4 - t3:.6f} s")

"""
Setup time:                 0.000137 s
Initial render time:        0.119352 s
Re-render pillar loop time: 0.059887 s
Re-render line loop time:   0.223204 s
"""