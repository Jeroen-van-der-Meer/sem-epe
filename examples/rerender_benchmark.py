"""How long does it take to re-render an image?"""

import time
import matplotlib.pyplot as plt
import math

from sem_epe import Layer, Layout, Line, Pillar, Orientation

t0 = time.perf_counter()

# --- Setup ---

imsize = 2048
pitch = 32

layout = Layout(height=imsize, width=imsize, background=0.05)

metal1 = Layer(name="metal1", gray_value=0.3, z_order=1)
for col in range(pitch // 2, imsize, 2 * pitch):
    metal1.add_feature(Line(Orientation.VERTICAL, thickness=pitch//2, position=col))
nfeat_m1 = len(metal1.features)
layout.add_layer(metal1)

via1 = Layer(name="via1", gray_value=0.8, z_order=2)
for col in range(pitch // 2, imsize, 2 * pitch):
    for row in range(pitch // 2, imsize, pitch):
        via1.add_feature(Pillar(x=col, y=row, diameter=pitch/math.sqrt(2)))
nfeat_v1 = len(via1.features)
layout.add_layer(via1)

metal2 = Layer(name="metal2", gray_value=0.2, z_order=3)
for row in range(pitch // 2, imsize, pitch):
    metal2.add_feature(Line(Orientation.HORIZONTAL, thickness=pitch//2, position=row))
nfeat_m2 = len(metal2.features)
layout.add_layer(metal2)

t1 = time.perf_counter()

# --- Initial render ---

img_initial = layout.render().copy()

t2 = time.perf_counter()

# --- Incremental re-render of M1 lines ---

for f in metal1.features:
    f.position += 1
    layout.rerender_feature(f)

t3 = time.perf_counter()

# --- Incremental re-render of V1 pillars ---

for f in via1.features:
    f.x += 1
    layout.rerender_feature(f)

t4 = time.perf_counter()

# --- Incremental re-render of M2 lines ---

for f in metal2.features:
    f.position += 1
    layout.rerender_feature(f)

t5 = time.perf_counter()

print(f"Setup time:           {((t1 - t0) * 1000):.3f} ms")
print(f"Initial render time:  {((t2 - t1) * 1000):.3f} ms")
print(f"Average re-render M1: {((t3 - t2) / nfeat_m1 * 1000):.3f} ms")
print(f"Average re-render V1: {((t4 - t3) / nfeat_v1 * 1000):.3f} ms")
print(f"Average re-render M2: {((t5 - t4) / nfeat_m2 * 1000):.3f} ms")

"""
Setup time:           2.282 ms
Initial render time:  91.249 ms
Average re-render M1: 0.645 ms
Average re-render V1: 0.089 ms
Average re-render M2: 0.095 ms
"""

plt.imshow(img_initial, cmap="gray", vmin=0, vmax=1)
plt.show()