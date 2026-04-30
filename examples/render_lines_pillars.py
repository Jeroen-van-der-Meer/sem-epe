"""Example: Render lines and pillars."""

import matplotlib.pyplot as plt
import numpy as np

from sem_epe import Layer, Layout, Line, Pillar, Orientation

layout = Layout(height=256, width=256, background=0.1)

metal1 = Layer(name="metal1", gray_value=0.5, z_order=0)
for col in [64, 128, 192]:
    metal1.add_feature(Line(Orientation.VERTICAL, thickness=20, position=col))
layout.add_layer(metal1)

metal2 = Layer(name="metal2", gray_value=0.9, z_order=1)
for row in [80, 160]:
    metal2.add_feature(Line(Orientation.HORIZONTAL, thickness=16, position=row))
layout.add_layer(metal2)

metal1.add_feature(Pillar(x=220, y=80, diameter=100))
metal2.add_feature(Pillar(x=170, y=80, diameter=40))

img_initial = layout.render().copy()
print(f"Layout:  {layout}")
print(f"Image:   shape={img_initial.shape}, "
      f"min={img_initial.min():.2f}, max={img_initial.max():.2f}")

vline = metal1.features[0]
print(f"\nMoving {vline} by +5 px …")
vline.position += 5
img_updated = layout.rerender_feature(vline).copy()

smallpillar = metal2.features[2]
print(f"\nMoving {smallpillar} by -25 px …")
smallpillar.x -= 50
img_updated = layout.rerender_feature(smallpillar).copy()

diff = np.abs(img_updated.astype(float) - img_initial.astype(float))
changed_px = int((diff > 0).sum())
print(f"Pixels changed: {changed_px} / {img_initial.size}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_initial, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Initial render")
axes[1].imshow(img_updated, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Vertical line +5 px; pillar -25 px")
axes[2].imshow(diff, cmap="hot")
axes[2].set_title("Pixel difference")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
