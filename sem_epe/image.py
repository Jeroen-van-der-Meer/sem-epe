from __future__ import annotations

from typing import Tuple

import numpy as np

class SEMImage:
    """
    A real SEM image together with its scale calibration.

    Parameters
    ----------
    image : np.ndarray, shape=(H, W)
        Normalised grayscale image with values in [0, 1].  Converted to
        float32 on construction.
    nm_per_pixel : float
        Physical pixel size in nanometres.  Used to convert pixel-space
        distances (e.g. EPE) to physical units.
    """

    def __init__(self, image: np.ndarray, nm_per_pixel: float) -> None:
        if image.ndim != 2:
            raise ValueError(f"expected 2-D image, got shape {image.shape}")
        if not (0.0 <= image.min() and image.max() <= 1.0):
            raise ValueError("image values must be normalised to [0, 1]")
        self.image: np.ndarray = image.astype(np.float32)
        self.nm_per_pixel: float = float(nm_per_pixel)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.image.shape

    def __repr__(self) -> str:
        h, w = self.shape
        return f"SEMImage({h}x{w}, nm_per_pixel={self.nm_per_pixel})"
