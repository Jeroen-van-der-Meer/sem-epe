"""
sem_epe — Layer-aware SEM image EPE (Edge Placement Error) analysis.

Defines a layout as an ordered stack of layers containing geometric features
(lines, pillars).  Renders a synthetic grayscale top-down SEM image and
supports efficient partial re-rendering when individual feature parameters
are modified — a prerequisite for nonlinear least-squares fitting.

Coordinate convention
---------------------
Images are H × W NumPy arrays indexed as ``image[row, col]`` with the
origin at the top-left corner.  Feature positions use the same convention:

    x  →  column index (horizontal)
    y  ↓  row index    (vertical)

Grayscale values are normalised floats in [0, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "Orientation",
    "Feature",
    "Line",
    "Pillar",
    "Layer",
    "Layout",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Orientation(Enum):
    """Orientation of a line feature."""
    HORIZONTAL = "horizontal"
    VERTICAL   = "vertical"


# ---------------------------------------------------------------------------
# Abstract Feature
# ---------------------------------------------------------------------------

class Feature(ABC):
    """
    Base class for a single geometric layout feature.

    A feature belongs to exactly one :class:`Layer`; its rendered gray value
    is always that of its owning layer.  Sub-classes define the shape through
    :meth:`render_mask` and :meth:`bounding_box`.
    """

    def __init__(self) -> None:
        self.layer: "Layer"

    @abstractmethod
    def render_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Pixel footprint of this feature.

        Parameters
        ----------
        shape : (H, W)
            Image dimensions in pixels.

        Returns
        -------
        np.ndarray, dtype=bool, shape=(H, W)
            ``True`` at every pixel covered by this feature.
        """

    @abstractmethod
    def bounding_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Tight axis-aligned bounding box of the feature, clipped to *shape*.

        Returns
        -------
        (row_min, col_min, row_max, col_max) : int
            Half-open intervals: ``image[row_min:row_max, col_min:col_max]``
            contains the entire feature.
        """


# ---------------------------------------------------------------------------
# Line
# ---------------------------------------------------------------------------

class Line(Feature):
    """
    A rectangular line (trace) running parallel to one image axis.

    Parameters
    ----------
    orientation : Orientation
        HORIZONTAL → runs left-to-right (constant row band).
        VERTICAL   → runs top-to-bottom (constant column band).
    thickness : float
        Critical dimension (CD): full width of the line in pixels.
    position : float
        Centre of the line along the *perpendicular* axis.
        Row-coordinate for HORIZONTAL; column-coordinate for VERTICAL.
    extent : (float, float) or None
        ``(start, end)`` along the *parallel* axis in pixels.
        ``None`` → line spans the full image dimension.
    """

    def __init__(
        self,
        orientation: Orientation,
        thickness: float,
        position: float,
        extent: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.orientation: Orientation = orientation
        self.thickness: float = float(thickness)
        self.position: float = float(position)
        self.extent: Optional[Tuple[float, float]] = extent

    def render_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros(shape, dtype=bool)
        half = self.thickness / 2.0

        if self.orientation == Orientation.HORIZONTAL:
            r0 = max(0, int(np.floor(self.position - half)))
            r1 = min(h, int(np.ceil(self.position + half)))
            if self.extent is None:
                c0, c1 = 0, w
            else:
                c0 = max(0, int(np.floor(self.extent[0])))
                c1 = min(w, int(np.ceil(self.extent[1])))
        else:  # VERTICAL
            c0 = max(0, int(np.floor(self.position - half)))
            c1 = min(w, int(np.ceil(self.position + half)))
            if self.extent is None:
                r0, r1 = 0, h
            else:
                r0 = max(0, int(np.floor(self.extent[0])))
                r1 = min(h, int(np.ceil(self.extent[1])))

        if r0 < r1 and c0 < c1:
            mask[r0:r1, c0:c1] = True
        return mask

    def bounding_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = shape
        half = self.thickness / 2.0
        if self.orientation == Orientation.HORIZONTAL:
            r0 = max(0, int(np.floor(self.position - half)))
            r1 = min(h, int(np.ceil(self.position + half)))
            c0 = 0 if self.extent is None else max(0, int(np.floor(self.extent[0])))
            c1 = w if self.extent is None else min(w, int(np.ceil(self.extent[1])))
        else:
            c0 = max(0, int(np.floor(self.position - half)))
            c1 = min(w, int(np.ceil(self.position + half)))
            r0 = 0 if self.extent is None else max(0, int(np.floor(self.extent[0])))
            r1 = h if self.extent is None else min(h, int(np.ceil(self.extent[1])))
        return r0, c0, r1, c1

    def __repr__(self) -> str:
        return (
            f"Line({self.orientation.value}, pos={self.position:.2f}, "
            f"cd={self.thickness:.2f})"
        )


# ---------------------------------------------------------------------------
# Pillar
# ---------------------------------------------------------------------------

class Pillar(Feature):
    """
    A circular pillar (via / contact hole).

    Parameters
    ----------
    x : float
        Column coordinate of the pillar centre in pixels.
    y : float
        Row coordinate of the pillar centre in pixels.
    diameter : float
        Critical dimension (CD) of the pillar in pixels.
    """

    def __init__(
        self,
        x: float,
        y: float,
        diameter: float,
    ) -> None:
        super().__init__()
        self.x: float = float(x)
        self.y: float = float(y)
        self.diameter: float = float(diameter)

    def render_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        rows, cols = np.ogrid[:h, :w]
        radius = self.diameter / 2.0
        return (cols - self.x) ** 2 + (rows - self.y) ** 2 <= radius ** 2

    def bounding_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = shape
        r = self.diameter / 2.0
        r0 = max(0, int(np.floor(self.y - r)))
        c0 = max(0, int(np.floor(self.x - r)))
        r1 = min(h, int(np.ceil(self.y + r)))
        c1 = min(w, int(np.ceil(self.x + r)))
        return r0, c0, r1, c1

    def __repr__(self) -> str:
        return f"Pillar(x={self.x:.2f}, y={self.y:.2f}, d={self.diameter:.2f})"


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class Layer:
    """
    A physical material layer rendered with a uniform grayscale value.

    Within a :class:`Layout`, layers are stacked in ascending *z_order*:
    features in higher-z layers obscure those in lower-z layers.

    Parameters
    ----------
    name : str
    gray_value : float
        Normalised intensity in [0, 1].
    z_order : int
        Stacking position.  Higher = rendered on top.
    """

    def __init__(self, name: str, gray_value: float, z_order: int) -> None:
        self.name: str = name
        self.gray_value: float = float(gray_value)
        self.z_order: int = int(z_order)
        self.features: List[Feature] = []

    def add_feature(self, feature: Feature) -> "Layer":
        """Attach *feature* to this layer.  Returns ``self`` for chaining."""
        feature.layer = self
        self.features.append(feature)
        return self

    def render_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Union pixel mask of every feature in this layer."""
        mask = np.zeros(shape, dtype=bool)
        for f in self.features:
            mask |= f.render_mask(shape)
        return mask

    def __repr__(self) -> str:
        return (
            f"Layer(name={self.name!r}, gray={self.gray_value:.3f}, "
            f"z={self.z_order}, n_features={len(self.features)})"
        )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class Layout:
    """
    Full expected layout: an ordered stack of :class:`Layer` objects.

    Rendering proceeds bottom-to-top (ascending *z_order*).  Each layer's
    features paint over lower ones, modelling physical material stacking:
    the topmost layer covering a pixel determines its gray value.

    Mask caches
    -----------
    :meth:`render` populates two dictionaries:

    * ``_feature_masks`` — one boolean (H, W) array per :class:`Feature`.
    * ``_layer_masks``   — union boolean (H, W) array per :class:`Layer`.

    :meth:`rerender_feature` uses these caches to limit recompositing to
    only the pixels whose coverage state has changed, which is critical for
    the inner loop of nonlinear least-squares fitting.

    Parameters
    ----------
    height, width : int
        Canvas dimensions in pixels.
    background : float
        Gray value [0, 1] of pixels not covered by any feature.
    """

    def __init__(self, height: int, width: int, background: float = 0.0) -> None:
        self.height: int = height
        self.width: int = width
        self.background: float = float(background)
        self.layers: List[Layer] = []

        self._feature_masks: Dict[Feature, np.ndarray] = {}
        self._layer_masks: Dict[Layer, np.ndarray] = {}
        self._image: Optional[np.ndarray] = None

    def add_layer(self, layer: Layer) -> "Layout":
        """Register *layer* and maintain the list sorted by z_order."""
        self.layers.append(layer)
        self.layers.sort(key=lambda l: l.z_order)
        return self

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def render(self) -> np.ndarray:
        """
        Render all layers from scratch and refresh all mask caches.

        Must be called at least once before :meth:`rerender_feature`.

        Returns
        -------
        np.ndarray, dtype=float32, shape=(H, W)
            Rendered image with pixel values in [0, 1].
        """
        image = np.full(self.shape, self.background, dtype=np.float32)

        for layer in self.layers:
            lm = np.zeros(self.shape, dtype=bool)
            for f in layer.features:
                fm = f.render_mask(self.shape)
                self._feature_masks[f] = fm
                lm |= fm # Layers in self.layers are sorted in z
            self._layer_masks[layer] = lm
            image[lm] = layer.gray_value

        self._image = image
        return image.copy()

    def rerender_feature(self, feature: Feature) -> np.ndarray:
        """
        Re-render after exactly one feature's parameters have changed.

        The caller must update the feature's attributes *before* calling
        this method.  The update is then propagated to the cached image
        with minimal work.

        Algorithm
        ---------
        1. Recompute the feature's pixel mask.
        2. Rebuild the owning layer's combined mask from the updated
           per-feature cache.
        3. Compute ``delta = old_layer_mask XOR new_layer_mask``: the set
           of pixels whose layer-coverage state flipped.
        4. Re-composite *only* the delta pixels across all layers in
           z-order, restricted to their tight bounding box.

        Layer-awareness guarantee
        -------------------------
        Because step 4 applies all layers bottom-to-top, a pixel obscured
        by a higher-z layer is correctly handled: the higher layer wins
        regardless of what changed below it.

        Parameters
        ----------
        feature : Feature
            The feature whose parameters were just modified.

        Returns
        -------
        np.ndarray, dtype=float32, shape=(H, W)
            Updated rendered image.
        """
        layer = feature.layer

        new_fm = feature.render_mask(self.shape)
        self._feature_masks[feature] = new_fm

        old_lm = self._layer_masks[layer]
        new_lm = np.zeros(self.shape, dtype=bool)
        for f in layer.features:
            new_lm |= self._feature_masks[f]
        self._layer_masks[layer] = new_lm

        delta = old_lm ^ new_lm

        rows, cols = np.where(delta)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1

        sub_delta = delta[r0:r1, c0:c1]

        sub_img = self._image[r0:r1, c0:c1]
        sub_img[sub_delta] = self.background
        for lyr in self.layers:
            sub_lm = self._layer_masks[lyr][r0:r1, c0:c1]
            covered = sub_delta & sub_lm
            if covered.any():
                sub_img[covered] = lyr.gray_value

        return self._image.copy()

    @property
    def image(self) -> Optional[np.ndarray]:
        """Current cached image, or ``None`` if :meth:`render` has not been called."""
        return None if self._image is None else self._image.copy()

    def __repr__(self) -> str:
        names = ", ".join(l.name for l in self.layers)
        return f"Layout({self.height}x{self.width}, layers=[{names}])"
