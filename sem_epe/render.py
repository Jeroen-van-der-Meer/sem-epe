from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

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
    def render_mask(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Pixel footprint of this feature within *roi*.

        Parameters
        ----------
        roi : (r0, c0, r1, c1)
            Subregion of the canvas to render, in image-space pixel
            coordinates.  Pass ``(0, 0, height, width)`` for the full image.

        Returns
        -------
        np.ndarray, dtype=bool, shape=(r1-r0, c1-c0)
            ``True`` at every pixel within *roi* covered by this feature.
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
        HORIZONTAL
        VERTICAL
    thickness : float
        Critical dimension (CD): full width of the line in pixels.
    position : float
        Centre of the line along the perpendicular axis.
    extent : (float, float) or None
        ``(start, end)`` along the parallel axis in pixels. (FIXME: May remove)
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

    def render_mask(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = roi
        h, w = r1 - r0, c1 - c0
        mask = np.zeros((h, w), dtype=bool)
        half = self.thickness / 2.0

        if self.orientation == Orientation.HORIZONTAL:
            mr0 = max(0, int(np.floor(self.position - half)) - r0)
            mr1 = min(h, int(np.ceil(self.position + half)) - r0)
            if self.extent is None:
                mc0, mc1 = 0, w
            else:
                mc0 = max(0, int(np.floor(self.extent[0])) - c0)
                mc1 = min(w, int(np.ceil(self.extent[1])) - c0)
        else:  # VERTICAL
            mc0 = max(0, int(np.floor(self.position - half)) - c0)
            mc1 = min(w, int(np.ceil(self.position + half)) - c0)
            if self.extent is None:
                mr0, mr1 = 0, h
            else:
                mr0 = max(0, int(np.floor(self.extent[0])) - r0)
                mr1 = min(h, int(np.ceil(self.extent[1])) - r0)

        if mr0 < mr1 and mc0 < mc1:
            mask[mr0:mr1, mc0:mc1] = True
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

    def render_mask(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = roi
        h, w = r1 - r0, c1 - c0
        rows, cols = np.ogrid[:h, :w]
        radius = self.diameter / 2.0
        return (cols - (self.x - c0)) ** 2 + (rows - (self.y - r0)) ** 2 <= radius ** 2

    def bounding_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = shape
        r = self.diameter / 2.0
        r0 = max(0, int(np.floor(self.y - r)))
        c0 = max(0, int(np.floor(self.x - r)))
        r1 = min(h, int(np.floor(self.y + r)) + 1)
        c1 = min(w, int(np.floor(self.x + r)) + 1)
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

    def render_mask(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Union pixel mask of every feature in this layer within *roi*."""
        r0, c0, r1, c1 = roi
        mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for f in self.features:
            mask |= f.render_mask(roi)
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
    :meth:`render` populates three dictionaries:

    * ``_feature_masks``  — compact boolean array per :class:`Feature`, sized
      to the feature's bounding box.
    * ``_layer_masks``    — union boolean (H, W) array per :class:`Layer`.
    * ``_layer_bboxes``   — (N, 4) int32 array of bboxes for every feature in
      each layer, kept in sync with ``_feature_masks``.
    * ``_feature_index``  — maps each :class:`Feature` to its row index in its
      layer's ``_layer_bboxes`` array.

    :meth:`rerender_feature` uses these caches to restrict all work to the
    dirty subregion (union of old and new bounding boxes).  The layer mask is
    rebuilt by splatting only the features whose bboxes overlap the dirty
    region.

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
        self._layer_bboxes: Dict[Layer, np.ndarray] = {}
        self._feature_index: Dict[Feature, int] = {}
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
            The internal image buffer.
        """
        image = np.full(self.shape, self.background, dtype=np.float32)

        for layer in self.layers:
            lm = np.zeros(self.shape, dtype=bool)
            bboxes = np.empty((len(layer.features), 4), dtype=np.int32)
            for i, f in enumerate(layer.features):
                bbox = f.bounding_box(self.shape)
                r0, c0, r1, c1 = bbox
                fm = f.render_mask(bbox)        # compact: shape (r1-r0, c1-c0)
                self._feature_masks[f] = fm
                self._feature_index[f] = i
                bboxes[i] = bbox
                lm[r0:r1, c0:c1] |= fm
            self._layer_bboxes[layer] = bboxes
            self._layer_masks[layer] = lm
            image[lm] = layer.gray_value

        self._image = image
        return self._image

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

        Parameters
        ----------
        feature : Feature
            The feature whose parameters were just modified.

        Returns
        -------
        np.ndarray, dtype=float32, shape=(H, W)
            The internal image buffer, updated in place.
        """
        layer = feature.layer
        feat_idx = self._feature_index[feature]
        layer_bboxes = self._layer_bboxes[layer]

        # Dirty region = union of old and new bounding boxes.
        old_r0, old_c0, old_r1, old_c1 = layer_bboxes[feat_idx]
        new_r0, new_c0, new_r1, new_c1 = feature.bounding_box(self.shape)
        dr0 = int(min(old_r0, new_r0))
        dc0 = int(min(old_c0, new_c0))
        dr1 = int(max(old_r1, new_r1))
        dc1 = int(max(old_c1, new_c1))

        # Compact new feature mask; update the per-layer bbox array in-place.
        self._feature_masks[feature] = feature.render_mask((new_r0, new_c0, new_r1, new_c1))
        layer_bboxes[feat_idx] = (new_r0, new_c0, new_r1, new_c1)

        # Snapshot old layer mask in dirty region.
        old_lm_sub = self._layer_masks[layer][dr0:dr1, dc0:dc1].copy()

        # Vectorised overlap test.
        hit = ((layer_bboxes[:, 0] < dr1) & (layer_bboxes[:, 2] > dr0) &
               (layer_bboxes[:, 1] < dc1) & (layer_bboxes[:, 3] > dc0))

        # Rebuild layer mask in dirty region by splatting only overlapping features.
        lm_sub = self._layer_masks[layer][dr0:dr1, dc0:dc1]
        lm_sub[:] = False
        for idx in np.where(hit)[0]:
            fr0, fc0, fr1, fc1 = layer_bboxes[idx]
            ir0, ic0 = max(fr0, dr0), max(fc0, dc0)
            ir1, ic1 = min(fr1, dr1), min(fc1, dc1)
            lm_sub[ir0-dr0:ir1-dr0, ic0-dc0:ic1-dc0] |= (
                self._feature_masks[layer.features[idx]][ir0-fr0:ir1-fr0, ic0-fc0:ic1-fc0]
            )

        delta_sub = old_lm_sub ^ lm_sub

        if not delta_sub.any():
            return self._image

        sub_img = self._image[dr0:dr1, dc0:dc1]
        sub_img[delta_sub] = self.background
        for lyr in self.layers:
            sub_lm = self._layer_masks[lyr][dr0:dr1, dc0:dc1]
            covered = delta_sub & sub_lm
            if covered.any():
                sub_img[covered] = lyr.gray_value

        return self._image

    @property
    def image(self) -> Optional[np.ndarray]:
        """The internal image buffer, or ``None`` if :meth:`render` has not been called."""
        return self._image

    def __repr__(self) -> str:
        names = ", ".join(l.name for l in self.layers)
        return f"Layout({self.height}x{self.width}, layers=[{names}])"
