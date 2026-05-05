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
        Sub-pixel coverage fraction of this feature within *roi*.

        Parameters
        ----------
        roi : (r0, c0, r1, c1)
            Subregion of the canvas to render, in image-space pixel
            coordinates.  Pass ``(0, 0, height, width)`` for the full image.

        Returns
        -------
        np.ndarray, dtype=float32, shape=(r1-r0, c1-c0)
            Coverage fraction in [0, 1] for each pixel within *roi*.
            0 = fully outside, 1 = fully inside; values in between occur
            at edge pixels.
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
        half = self.thickness / 2.0
        rows = np.arange(r0, r1).reshape(-1, 1)  # (h, 1)
        cols = np.arange(c0, c1).reshape(1, -1)  # (1, w)

        if self.orientation == Orientation.HORIZONTAL:
            row_cov = np.clip(
                np.minimum(rows + 0.5, self.position + half) -
                np.maximum(rows - 0.5, self.position - half),
                0.0, 1.0,
            )
            if self.extent is None:
                col_cov = 1.0
            else:
                col_cov = np.clip(
                    np.minimum(cols + 0.5, self.extent[1]) -
                    np.maximum(cols - 0.5, self.extent[0]),
                    0.0, 1.0,
                )
        else:  # VERTICAL
            col_cov = np.clip(
                np.minimum(cols + 0.5, self.position + half) -
                np.maximum(cols - 0.5, self.position - half),
                0.0, 1.0,
            )
            if self.extent is None:
                row_cov = 1.0
            else:
                row_cov = np.clip(
                    np.minimum(rows + 0.5, self.extent[1]) -
                    np.maximum(rows - 0.5, self.extent[0]),
                    0.0, 1.0,
                )

        return np.broadcast_to(row_cov * col_cov, (h, w)).astype(np.float32)

    def bounding_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = shape
        half = self.thickness / 2.0
        if self.orientation == Orientation.HORIZONTAL:
            r0 = max(0, int(np.floor(self.position - half)))
            r1 = min(h, int(np.ceil(self.position + half)) + 1)
            c0 = 0 if self.extent is None else max(0, int(np.floor(self.extent[0])))
            c1 = w if self.extent is None else min(w, int(np.ceil(self.extent[1])) + 1)
        else:
            c0 = max(0, int(np.floor(self.position - half)))
            c1 = min(w, int(np.ceil(self.position + half)) + 1)
            r0 = 0 if self.extent is None else max(0, int(np.floor(self.extent[0])))
            r1 = h if self.extent is None else min(h, int(np.ceil(self.extent[1])) + 1)
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
        dist = np.sqrt((cols - (self.x - c0)) ** 2 + (rows - (self.y - r0)) ** 2)
        # Linear ramp over a 1-pixel band: full inside, zero outside,
        # interpolated within [radius - 0.5, radius + 0.5].
        return np.clip(radius + 0.5 - dist, 0.0, 1.0).astype(np.float32)

    def bounding_box(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        h, w = shape
        r = self.diameter / 2.0
        r0 = max(0, int(np.floor(self.y - r)) - 1)
        c0 = max(0, int(np.floor(self.x - r)) - 1)
        r1 = min(h, int(np.ceil(self.y + r)) + 1)
        c1 = min(w, int(np.ceil(self.x + r)) + 1)
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

    def __init__(self, name: str, gray_value: Optional[float] = None, z_order: int = 0) -> None:
        self.name: str = name
        self.gray_value: Optional[float] = None if gray_value is None else float(gray_value)
        self.z_order: int = int(z_order)
        self.features: List[Feature] = []

    def add_feature(self, feature: Feature) -> "Layer":
        """Attach *feature* to this layer.  Returns ``self`` for chaining."""
        feature.layer = self
        self.features.append(feature)
        return self

    def render_mask(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Float32 coverage mask (union of all features) within *roi*."""
        r0, c0, r1, c1 = roi
        mask = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
        for f in self.features:
            mask += f.render_mask(roi)
            np.minimum(mask, 1.0, out=mask)
        return mask

    def __repr__(self) -> str:
        gray = f"{self.gray_value:.3f}" if self.gray_value is not None else "?"
        return (
            f"Layer(name={self.name!r}, gray={gray}, "
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
            if layer.gray_value is None:
                raise ValueError(
                    f"Layer {layer.name!r} has no gray_value set. "
                    f"Call epe.align() to discover it, or set layer.gray_value manually."
                )
            lm = np.zeros(self.shape, dtype=np.float32)
            bboxes = np.empty((len(layer.features), 4), dtype=np.int32)
            for i, f in enumerate(layer.features):
                bbox = f.bounding_box(self.shape)
                r0, c0, r1, c1 = bbox
                fm = f.render_mask(bbox)        # compact: shape (r1-r0, c1-c0)
                self._feature_masks[f] = fm
                self._feature_index[f] = i
                bboxes[i] = bbox
                sub = lm[r0:r1, c0:c1]
                sub += fm
                np.minimum(sub, 1.0, out=sub)
            self._layer_bboxes[layer] = bboxes
            self._layer_masks[layer] = lm
            # Alpha composite: image = (1 - lm) * image + lm * gray_value
            image += lm * (layer.gray_value - image)

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

        old_r0, old_c0, old_r1, old_c1 = layer_bboxes[feat_idx]
        new_r0, new_c0, new_r1, new_c1 = feature.bounding_box(self.shape)
        new_fm = feature.render_mask((new_r0, new_c0, new_r1, new_c1))

        # Early exit: compact mask unchanged → image unchanged.
        if (new_r0 == old_r0 and new_c0 == old_c0 and
                new_r1 == old_r1 and new_c1 == old_c1 and
                np.array_equal(self._feature_masks[feature], new_fm)):
            return self._image

        self._feature_masks[feature] = new_fm
        layer_bboxes[feat_idx] = (new_r0, new_c0, new_r1, new_c1)

        # Dirty region = union of old and new bounding boxes.
        dr0 = int(min(old_r0, new_r0))
        dc0 = int(min(old_c0, new_c0))
        dr1 = int(max(old_r1, new_r1))
        dc1 = int(max(old_c1, new_c1))

        # Vectorised overlap test.
        hit = ((layer_bboxes[:, 0] < dr1) & (layer_bboxes[:, 2] > dr0) &
               (layer_bboxes[:, 1] < dc1) & (layer_bboxes[:, 3] > dc0))

        # Rebuild layer coverage in dirty region by splatting overlapping features.
        lm_sub = self._layer_masks[layer][dr0:dr1, dc0:dc1]
        lm_sub[:] = 0.0
        for idx in np.where(hit)[0]:
            fr0, fc0, fr1, fc1 = layer_bboxes[idx]
            ir0, ic0 = max(fr0, dr0), max(fc0, dc0)
            ir1, ic1 = min(fr1, dr1), min(fc1, dc1)
            dst = lm_sub[ir0-dr0:ir1-dr0, ic0-dc0:ic1-dc0]
            src = self._feature_masks[layer.features[idx]][ir0-fr0:ir1-fr0, ic0-fc0:ic1-fc0]
            dst += src
            np.minimum(dst, 1.0, out=dst)

        # Recomposite dirty region from scratch across all layers.
        sub_img = self._image[dr0:dr1, dc0:dc1]
        sub_img[:] = self.background
        for lyr in self.layers:
            if lyr.gray_value is None:
                raise ValueError(
                    f"Layer {lyr.name!r} has no gray_value set. "
                    f"Call epe.align() to discover it, or set layer.gray_value manually."
                )
            alpha = self._layer_masks[lyr][dr0:dr1, dc0:dc1]
            sub_img += alpha * (lyr.gray_value - sub_img)

        return self._image

    def region_mask(
        self, threshold: float = 0.9
    ) -> Tuple[np.ndarray, List[Optional["Layer"]]]:
        """
        Assign each pixel to a layout region using geometry alone.

        Layers are processed from top (highest z_order) to bottom.  A pixel
        is assigned to layer k if that layer's coverage exceeds *threshold*
        and no layer above it has coverage exceeding ``1 - threshold``.
        Pixels not cleanly inside any single layer keep the sentinel value
        -1 and are excluded from region-based computations (e.g. alignment).
        Pixels covered by no layer to any significant degree are assigned to
        the background region (index 0).

        This method uses only :meth:`Layer.render_mask` and is therefore
        independent of :attr:`Layer.gray_value`; it can be called before
        gray values are known.

        Parameters
        ----------
        threshold : float, optional
            Coverage fraction above which a pixel is considered "inside" a
            layer.  Default: 0.9.

        Returns
        -------
        region : np.ndarray, shape (H, W), dtype int32
            Per-pixel region index.  -1 = edge/excluded, 0 = background,
            1 … n = layers in descending z_order (topmost first).
        region_owners : list of Optional[Layer], length n+1
            ``region_owners[0]`` is ``None`` (background).
            ``region_owners[k]`` for k ≥ 1 is the :class:`Layer` whose
            interior pixels carry region index k.
        """
        H, W = self.shape
        full_roi = (0, 0, H, W)
        sorted_layers = sorted(self.layers, key=lambda l: l.z_order, reverse=True)

        region = np.full((H, W), -1, dtype=np.int32)
        higher_coverage = np.zeros((H, W), dtype=np.float32)
        region_owners: List[Optional[Layer]] = [None]  # index 0 = background

        for layer in sorted_layers:
            layer_mask = layer.render_mask(full_roi)
            region_idx = len(region_owners)
            is_pure = (layer_mask > threshold) & (higher_coverage < (1.0 - threshold))
            region[is_pure] = region_idx
            region_owners.append(layer)
            np.maximum(higher_coverage, layer_mask, out=higher_coverage)

        region[higher_coverage < (1.0 - threshold)] = 0
        return region, region_owners

    @property
    def image(self) -> Optional[np.ndarray]:
        """The internal image buffer, or ``None`` if :meth:`render` has not been called."""
        return self._image

    def __repr__(self) -> str:
        names = ", ".join(l.name for l in self.layers)
        return f"Layout({self.height}x{self.width}, layers=[{names}])"
