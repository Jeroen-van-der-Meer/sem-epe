from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .render import Layer, Layout


def _build_region_mask(
    layout: Layout,
    threshold: float = 0.9,
) -> Tuple[np.ndarray, List[Optional[Layer]]]:
    """
    Assign each pixel to a layout region using geometry alone.

    Layers are processed from top (highest z_order) to bottom.  A pixel is
    assigned to layer k if that layer's coverage exceeds *threshold* and no
    layer above it has coverage exceeding ``1 - threshold``.  Pixels not
    cleanly inside any single layer (edge / mixed-coverage pixels) keep the
    sentinel value -1 and are excluded from the alignment objective.  Pixels
    that no layer covers to any significant degree are assigned to the
    background region (index 0).

    This function calls only ``layer.render_mask`` and is therefore
    independent of the layer gray values; it can be used before those values
    are known.

    Parameters
    ----------
    layout : Layout
    threshold : float, optional
        Coverage fraction above which a pixel is considered "inside" a
        layer.  ``1 - threshold`` is used as the complementary bound.
        Default: 0.9.

    Returns
    -------
    region : np.ndarray, shape (H, W)
        Per-pixel region index.  -1 means excluded (edge pixel).  0 means
        background.  Indices 1 … n correspond to layers in descending
        z_order order (topmost layer first).
    region_owners : list of Optional[Layer], length n+1
        ``region_owners[0]`` is ``None`` (background).
        ``region_owners[k]`` for ``k >= 1`` is the :class:`Layer` whose
        interior pixels carry region index ``k``.
    """
    H, W = layout.shape
    full_roi = (0, 0, H, W)

    sorted_layers = sorted(layout.layers, key=lambda l: l.z_order, reverse=True)

    region = np.full((H, W), -1, dtype=int)
    higher_coverage = np.zeros((H, W), dtype=float)

    region_owners: List[Optional[Layer]] = [None]  # index 0 reserved for background

    for layer in sorted_layers:
        layer_mask = layer.render_mask(full_roi)
        region_idx = len(region_owners)
        is_pure = (layer_mask > threshold) & (higher_coverage < (1.0 - threshold))
        region[is_pure] = region_idx
        region_owners.append(layer)
        np.maximum(higher_coverage, layer_mask, out=higher_coverage)

    region[higher_coverage < (1.0 - threshold)] = 0

    return region, region_owners


def _poly_features(H: int, W: int, degree: int) -> np.ndarray:
    """
    Build the polynomial basis matrix for every pixel in an (H, W) image.

    Pixel coordinates are normalised to [-1, 1] before computing the basis,
    so all monomials stay O(1) regardless of image size and the coefficient
    vectors for both axes have comparable magnitudes.

    The monomials are ordered by total degree, then by decreasing power of
    the row coordinate::

        degree 1 : 1,  r,  c               → 3 columns
        degree 2 : 1,  r,  c,  r², rc, c²  → 6 columns
        degree d : (d+1)(d+2)/2 columns

    Parameters
    ----------
    H, W : int
        Image dimensions.
    degree : int
        Polynomial degree.  1 = affine, 2 = second-order.

    Returns
    -------
    np.ndarray, shape (H*W, n_coeffs), dtype float64
        Row ``i`` corresponds to layout pixel ``(i // W, i % W)``.
    """
    rr, cc = np.mgrid[0:H, 0:W]
    r_n = (2.0 * rr / max(H - 1, 1) - 1.0).ravel()
    c_n = (2.0 * cc / max(W - 1, 1) - 1.0).ravel()

    cols = [np.ones(H * W)]
    for d in range(1, degree + 1):
        for k in range(d + 1):
            cols.append(r_n ** (d - k) * c_n ** k)

    return np.column_stack(cols)  # shape (H*W, n_coeffs)


def _apply_warp(
    poly_feat: np.ndarray,
    coeffs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the polynomial warp at every layout pixel.

    For each layout pixel the warp returns the row and column position in the
    distorted SEM image where the corrected value should be sampled.

    Parameters
    ----------
    poly_feat : np.ndarray, shape (H*W, n_coeffs)
        Polynomial basis matrix from :func:`_poly_features`.
    coeffs : np.ndarray, shape (2, n_coeffs)
        Polynomial coefficients.  ``coeffs[0]`` maps to the SEM row axis;
        ``coeffs[1]`` maps to the SEM column axis.

    Returns
    -------
    r_sem, c_sem : np.ndarray, each shape (H*W,)
        Sample positions in the distorted SEM image for each layout pixel,
        in row-major order matching *poly_feat*.
    """
    coords = poly_feat @ coeffs.T  # (H*W, 2)
    return coords[:, 0], coords[:, 1]
