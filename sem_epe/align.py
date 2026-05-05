from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.ndimage
import scipy.optimize

from .image import SEMImage
from .render import Layout


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class AlignResult:
    """
    Output of a completed alignment.

    Attributes
    ----------
    coefficients : np.ndarray, shape (2, n_coeffs)
        Optimal polynomial coefficients.  ``coefficients[0]`` maps normalised
        layout coordinates to SEM row samples; ``coefficients[1]`` to SEM
        column samples.  Monomials follow the ordering of
        :func:`_poly_features`.
    corrected : np.ndarray, dtype float32, shape (H, W)
        SEM image with distortion corrected, resampled to layout coordinates.
        Pass this as the target to :func:`~sem_epe.fit`.
    cost : float
        Final within-region intensity variance at convergence (lower is
        better; zero would be a perfect piecewise-constant match).
    """
    coefficients: np.ndarray
    corrected: np.ndarray
    cost: float


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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
    Evaluate the polynomial warp at every pixel described by *poly_feat*.

    For each output pixel the warp returns the row and column position in the
    distorted SEM image where the corrected value should be sampled.

    Parameters
    ----------
    poly_feat : np.ndarray, shape (N, n_coeffs)
        Polynomial basis matrix from :func:`_poly_features`, or a row-subset
        thereof.
    coeffs : np.ndarray, shape (2, n_coeffs)
        Polynomial coefficients.  ``coeffs[0]`` maps to the SEM row axis;
        ``coeffs[1]`` maps to the SEM column axis.

    Returns
    -------
    r_sem, c_sem : np.ndarray, each shape (N,)
        Sample positions in the distorted SEM image.
    """
    coords = poly_feat @ coeffs.T  # (N, 2)
    return coords[:, 0], coords[:, 1]


def _initial_coeffs(
    H: int, W: int, H_sem: int, W_sem: int, degree: int
) -> np.ndarray:
    """
    Polynomial coefficients for the identity-like starting transform.

    Maps each layout pixel (r, c) to the corresponding position in a SEM
    image of size (H_sem, W_sem) with the layout centred inside it.  Scale
    is 1 and there is no rotation or higher-order distortion.

    Parameters
    ----------
    H, W : int
        Layout dimensions.
    H_sem, W_sem : int
        SEM image dimensions.
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray, shape (2, n_coeffs)
    """
    n_coeffs = (degree + 1) * (degree + 2) // 2
    coeffs = np.zeros((2, n_coeffs))
    # Columns 0, 1, 2 are the constant, r̃, and c̃ terms respectively.
    coeffs[0, 0] = (H_sem - 1) / 2.0  # row offset: layout centre → SEM centre
    coeffs[0, 1] = (H - 1) / 2.0      # row scale:  r̃ = ±1 spans layout rows
    coeffs[1, 0] = (W_sem - 1) / 2.0  # col offset
    coeffs[1, 2] = (W - 1) / 2.0      # col scale:  c̃ = ±1 spans layout cols
    return coeffs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align(
    image: SEMImage,
    layout: Layout,
    *,
    degree: int = 2,
    region_threshold: float = 0.9,
    **solver_kwargs,
) -> AlignResult:
    """
    Correct SEM image distortion by aligning it to the expected layout
    geometry, and discover the per-layer gray values.

    The alignment is formulated as a nonlinear least-squares problem: find
    the polynomial warp T such that, after resampling the SEM image through
    T, each layout region (one per layer + background) is as
    intensity-uniform as possible.  The residual for pixel j in region k is
    ``I_j − μ_k``, where ``μ_k`` is the mean intensity of all valid pixels
    in that region under the current warp.  Minimising the sum of squared
    residuals is equivalent to minimising the total within-region variance.

    This self-supervised objective requires no knowledge of gray values in
    advance; they are extracted as per-region means once the warp converges.

    The warp uses the inverse / pull model: for each output pixel at layout
    coordinate (r, c), T(r, c) gives the position to sample in the distorted
    SEM image.  The resulting corrected image has the same shape as the
    layout and can be passed directly to :func:`~sem_epe.fit`.

    After convergence, ``layout.background`` and each ``layer.gray_value``
    are updated in place with the discovered values so that
    :func:`~sem_epe.fit` can be called immediately without further setup.

    Parameters
    ----------
    image : SEMImage
        The distorted SEM image to align.  May be larger than the layout.
    layout : Layout
        Expected layout geometry.  Gray values are not used and will be
        overwritten.
    degree : int, optional
        Polynomial degree of the warp.  1 = affine (translation, rotation,
        scale), 2 = adds second-order lens distortion terms.  Default: 2.
    region_threshold : float, optional
        Coverage fraction above which a pixel is considered "inside" a layer
        for the region mask.  Default: 0.9.
    **solver_kwargs
        Forwarded to ``scipy.optimize.least_squares`` (e.g. ``ftol``,
        ``max_nfev``).

    Returns
    -------
    AlignResult
    """
    sem_arr = image.image.astype(np.float64)
    H_sem, W_sem = image.shape
    H, W = layout.shape

    region, region_owners = layout.region_mask(threshold=region_threshold)
    region_flat = region.ravel()
    n_regions = len(region_owners)

    # Build full basis matrix once; slice out the included-pixel subset for
    # the residual function, which is evaluated many times during optimisation.
    poly_feat = _poly_features(H, W, degree)

    included_pixels = np.where(region_flat != -1)[0]
    included_region = region_flat[included_pixels]
    included_poly = poly_feat[included_pixels]

    # Precompute which positions within included_pixels belong to each region.
    region_rel_indices = [
        np.where(included_region == k)[0] for k in range(n_regions)
    ]

    x0 = _initial_coeffs(H, W, H_sem, W_sem, degree).ravel()

    def residuals(x: np.ndarray) -> np.ndarray:
        coeffs = x.reshape(2, -1)
        r_sem, c_sem = _apply_warp(included_poly, coeffs)
        sampled = scipy.ndimage.map_coordinates(
            sem_arr, [r_sem, c_sem], order=1, mode='constant', cval=np.nan,
            prefilter=False,
        )
        result = np.zeros(len(included_pixels))
        for rel_idx in region_rel_indices:
            if len(rel_idx) < 2:
                continue
            vals = sampled[rel_idx]
            finite = np.isfinite(vals)
            finite_vals = vals[finite]
            if len(finite_vals) > 1:
                mu = np.mean(finite_vals)
                result[rel_idx[finite]] = finite_vals - mu
        return result

    opt = scipy.optimize.least_squares(
        residuals, x0, method='trf', jac='2-point', **solver_kwargs
    )

    opt_coeffs = opt.x.reshape(2, -1)

    # Produce the corrected image for all layout pixels.
    r_sem_full, c_sem_full = _apply_warp(poly_feat, opt_coeffs)
    corrected = scipy.ndimage.map_coordinates(
        sem_arr, [r_sem_full, c_sem_full], order=1, mode='nearest',
        prefilter=False,
    ).reshape(H, W).astype(np.float32)

    # Extract and write back gray values from valid (in-bounds) samples.
    corrected_flat = corrected.ravel()
    for owner, rel_idx in zip(region_owners, region_rel_indices):
        if len(rel_idx) == 0:
            continue
        gray = float(np.mean(corrected_flat[included_pixels[rel_idx]]))
        if owner is None:
            layout.background = gray
        else:
            owner.gray_value = gray

    return AlignResult(
        coefficients=opt_coeffs,
        corrected=corrected,
        cost=float(opt.cost),
    )
