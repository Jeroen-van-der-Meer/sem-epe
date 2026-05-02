from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.optimize

from .image import SEMImage
from .render import Layout
from .tune import ParameterSet, Tuner

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Output of a completed fit call.

    Attributes
    ----------
    optimizer_result : scipy.optimize.OptimizeResult
        scipy result from the final optimisation step; inspect ``.cost``,
        ``.nfev``, ``.message``, ``.success``, etc.
    residual : np.ndarray, dtype=float32, shape=(H, W)
        ``render(fitted) - target``: signed pixel-wise error at convergence.
    starting_render : np.ndarray, dtype=float32, shape=(H, W)
        Layout render at the nominal starting point, before any fitting.
    """

    optimizer_result: scipy.optimize.OptimizeResult
    residual: np.ndarray
    starting_render: np.ndarray

    @property
    def success(self) -> bool:
        return bool(self.optimizer_result.success)

    @property
    def rms_error(self) -> float:
        """Root-mean-square pixel error of the final fit."""
        return float(np.sqrt(np.mean(self.residual ** 2)))


# ---------------------------------------------------------------------------
# Parameter-group helpers
# ---------------------------------------------------------------------------

def _layer_param_groups(params: ParameterSet) -> List[ParameterSet]:
    """
    Split *params* into one ParameterSet per layer, ordered top-to-bottom
    (descending z_order).  Bounds and nominals are preserved from *params*.
    """
    layer_indices: defaultdict = defaultdict(list)
    for i, (f, _) in enumerate(params._params):
        layer_indices[id(f.layer)].append(i)

    layers = {id(f.layer): f.layer for f, _ in params._params}
    sorted_layers = sorted(layers.values(), key=lambda l: l.z_order, reverse=True)

    groups = []
    for layer in sorted_layers:
        idxs = layer_indices[id(layer)]
        entries = [
            (params._params[i][0], params._params[i][1],
             params._lo_devs[i], params._hi_devs[i])
            for i in idxs
        ]
        groups.append(ParameterSet(entries))
    return groups


def _per_feature_groups(params: ParameterSet) -> List[ParameterSet]:
    """
    Split *params* into one ParameterSet per feature, in first-appearance
    order.  Bounds and nominals are preserved from *params*.
    """
    feature_indices: defaultdict = defaultdict(list)
    for i, (f, _) in enumerate(params._params):
        feature_indices[id(f)].append(i)

    groups = []
    for feature in params.features:
        idxs = feature_indices[id(feature)]
        entries = [
            (params._params[i][0], params._params[i][1],
             params._lo_devs[i], params._hi_devs[i])
            for i in idxs
        ]
        groups.append(ParameterSet(entries))
    return groups


def _feature_roi(feature, layout_shape: tuple) -> tuple:
    """Bounding box expanded to twice its size (half-dimension margin each side), clamped to image."""
    r0, c0, r1, c1 = feature.bounding_box(layout_shape)
    h, w = r1 - r0, c1 - c0
    H, W = layout_shape
    return (
        max(0, r0 - h // 2),
        max(0, c0 - w // 2),
        min(H, r1 + h // 2),
        min(W, c1 + w // 2),
    )


def _setup(image: SEMImage, layout: Layout) -> np.ndarray:
    """Validate shapes, render layout, return a copy of the starting render."""
    if image.shape != layout.shape:
        raise ValueError(
            f"image shape {image.shape} does not match layout shape {layout.shape}"
        )
    layout.render()
    return layout.image.copy()


# ---------------------------------------------------------------------------
# Fit strategies
# ---------------------------------------------------------------------------

def fit_global(
    image: SEMImage,
    layout: Layout,
    params: ParameterSet,
    **solver_kwargs,
) -> FitResult:
    """
    Fit *layout* to *image* by adjusting all free parameters jointly in a
    single ``least_squares`` call.

    Parameters
    ----------
    image : SEMImage
    layout : Layout
    params : ParameterSet
    **solver_kwargs
        Forwarded to ``least_squares`` (e.g. ``ftol``, ``max_nfev``).
    """
    starting_render = _setup(image, layout)

    tuner = Tuner(layout, image.image, params)
    last_opt = tuner.fit(**solver_kwargs)

    layout.render()
    residual = layout.image - image.image
    return FitResult(optimizer_result=last_opt, residual=residual,
                     starting_render=starting_render)


def fit_per_layer(
    image: SEMImage,
    layout: Layout,
    params: ParameterSet,
    **solver_kwargs,
) -> FitResult:
    """
    Fit *layout* to *image* by calling ``least_squares`` once per layer,
    from the topmost layer (highest z_order) down to the buried layers.

    Parameters
    ----------
    image : SEMImage
    layout : Layout
    params : ParameterSet
    **solver_kwargs
        Forwarded to ``least_squares`` (e.g. ``ftol``, ``max_nfev``).
    """
    starting_render = _setup(image, layout)
    last_opt = None

    for group in _layer_param_groups(params):
        tuner = Tuner(layout, image.image, group)
        last_opt = tuner.fit(**solver_kwargs)
        layout.render()

    residual = layout.image - image.image
    return FitResult(optimizer_result=last_opt, residual=residual,
                     starting_render=starting_render)

def fit_per_feature(
    image: SEMImage,
    layout: Layout,
    params: ParameterSet,
    **solver_kwargs,
) -> FitResult:
    """
    Fit *layout* to *image* by calling ``least_squares`` once per feature,
    from the topmost layer (highest z_order) down, and within each layer in
    first-appearance order.

    Parameters
    ----------
    image : SEMImage
    layout : Layout
    params : ParameterSet
    **solver_kwargs
        Forwarded to ``least_squares`` (e.g. ``ftol``, ``max_nfev``).
    """
    starting_render = _setup(image, layout)
    last_opt = None

    for group in _layer_param_groups(params):
        for fg in _per_feature_groups(group):
            roi = _feature_roi(fg.features[0], layout.shape)
            tuner = Tuner(layout, image.image, fg, roi=roi)
            last_opt = tuner.fit(**solver_kwargs)
        layout.render()

    residual = layout.image - image.image
    return FitResult(optimizer_result=last_opt, residual=residual,
                     starting_render=starting_render)


# Backwards-compatible alias.
fit = fit_per_feature
