from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .image import SEMImage
from .render import Layout
from .tune import FeatureResult, Parameter, tune


@dataclass
class FitResult:
    """
    Output of a completed fit call.

    Attributes
    ----------
    starting_render : np.ndarray, shape=(H, W)
        Layout render at the nominal starting point, before any fitting.
    final_render : np.ndarray, shape=(H, W)
        Layout render after fitting.
    feature_results : list of FeatureResult
        Per-feature optimisation outcomes, in the order features were
        processed (top layer first, then descending z_order).
    """
    starting_render: np.ndarray
    final_render: np.ndarray
    feature_results: List[FeatureResult] = field(default_factory=list)


def fit(
    image: SEMImage,
    layout: Layout,
    params: List[Parameter],
    regularization: float = 0.0,
    **solver_kwargs,
) -> FitResult:
    """
    Fit *layout* to *image* by optimising each feature independently,
    processing layers from top (highest z_order) to bottom.

    Parameters
    ----------
    image : SEMImage
    layout : Layout
    params : list of Parameter
    regularization : float, optional
        Passed to :func:`tune`.
    **solver_kwargs
        Forwarded to ``scipy.optimize.minimize`` via ``options``
        (e.g. ``xtol``, ``ftol``, ``maxfev``).
    """
    if image.shape != layout.shape:
        raise ValueError(
            f"image shape {image.shape} does not match layout shape {layout.shape}"
        )

    by_layer: defaultdict = defaultdict(list)
    layer_map = {}
    for p in params:
        lid = id(p.feature.layer)
        by_layer[lid].append(p)
        layer_map[lid] = p.feature.layer

    sorted_layers = sorted(layer_map.values(), key=lambda l: l.z_order, reverse=True)

    layout.render()
    starting_render = layout.image.copy()
    all_feature_results: List[FeatureResult] = []
    for layer in sorted_layers:
        layer_results = tune(
            layout, image.image, by_layer[id(layer)],
            regularization=regularization,
            **solver_kwargs,
        )
        all_feature_results.extend(layer_results)
        final_render = layout.render()
    return FitResult(starting_render, final_render, all_feature_results)
