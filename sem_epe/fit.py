from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np

from .image import SEMImage
from .render import Layout
from .tune import Parameter, tune


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
    """
    starting_render: np.ndarray
    final_render: np.ndarray


def fit(
    image: SEMImage,
    layout: Layout,
    params: List[Parameter],
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
    **solver_kwargs
        Forwarded to ``scipy.optimize.least_squares`` (e.g. ``ftol``, ``max_nfev``).

    Returns
    -------
    starting_render : np.ndarray
        A copy of ``layout.image`` at the nominal starting point, before any
        optimisation.  Useful for visualising how much the fit moved.
    """
    if image.shape != layout.shape:
        raise ValueError(
            f"image shape {image.shape} does not match layout shape {layout.shape}"
        )

    # FIXME: May move to `tune.py`.
    by_layer: defaultdict = defaultdict(list)
    layer_map = {}
    for p in params:
        lid = id(p.feature.layer)
        by_layer[lid].append(p)
        layer_map[lid] = p.feature.layer

    sorted_layers = sorted(layer_map.values(), key=lambda l: l.z_order, reverse=True)

    layout.render()
    starting_render = layout.image.copy()
    for layer in sorted_layers:
        tune(layout, image.image, by_layer[id(layer)], **solver_kwargs)
        final_render = layout.render()
    return FitResult(starting_render, final_render)
