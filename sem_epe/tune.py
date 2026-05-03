from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import scipy.optimize

from .render import Feature, Layout


@dataclass
class Parameter:
    """A single free parameter for fitting: one attribute on one feature."""
    feature: Feature
    attribute: str
    lo: float = -math.inf
    hi: float = math.inf


def tune(
    layout: Layout,
    target: np.ndarray,
    params: List[Parameter],
    **kwargs,
) -> None:
    """
    Fit feature attributes to minimise residuals against *target*, in-place.

    Features are processed in first-appearance order.  For each feature,
    optimisation is restricted to a ROI twice the size of its bounding box
    (half-dimension margin on each side, clamped to image bounds).

    Parameters
    ----------
    layout : Layout
        Must have been rendered before calling this function.
    target : np.ndarray
        The SEM image to fit against.  Must have the same shape as ``layout``.
    params : list of Parameter
        Free parameters.  Multiple parameters per feature are supported and
        are optimised jointly for that feature.
    **kwargs
        Forwarded to ``scipy.optimize.least_squares`` (e.g. ``ftol``, ``max_nfev``).
    """
    by_feature: Dict[int, List[Parameter]] = {}
    for p in params:
        fid = id(p.feature)
        if fid not in by_feature:
            by_feature[fid] = []
        by_feature[fid].append(p)

    H, W = layout.shape

    for feature_params in by_feature.values():
        feature = feature_params[0].feature

        r0, c0, r1, c1 = feature.bounding_box(layout.shape)
        h, w = r1 - r0, c1 - c0
        r0 = max(0, r0 - h // 2)
        c0 = max(0, c0 - w // 2)
        r1 = min(H, r1 + h // 2)
        c1 = min(W, c1 + w // 2)
        target_patch = target[r0:r1, c0:c1]

        nominals = [getattr(p.feature, p.attribute) for p in feature_params]
        x0 = np.array(nominals)
        lo_bounds = np.array([n + p.lo for n, p in zip(nominals, feature_params)])
        hi_bounds = np.array([n + p.hi for n, p in zip(nominals, feature_params)])

        def residuals(x, _fp=feature_params, _f=feature, _r0=r0, _c0=c0, _r1=r1, _c1=c1, _tp=target_patch):
            for p, v in zip(_fp, x):
                setattr(p.feature, p.attribute, float(v))
            layout.rerender_feature(_f)
            return (layout.image[_r0:_r1, _c0:_c1] - _tp).ravel()

        result = scipy.optimize.least_squares(
            residuals,
            x0=x0,
            bounds=(lo_bounds, hi_bounds),
            method="trf",
            jac="3-point",
            **kwargs,
        )

        for p, v in zip(feature_params, result.x):
            setattr(p.feature, p.attribute, float(v))
        layout.rerender_feature(feature)
