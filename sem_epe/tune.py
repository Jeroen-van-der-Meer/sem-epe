from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.optimize

from .render import Feature, Layout


@dataclass
class Parameter:
    """A single free parameter for fitting: one attribute on one feature."""
    feature: Feature
    attribute: str


@dataclass
class FeatureResult:
    """
    Optimisation outcome for a single feature.

    Attributes
    ----------
    feature : Feature
        The feature that was optimised.
    params : list of Parameter
        The parameters that were adjusted.
    success : bool
        Whether the optimiser reported convergence.
    n_evaluations : int
        Total number of objective function evaluations.
    final_cost : float
        L1 cost (sum of absolute residuals) at convergence.
    """
    feature: Feature
    params: List[Parameter]
    success: bool
    n_evaluations: int
    final_cost: float


def tune(
    layout: Layout,
    target: np.ndarray,
    params: List[Parameter],
    regularization: float = 0.0,
    **kwargs,
) -> List[FeatureResult]:
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
    regularization : float, optional
        L2 regularization weight (λ).  Adds ``λ · sum((x − x₀)²)`` to the
        objective, where *x₀* are the nominal parameter values at the start
        of each feature's optimisation.
    **kwargs
        Forwarded to ``scipy.optimize.minimize`` via ``options``
        (e.g. ``xtol``, ``ftol``, ``maxfev``).

    Returns
    -------
    list of FeatureResult
        One entry per optimised feature, in first-appearance order.
    """
    by_feature: Dict[int, List[Parameter]] = {}
    for p in params:
        fid = id(p.feature)
        if fid not in by_feature:
            by_feature[fid] = []
        by_feature[fid].append(p)

    H, W = layout.shape
    results: List[FeatureResult] = []

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

        def residuals(x, _fp=feature_params, _f=feature, _r0=r0, _c0=c0, _r1=r1, _c1=c1, _tp=target_patch):
            for p, v in zip(_fp, x):
                setattr(p.feature, p.attribute, v)
            layout.rerender_feature(_f)
            return (layout.image[_r0:_r1, _c0:_c1] - _tp).ravel()

        def objective(x, _x0=x0):
            data_cost = np.sum(np.abs(residuals(x)))
            if regularization:
                return data_cost + regularization * np.sum((x - _x0) ** 2)
            return data_cost

        result = scipy.optimize.minimize(
            objective,
            x0=x0,
            method='Powell',
            options=kwargs or None,
        )

        for p, v in zip(feature_params, result.x):
            setattr(p.feature, p.attribute, v)
        layout.rerender_feature(feature)

        data_cost = np.sum(np.abs(residuals(result.x)))
        results.append(FeatureResult(
            feature=feature,
            params=feature_params,
            success=result.success,
            n_evaluations=result.nfev,
            final_cost=data_cost,
        ))

    return results
