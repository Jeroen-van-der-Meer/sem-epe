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
    Output of a completed :func:`fit` call.

    Attributes
    ----------
    optimizer_result : scipy.optimize.OptimizeResult
        scipy result from the final optimisation step; inspect ``.cost``,
        ``.nfev``, ``.message``, ``.success``, etc.
    residual : np.ndarray, dtype=float32, shape=(H, W)
        ``render(fitted) - target``: signed pixel-wise error at convergence.
    starting_render : np.ndarray, dtype=float32, shape=(H, W)
        Layout render at the perturbed starting point, before any fitting.
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


def fit(
    image: SEMImage,
    layout: Layout,
    params: ParameterSet,
    *,
    convergence_tol: float = 1e-6,
    max_passes: int = 20,
    **solver_kwargs,
) -> FitResult:
    """
    Fit *layout* to *image* by adjusting the free parameters in *params*.

    Parameters are grouped by layer and fitted sequentially from the topmost
    layer (highest z_order) down to the buried layers.  This exposes the
    buried layers to a cleaner residual before they are fitted.

    Bounds are taken from ``params`` (set via deviations at construction
    time); see :class:`~sem_epe.tune.ParameterSet`.

    Parameters
    ----------
    image : SEMImage
        The real SEM image to fit against.
    layout : Layout
        Expected layout, fully populated with layers and features.  Its
        pixel dimensions must match ``image.shape``.
    params : ParameterSet
        Declares which feature attributes are free parameters, together
        with their allowed deviations from the nominal values.
    convergence_tol : float, optional
        Stop iterating passes when the largest top-layer parameter change
        between consecutive passes drops below this threshold (pixels).
        Default: ``1e-6``.
    max_passes : int, optional
        Hard upper bound on the number of top-to-bottom passes.
        Default: ``20``.
    **solver_kwargs
        Additional keyword arguments forwarded to ``least_squares``
        (e.g. ``ftol``, ``xtol``, ``gtol``, ``max_nfev``, ``verbose``).

    Returns
    -------
    FitResult
        ``starting_render`` is captured before pass 1; ``residual`` and
        ``optimizer_result`` reflect the state after the final pass.
    """
    if image.shape != layout.shape:
        raise ValueError(
            f"image shape {image.shape} does not match layout {layout.shape}"
        )

    layer_groups = _layer_param_groups(params)

    layout.render()
    starting_render = layout.image.copy()

    last_opt = None
    top_group = layer_groups[0]

    for _ in range(max_passes):
        top_before = top_group.get().copy()

        tuner = Tuner(layout, image.image, top_group)
        last_opt = tuner.fit(**solver_kwargs)

        # If the top layer didn't move, the lower layers have also stabilised,
        # and we can stop the loop.
        if np.max(np.abs(top_group.get() - top_before)) < convergence_tol:
            break

        for group in layer_groups[1:]:
            tuner = Tuner(layout, image.image, group)
            last_opt = tuner.fit(**solver_kwargs)

    residual = layout.image - image.image
    return FitResult(optimizer_result=last_opt, residual=residual,
                     starting_render=starting_render)
