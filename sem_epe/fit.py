from __future__ import annotations

from dataclasses import dataclass

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
        Full scipy result; inspect ``.cost``, ``.nfev``, ``.message``,
        ``.success``, etc.
    residual : np.ndarray, dtype=float32, shape=(H, W)
        ``render(fitted) - target``: signed pixel-wise error at convergence.
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


def fit(
    image: SEMImage,
    layout: Layout,
    params: ParameterSet,
    **solver_kwargs,
) -> FitResult:
    """
    Fit *layout* to *image* by adjusting the free parameters in *params*.

    This is the main entry point for SEM EPE analysis.  It renders the
    layout from scratch, runs the nonlinear least-squares optimiser, and
    returns a :class:`FitResult` whose ``.residual`` and ``.rms_error``
    summarise the quality of the fit.  On return the best-found parameter
    values are written back into the feature objects.

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
    **solver_kwargs
        Additional keyword arguments forwarded to ``least_squares``
        (e.g. ``ftol``, ``xtol``, ``gtol``, ``max_nfev``, ``verbose``).

    Returns
    -------
    FitResult
    """
    if image.shape != layout.shape:
        raise ValueError(
            f"image shape {image.shape} does not match layout {layout.shape}"
        )
    layout.render()
    starting_render = layout.image.copy()
    tuner = Tuner(layout, image.image, params)
    opt = tuner.fit(**solver_kwargs)
    residual = layout.image - image.image
    return FitResult(optimizer_result=opt, residual=residual, starting_render=starting_render)
