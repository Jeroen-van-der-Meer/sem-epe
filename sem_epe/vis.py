from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .fit import FitResult
    from .image import SEMImage


def plot_fit(
    target: "SEMImage",
    result: "FitResult",
) -> None:
    """
    Show a four-panel figure summarising a completed fit.

    Panels
    ------
    SEM image
        The SEM image passed to :func:`~sem_epe.fit`.
    Starting render
        The layout render at the perturbed starting point.
    Final render
        The layout render at the optimised parameters.
    Residual
        Delta between SEM imgae and converged render; a perfect fit
        produces a uniformly grey panel.

    Parameters
    ----------
    target : SEMImage
        The target image used during fitting.
    result : FitResult
        Return value of :func:`~sem_epe.fit`.
    """
    sem_image       = target.image
    starting_render = result.starting_render
    final_render    = target.image + result.residual
    delta           = result.residual
    vmax_delta      = float(np.abs(delta).max()) or 1e-6

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("SEM image",       sem_image,       dict(cmap="gray", vmin=0, vmax=1)),
        ("Starting render", starting_render, dict(cmap="gray", vmin=0, vmax=1)),
        ("Final render",    final_render,    dict(cmap="gray", vmin=0, vmax=1)),
        ("Residual",        delta,           dict(cmap="RdBu", vmin=-vmax_delta, vmax=vmax_delta)),
    ]
    for ax, (title, img, kw) in zip(axes, panels):
        im = ax.imshow(img, **kw)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.show()
