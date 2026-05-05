from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .fit import FitResult
    from .image import SEMImage
    from .render import Layout


def plot_fit(
    target: "SEMImage",
    result: "FitResult",
) -> None:
    """
    Show a four-panel figure summarising a completed fit.

    Panels
    ------
    Starting render
        The layout render at the nominal starting point.
    SEM image
        The input SEM image.
    Final render
        The layout render at the optimised parameters.
    Residual
        Delta between SEM image and converged render; a perfect fit
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
    final_render    = result.final_render
    delta           = sem_image - final_render
    vmax_delta      = float(np.abs(delta).max()) or 1e-6

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    panels = [
        ("Starting render", starting_render, dict(cmap="gray", vmin=0, vmax=1)),
        ("SEM image",       sem_image,       dict(cmap="gray", vmin=0, vmax=1)),
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


def plot_overlay(
    target: "SEMImage",
    layout: "Layout",
    *,
    contour_color: str = "red",
    linewidth: float = 0.8,
) -> None:
    """
    Show the SEM image with one subplot per layer, each with that layer's
    feature outlines overlaid.

    Parameters
    ----------
    target : SEMImage
        The SEM image to use as the background.
    layout : Layout
        Layout after fitting.
    contour_color : str, optional
        Matplotlib color for the contour lines.  Default: ``"red"``.
    linewidth : float, optional
        Width of the contour lines in points.  Default: 0.8.
    """
    H, W = layout.shape
    layers = layout.layers
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], layers):
        ax.imshow(target.image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        mask = layer.render_mask((0, 0, H, W))
        ax.contour(mask, levels=[0.5], colors=contour_color, linewidths=linewidth)
        ax.set_title(layer.name)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
