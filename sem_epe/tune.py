from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.optimize
import scipy.sparse

from .render import Feature, Layout

class ParameterSet:
    """
    Maps a flat optimisation vector to individual feature attributes,
    together with per-parameter bounds expressed as deviations from the
    nominal (design) values.

    Each entry is a ``(feature, attr)`` or ``(feature, attr, lo, hi)``
    tuple, where *lo* and *hi* are signed deviations from the nominal value
    (default: ``-inf`` / ``+inf``).  Nominal values are captured from the
    feature attributes at construction time.

    Parameters
    ----------
    params : list of (Feature, str) or (Feature, str, float, float)
        Each entry names a writable float attribute on the feature
        (e.g. ``'position'``, ``'thickness'``, ``'x'``, ``'diameter'``).
        The optional third and fourth elements are the lower and upper
        deviation bounds (e.g. ``-10, +10`` means ±10 pixels from nominal).
    """

    def __init__(self, params: List) -> None:
        self._params: List[Tuple[Feature, str]] = []
        self._lo_devs: List[float] = []
        self._hi_devs: List[float] = []

        for entry in params:
            if len(entry) == 2:
                f, a = entry
                lo, hi = -np.inf, np.inf
            elif len(entry) == 4:
                f, a, lo, hi = entry
            else:
                raise ValueError(
                    f"each entry must be (feature, attr) or "
                    f"(feature, attr, lo, hi); got {entry!r}"
                )
            self._params.append((f, a))
            self._lo_devs.append(float(lo))
            self._hi_devs.append(float(hi))

        seen: Dict[int, Feature] = {}
        for f, _ in self._params:
            if id(f) not in seen:
                seen[id(f)] = f
        self._features: List[Feature] = list(seen.values())

        self._nominals: np.ndarray = np.array(
            [getattr(f, a) for f, a in self._params], dtype=np.float64
        )

    @classmethod
    def for_features(
        cls,
        features: Sequence[Feature],
        attrs: Sequence[str],
        *,
        deviation: "float | Dict[str, float]" = np.inf,
    ) -> "ParameterSet":
        """
        Register the same *attrs* for every feature in *features*.

        Parameters
        ----------
        deviation : float or dict of {attr: float}, optional
            Symmetric deviation bound(s).  A scalar applies to every
            parameter; a dict maps attribute names to individual bounds.
            Default: unbounded (``inf``).
        """
        entries = []
        for f in features:
            for a in attrs:
                dev = deviation if np.isscalar(deviation) else deviation[a]
                entries.append((f, a, -float(dev), float(dev)))
        return cls(entries)

    def __len__(self) -> int:
        return len(self._params)

    @property
    def features(self) -> List[Feature]:
        """Unique features, in first-appearance order."""
        return self._features

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Absolute ``(lower, upper)`` bound arrays for
        ``scipy.optimize.least_squares``, computed as nominal ± deviation.
        """
        lo = self._nominals + np.array(self._lo_devs)
        hi = self._nominals + np.array(self._hi_devs)
        return lo, hi

    def get(self) -> np.ndarray:
        """Read current attribute values out of the features into a float64 vector."""
        return np.array([getattr(f, a) for f, a in self._params], dtype=np.float64)

    def set(self, x: np.ndarray) -> None:
        """Write values from *x* back into the feature attributes."""
        for (f, a), v in zip(self._params, x):
            setattr(f, a, float(v))


class Tuner:
    """
    Nonlinear least-squares fitter: tunes the feature parameters to
    minimise the MSE between the rendered layout and a target SEM image.

    The Jacobian sparsity structure is derived from feature bounding boxes
    and passed to ``scipy.optimize.least_squares``, which uses graph-coloured
    finite differences internally.  For a layout of non-overlapping lines
    within each layer (horizontal M1, vertical M2, …) the conflict graph is
    bipartite across layers, giving a chromatic number of O(n_layers) rather
    than O(n_features).  The full Jacobian is therefore computed in O(n_layers)
    residual evaluations per optimisation step, regardless of how many features
    are present.

    Parameters
    ----------
    layout : Layout
        Must have :meth:`~Layout.render` called before :meth:`fit`.
    target : np.ndarray, dtype=float32, shape=(H, W)
        The real SEM image to fit against.
    params : ParameterSet
        Declares which feature attributes are free parameters.
    """

    def __init__(
        self,
        layout: Layout,
        target: np.ndarray,
        params: ParameterSet,
        roi: "Tuple[int, int, int, int] | None" = None,
    ) -> None:
        if target.shape != layout.shape:
            raise ValueError(
                f"target shape {target.shape} does not match layout {layout.shape}"
            )
        self.layout = layout
        self.target = target.astype(np.float32)
        self.params = params
        self._roi = roi
        if roi is not None:
            r0, c0, r1, c1 = roi
            self._target_roi: np.ndarray = self.target[r0:r1, c0:c1]

    # ------------------------------------------------------------------

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ``render(x) - target``, flattened to 1-D.

        Applies *x* to the feature attributes and propagates every changed
        feature through :meth:`~Layout.rerender_feature`.  Because the dirty-
        region optimisation in ``rerender_feature`` fires a cheap early exit
        when parameters are unchanged, calling it unconditionally for all
        features is safe even when scipy only perturbs a subset of them.
        """
        self.params.set(x)
        for feature in self.params.features:
            self.layout.rerender_feature(feature)
        if self._roi is not None:
            r0, c0, r1, c1 = self._roi
            return (self.layout.image[r0:r1, c0:c1] - self._target_roi).ravel()
        return (self.layout.image - self.target).ravel()

    def jacobian_sparsity(self) -> scipy.sparse.spmatrix:
        """
        Build the ``(H*W) × n_params`` sparsity mask of the Jacobian.

        Parameter *i* can only affect pixels inside the bounding box of its
        feature.  scipy uses this mask to colour the conflict graph and batch
        all same-colour parameters into a single finite-difference perturbation.
        """
        h, w = self.layout.shape
        rows_list: List[np.ndarray] = []
        cols_list: List[np.ndarray] = []

        for i, (feature, _) in enumerate(self.params._params):
            r0, c0, r1, c1 = feature.bounding_box(self.layout.shape)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            pixel_idx = (rr * w + cc).ravel()
            rows_list.append(pixel_idx)
            cols_list.append(np.full(len(pixel_idx), i, dtype=np.intp))

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        return scipy.sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(h * w, len(self.params)),
        )

    # ------------------------------------------------------------------

    def fit(self, **kwargs) -> scipy.optimize.OptimizeResult:
        """
        Run the optimisation and return scipy's ``OptimizeResult``.

        Uses the trust-region reflective method (``'trf'``), the only scipy
        method that accepts ``jac_sparsity``.  Extra keyword arguments are
        forwarded to ``scipy.optimize.least_squares`` (e.g. ``ftol``,
        ``max_nfev``).

        On return the best-found parameters are written back into the features
        and the layout image is updated incrementally via
        :meth:`~Layout.rerender_feature`.  Callers that need a fully
        consistent render (e.g. after all features have been fitted) should
        call ``layout.render()`` themselves.
        """
        jac_sparsity = None if self._roi is not None else self.jacobian_sparsity()
        result = scipy.optimize.least_squares(
            self.residuals,
            x0=self.params.get(),
            method="trf",
            jac="3-point",
            jac_sparsity=jac_sparsity,
            bounds=self.params.bounds,
            **kwargs,
        )

        self.params.set(result.x)
        for feature in self.params.features:
            self.layout.rerender_feature(feature)
        return result

