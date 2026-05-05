import numpy as np
import pytest

import sem_epe as epe
from sem_epe.align import _poly_features, _apply_warp, _initial_coeffs


# ---------------------------------------------------------------------------
# Layout.region_mask
# ---------------------------------------------------------------------------

class TestRegionMask:
    def _line_layout(self):
        """10×5 layout with a single horizontal line at position=5, thickness=4."""
        layer = epe.Layer("l", z_order=1)
        layer.add_feature(epe.Line(epe.Orientation.HORIZONTAL, thickness=4.0, position=5.0))
        layout = epe.Layout(10, 5, background=0.0)
        layout.add_layer(layer)
        return layout, layer

    def test_owners_structure(self):
        layout, layer = self._line_layout()
        region, owners = layout.region_mask()
        assert owners[0] is None          # index 0 = background
        assert owners[1] is layer         # index 1 = the single layer

    def test_interior_pixels(self):
        """Pixels fully inside the line (coverage = 1.0) get the layer's index."""
        layout, _ = self._line_layout()
        region, _ = layout.region_mask(threshold=0.9)
        # For position=5, half=2: interior rows are 4, 5, 6 (coverage 1.0)
        for row in [4, 5, 6]:
            assert np.all(region[row, :] == 1), f"row {row} should be layer region"

    def test_background_pixels(self):
        """Pixels far from the line (coverage = 0.0) get region index 0."""
        layout, _ = self._line_layout()
        region, _ = layout.region_mask(threshold=0.9)
        for row in [0, 1, 2, 8, 9]:
            assert np.all(region[row, :] == 0), f"row {row} should be background"

    def test_edge_pixels_excluded(self):
        """Boundary pixels (coverage ≈ 0.5) are excluded (region = -1)."""
        layout, _ = self._line_layout()
        region, _ = layout.region_mask(threshold=0.9)
        for row in [3, 7]:
            assert np.all(region[row, :] == -1), f"row {row} should be excluded"

    def test_two_layers_top_wins(self):
        """Where both layers overlap, the top (higher z_order) layer is assigned."""
        bottom = epe.Layer("bottom", z_order=0)
        top = epe.Layer("top", z_order=1)
        # Thick horizontal line in the bottom layer covers the whole width.
        bottom.add_feature(
            epe.Line(epe.Orientation.HORIZONTAL, thickness=20.0, position=25.0)
        )
        # Narrow vertical line in the top layer runs through the centre.
        top.add_feature(
            epe.Line(epe.Orientation.VERTICAL, thickness=4.0, position=25.0)
        )
        layout = epe.Layout(50, 50, background=0.0)
        layout.add_layer(bottom)
        layout.add_layer(top)

        region, owners = layout.region_mask(threshold=0.9)

        assert owners[1] is top     # topmost layer processed first → index 1
        assert owners[2] is bottom

        # Centre of the vertical top-layer line.
        assert region[25, 25] == 1
        # Inside the horizontal bottom-layer line, far from the vertical line.
        assert region[25, 2] == 2

    def test_no_gray_value_required(self):
        """region_mask works even when gray_value is None (not yet set)."""
        layer = epe.Layer("l", z_order=1)   # gray_value defaults to None
        layer.add_feature(epe.Line(epe.Orientation.HORIZONTAL, thickness=6.0, position=15.0))
        layout = epe.Layout(30, 20, background=0.0)
        layout.add_layer(layer)
        # Should not raise even though gray_value is None.
        region, owners = layout.region_mask()
        assert region.shape == (30, 20)


# ---------------------------------------------------------------------------
# _poly_features
# ---------------------------------------------------------------------------

class TestPolyFeatures:
    def test_shape_degree1(self):
        assert _poly_features(10, 8, degree=1).shape == (80, 3)

    def test_shape_degree2(self):
        assert _poly_features(10, 8, degree=2).shape == (80, 6)

    def test_constant_column_is_ones(self):
        P = _poly_features(7, 9, degree=2)
        np.testing.assert_array_equal(P[:, 0], 1.0)

    def test_coordinates_normalised(self):
        """Row and column normalised coordinates span exactly [-1, 1]."""
        P = _poly_features(5, 7, degree=1)
        # Column 1 = r̃, column 2 = c̃
        assert P[:, 1].min() == pytest.approx(-1.0)
        assert P[:, 1].max() == pytest.approx(1.0)
        assert P[:, 2].min() == pytest.approx(-1.0)
        assert P[:, 2].max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _apply_warp with identity coefficients
# ---------------------------------------------------------------------------

class TestApplyWarp:
    def test_identity_same_size(self):
        """Initial coefficients for a same-size SEM map each pixel to itself."""
        H, W = 20, 30
        poly = _poly_features(H, W, degree=2)
        coeffs = _initial_coeffs(H, W, H, W, degree=2)
        r_sem, c_sem = _apply_warp(poly, coeffs)

        expected_r = np.repeat(np.arange(H, dtype=float), W)
        expected_c = np.tile(np.arange(W, dtype=float), H)
        np.testing.assert_allclose(r_sem, expected_r, atol=1e-10)
        np.testing.assert_allclose(c_sem, expected_c, atol=1e-10)

    def test_centered_in_larger_sem(self):
        """Initial coefficients centre the layout inside a larger SEM."""
        H, W, PAD = 20, 30, 5
        H_sem, W_sem = H + 2 * PAD, W + 2 * PAD
        poly = _poly_features(H, W, degree=1)
        coeffs = _initial_coeffs(H, W, H_sem, W_sem, degree=1)
        r_sem, c_sem = _apply_warp(poly, coeffs)

        # Top-left layout pixel (0, 0) should map to SEM pixel (PAD, PAD).
        assert r_sem[0] == pytest.approx(PAD, abs=1e-10)
        assert c_sem[0] == pytest.approx(PAD, abs=1e-10)
        # Bottom-right layout pixel (H-1, W-1) → (H-1+PAD, W-1+PAD).
        assert r_sem[-1] == pytest.approx(H - 1 + PAD, abs=1e-10)
        assert c_sem[-1] == pytest.approx(W - 1 + PAD, abs=1e-10)


# ---------------------------------------------------------------------------
# align — end-to-end
# ---------------------------------------------------------------------------

class TestAlign:
    def _simple_layout(self):
        """Layout with one horizontal line; gray_value left unset."""
        layer = epe.Layer("l", z_order=1)
        layer.add_feature(
            epe.Line(epe.Orientation.HORIZONTAL, thickness=20.0, position=50.0)
        )
        layout = epe.Layout(100, 100, background=0.0)
        layout.add_layer(layer)
        return layout, layer

    def test_gray_values_recovered(self):
        """
        When the SEM image equals the layout render, align() recovers the
        correct gray values and writes them back to the layout.
        """
        layout, layer = self._simple_layout()

        TRUE_GRAY = 0.7
        TRUE_BG   = 0.1

        # Render with known gray values, then strip them.
        layer.gray_value = TRUE_GRAY
        layout.background = TRUE_BG
        render = layout.render().copy()
        layer.gray_value = None
        layout.background = 0.0

        sem = epe.SEMImage(render, nm_per_pixel=1.0)
        epe.align(sem, layout, degree=1)

        assert layer.gray_value == pytest.approx(TRUE_GRAY, abs=0.01)
        assert layout.background == pytest.approx(TRUE_BG, abs=0.01)

    def test_corrected_image_matches_render(self):
        """Corrected image equals the original render for a distortion-free SEM."""
        layout, layer = self._simple_layout()

        layer.gray_value = 0.6
        layout.background = 0.15
        render = layout.render().copy()
        layer.gray_value = None
        layout.background = 0.0

        sem = epe.SEMImage(render, nm_per_pixel=1.0)
        result = epe.align(sem, layout, degree=1)

        np.testing.assert_allclose(result.corrected, render, atol=0.01)

    def test_larger_sem_centered(self):
        """align() succeeds when the SEM is larger and the layout is centred."""
        layout, layer = self._simple_layout()
        PAD = 10

        TRUE_GRAY = 0.8
        layer.gray_value = TRUE_GRAY
        layout.background = 0.05
        render = layout.render().copy()
        layer.gray_value = None
        layout.background = 0.0

        # Pad the render symmetrically to simulate a larger SEM FOV.
        sem_arr = np.pad(render, PAD, constant_values=0.05).astype(np.float32)
        sem = epe.SEMImage(sem_arr, nm_per_pixel=1.0)

        result = epe.align(sem, layout, degree=1)

        assert layer.gray_value == pytest.approx(TRUE_GRAY, abs=0.02)
        np.testing.assert_allclose(result.corrected, render, atol=0.02)

    def test_layout_renderable_after_align(self):
        """After align(), the layout has valid gray values and render() succeeds."""
        layout, layer = self._simple_layout()
        layer.gray_value = 0.5
        layout.background = 0.1
        render = layout.render().copy()
        layer.gray_value = None
        layout.background = 0.0

        sem = epe.SEMImage(render, nm_per_pixel=1.0)
        epe.align(sem, layout, degree=1)

        # Should not raise.
        layout.render()
        assert layer.gray_value is not None
