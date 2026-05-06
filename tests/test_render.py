import numpy as np
import pytest

from sem_epe import Layout, Layer, Line, Pillar, Orientation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def single_layer_layout(feature, *, size=40, gray=0.6, background=0.1):
    layer = Layer("l", gray_value=gray, z_order=1)
    layer.add_feature(feature)
    layout = Layout(size, size, background=background)
    layout.add_layer(layer)
    return layout


def assert_rerender_matches_render(layout, feature, **new_attrs):
    """Core invariant: rerender_feature must give the same image as a fresh render."""
    layout.render()
    for k, v in new_attrs.items():
        setattr(feature, k, v)
    layout.rerender_feature(feature)
    incremental = layout.image.copy()
    layout.render()
    np.testing.assert_array_equal(incremental, layout.image)


# ---------------------------------------------------------------------------
# render_mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feature", [
    Line(Orientation.HORIZONTAL, thickness=4, position=10),
    Line(Orientation.VERTICAL,   thickness=4, position=10),
    Pillar(x=10, y=10, diameter=6),
], ids=["h-line", "v-line", "pillar"])
def test_mask_shape_matches_roi(feature):
    assert feature.render_mask((3, 5, 14, 18)).shape == (11, 13)


@pytest.mark.parametrize("feature", [
    Line(Orientation.HORIZONTAL, thickness=6, position=15),
    Line(Orientation.VERTICAL,   thickness=6, position=15),
    Line(Orientation.HORIZONTAL, thickness=6, position=15, extent=(5, 25)),
    Pillar(x=15, y=15, diameter=8),
], ids=["h-line", "v-line", "h-line-extent", "pillar"])
def test_mask_roi_offset_matches_full_image_subregion(feature):
    """Fundamental contract for the dirty-region optimisation in rerender_feature."""
    full = feature.render_mask((0, 0, 30, 30))
    r0, c0, r1, c1 = 5, 4, 25, 26
    np.testing.assert_array_equal(full[r0:r1, c0:c1], feature.render_mask((r0, c0, r1, c1)))


# ---------------------------------------------------------------------------
# bounding_box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feature", [
    Line(Orientation.HORIZONTAL, thickness=6, position=15),
    Line(Orientation.VERTICAL,   thickness=6, position=15),
    Line(Orientation.HORIZONTAL, thickness=6, position=15, extent=(5, 25)),
    Pillar(x=15, y=15, diameter=8),
], ids=["h-line", "v-line", "h-line-extent", "pillar"])
def test_bbox_contains_all_covered_pixels(feature):
    shape = (30, 30)
    r0, c0, r1, c1 = feature.bounding_box(shape)
    rows, cols = np.where(feature.render_mask((0, 0, *shape)))
    if rows.size:
        assert rows.min() >= r0 and rows.max() < r1
        assert cols.min() >= c0 and cols.max() < c1


@pytest.mark.parametrize("feature", [
    Line(Orientation.HORIZONTAL, thickness=6, position=15),
    Line(Orientation.VERTICAL,   thickness=6, position=15),
    Pillar(x=15, y=15, diameter=8),
], ids=["h-line", "v-line", "pillar"])
def test_bbox_render_mask_consistent(feature):
    """render_mask(bbox) must equal the bbox subregion of the full-image mask."""
    shape = (30, 30)
    r0, c0, r1, c1 = feature.bounding_box(shape)
    full = feature.render_mask((0, 0, *shape))
    np.testing.assert_array_equal(full[r0:r1, c0:c1], feature.render_mask((r0, c0, r1, c1)))


def test_bbox_clipped_at_image_boundary():
    assert Line(Orientation.HORIZONTAL, thickness=10, position=2 ).bounding_box((20, 20))[0] == 0
    assert Line(Orientation.HORIZONTAL, thickness=10, position=18).bounding_box((20, 20))[2] == 20


# ---------------------------------------------------------------------------
# Layout.render
# ---------------------------------------------------------------------------

def test_render_shape_and_dtype():
    img = Layout(20, 30, background=0.0).render()
    assert img.shape == (20, 30) and img.dtype == float


def test_render_empty_layout_is_background():
    np.testing.assert_array_equal(Layout(20, 20, background=0.5).render(), 0.5)


def test_render_single_feature_gray_value():
    f = Line(Orientation.HORIZONTAL, thickness=4, position=10)
    img = single_layer_layout(f, size=20, gray=0.7, background=0.0).render()
    assert img[10, 10] == pytest.approx(0.7)
    assert img[0,  0 ] == pytest.approx(0.0)


def test_render_z_order_higher_wins():
    bot = Layer("bot", gray_value=0.3, z_order=1)
    bot.add_feature(Pillar(x=10, y=10, diameter=10))
    top = Layer("top", gray_value=0.9, z_order=2)
    top.add_feature(Pillar(x=10, y=10, diameter=4))
    layout = Layout(20, 20, background=0.0)
    layout.add_layer(bot).add_layer(top)
    img = layout.render()
    assert img[10, 10] == pytest.approx(0.9)  # both cover centre; top wins
    assert img[10, 14] == pytest.approx(0.3)  # only bottom covers this pixel


def test_render_z_order_independent_of_insertion_order():
    lo = Layer("lo", gray_value=0.2, z_order=1)
    hi = Layer("hi", gray_value=0.8, z_order=2)
    lo.add_feature(Pillar(x=10, y=10, diameter=8))
    hi.add_feature(Pillar(x=10, y=10, diameter=4))
    img_ab = Layout(20, 20).add_layer(lo).add_layer(hi).render()
    img_ba = Layout(20, 20).add_layer(hi).add_layer(lo).render()
    np.testing.assert_array_equal(img_ab, img_ba)


# ---------------------------------------------------------------------------
# Layout.rerender_feature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("new_attrs", [
    {"position": 25},
    {"thickness": 8},
    {"position": 1},    # near image boundary
], ids=["shift", "widen", "boundary"])
def test_rerender_horizontal_line(new_attrs):
    f = Line(Orientation.HORIZONTAL, thickness=4, position=20)
    assert_rerender_matches_render(single_layer_layout(f), f, **new_attrs)


@pytest.mark.parametrize("new_attrs", [
    {"position": 25},
    {"thickness": 8},
], ids=["shift", "widen"])
def test_rerender_vertical_line(new_attrs):
    f = Line(Orientation.VERTICAL, thickness=4, position=20)
    assert_rerender_matches_render(single_layer_layout(f), f, **new_attrs)


@pytest.mark.parametrize("new_attrs", [
    {"x": 25},
    {"diameter": 14},
    {"x": 1, "y": 1},  # near corner
], ids=["shift-x", "grow", "corner"])
def test_rerender_pillar(new_attrs):
    f = Pillar(x=20, y=20, diameter=8)
    assert_rerender_matches_render(single_layer_layout(f), f, **new_attrs)


def test_rerender_noop_leaves_image_unchanged():
    f = Pillar(x=20, y=20, diameter=8)
    layout = single_layer_layout(f)
    layout.render()
    before = layout.image.copy()
    layout.rerender_feature(f)
    np.testing.assert_array_equal(before, layout.image)


def test_rerender_sequential_stays_consistent():
    f = Pillar(x=20, y=20, diameter=6)
    layout = single_layer_layout(f)
    layout.render()
    for _ in range(5):
        f.x += 1
        layout.rerender_feature(f)
    incremental = layout.image.copy()
    layout.render()
    np.testing.assert_array_equal(incremental, layout.image)


def test_rerender_lower_layer_obscured_by_upper():
    """Moving a bottom-layer feature under a top-layer feature: z-order must hold."""
    bot = Layer("bot", gray_value=0.3, z_order=1)
    f = Pillar(x=10, y=10, diameter=6)
    bot.add_feature(f)
    top = Layer("top", gray_value=0.9, z_order=2)
    top.add_feature(Pillar(x=25, y=25, diameter=6))
    layout = Layout(40, 40, background=0.0)
    layout.add_layer(bot).add_layer(top)
    assert_rerender_matches_render(layout, f, x=25, y=25)


# ---------------------------------------------------------------------------
# Zero-width / zero-diameter edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thickness", [0.0, -1.0, -10.0])
@pytest.mark.parametrize("orientation", [Orientation.HORIZONTAL, Orientation.VERTICAL],
                         ids=["horizontal", "vertical"])
def test_nonpositive_thickness_line_renders_empty(orientation, thickness):
    """A line with thickness ≤ 0 must produce an all-zero mask without crashing."""
    f = Line(orientation, thickness=thickness, position=10.0)
    mask = f.render_mask((0, 0, 20, 20))
    assert mask.shape == (20, 20)
    assert mask.max() == 0.0


@pytest.mark.parametrize("diameter", [0.0, -1.0, -10.0])
def test_nonpositive_diameter_pillar_renders_empty(diameter):
    """A pillar with diameter ≤ 0 must produce an all-zero mask without crashing."""
    f = Pillar(x=10.0, y=10.0, diameter=diameter)
    mask = f.render_mask((0, 0, 20, 20))
    assert mask.shape == (20, 20)
    assert mask.max() == 0.0


@pytest.mark.parametrize("thickness", [0.0, -1.0, -10.0])
@pytest.mark.parametrize("orientation", [Orientation.HORIZONTAL, Orientation.VERTICAL],
                         ids=["horizontal", "vertical"])
def test_nonpositive_thickness_line_layout_is_background(orientation, thickness):
    """A layout with only a thickness ≤ 0 line must render to pure background."""
    f = Line(orientation, thickness=thickness, position=10.0)
    img = single_layer_layout(f, size=20, gray=0.7, background=0.1).render()
    np.testing.assert_array_equal(img, 0.1)


@pytest.mark.parametrize("diameter", [0.0, -1.0, -10.0])
def test_nonpositive_diameter_pillar_layout_is_background(diameter):
    """A layout with only a diameter ≤ 0 pillar must render to pure background."""
    f = Pillar(x=10.0, y=10.0, diameter=diameter)
    img = single_layer_layout(f, size=20, gray=0.7, background=0.1).render()
    np.testing.assert_array_equal(img, 0.1)


# ---------------------------------------------------------------------------
# Layer transparency
# ---------------------------------------------------------------------------

def _two_layer_layout(*, bot_gray, top_gray, top_transparency, size=20):
    """Background=0, bottom layer full coverage, top layer full coverage."""
    bot = Layer("bot", gray_value=bot_gray, z_order=1)
    bot.add_feature(Line(Orientation.HORIZONTAL, thickness=size, position=size / 2))
    top = Layer("top", gray_value=top_gray, z_order=2, transparency=top_transparency)
    top.add_feature(Line(Orientation.HORIZONTAL, thickness=size, position=size / 2))
    layout = Layout(size, size, background=0.0)
    layout.add_layer(bot)
    layout.add_layer(top)
    return layout


def test_fully_opaque_layer_covers_below():
    """transparency=0 (default): top layer completely replaces bottom."""
    img = _two_layer_layout(bot_gray=0.2, top_gray=0.8, top_transparency=0.0).render()
    assert img[10, 10] == pytest.approx(0.8)


def test_fully_transparent_layer_has_no_effect():
    """transparency=1: top layer is invisible; bottom shows through everywhere."""
    img = _two_layer_layout(bot_gray=0.2, top_gray=0.8, top_transparency=1.0).render()
    assert img[10, 10] == pytest.approx(0.2)


def test_half_transparent_layer_blends():
    """transparency=0.5: pixel value should be halfway between bottom and top gray."""
    bot_gray, top_gray = 0.2, 0.8
    img = _two_layer_layout(bot_gray=bot_gray, top_gray=top_gray, top_transparency=0.5).render()
    # image = bot_gray + lm * 0.5 * (top_gray - bot_gray) = 0.2 + 0.5 * 0.6 = 0.5
    assert img[10, 10] == pytest.approx(bot_gray + 0.5 * (top_gray - bot_gray))


def test_transparency_rerender_matches_render():
    """rerender_feature must give the same result as a fresh render for transparent layers."""
    f = Pillar(x=10, y=10, diameter=6)
    layer = Layer("l", gray_value=0.9, z_order=1, transparency=0.4)
    layer.add_feature(f)
    layout = Layout(20, 20, background=0.1)
    layout.add_layer(layer)
    assert_rerender_matches_render(layout, f, x=14, y=14)


# ---------------------------------------------------------------------------
# Layer invert (holes)
# ---------------------------------------------------------------------------

def test_inverted_layer_solid_outside_features():
    """Outside all features an inverted layer paints its gray_value."""
    f = Pillar(x=10, y=10, diameter=4)
    layer = Layer("l", gray_value=0.7, z_order=1, invert=True)
    layer.add_feature(f)
    layout = Layout(20, 20, background=0.0)
    layout.add_layer(layer)
    img = layout.render()
    # Far corner: no feature → inverted layer fully covers → gray_value
    assert img[0, 0] == pytest.approx(0.7)


def test_inverted_layer_transparent_inside_features():
    """Centre of a feature in an inverted layer is a hole — background shows through."""
    f = Pillar(x=10, y=10, diameter=6)
    layer = Layer("l", gray_value=0.7, z_order=1, invert=True)
    layer.add_feature(f)
    layout = Layout(20, 20, background=0.2)
    layout.add_layer(layer)
    img = layout.render()
    assert img[10, 10] == pytest.approx(0.2)


def test_inverted_layer_over_lower_layer():
    """Hole in inverted top layer exposes the lower layer's gray value."""
    bot = Layer("bot", gray_value=0.4, z_order=1)
    bot.add_feature(Line(Orientation.HORIZONTAL, thickness=20, position=10))
    top = Layer("top", gray_value=0.9, z_order=2, invert=True)
    top.add_feature(Pillar(x=10, y=10, diameter=6))
    layout = Layout(20, 20, background=0.0)
    layout.add_layer(bot).add_layer(top)
    img = layout.render()
    # Centre pixel: hole in top → shows bot gray
    assert img[10, 10] == pytest.approx(0.4)
    # Away from hole: top layer solid → top gray
    assert img[0, 0] == pytest.approx(0.9)


def test_inverted_layer_rerender_matches_render():
    """rerender_feature must give the same result as a fresh render for inverted layers."""
    f = Pillar(x=10, y=10, diameter=6)
    layer = Layer("l", gray_value=0.8, z_order=1, invert=True)
    layer.add_feature(f)
    layout = Layout(20, 20, background=0.1)
    layout.add_layer(layer)
    assert_rerender_matches_render(layout, f, x=14, y=14)
