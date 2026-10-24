from pathlib import Path

import pytest

from examples.make_sample_dxf import build
from papa.toolpath import blocks as blockmod
from papa.toolpath import dxf_loader, fit


@pytest.fixture(scope="module")
def strokes(tmp_path_factory):
    dxf = build(tmp_path_factory.mktemp("dxf") / "sample.dxf")
    return dxf_loader.load(dxf)


def test_square_drawn_as_four_lines_becomes_one_stroke(strokes):
    # the closed square should chain into a single pen-down path
    closed = [s for s in strokes if len(s.points) >= 5 and s.points[0] == s.points[-1]]
    assert closed, "four shared-endpoint lines should have chained into one stroke"


def test_fit_is_uniform_and_inside_the_area(strokes):
    r = fit.fit(strokes, 25.0, 25.0)
    xs = [p[0] for s in r.strokes for p in s.points]
    ys = [p[1] for s in r.strokes for p in s.points]
    assert min(xs) >= -1e-9 and max(xs) <= 25.0 + 1e-9
    assert min(ys) >= -1e-9 and max(ys) <= 25.0 + 1e-9
    # uniform scale: aspect ratio preserved
    sw, sh = r.source_size
    fw, fh = r.fitted_size
    assert abs((fw / fh) - (sw / sh)) < 1e-6


def test_fit_flips_dxf_y_up_into_slide_y_down(strokes):
    r = fit.fit(strokes, 25.0, 25.0)
    src_top = max(p[1] for s in strokes for p in s.points)
    # the highest DXF point must land near y=0 in slide-local coordinates
    top_pt = min(p[1] for s in r.strokes for p in s.points)
    assert src_top > 0 and top_pt < 1.0


def test_pattern_square_change_rescales(strokes):
    small = fit.fit(strokes, 10.0, 10.0)
    big = fit.fit(strokes, 25.0, 25.0)
    assert big.scale > small.scale
    assert max(p[0] for s in small.strokes for p in s.points) <= 10.0 + 1e-9


def test_blocks_are_short_contiguous_and_cover_the_path(strokes):
    r = fit.fit(strokes, 25.0, 25.0)
    bl = blockmod.make_blocks(r.strokes, target_mm=3.0)
    assert bl and all(b.length <= 3.0 + 1e-6 for b in bl)
    total_path = sum(s.length for s in r.strokes)
    assert abs(sum(b.length for b in bl) - total_path) < 1e-6
    assert [b.index for b in bl] == list(range(len(bl)))
    # each stroke must start and end exactly once
    for si in {b.stroke_index for b in bl}:
        own = [b for b in bl if b.stroke_index == si]
        assert sum(b.starts_stroke for b in own) == 1
        assert sum(b.ends_stroke for b in own) == 1


def test_resume_backs_up_so_no_gap_is_left(strokes):
    r = fit.fit(strokes, 25.0, 25.0)
    bl = blockmod.make_blocks(r.strokes, target_mm=3.0)
    pts = blockmod.resume_points(bl[5], 0.3)
    assert len(pts) == len(bl[5].points) + 1
    a, b = pts[0], pts[1]
    assert 0 < ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= 0.3 + 1e-9


def test_unsupported_entities_are_rejected_by_name(tmp_path):
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.modelspace().add_circle((0, 0), 5)
    p = tmp_path / "circle.dxf"
    doc.saveas(str(p))
    with pytest.raises(dxf_loader.UnsupportedEntity, match="CIRCLE"):
        dxf_loader.load(p)
