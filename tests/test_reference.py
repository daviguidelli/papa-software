"""Reference substrate: red-dot matching, rigid fit and moving a G-code file."""
import math

import numpy as np
import pytest

from papa.ops import reference as rf

PL = {"x_min": 162.5, "y_min": 162.1, "size": 25.0}


def turned(ref, deg, shift, about=None):
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    c = ref.mean(0) if about is None else np.asarray(about)
    return (ref - c) @ R.T + c + shift


@pytest.mark.parametrize("deg", [0.0, 1.7, -3.0, 90.0, 180.0, -90.0])
def test_red_dot_fixes_the_orientation_any_way_round(deg):
    ref = rf.reference_mm(PL)
    new = turned(ref, deg, [20, -15])
    m = rf.match(ref, 2, new[[1, 3, 0, 2]], new[2], 25)          # corners in any order
    assert abs((m.rotation_deg - deg + 180) % 360 - 180) < 1e-9
    assert m.rms_mm == pytest.approx(0, abs=1e-9) and m.red_found and not m.warnings
    p = np.array([170.0, 175.0])
    assert m.R @ p + m.t == pytest.approx(turned(p[None], deg, [20, -15], ref.mean(0))[0])


def test_without_the_dot_the_smallest_turn_is_taken_and_flagged():
    ref = rf.reference_mm(PL)
    m = rf.match(ref, 0, turned(ref, 2.5, [1, 2]), None, 25)
    assert m.rotation_deg == pytest.approx(2.5) and not m.red_found
    assert any("Red dot not found" in w for w in m.warnings)


def test_wrong_size_is_flagged():
    ref = rf.reference_mm(PL)
    big = (ref - ref.mean(0)) * 1.04 + ref.mean(0)
    assert any("Glass measures 26.00" in w for w in rf.match(ref, 0, big, big[0], 25).warnings)


def test_gcode_moves_rotate_including_single_axis_relative_and_arcs():
    g = "\n".join([
        "G21 ; mm", "G90", "G0 Z10 F300", "G0 X10 Y0 F3000", "G1 Z4.5", "G1 X20 F600 ; only X",
        "G1 Y5", "G2 X25 Y10 I5 J0", "G91", "G1 X1 Y0", "G90", "G28",
    ])
    th = math.radians(90)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out, segs = rf.transform_gcode(g, R, np.array([100.0, 0.0]))
    lines = out.splitlines()
    assert lines[3] == "G0 X100.000 Y10.000 F3000"
    assert lines[5] == "G1 X100.000 Y20.000 F600 ; only X"           # both axes written out
    assert lines[6] == "G1 X95.000 Y20.000"
    assert lines[7] == "G2 X90.000 Y25.000 I0.000 J5.000"
    assert lines[9] == "G1 X0.000 Y1.000"                               # relative: turned only
    assert lines[2] == "G0 Z10 F300" and lines[11] == "G28"
    assert [s["cut"] for s in segs] == [True, True, True, True]


def test_unsafe_files_are_refused():
    with pytest.raises(rf.ReferenceError, match="G20"):
        rf.transform_gcode("G20\nG0 X1 Y1\n")
    with pytest.raises(rf.ReferenceError, match="G92"):
        rf.transform_gcode("G92 X0 Y0\n")


def test_red_dot_found_in_the_reference_photo():
    """The phone photo of the reference glass: the dot is at its bottom-right corner."""
    from pathlib import Path

    import cv2

    img = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "data" / "reference_substrate.png")), cv2.COLOR_BGR2RGB)
    corners = [[323, 463], [663, 463], [663, 806], [323, 806]]       # the glass square, by eye
    assert rf.red_corner(img, corners) == 2


def test_array_on_glass_turn_offset_and_swap():
    import math

    import numpy as np

    from papa.ops import reference as refmod

    glass = np.array([[0, 0], [25, 0], [25, 25], [0, 25]], float)
    arr = np.array([[3, 8], [13, 8], [13, 18], [3, 18]], float)          # left of the glass centre
    # new glass: turned 30 deg and moved; array moved 1 mm right and turned 2 deg on the glass
    th = math.radians(30)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = np.array([100.0, 50.0])
    a2 = math.radians(2)
    R2 = np.array([[math.cos(a2), -math.sin(a2)], [math.sin(a2), math.cos(a2)]])
    arr_on_glass = (arr - arr.mean(0)) @ R2.T + arr.mean(0) + [1.0, 0]
    m = refmod.match(glass, 0, glass @ R.T + t, (glass @ R.T + t)[0], 25)
    g = refmod.array_on_glass(glass, arr, arr_on_glass @ R.T + t, m)
    assert g["rotation_deg"] == pytest.approx(2, abs=0.05)
    assert g["offset_mm"] == pytest.approx([1, 0], abs=0.01)
    assert g["swapped"] == []
    # the same array on the right half of the glass: left/right swapped
    mirrored = arr.copy(); mirrored[:, 0] = 25 - mirrored[:, 0]
    g = refmod.array_on_glass(glass, arr, mirrored @ R.T + t, m)
    assert g["swapped"] == ["left/right"]
