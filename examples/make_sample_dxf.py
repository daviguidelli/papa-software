"""Write a sample DXF: a square, a cross and an open polyline.

Straight segments only, so it exercises the real pipeline (chaining, fitting,
blocking) without needing a drawing from CAD.
"""
from pathlib import Path

import ezdxf


def build(path: Path) -> Path:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # square as four separate LINEs: chaining should join them into one stroke
    pts = [(0, 0), (40, 0), (40, 40), (0, 40)]
    for a, b in zip(pts, pts[1:] + [pts[0]]):
        msp.add_line(a, b)
    msp.add_line((8, 20), (32, 20))
    msp.add_line((20, 8), (20, 32))
    msp.add_lwpolyline([(4, 4), (12, 10), (20, 4), (28, 10)])
    doc.saveas(str(path))
    return path


if __name__ == "__main__":
    out = build(Path(__file__).parent / "sample.dxf")
    print(f"wrote {out}")
