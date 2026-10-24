"""Split strokes into short, resumable blocks.

The Pi streams blocks rather than uploading one G-code file, for two reasons:
wear is checked between blocks, and the index of the last finished block is
the resume point after a swab change or a crash.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dxf_loader import Stroke

Point = tuple[float, float]


@dataclass(frozen=True)
class Block:
    """A run of pen-down points inside one stroke, in slide-local mm."""
    index: int
    stroke_index: int
    points: list[Point]
    starts_stroke: bool
    ends_stroke: bool

    @property
    def length(self) -> float:
        return sum(
            ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            for a, b in zip(self.points, self.points[1:])
        )


def _split_segment(a: Point, b: Point, target: float) -> list[Point]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= target or dist == 0:
        return [b]
    n = int(dist // target) + (1 if dist % target else 0)
    return [(a[0] + dx * i / n, a[1] + dy * i / n) for i in range(1, n + 1)]


def make_blocks(strokes: list[Stroke], target_mm: float = 3.0) -> list[Block]:
    """Dense points first, then group them into blocks of about target_mm."""
    blocks: list[Block] = []
    idx = 0
    for si, stroke in enumerate(strokes):
        dense: list[Point] = [stroke.points[0]]
        for a, b in zip(stroke.points, stroke.points[1:]):
            dense.extend(_split_segment(a, b, target_mm))

        cur: list[Point] = [dense[0]]
        acc = 0.0
        first = True
        for a, b in zip(dense, dense[1:]):
            step = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            # close the block BEFORE going over target, so no block is longer
            # than target_mm and the wear check keeps its granularity
            if acc + step > target_mm + 1e-9 and len(cur) > 1:
                blocks.append(Block(idx, si, cur, first, False))
                idx += 1
                cur, acc, first = [a], 0.0, False
            acc += step
            cur.append(b)
        if len(cur) > 1:
            blocks.append(Block(idx, si, cur, first, True))
            idx += 1
        elif blocks:
            last = blocks[-1]
            blocks[-1] = Block(
                last.index, last.stroke_index, last.points, last.starts_stroke, True
            )
    return blocks


def resume_points(block: Block, overlap_mm: float) -> list[Point]:
    """Block points with a short lead-in, so a resume leaves no gap.

    Backing up along the path already scribed costs nothing; starting exactly
    where the last block ended risks a visible break if the swab shifted.
    """
    if overlap_mm <= 0 or len(block.points) < 2:
        return list(block.points)
    a, b = block.points[0], block.points[1]
    dx, dy = a[0] - b[0], a[1] - b[1]
    d = (dx * dx + dy * dy) ** 0.5
    if d == 0:
        return list(block.points)
    k = min(overlap_mm, d) / d
    lead = (a[0] + dx * k, a[1] + dy * k)
    return [lead] + list(block.points)
