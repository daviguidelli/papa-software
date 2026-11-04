"""Substrate detection from red corner tape.

A port of the team's `substrateRecognition` script, same algorithm and defaults,
split into functions the GUI and the orchestrator can call on any frame instead
of only on a fresh 4056x3040 still:

1. red markers: HSV threshold for red (hue wraps, so two ranges), open to drop
   specks, close to join a fragmented marker, contours, enclosing circle.
2. pairing: each substrate has two markers on a diagonal, one upper-right and
   one lower-left. Every valid (upper-right, lower-left) pair is a candidate,
   best score first, each marker used once. The score is the distance between
   the pair, or, when the real spacing is known, how far the pair is from it.
3. numbering: sort pair centres into rows, then left to right, so S1, S2, ...
   read like text.

All pixel settings in SubstrateParams are written for a 4056 px wide frame, as
in the script. detect() rescales them to the frame it is given.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

REFERENCE_WIDTH = 4056


@dataclass(frozen=True)
class SubstrateParams:
    min_red_area: float = 100.0
    red_close_kernel: int = 31
    min_diagonal_dx: float = 10.0
    min_diagonal_dy: float = 10.0
    max_pair_distance: float = 1200.0
    expected_dx: float | None = None      # marker spacing in pixels at 4056 px wide
    expected_dy: float | None = None
    pair_tolerance: float = 0.30
    row_tolerance: float = 150.0

    def scaled(self, frame_width: int) -> "SubstrateParams":
        k = frame_width / REFERENCE_WIDTH
        kern = max(3, int(round(self.red_close_kernel * k)) | 1)
        return replace(
            self,
            min_red_area=self.min_red_area * k * k,
            red_close_kernel=kern,
            min_diagonal_dx=self.min_diagonal_dx * k,
            min_diagonal_dy=self.min_diagonal_dy * k,
            max_pair_distance=self.max_pair_distance * k,
            expected_dx=None if self.expected_dx is None else self.expected_dx * k,
            expected_dy=None if self.expected_dy is None else self.expected_dy * k,
            row_tolerance=self.row_tolerance * k,
        )


def detect_red_markers(bgr: np.ndarray, p: SubstrateParams) -> list[dict]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.red_close_kernel, p.red_close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < p.min_red_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        markers.append({"x": round(float(x), 2), "y": round(float(y), 2),
                        "radius": round(float(r), 2), "area": round(float(area), 2)})
    markers.sort(key=lambda m: (m["y"], m["x"]))
    return markers


def pair_markers(markers: list[dict], p: SubstrateParams) -> tuple[list[dict], list[int]]:
    cands = []
    for i, a in enumerate(markers):
        for j, b in enumerate(markers):
            if i == j:
                continue
            dx = a["x"] - b["x"]          # a is the upper-right marker
            dy = b["y"] - a["y"]          # b is the lower-left marker
            if dx < p.min_diagonal_dx or dy < p.min_diagonal_dy:
                continue
            dist = float(np.hypot(dx, dy))
            if dist > p.max_pair_distance:
                continue
            if p.expected_dx is not None and abs(dx - p.expected_dx) > p.expected_dx * p.pair_tolerance:
                continue
            if p.expected_dy is not None and abs(dy - p.expected_dy) > p.expected_dy * p.pair_tolerance:
                continue
            if p.expected_dx is not None and p.expected_dy is not None:
                score = abs(dx - p.expected_dx) / p.expected_dx + abs(dy - p.expected_dy) / p.expected_dy
            else:
                score = dist
            cands.append((score, i, j, dist))
    cands.sort(key=lambda c: c[0])
    used: set[int] = set()
    subs = []
    for _score, i, j, dist in cands:
        if i in used or j in used:
            continue
        a, b = markers[i], markers[j]
        subs.append({"marker_1": a, "marker_2": b,
                     "center_x": round((a["x"] + b["x"]) / 2, 2),
                     "center_y": round((a["y"] + b["y"]) / 2, 2),
                     "pair_distance_px": round(dist, 2)})
        used.update((i, j))
    return subs, [i for i in range(len(markers)) if i not in used]


def number_substrates(subs: list[dict], p: SubstrateParams) -> list[dict]:
    rows: list[dict] = []
    for s in sorted(subs, key=lambda s: (s["center_y"], s["center_x"])):
        row = next((r for r in rows if abs(s["center_y"] - r["y"]) <= p.row_tolerance), None)
        if row is None:
            rows.append({"y": s["center_y"], "items": [s]})
        else:
            row["items"].append(s)
            row["y"] = float(np.mean([i["center_y"] for i in row["items"]]))
    out: list[dict] = []
    for row in sorted(rows, key=lambda r: r["y"]):
        for s in sorted(row["items"], key=lambda s: s["center_x"]):
            s["substrate_id"] = len(out) + 1
            out.append(s)
    return out


def annotate(bgr: np.ndarray, markers: list[dict], subs: list[dict]) -> np.ndarray:
    out = bgr.copy()
    k = max(0.25, out.shape[1] / REFERENCE_WIDTH)
    for m in markers:
        cv2.circle(out, (int(m["x"]), int(m["y"])), max(6, int(m["radius"]) + int(5 * k)),
                   (0, 255, 0), max(2, int(5 * k)))
    scale, thick = 2.0 * k * 1.6, max(2, int(5 * k))
    for s in subs:
        label = f"S{s['substrate_id']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(out, label, (int(s["center_x"]) - tw // 2, int(s["center_y"]) + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def detect(rgb: np.ndarray, params: SubstrateParams | None = None) -> dict[str, Any]:
    """Run the whole pipeline on an RGB frame. Pixel results are in this frame's
    own pixels; the annotated image is BGR, ready for cv2.imwrite."""
    h, w = rgb.shape[:2]
    p = (params or SubstrateParams()).scaled(w)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    markers = detect_red_markers(bgr, p)
    subs, unmatched = pair_markers(markers, p)
    subs = number_substrates(subs, p)
    return {
        "image_width": w, "image_height": h, "coordinate_units": "pixels",
        "marker_count": len(markers), "substrate_count": len(subs),
        "pairing_verified": False,
        "substrates": subs,
        "unmatched_markers": [markers[i] for i in unmatched],
        "_markers": markers, "_bgr": bgr,
    }
