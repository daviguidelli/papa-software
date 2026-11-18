"""Electrode array detection: a block of parallel lines on a white holder.

Runs on the corrected (straight-down) view, where an array is a true
rectangle. The lines' colour depends on the light and exposure (copper in one
photo, near black in another), so colour is only one of two cues. Steps:

1. mark line pixels: copper-coloured pixels (HSV window) OR pixels clearly
   darker than their surroundings (a black-hat filter, which picks out thin
   dark lines whatever their colour),
2. close the gaps between electrode lines so each array is one solid blob,
3. keep blobs of the right size and shape that are mostly line pixels, sit on
   a bright holder, and show a row of evenly spaced parallel stripes (this is
   what rejects the plate's holes and slots, label text and shadows), and fit
   a rotated rectangle to each,
4. measure the exact outline from the electrode lines themselves (see
   measure): the blob only says roughly where to look, so a pocket edge or a
   shadow next to the array can no longer make the outline too big.

Sizes are fractions of the image, so the same settings work at any frame size.
Every setting can be overridden from calibration.yaml under
detection.electrodes (same names as ElectrodeParams).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ElectrodeParams:
    h_min: int = 5  # OpenCV hue, 0-179
    h_max: int = 28
    s_min: int = 45
    v_min: int = 70
    min_area_frac: float = 0.0002  # blob area / image area
    max_area_frac: float = 0.05
    close_frac: float = 0.012  # gap-closing size / image width; wider than the gap
    # between lines, narrower than the gap between arrays
    min_fill: float = 0.55  # blob area / rotated-rectangle area
    max_aspect: float = 2.5  # long side / short side
    min_copper: float = 0.25  # line pixels (copper or dark) / rectangle area
    min_surround_v: int = 100  # brightness around the array (white holder; real lighting reads ~140)
    line_contrast: int = 25  # grey levels a line must be darker than its surroundings
    min_stripes: int = 4  # evenly spaced parallel lines needed to call it an array

    @classmethod
    def from_cfg(cls, d: dict | None) -> "ElectrodeParams":
        names = {f.name: f.type for f in fields(cls)}
        out = cls()
        for k, v in (d or {}).items():
            if k in names and v is not None:
                out = replace(out, **{k: type(getattr(out, k))(v)})
        return out


@dataclass
class Array:
    corners: np.ndarray  # 4 x 2, pixels of the image searched, minAreaRect order
    center: tuple[float, float]
    copper: float  # fraction of the rectangle that is copper


def copper_mask(rgb: np.ndarray, p: ElectrodeParams) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (p.h_min, p.s_min, p.v_min), (p.h_max, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def copper_score(rgb: np.ndarray, p: ElectrodeParams) -> np.ndarray:
    """0-1 copper score per pixel. Soft (saturation of copper-hued pixels):
    blended edge pixels keep copper's hue at lower saturation and so count
    partly, which is where the sub-pixel detail comes from."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_ok = ((hsv[:, :, 0] >= p.h_min) & (hsv[:, :, 0] <= p.h_max) & (hsv[:, :, 2] >= p.v_min)).astype(
        np.float32
    )
    return hue_ok * np.clip(hsv[:, :, 1] / 255.0, 0, 1)


def _peaks(P, floor):
    out = []
    for i in range(1, len(P) - 1):
        if P[i] >= P[i - 1] and P[i] > P[i + 1] and P[i] > floor:
            d = P[i - 1] - 2 * P[i] + P[i + 1]
            off = 0.5 * (P[i - 1] - P[i + 1]) / d if d != 0 else 0
            out.append((i + off, float(P[i])))
    return out


def _even_run(pk):
    best = []
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            d = pk[j][0] - pk[i][0]
            if d < 2:
                continue
            run = [pk[i], pk[j]]
            for q in pk[j + 1 :]:
                g = q[0] - run[-1][0]
                if abs(g - d) < 0.25 * d:
                    run.append(q)
                elif g > 1.25 * d:
                    break
            if len(run) > len(best):
                best = run
    return best


def _cross(P, i, step, level):
    i = int(round(i))
    while 0 <= i + step < len(P) and P[i + step] >= level:
        i += step
    j = i + step
    if not 0 <= j < len(P) or P[i] == P[j]:
        return float(i)
    return i + step * (P[i] - level) / (P[i] - P[j])


def _window(score, center, theta, S):
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), math.degrees(theta), 1.0)
    M[:, 2] += [S / 2 - center[0], S / 2 - center[1]]
    return cv2.warpAffine(score, M, (S, S), flags=cv2.INTER_LINEAR), M


def _line_set(win, S, size):
    mid = slice(int(S / 2 - 0.25 * size), int(S / 2 + 0.25 * size))
    P = win[:, mid].mean(1)
    pk = _peaks(P, 0.15 * P.max())
    run = _even_run([p for p in pk if p[1] > 0.3 * P.max()])
    if len(run) < 4:
        return None, P
    pitch = float(np.median(np.diff([y for y, _ in run])))
    amp = float(np.median([a for _, a in run]))

    def extent(y):
        row = win[max(0, int(round(y)) - 1) : int(round(y)) + 2].mean(0)
        lvl = 0.5 * float(np.median(row[mid]))
        return _cross(row, S / 2, -1, lvl), _cross(row, S / 2, 1, lvl)

    ext = [extent(y) for y, _ in run]
    L = float(np.median([b - a for a, b in ext]))

    def ok(p):
        a, b = extent(p[0])
        return p[1] > 0.35 * amp and abs((b - a) - L) < 0.2 * L

    lines = list(run)
    while True:  # weaker outer lines (glare, bus bars) one pitch out, same length
        n = [p for p in pk if 0.7 * pitch < lines[0][0] - p[0] < 1.45 * pitch and ok(p)]
        if not n:
            break
        lines.insert(0, n[0])
    while True:
        n = [p for p in pk if 0.7 * pitch < p[0] - lines[-1][0] < 1.45 * pitch and ok(p)]
        if not n:
            break
        lines.append(n[0])
    return lines, P


def measure(lines_img: np.ndarray, center, size: float, copper: np.ndarray | None = None):
    """The exact outline of one array, measured from its electrode lines.

    lines_img is the dark-line score (dark_lines), centre and size a rough
    position. Steps: the line direction from the image structure; a window
    turned so the lines run level; the evenly spaced, equally long lines
    across it (anything else, such as a pocket edge or a hole, breaks the
    spacing or the length and is left out); the angle refined until the
    lines are level to 0.01 deg; the outline is the outer edges of the first
    and last line and the median line ends. If the array looks copper (a
    copper film between the lines), the film's extent is used instead, as
    pale outer lines can vanish in glare. Returns (corners 4 x 2, number of
    lines) or None if there is no regular comb of at least 4 lines.
    """
    theta = _orientation(lines_img, center, int(0.6 * size))
    score = lines_img
    S = int(1.7 * size)
    for _ in range(4):
        win, M = _window(score, center, theta, S)
        lines, P = _line_set(win, S, size)
        if not lines:
            return None
        # slope of the lines: centroid row in a left and a right band
        sl = []
        for y, _ in lines:
            r0 = max(0, int(round(y)) - 2)
            band = win[r0 : r0 + 5]

            def cy(x0, x1):
                w = band[:, x0:x1].sum(1)
                return r0 + (np.arange(len(w)) * w).sum() / max(w.sum(), 1e-6)

            xa, xb = int(S / 2 - 0.35 * size), int(S / 2 - 0.1 * size)
            xc, xd = int(S / 2 + 0.1 * size), int(S / 2 + 0.35 * size)
            sl.append((cy(xc, xd) - cy(xa, xb)) / ((xc + xd) / 2 - (xa + xb) / 2))
        slope = float(np.median(sl))
        if abs(slope) < 2e-4:
            break
        theta += math.atan(slope)  # positive slope (falls to the right) -> rotate the window more
    top = _cross(P, lines[0][0], -1, lines[0][1] / 2)
    bot = _cross(P, lines[-1][0], 1, lines[-1][1] / 2)
    mid = slice(int(S / 2 - 0.25 * size), int(S / 2 + 0.25 * size))
    ends = []
    for y, _ in lines:
        row = win[max(0, int(round(y)) - 1) : int(round(y)) + 2].mean(0)
        lvl = 0.5 * float(np.median(row[mid]))
        ends.append((_cross(row, S / 2, -1, lvl), _cross(row, S / 2, 1, lvl)))
    left = float(np.median([a for a, _ in ends]))
    right = float(np.median([b for _, b in ends]))
    if copper is not None:  # copper-looking arrays: the copper film marks the whole array
        cw = cv2.warpAffine(copper, M, (S, S), flags=cv2.INTER_LINEAR)
        xs = slice(int(left + 0.2 * (right - left)), int(right - 0.2 * (right - left)) + 1)
        gaps = [int(round((a[0] + b[0]) / 2)) for a, b in zip(lines[:-1], lines[1:])]
        rows = [int(round(y)) for y, _ in lines]
        film = float(np.median(cw[gaps, xs])) if gaps else 0.0
        on_lines = float(np.median(cw[rows, xs]))
        if film > 0.12 and on_lines > 0.12:  # lines and gaps both copper: a film, not bare holder
            yc, xc = int((top + bot) / 2), int((left + right) / 2)
            inner = cw[int(top) : int(bot) + 1, int(left) : int(right) + 1]
            lvl = 0.5 * float(np.median(inner))
            Pc = np.convolve(cw[:, xs].mean(1), np.ones(3) / 3, mode="same")
            Cc = np.convolve(
                cw[int(top + 0.2 * (bot - top)) : int(bot - 0.2 * (bot - top)) + 1].mean(0),
                np.ones(3) / 3,
                mode="same",
            )
            t2, b2 = _cross(Pc, yc, -1, lvl), _cross(Pc, yc, 1, lvl)
            l2, r2 = _cross(Cc, xc, -1, lvl), _cross(Cc, xc, 1, lvl)
            pitch = (lines[-1][0] - lines[0][0]) / max(1, len(lines) - 1)
            # The lines give the sharper edge; the film only takes over on a side where
            # it reaches clearly further out, i.e. where outer lines were lost to glare.
            if 0.6 * pitch < top - t2 < 4 * pitch:
                top = t2
            if 0.6 * pitch < b2 - bot < 4 * pitch:
                bot = b2
            if 0.6 * pitch < left - l2 < 4 * pitch:
                left = l2
            if 0.6 * pitch < r2 - right < 4 * pitch:
                right = r2
    rect = np.array([[left, top], [right, top], [right, bot], [left, bot]])
    Minv = cv2.invertAffineTransform(M)
    return np.c_[rect, np.ones(4)] @ Minv.T, len(lines)


def _orientation(score, center, half):
    x0, y0 = int(center[0] - half), int(center[1] - half)
    patch = score[max(0, y0) : y0 + 2 * half, max(0, x0) : x0 + 2 * half]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    phi = 0.5 * math.atan2(2 * (gx * gy).sum(), (gx * gx).sum() - (gy * gy).sum())
    return phi + math.pi / 2


def dark_lines(gray: np.ndarray, k: int) -> np.ndarray:
    """How much darker each pixel is than its surroundings (black-hat), in grey levels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_BLACKHAT, kernel)


def stripes(lines: np.ndarray, corners: np.ndarray) -> int:
    """Evenly spaced parallel lines inside a rotated rectangle, counted across
    whichever direction has more (0 if they are not evenly spaced)."""
    c = np.asarray(corners, dtype=np.float32)
    rw, rh = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])
    W, H = max(8, int(round(rw))), max(8, int(round(rh)))
    M = cv2.getPerspectiveTransform(c, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    patch = cv2.warpPerspective(lines, M, (W, H))
    best = 0
    for prof in (patch.mean(axis=1), patch.mean(axis=0)):  # across rows, across columns
        n = len(prof)
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")[int(0.05 * n) : n - int(0.05 * n)]
        if len(prof) < 8 or prof.max() - prof.min() < 3:
            continue
        above = prof > (prof.max() + prof.min()) / 2
        starts = np.nonzero(above[1:] & ~above[:-1])[0]
        if above[0]:
            starts = np.r_[0, starts]
        if len(starts) >= 3:
            gaps = np.diff(starts)
            if gaps.std() / gaps.mean() > 0.35:  # uneven: not an electrode comb
                continue
        best = max(best, len(starts))
    return best


def detect(rgb: np.ndarray, p: ElectrodeParams | None = None, coarse_width: int = 2000) -> list[Array]:
    """Arrays in an RGB image (normally the corrected view), largest first.

    A big picture (a full-resolution scan) is searched on a smaller copy
    about coarse_width wide, which is many times faster, and each array
    found is then measured on the full picture in a window around it, so the
    outline keeps full-resolution accuracy."""
    p = p or ElectrodeParams()
    h, w = rgb.shape[:2]
    if w <= 1.5 * coarse_width:
        return _detect(rgb, p)
    f = coarse_width / w
    small = cv2.resize(rgb, (coarse_width, max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(p.close_frac * w)) | 1)
    out = []
    for a in _detect(small, p):
        rough = a.corners / f
        size = float(max(np.linalg.norm(rough[1] - rough[0]), np.linalg.norm(rough[2] - rough[1])))
        centre = rough.mean(0)
        m = 1.0 * size + k
        x0, y0 = int(max(0, centre[0] - m)), int(max(0, centre[1] - m))
        x1, y1 = int(min(w, centre[0] + m)), int(min(h, centre[1] + m))
        win = rgb[y0:y1, x0:x1]
        lines = dark_lines(cv2.cvtColor(win, cv2.COLOR_RGB2GRAY), k)
        measured = measure(lines, centre - [x0, y0], size, copper_score(win, p))
        local = np.asarray(measured[0], np.float64) if measured is not None else rough - [x0, y0]
        cop = copper_mask(win, p)
        inside = np.zeros(cop.shape, np.uint8)
        cv2.fillPoly(inside, [np.round(local).astype(np.int32)], 255)
        if cv2.countNonZero(cv2.bitwise_and(cop, inside)) >= 0.3 * max(1, cv2.countNonZero(inside)):
            local = trim_to_copper(local, cop)            # same clean-up as on the small copy
        corners = local + [x0, y0]
        out.append(
            Array(
                np.asarray(corners, dtype=np.float64),
                tuple(map(float, np.asarray(corners).mean(0))),
                a.copper,
            )
        )
    return out


def trim_to_copper(corners: np.ndarray, cop: np.ndarray, step: float = 0.04, max_cut: float = 0.3,
                   min_band: float = 0.15) -> np.ndarray:
    """Pull in any side of an outline whose edge band has (almost) no copper:
    a clamp, clip or holder edge beside the array that looked like more lines.
    corners TL, TR, BR, BL (any consistent order around the box)."""
    c = np.asarray(corners, np.float64).copy()
    for _ in range(int(max_cut / step) * 4):
        changed = False
        for i in range(4):                       # side i runs c[i] -> c[i+1]; inward is towards the opposite side
            a, b = c[i], c[(i + 1) % 4]
            d, e = c[(i + 3) % 4], c[(i + 2) % 4]  # the opposite side
            band = np.array([a, b, b + (e - b) * step, a + (d - a) * step], np.float32)
            m = np.zeros(cop.shape, np.uint8)
            cv2.fillPoly(m, [np.round(band).astype(np.int32)], 255)
            n = cv2.countNonZero(m)
            if n and cv2.countNonZero(cv2.bitwise_and(cop, m)) < min_band * n:
                full = np.linalg.norm(e - b)
                if full <= 0:
                    continue
                c[i], c[(i + 1) % 4] = a + (d - a) * step, b + (e - b) * step
                changed = True
        if not changed:
            break
    w0 = np.linalg.norm(np.asarray(corners)[1] - np.asarray(corners)[0])
    h0 = np.linalg.norm(np.asarray(corners)[3] - np.asarray(corners)[0])
    if np.linalg.norm(c[1] - c[0]) < (1 - max_cut) * w0 or np.linalg.norm(c[3] - c[0]) < (1 - max_cut) * h0:
        return np.asarray(corners, np.float64)   # cut too much: keep the measured outline
    return c


def _detect(rgb: np.ndarray, p: ElectrodeParams) -> list[Array]:
    """detect() on one picture, at its own resolution."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = max(3, int(round(p.close_frac * w)) | 1)
    lines = dark_lines(gray, k)
    cop_only = copper_mask(rgb, p)
    both = cv2.bitwise_or(cop_only, (lines > p.line_contrast).astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # candidates from copper colour + dark lines, then from copper colour alone:
    # the second finds an array whose blob merged with a holder or a clamp beside it
    cands = []
    for mask in (both, cop_only):
        blobs = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands += [(c, mask) for c in contours]
    found: list[Array] = []
    for c, mask in cands:
        if any(cv2.pointPolygonTest(a.corners.astype(np.float32), tuple(map(float, c.reshape(-1, 2).mean(0))), False) >= 0
               for a in found):
            continue                # already found from the other mask
        area = cv2.contourArea(c)
        if not p.min_area_frac * w * h <= area <= p.max_area_frac * w * h:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) == 0:
            continue
        if area / (rw * rh) < p.min_fill or max(rw, rh) / min(rw, rh) > p.max_aspect:
            continue
        corners = cv2.boxPoints(rect)
        box = np.zeros_like(mask)
        cv2.fillPoly(box, [corners.astype(np.int32)], 255)
        copper = cv2.countNonZero(cv2.bitwise_and(mask, box)) / max(1, cv2.countNonZero(box))
        if copper < p.min_copper:
            continue
        ring = cv2.subtract(cv2.dilate(box, kernel), box)
        if cv2.mean(gray, mask=ring)[0] < p.min_surround_v:
            continue
        size = float(max(rw, rh))
        measured = measure(lines, corners.mean(0), size, copper_score(rgb, p))
        if measured is None:
            continue
        # count the lines on the rough box, or, when that box also takes in a bit
        # of a holder or shadow beside the array, on the exact outline
        exact = np.asarray(measured[0], np.float32)
        ew, eh = float(np.linalg.norm(exact[1] - exact[0])), float(np.linalg.norm(exact[3] - exact[0]))
        if min(ew, eh) <= 0 or max(ew, eh) / min(ew, eh) > p.max_aspect or ew * eh < p.min_area_frac * w * h:
            continue                # the exact outline must still look like an array
        if stripes(lines, corners) < p.min_stripes:
            if max(measured[1], stripes(lines, exact)) < p.min_stripes:
                continue
            inside = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(inside, [np.round(exact).astype(np.int32)], 255)
            if cv2.countNonZero(cv2.bitwise_and(cop_only, inside)) < 0.3 * max(1, cv2.countNonZero(inside)):
                continue            # evenly striped but not copper: a vent grille, a printed label
        if min(corners[:, 0].min(), corners[:, 1].min()) < 2 or corners[:, 0].max() > w - 3 or corners[:, 1].max() > h - 3:
            continue                # cut off by the picture's edge
        corners = np.asarray(measured[0], dtype=np.float64)
        inside = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(inside, [np.round(corners).astype(np.int32)], 255)
        if cv2.countNonZero(cv2.bitwise_and(cop_only, inside)) >= 0.3 * max(1, cv2.countNonZero(inside)):
            corners = trim_to_copper(corners, cop_only)       # a copper array: drop non-copper edge strips
        found.append(Array(corners, tuple(map(float, corners.mean(0))), float(copper)))
    found.sort(key=lambda a: -cv2.contourArea(a.corners.astype(np.float32)))
    return found


def detect_loose(rgb: np.ndarray, p: ElectrodeParams | None = None) -> list[Array]:
    """A gentler search for close-up photos (a phone photo of one substrate),
    where the lines can look brighter than the gaps and the strict line test
    of detect() fails: the biggest roughly rectangular copper-coloured patch.
    Tries detect() first."""
    strict = detect(rgb, p)
    p = p or ElectrodeParams()
    loose = ElectrodeParams(**{**p.__dict__, "s_min": min(p.s_min, 30), "v_min": min(p.v_min, 60)})
    h, w = rgb.shape[:2]
    m = copper_mask(rgb, loose)
    k = max(3, int(round(p.close_frac * w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < 0.005 * w * h:
            break
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if min(rw, rh) == 0 or area / (rw * rh) < 0.5 or max(rw, rh) / min(rw, rh) > p.max_aspect:
            continue
        corners = cv2.boxPoints(rect).astype(np.float64)
        out.append(Array(corners, tuple(map(float, corners.mean(0))), float(area / (rw * rh))))
    if not strict:
        return out[:1]
    if out:
        big = out[0]
        a_big = cv2.contourArea(big.corners.astype(np.float32))
        a_str = cv2.contourArea(strict[0].corners.astype(np.float32))
        inside = cv2.pointPolygonTest(big.corners.astype(np.float32), strict[0].center, False) >= 0
        if inside and a_big > 2.0 * a_str:
            return [big]        # the strict find is only part of a bigger copper patch: one array, seen close up
    return strict
