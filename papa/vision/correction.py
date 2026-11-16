"""Lens distortion and tilt correction for the plate camera, plus the
camera-to-printer mapping. Ported from cameraUpdates/camera_correction on main.

Three frames:
  photo   pixels of a camera frame, any supported size (see SENSOR_MODES)
  plate   the reference straight-down view: lens distortion and camera tilt
          removed, so equal distances on the plate are equal distances here
          (135 px per checkerboard square)
  printer machine X, Y in mm, the coordinates the GUI jogs in

photo <-> plate is the lens/tilt calibration below (made from 1920 x 1080
frames, valid while focus, zoom, camera and plate height are unchanged).
plate <-> printer is PrinterAxes, fitted from taught points: the tool tip is
jogged over a mark and the mark is clicked in a camera frame. Because the
plate view is already true to scale, that fit is a rotation, uniform scale and
shift (plus a mirror when printer Y runs opposite to image y): two points
define it, more points check it.

Points on raised parts (a substrate on the plate) look slightly larger and
lean outward. Teach the camera points at substrate height when you can.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# ---- Camera --------------------------------------------------------------------
SENSOR_SIZE = (4056, 3040)  # IMX477 active pixels
PIXEL_SIZE_MM = 0.00155     # 1.55 um
LENS_FOCAL_MM = 6.0         # nominal focal length of the lens

# Sensor modes: photo size -> (sensor crop left, crop top, binning)
SENSOR_MODES = {
    (4056, 3040): (0, 0, 1),
    (2028, 1520): (0, 0, 2),
    (2028, 1080): (0, 440, 2),
    (1332, 990): (696, 528, 2),
    (1920, 1080): (108, 440, 2),  # 2028 x 1080 mode cropped to 16:9
}

# ---- Calibration (made from 1920 x 1080 frames) -------------------------------
CALIBRATED_SIZE = (1920, 1080)
IMAGE_SIZE = CALIBRATED_SIZE  # kept for older scripts

# Lens model: radial distortion around the optical center, in calibration-frame
# pixels (2 x 2 binned, so 3.1 um each). FOCAL is the lens's nominal focal
# length in those pixels; K1 and K2 are relative to it.
FOCAL = LENS_FOCAL_MM / (PIXEL_SIZE_MM * 2)  # 1935.48 px
CENTER = np.array([969.9285797659654, 501.59325852544356])
K1 = -0.44591581843016725
K2 = 0.17186869491941814

# Reference corrected-view pixel -> straightened 1080p pixel (removes camera tilt).
TOPDOWN_MATRIX = np.array([
    [1.0040645145924654, -0.0244083509270049, -196.84154759218328],
    [0.022885544720503925, 0.9995164011316575, -148.81978016596764],
    [-4.829675708607278e-06, 1.884413385863845e-06, 1.0],
])
TOPDOWN_SIZE = (2323, 1362)  # reference corrected view for 1080p frames
PIXELS_PER_SQUARE = 135.0    # scale of the reference corrected view


# -------------------------------------------------------------------------------


def _photo_mode(size):
    """How a photo size relates to the calibration frame: frame px = (photo px - offset) / scale."""
    left, top, binning = SENSOR_MODES[size]
    cal_left, cal_top, cal_binning = SENSOR_MODES[CALIBRATED_SIZE]
    # A binned pixel's centre on the sensor is at  crop + binning * pixel + (binning - 1) / 2.
    scale = cal_binning / binning
    offset = [(cal_start + (cal_binning - 1) / 2 - start - (binning - 1) / 2) / binning
              for cal_start, start in ((cal_left, left), (cal_top, top))]
    return tuple(offset), scale


PHOTO_MODES = {size: _photo_mode(size) for size in SENSOR_MODES}

_INVERSE_TOPDOWN = np.linalg.inv(TOPDOWN_MATRIX)


def _apply(matrix, points):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _distort(points):
    """Straightened 1080p pixel -> 1080p pixel as the lens records it."""
    normalized = (points - CENTER) / FOCAL
    r2 = (normalized ** 2).sum(axis=1, keepdims=True)
    return CENTER + FOCAL * normalized * (1 + K1 * r2 + K2 * r2 ** 2)


def _undistort(points):
    """1080p pixel as the lens records it -> straightened 1080p pixel."""
    normalized = (points - CENTER) / FOCAL
    straight = normalized.copy()
    for _ in range(30):
        r2 = (straight ** 2).sum(axis=1, keepdims=True)
        straight = normalized / (1 + K1 * r2 + K2 * r2 ** 2)
    return CENTER + FOCAL * straight


class Correction:
    """Correction for one photo size, producing a corrected view at a fixed scale."""

    def __init__(self, photo_size=CALIBRATED_SIZE, pixels_per_square=None):
        photo_size = tuple(int(v) for v in photo_size)
        if photo_size not in PHOTO_MODES:
            sizes = ", ".join(f"{w} x {h}" for w, h in PHOTO_MODES)
            raise ValueError(f"No calibration for {photo_size[0]} x {photo_size[1]} photos "
                             f"(supported: {sizes}).")
        (ox, oy), self._scale = PHOTO_MODES[photo_size]
        self._offset = np.array([ox, oy])
        self.photo_size = photo_size
        # By default, keep about the same detail as the original photo.
        self.pixels_per_square = float(pixels_per_square or PIXELS_PER_SQUARE * self._scale)
        self._step = PIXELS_PER_SQUARE / self.pixels_per_square  # output px -> reference px

        if photo_size == CALIBRATED_SIZE and self._step == 1.0:
            self._origin = np.zeros(2)
            self.output_size = TOPDOWN_SIZE
        else:  # cover the whole photo
            w, h = photo_size
            xs, ys = np.linspace(0, w - 1, 400), np.linspace(0, h - 1, 400)
            border = np.vstack([np.c_[xs, 0 * xs], np.c_[xs, 0 * xs + h - 1],
                                np.c_[0 * ys, ys], np.c_[0 * ys + w - 1, ys]])
            reference = self._photo_to_reference(border)
            low, high = reference.min(axis=0), reference.max(axis=0)
            self._origin = np.floor(low)
            size = np.ceil((high - self._origin) / self._step).astype(int) + 1
            self.output_size = (int(size[0]), int(size[1]))
        self._maps = None

    def _photo_to_reference(self, points):
        frame = (points - self._offset) / self._scale
        return _apply(_INVERSE_TOPDOWN, _undistort(frame))

    def _reference_to_photo(self, points):
        frame = _distort(_apply(TOPDOWN_MATRIX, points))
        return self._offset + self._scale * frame

    def to_corrected(self, points):
        """Photo pixel coordinates (N x 2) -> corrected-view pixel coordinates."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return (self._photo_to_reference(points) - self._origin) / self._step

    def to_original(self, points):
        """Corrected-view pixel coordinates (N x 2) -> photo pixel coordinates."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return self._reference_to_photo(self._origin + points * self._step)

    def to_plate(self, points):
        """Photo pixels (N x 2) -> plate frame (the reference corrected view)."""
        return self._photo_to_reference(np.asarray(points, dtype=np.float64).reshape(-1, 2))

    def from_plate(self, points):
        """Plate frame (N x 2) -> photo pixels."""
        return self._reference_to_photo(np.asarray(points, dtype=np.float64).reshape(-1, 2))

    def corrected_to_plate(self, points):
        """Corrected-view pixels of this correction (N x 2) -> plate frame."""
        return self._origin + np.asarray(points, dtype=np.float64).reshape(-1, 2) * self._step

    def _build_maps(self, grid=8):
        # The mapping is smooth, so compute it on a coarse grid and interpolate.
        width, height = self.output_size
        cols, rows = width // grid + 3, height // grid + 3
        gx, gy = np.meshgrid((np.arange(cols) + 0.5) * grid - 0.5 - grid,
                             (np.arange(rows) + 0.5) * grid - 0.5 - grid)
        source = self.to_original(np.c_[gx.ravel(), gy.ravel()])
        maps = []
        for channel in (0, 1):
            coarse = source[:, channel].reshape(rows, cols).astype(np.float32)
            fine = cv2.resize(coarse, (cols * grid, rows * grid), interpolation=cv2.INTER_LINEAR)
            maps.append(np.ascontiguousarray(fine[grid:grid + height, grid:grid + width]))
        return maps

    def correct_image(self, image):
        """Return the corrected, straight-down view. Areas the camera did not see are black."""
        if image.shape[1::-1] != self.photo_size:
            raise ValueError(f"Expected a {self.photo_size[0]} x {self.photo_size[1]} photo, got "
                             f"{image.shape[1]} x {image.shape[0]}.")
        if self._maps is None:
            self._maps = self._build_maps()
        return cv2.remap(image, self._maps[0], self._maps[1], cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT)


_corrections: dict = {}


def get_correction(photo_size=CALIBRATED_SIZE) -> Correction:
    """The (cached) Correction for a photo size."""
    key = tuple(int(v) for v in photo_size)
    if key not in _corrections:
        _corrections[key] = Correction(key)
    return _corrections[key]


def check_checkerboard(rgb, inner_corners=(4, 4)) -> dict | None:
    """Measure the 5 x 5 checkerboard's squares in the corrected view (as
    capture_and_correct.py --check does). None if the board is not found."""
    c = get_correction(rgb.shape[1::-1])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, inner_corners, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    window = 11 if rgb.shape[1] <= 2000 else 21
    corners = cv2.cornerSubPix(gray, corners, (window, window), (-1, -1), criteria)
    cols, rows = inner_corners
    grid = c.to_corrected(corners.reshape(-1, 2)).reshape(rows, cols, 2)
    sides = np.concatenate([np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                            np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()])
    low, high, mean, target = float(sides.min()), float(sides.max()), float(sides.mean()), c.pixels_per_square
    ok = abs(mean / target - 1) < 0.02 and high / low < 1.04
    return {"min_px": round(low, 2), "max_px": round(high, 2), "mean_px": round(mean, 2),
            "target_px": round(target, 1), "matches": bool(ok)}


def supported(photo_size) -> bool:
    return tuple(int(v) for v in photo_size) in PHOTO_MODES


# ---- Printer axes -----------------------------------------------------------------

GUI_PHOTO_SIZE = (1920, 1080)
MAX_POINT_ERROR_MM = 0.5     # Stage 6 pass limit in docs/VALIDATION.md


class AxesError(ValueError):
    pass


class PrinterAxes:
    """Plate frame <-> printer X, Y in mm, as a similarity (optionally mirrored)."""

    def __init__(self, plate_px, printer_mm) -> None:
        ref = np.asarray(plate_px, dtype=np.float64).reshape(-1, 2)
        mm = np.asarray(printer_mm, dtype=np.float64).reshape(-1, 2)
        if len(ref) != len(mm):
            raise AxesError(f"{len(ref)} pixel points but {len(mm)} printer points")
        if len(ref) < 2:
            raise AxesError(f"need at least 2 taught camera points, have {len(ref)}")
        self.count = len(ref)
        self._ref, self._mm = ref, mm
        # As complex numbers a similarity is mm = a*z + b, or a*conj(z) + b mirrored.
        z, w = ref @ [1, 1j], mm @ [1, 1j]
        best = None
        for mirrored in (False, True):
            zz = np.conj(z) if mirrored else z
            zc, wc = zz - zz.mean(), w - w.mean()
            denom = np.sum(np.abs(zc) ** 2)
            if denom < 1e-9:
                raise AxesError("the taught camera points are all at the same pixel")
            a = np.sum(np.conj(zc) * wc) / denom
            b = w.mean() - a * zz.mean()
            err = np.abs(a * zz + b - w)
            if best is None or err.sum() < best[3].sum():
                best = (mirrored, a, b, err)
        self.mirrored, self._a, self._b, self.errors_mm = best
        if abs(self._a) < 1e-12:
            raise AxesError("the taught printer positions are all the same")
        self.mm_per_plate_px = float(abs(self._a))

    @classmethod
    def from_detection(cls, detection: dict | None) -> "PrinterAxes":
        """From calibration.yaml detection.camera_to_machine (photo pixels + mm)."""
        m = (detection or {}).get("camera_to_machine") or {}
        px, mm = m.get("points_px") or [], m.get("points_mm") or []
        size = tuple(m.get("photo_size") or GUI_PHOTO_SIZE)
        if not supported(size):
            raise AxesError(f"camera points were taken on a {size[0]} x {size[1]} frame, "
                            "which has no lens calibration")
        if len(px) != len(mm):
            raise AxesError(f"{len(px)} pixel points but {len(mm)} printer points")
        plate = get_correction(size).to_plate(px) if len(px) else np.zeros((0, 2))
        return cls(plate, mm)

    def plate_to_mm(self, points):
        z = np.asarray(points, dtype=np.float64).reshape(-1, 2) @ [1, 1j]
        w = self._a * (np.conj(z) if self.mirrored else z) + self._b
        return np.c_[w.real, w.imag]

    def mm_to_plate(self, points):
        w = np.asarray(points, dtype=np.float64).reshape(-1, 2) @ [1, 1j]
        z = (w - self._b) / self._a
        z = np.conj(z) if self.mirrored else z
        return np.c_[z.real, z.imag]

    def report(self) -> dict[str, Any]:
        """The fit and how far to trust it, for the GUI."""
        d = self.mm_to_plate([[1, 0], [0, 0]])
        x_angle = float(np.degrees(np.arctan2(*(d[0] - d[1])[::-1])))
        out: dict[str, Any] = {
            "points": self.count,
            "mirrored": bool(self.mirrored),
            "x_axis_deg_in_image": round(x_angle, 3),
            "mm_per_square": round(float(self.mm_per_plate_px * PIXELS_PER_SQUARE), 4),
            "errors_mm": [round(float(e), 4) for e in self.errors_mm] if self.count >= 3 else [],
            "warnings": [],
        }
        if self.count < 3:
            out["warnings"].append("Only 2 camera points: nothing checks them. Teach a 3rd and 4th.")
        elif self.errors_mm.max() > MAX_POINT_ERROR_MM:
            out["warnings"].append(f"A camera point is {self.errors_mm.max():.2f} mm off the fit "
                                   f"(limit {MAX_POINT_ERROR_MM} mm). Re-teach it.")
        if self.count >= 3:
            A, *_ = np.linalg.lstsq(np.c_[self._ref, np.ones(self.count)], self._mm, rcond=None)
            try:
                jx, jy = np.linalg.inv(A[:2].T)
                ratio = np.linalg.norm(jx) / np.linalg.norm(jy)
                cos = abs(jx @ jy) / np.linalg.norm(jx) / np.linalg.norm(jy)
                skew = 90 - float(np.degrees(np.arccos(min(1.0, cos))))
                out["xy_scale_diff_pct"] = round(float(abs(ratio - 1) * 100), 3)
                out["axes_off_square_deg"] = round(skew, 3)
                if abs(ratio - 1) > 0.01 or skew > 0.5:
                    out["warnings"].append("X and Y disagree in scale or squareness: check steps/mm, "
                                           "gantry squareness, or the camera points.")
            except np.linalg.LinAlgError:
                out["warnings"].append("The camera points are in a line. Spread them over the plate.")
        return out


def photo_to_mm(points, photo_size, axes: PrinterAxes):
    return axes.plate_to_mm(get_correction(photo_size).to_plate(points))


def mm_to_photo(points_mm, photo_size, axes: PrinterAxes):
    return get_correction(photo_size).from_plate(axes.mm_to_plate(points_mm))
