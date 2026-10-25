"""Camera. Needs: Pi + Camera Module 3 mounted over the bed.

  python -m scripts.bench.b4_camera                  # stills, sharpness, stability
  python -m scripts.bench.b4_camera --sample 960,540 # HSV band from a pixel
Look at the saved PNGs in var/bench/ (copy them to your laptop with scp).
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from papa.config import load_calibration
from papa.hw import make_camera
from papa.vision.presence import HsvBand, blue_mask, sample_hsv

from ._common import OUT, Report, main


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="x,y pixel on a blue film or stem")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    r = Report("b4_camera")
    cal = load_calibration()
    cam = make_camera()
    frames = [cam.frame() for _ in range(args.frames)]
    f = frames[-1]
    path = OUT / f"camera-{r.stamp}.png"
    cv2.imwrite(str(path), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    r.note(f"saved {path}")
    r.check("frames are 3-channel and non-empty", f.ndim == 3 and f.shape[2] == 3, str(f.shape))

    gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    r.check("exposure sane (mean 60-190 of 255)", 60 <= mean <= 190, f"{mean:.0f}")
    clipped = float((gray >= 250).mean())
    r.check("little blown-out glare (<2% of pixels)", clipped < 0.02, f"{clipped * 100:.2f}%")
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    r.note(f"sharpness (Laplacian variance) {sharp:.0f}: only meaningful as a "
           "comparison. Adjust focus, re-run, keep the highest.")

    diffs = [float(np.abs(a.astype(int) - b.astype(int)).mean())
             for a, b in zip(frames, frames[1:])]
    r.check("scene is stable frame to frame (mean diff < 2)", max(diffs) < 2.0,
            f"worst {max(diffs):.2f}")

    band = HsvBand.from_cfg(cal.detection["film_hsv"]) if "film_hsv" in cal.detection else None
    if band is not None:
        mask = blue_mask(f, band)
        r.note(f"blue pixels under the current film band: {100 * (mask > 0).mean():.1f}%")
        cv2.imwrite(str(OUT / f"mask-{r.stamp}.png"), mask)

    if args.sample:
        x, y = (int(v) for v in args.sample.split(","))
        b = sample_hsv(f, (x, y))
        r.note(f"HSV band at ({x},{y}): {b}")
        r.data["sampled_band"] = str(b)
        r.note("hue is OpenCV units 0-179 (degrees / 2)")

    r.ask("the saved image shows the whole bed, in focus, without glare on the glass")
    r.ask("the 2.54 mm swab stem spans at least ~10 pixels in the image")
    return r.finish()


if __name__ == "__main__":
    main(run)
