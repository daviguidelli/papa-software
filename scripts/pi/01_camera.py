"""Does the camera turn on? Standalone: needs only picamera2 (+ opencv/numpy to save and check).

  python3 scripts/pi/01_camera.py            # one still -> camera_test.jpg
  python3 scripts/pi/01_camera.py --preview  # live window (needs a monitor on the Pi)
"""
import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--out", default="camera_test.jpg")
    args = ap.parse_args()

    try:
        from picamera2 import Picamera2
    except ImportError:
        print("FAIL: picamera2 not importable. On Raspberry Pi OS: sudo apt install -y python3-picamera2")
        print("      In a venv, create it with --system-site-packages")
        return 1

    cams = Picamera2.global_camera_info()
    print(f"cameras found: {len(cams)}")
    for c in cams:
        print("  ", c)
    if not cams:
        print("FAIL: no camera detected. Power off, reseat the ribbon cable (contacts toward the board), check the cable type for a Pi 5.")
        return 1

    cam = Picamera2()
    if args.preview:
        import os
        from picamera2 import Preview
        cam.configure(cam.create_preview_configuration())
        cam.start_preview(Preview.QTGL if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else Preview.DRM)
        cam.start()
        print("live preview on the Pi's screen. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        cam.stop()
        return 0

    cam.configure(cam.create_still_configuration(main={"size": (1920, 1080), "format": "RGB888"}))
    cam.start()
    time.sleep(2)                       # let exposure settle
    frame = cam.capture_array("main")
    cam.stop()
    print(f"frame: shape={frame.shape} dtype={frame.dtype} mean={frame.mean():.1f}")

    try:
        import cv2
        cv2.imwrite(args.out, frame)    # RGB888 from picamera2 is BGR order in memory, which is what OpenCV expects
        print(f"saved {args.out}  (copy to your laptop: scp pi@<ip>:{args.out} .)")
    except ImportError:
        cam.close()
        print("opencv missing, skipping save: sudo apt install -y python3-opencv")
        return 0
    cam.close()
    ok = 20 < frame.mean() < 235
    print("PASS" if ok else "WARN: image is nearly black or white - lens cap, or exposure")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
