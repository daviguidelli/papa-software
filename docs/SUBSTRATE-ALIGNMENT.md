# Substrate alignment (Substrates tab)

One photo of the plate gives every substrate its own toolpath: the camera finds
each copper electrode array, and that substrate's bridge sketch is turned and
shifted to sit exactly on it before the scrub runs.

## Flow

1. **Photo.** `Scan substrates` takes one full-resolution photo (4056 × 3040).
2. **Find the arrays.** In the corrected (straight-down) view, an array is a
   block of evenly spaced, equally long parallel lines on the white holder,
   whether the lines look copper or dark. Its outline is measured from the
   lines themselves: the outer edges of the first and last line and the line
   ends, with the angle refined until the lines run level
   (`papa/vision/electrodes.py`). Arrays are numbered S1, S2, ... in reading
   order of that view, with or without camera points.
3. **Your geometry.** Each substrate has a pattern: array width × height and
   the bridges as keep-outs. Press **Sketch** on a substrate's row: the bridge
   sketcher opens on the scanned substrates in order, each cut from the scan
   photo, straightened and zoomed (no uploading). Before the camera points
   are taught the array's size is not measured, so the sketcher asks for it;
   sketches drawn then are checked once the points exist, and flagged if the
   camera turns out to be turned relative to the printer. Draw the bridges and press
   Save to send them back. Or tick substrates and load a sketch file, or copy
   one substrate's pattern to others. Substrates without their own pattern use
   the Scrub tab's sketch.
4. **Compare.** The array's corners go to printer X/Y mm through the camera
   calibration. The pattern is fitted onto them as a rotation plus a shift (it
   keeps its sketched size). The table shows the measured size against the
   pattern, the turn from the printer axes, where the pattern's top-left corner
   lands, and how well the corners fit.
5. **Place the toolpath.** The scrub plan is unchanged in the substrate frame.
   It is placed on the machine at that corner and turned by that angle
   (`rotation_deg` in `papa/ops/geometry.py:to_machine`). Contact Z, speeds and
   swab size come from the Scrub tab. `Scrub` uses the swab already held;
   `Routine 1` grips, scrubs and disposes.

A substrate whose size differs from its pattern by more than the tolerance, or
that is turned more than the limit (Checks), is flagged in red and runs only
after you confirm.

## Frames

- The pattern's origin is the array's **top-left** corner, with local +x and
  +y mapped to the machine by `axes` in `machine.yaml` (now `+X`, `-Y`).
  Arrays need to sit within 45° of that orientation.
- Turn angles are positive from printer +X toward +Y.

## Calibration it needs

- **Lens and tilt** (`papa/vision/correction.py`, from
  `cameraUpdates/camera_correction`). Valid while focus, zoom, the camera
  mount and the plate height stay as calibrated.
- **Camera points** (the Camera points fold on the tab, Stage 6 in
  `VALIDATION.md`). Jog the swab tip over a small mark, click the mark in the
  photo, press Pair. Two points work; four spread over the plate plus a fifth to
  check are better. The fit reports each point's error (limit 0.5 mm) and
  whether X and Y agree in scale and squareness. Points are stored in
  `calibration.yaml` under `detection.camera_to_machine`, with the frame size
  they were clicked on. Teach them at substrate height if you can: raised parts
  look slightly larger in the photo.

## Detection settings

Override any of the `ElectrodeParams` names in `calibration.yaml`:

```yaml
detection:
  electrodes:
    h_min: 5        # copper hue window (OpenCV 0-179)
    h_max: 28
    s_min: 45
    min_area_frac: 0.0002
```

## Simulation

`scripts/run-sim.sh` sets `PAPA_SIM_ARRAYS=1`, so the mock camera draws four
arrays through the taught camera points. The simulated calibration's camera
points are only for the old presence check and are not geometrically
consistent; clear them and teach a few to see a clean fit.

## Capture and correct (Camera tab)

The Camera tab's **Capture and correct** does what `capture_and_correct.py`
did: a full-resolution photo, saved with its corrected straight-down view in
`var/captures/`, with an optional 5 × 5 checkerboard check against the
calibration (squares should measure about 270 px at full resolution). Both
files can be downloaded from the page. The **Corrected** box shows the live
view corrected too.
