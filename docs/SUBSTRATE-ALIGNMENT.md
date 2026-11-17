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
   camera turns out to be turned relative to the printer.
   In the sketcher, thumbnails of every substrate (✓ when sketched) open it
   with a click; [ and ] (or Page Up / Page Down) go to the previous / next.
   **Set its 4 corners** shows the photo around the substrate with the found
   corners: drag them (or clear and click all four) and press Use these
   corners. **+ Add a substrate** does the same on the whole photo for one the
   scan missed, and **Not a substrate** removes a false find (later substrates
   move up one number, their sketches with them). Hand-set corners hold for
   that scan; a new scan finds the arrays again. Draw the bridges and press
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

## Reference substrate and G-code (Reference fold)

One substrate whose toolpath already exists as G-code is the reference. Every
scan compares each new substrate's **glass square** to it and moves the G-code
by the same turn and shift.

1. **Photo of the reference** (any camera, from straight above). The browser
   converts it to JPEG (Safari also reads iPhone HEIC). Its electrode array is
   found by itself (`electrodes.detect_loose`, which also handles close-ups where
   the lines look brighter than the gaps); if the green outline is off, choose
   "electrode array corners" and click its 4 corners.
2. **Click the 4 glass corners.** The red-dot corner is found by itself
   (`reference.red_corner`); override it with the Red dot corner list.
3. **Placement**: where that glass sat on the printer for the G-code (left
   edge X, front edge Y, size; square to the printer). Default 162.5, 162.1, 25.
4. **Toolpath**, either or both:
   - a **bridge sketch** (.json from the sketcher) drawn over the whole glass,
     (0, 0) at the photo's top-left glass corner, x right, y down. Each scanned
     substrate without its own sketch then *runs* this sketch placed on its glass
     (`reference.glass_placement`), with Scrub / Routine 1, and its G-code ↓
     writes that scrub plan out (`scrub.to_gcode`; asks for a contact Z if the
     Scrub tab has none).
     **Reference G-code ↓** (next to the sketch) writes the known, unmodified
     toolpath: the sketch at the reference glass's place on the printer
     (`/api/reference/gcode`). Each substrate's **Compare to reference** shows
     that toolpath and the one modified for its glass overlaid, plus where each
     bridge moved (`/api/align/substrate/<id>/compare`); the same per-bridge
     list is in the header of its G-code.
     **Per-substrate geometry:** for a substrate with a glass match, the
     sketcher shows its whole glass turned like the reference photo, starting
     from the reference bridges (`frame: glass`). Move, resize, add or delete
     bridges and Save: the substrate then runs its own bridges on its glass, and
     Compare / the G-code header split each bridge's change into the glass move
     and your edit (bridges paired by id, else nearest within 3 mm; added and
     removed ones listed). A sketch drawn on the array (older patterns) stays on
     the array and is not compared.
   - a **G-code file** (G21 mm; G90/G91, G0/G1/G2/G3; G20 and G92 X/Y are
     refused). When loaded, G-code ↓ gives this file moved onto the glass.

Stored under `reference:` in `calibration.yaml` and in `<state>/reference/`
(photo.jpg, sketch.json, toolpath.gcode).

On each scan (needs the camera points), every substrate's glass is estimated
from its array, using the same glass-to-array layout as on the reference, unless
set by hand ("Set glass corners" in the table or "Set its glass corners" in
the sketcher). The red dot is looked for next to its corners. `reference.match`
fits new = R·ref + t with the red dot fixing which way round it is (any 90°
turn); without the dot the smallest turn is taken and flagged. Each row then
shows "Glass vs reference" (turn, shift, glass size, red dot, warnings) and a
**G-code ↓** link: the reference G-code with every X/Y (and arc I/J) moved by
R and t, Z and feeds unchanged, and a header saying what was done. On the
overview picture: cyan glass outline, red ring on the dot, orange moved cut path.

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
  `VALIDATION.md`). Click a small mark in the photo, jog the swab tip exactly
  over it, press Pair (Pair needs the printer connected and X/Y homed; the fold
  says which is missing). Two points work; four spread over the plate plus a fifth to
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
