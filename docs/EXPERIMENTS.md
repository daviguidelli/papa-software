# Numbers that must be measured

Everything here is a property of the swab, the film, the force and the speed.
None of it can be derived, and a guessed value is worse than an empty one
because it silently turns "these lines will merge" into "these lines are fine".
Each has a placeholder of `null` in `calibration.yaml` and the software says
"not measured" until it is filled in.

## 1. Kerf: how wide a line actually comes out

A swab does not scribe a mathematical line. At a given contact force it clears
a band of film of some width, centred on the toolpath. That width is the kerf,
and it sets the minimum spacing between two lines that are meant to stay
separate. Two paths closer together than one kerf come out as a single wide
clearing, and no amount of positional accuracy fixes that.

**Procedure.** On a sacrificial slide, scribe a comb: five pairs of parallel
20 mm lines, with gaps of 2.0, 1.0, 0.5, 0.25 and 0.1 mm. Run the whole comb at
one force and one speed, and write both on the slide.

- Under a microscope or a loupe, find the largest gap where the two lines have
  merged into one. The kerf is larger than that gap.
- Measure the width of a single isolated line directly. That is the kerf.
- The two should agree. If they do not, the swab is probably deforming under
  load, and the line width depends on how long that swab has been running.

Repeat at the extremes of the force range you intend to use. Kerf grows with
force, because the swab flattens.

Record in `calibration.yaml`:

```yaml
scribe:
  kerf_mm: 0.42
  measured_at_force_g: 110
  measured_at_speed_mm_s: 20
```

Once filled in, the DXF preview reports the closest approach in the pattern and
warns when two paths are closer than the kerf.

**What the software already checks, without the kerf:** the closest approach
between any two paths that do not cross. Paths that do cross are counted
separately, since lines that meet in the drawing are meant to meet.

## 2. Contact force: the working value, and what the cell can survive

These are two different numbers and both are currently unresolved.

The configured placeholders are `target_force_g: 5.0` and `abort_force_g: 20.0`.
A working force of **1.1 N is 112 gf**, which is twenty-two times the
placeholder. Before anything is trusted:

- **Confirm which TAL221 you have.** The family ships in several ratings. A
  100 g cell saturates at 0.98 N, which is *below* 1.1 N — it would be pinned
  at full scale before reaching the working force, and over its rating the
  reading is neither linear nor safe for the cell.
- If 1.1 N is really the working force, the cell must be rated well above it,
  and `abort_force_g` has to sit between the working force and the rating.
- If the cell is a 100 g part, then the working force has to come down, or the
  cell has to change.

`abort_force_g` exists to protect the cell, so it is only meaningful once the
rating is known.

**Procedure.** With a swab gripped, use the Jog and test tab: tare, then
calibrate the scale with a known weight in the working range. Lower onto the
cell in 0.05 mm steps and record force against depth. That curve gives the
swab's stiffness and tells you how much a 0.05 mm Z error costs in force.

## 3. Swab wear limits

`wear.max_path_mm` and `wear.max_contact_s` decide when a swab is retired
mid-slide. Scribe a long continuous path on a sacrificial slide and inspect at
intervals: the point where line quality visibly drops is the limit. Expect it
to depend on force.

Related, and worth checking at the same time: whether kerf grows as a swab
wears. If it does, line spacing has to allow for the widest kerf a swab reaches
before retirement, not the kerf it starts with.

## 4. Slide surface offset

`slide_surface_offset_mm` links the load cell Z to the glass: scribe height is
the probed contact height plus this offset. Measure the height difference
between the load cell top surface and the slide top surface. Without it, a
probe reading cannot be transferred to the slide at all.

## 5. Blue detection thresholds

Sample the real film and stem colours in the Calibration tab under the final
lighting, rather than typing hue numbers. Then check two things: that the
2.54 mm stem spans at least about ten pixels, and that glare on the glass does
not push a loaded slot below the threshold.

## 6. Scribe speed

Fast enough to be useful, slow enough to cut cleanly. Interacts with kerf and
with wear, so measure it alongside both rather than on its own.
