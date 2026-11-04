"""One FastAPI app, three tabs: Operation, Calibration, Jog and test.

They share this backend, so no logic is duplicated, and only one of them can
own the hardware at a time: while a job runs, the calibration and jog
endpoints refuse, which is what stops someone jogging the gantry mid-scribe.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from papa import config as cfgmod
from papa.hw import make_camera, make_gripper, make_loadcell, make_printer
from papa.ops.geometry import square, to_machine
from papa.ops.guard import Guard, GuardedPrinter, ZFloorError
from papa.ops.jobs import STATE_DIR, Job
from papa.ops.machine import Orchestrator, Status
from papa.ops.swabs import Rack
from papa.ops.zcal import GripFailed, OverForce, calibrate_z
from papa.hw.camera import LockedCamera
from papa.hw.loadcell import LoadCellSampler
from papa.hw.gripper import MAX_US, MIN_US, GripperNotCalibrated
from papa.ops.hold import HoldJog
from papa.vision.live import encode_jpeg
from papa.vision.substrates import REFERENCE_WIDTH, SubstrateParams, annotate, detect as detect_substrates
from papa.toolpath import blocks as blockmod
from papa.toolpath import dxf_loader, fit, spacing as spacingmod

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="PAPA")


class Device:
    """One lazily constructed device.

    Lazy on purpose: a laptop has no HX711, and building every driver at
    startup would mean a missing load cell stops you jogging the gantry. Each
    device is built on first use and a failure is remembered, so the endpoints
    that need it return 503 with the reason while everything else keeps working.
    """

    def __init__(self, name: str, factory: Callable[[], Any]) -> None:
        self.name = name
        self._factory = factory
        self._obj: Any = None
        self.error: str | None = None

    @property
    def built(self) -> bool:
        return self._obj is not None

    def get(self) -> Any:
        if self._obj is None and self.error is None:
            try:
                self._obj = self._factory()
            except Exception as exc:                     # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
        if self.error is not None:
            raise HTTPException(503, f"{self.name} is unavailable ({self.error})")
        return self._obj

    def describe(self) -> dict:
        return {
            "real": cfgmod.is_real(self.name),
            "built": self.built,
            "error": self.error,
        }


class Hub:
    """Owns the hardware handles and the one running job."""

    def __init__(self) -> None:
        self.machine = cfgmod.load_machine()
        self.cal = cfgmod.load_calibration()
        self.guard = Guard()
        self.last_probe: dict[str, float] | None = None
        self.devices = {
            "printer": Device(
                "printer", lambda: GuardedPrinter(make_printer(self.machine), self.guard)
            ),
            "loadcell": Device(
                "loadcell",
                lambda: make_loadcell(self.machine, self.cal, self.printer),
            ),
            "gripper": Device("gripper", lambda: make_gripper(self.machine, self.cal)),
            "camera": Device("camera", lambda: LockedCamera(make_camera(self.machine))),
        }
        self.orch: Orchestrator | None = None
        self.thread: threading.Thread | None = None
        self.status = Status()
        self.strokes: list[Any] = []
        self.dxf_name = ""
        self.lock = threading.Lock()
        self._sampler: LoadCellSampler | None = None
        self.hold = HoldJog(lambda: self.printer, self.machine, lambda: self.guard)
        self.servo_window = [1300, 1700]

    @property
    def printer(self) -> Any:
        return self.devices["printer"].get()

    @property
    def loadcell(self) -> Any:
        return self.devices["loadcell"].get()

    @property
    def gripper(self) -> Any:
        return self.devices["gripper"].get()

    @property
    def camera(self) -> Any:
        return self.devices["camera"].get()

    @property
    def sampler(self) -> LoadCellSampler:
        if self._sampler is None or self._sampler.cell is not self.loadcell:
            self._sampler = LoadCellSampler(self.loadcell)
        return self._sampler

    def stop_sampler(self) -> None:
        if self._sampler is not None:
            self._sampler.stop()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def require_idle(self) -> None:
        if self.busy:
            raise HTTPException(409, "a job is running; the hardware is busy")
        if self.hold.active:
            raise HTTPException(409, "release the held jog button first")

    def reload_calibration(self) -> None:
        self.cal = cfgmod.load_calibration()


hub = Hub()


# ---------- pages ----------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


# ---------- live camera ----------

_streams = 0
MAX_STREAMS = 3
BOUNDARY = "frame"


def _camera_host() -> str:
    """Base URL of a Pi to take footage from, when this server has no camera of
    its own (a laptop driving a real printer). Empty means use the local one."""
    return os.environ.get("PAPA_CAMERA_URL", "").rstrip("/")


def _grab_jpeg(width: int) -> bytes:
    return encode_jpeg(hub.camera.frame(), width, "REAL" if cfgmod.is_real("camera") else "MOCK")


async def _probe_host(base: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(base + "/api/camera/snapshot", params={"width": 160})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"camera host {base} is unreachable ({type(exc).__name__})") from exc
    if r.status_code != 200:
        raise HTTPException(502, f"camera host answered {r.status_code}: {r.text[:200]}")


@app.get("/api/camera/snapshot")
async def camera_snapshot(width: int = 1280) -> Response:
    width = min(max(width, 160), 4056)
    base = _camera_host()
    if base:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(base + "/api/camera/snapshot", params={"width": width})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"camera host {base} is unreachable ({type(exc).__name__})") from exc
        if r.status_code != 200:
            raise HTTPException(502, f"camera host answered {r.status_code}")
        jpeg = r.content
    else:
        jpeg = await asyncio.to_thread(_grab_jpeg, width)
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/camera/stream")
async def camera_stream(request: Request, width: int = 960, fps: float = 8.0,
                        frames: int = 0) -> StreamingResponse:
    """MJPEG for an <img>. `frames` caps the stream (0 is endless); tests use it."""
    global _streams
    width = min(max(width, 160), 1920)
    fps = min(max(fps, 1.0), 15.0)
    base = _camera_host()
    if _streams >= MAX_STREAMS:
        raise HTTPException(429, "too many live views open; close another tab")
    if base:
        await _probe_host(base)
    else:
        hub.camera          # builds it now, so a missing camera is a clean 503 with the reason

    async def local() -> Any:
        global _streams
        _streams += 1
        sent, period = 0, 1.0 / fps
        try:
            while not await request.is_disconnected():
                t0 = time.monotonic()
                try:
                    jpeg = await asyncio.to_thread(_grab_jpeg, width)
                except Exception:                                  # noqa: BLE001
                    break
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg + b"\r\n"
                sent += 1
                if frames and sent >= frames:
                    break
                await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))
        finally:
            _streams -= 1

    async def remote() -> Any:
        global _streams
        _streams += 1
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as c:
                async with c.stream("GET", base + "/api/camera/stream",
                                    params={"width": width, "fps": fps, "frames": frames}) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
        except httpx.HTTPError:
            return
        finally:
            _streams -= 1

    return StreamingResponse(
        remote() if base else local(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


# ---------- validation page ----------

def _next_scan_dir() -> tuple[Path, int]:
    root = STATE_DIR / "scans"
    root.mkdir(parents=True, exist_ok=True)
    ids = [int(p.name.split("_")[-1]) for p in root.glob("scan_*") if p.name.split("_")[-1].isdigit()]
    n = max(ids, default=0) + 1
    d = root / f"scan_{n:04d}"
    d.mkdir()
    return d, n


@app.post("/api/camera/detect")
async def camera_detect(payload: dict | None = None) -> JSONResponse:
    """Find the red corner tape on the current frame, pair it into substrates and
    number them. Same method as the team's substrateRecognition script."""
    import base64

    import cv2

    body = payload or {}
    remote = _camera_host()
    if remote:
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.post(f"{remote}/api/camera/detect", json=body)
            return JSONResponse(r.json(), status_code=r.status_code)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"camera host {remote} unreachable: {exc}"}, status_code=502)

    def work() -> dict:
        rgb = hub.camera.frame()
        w = rgb.shape[1]
        k = REFERENCE_WIDTH / w

        def px(name: str) -> float | None:
            v = body.get(name)
            return None if v in (None, "") else float(v) * k      # frame pixels -> reference pixels

        base = SubstrateParams()
        params = SubstrateParams(
            min_red_area=float(body.get("min_red_area") or base.min_red_area),
            expected_dx=px("expected_dx_px"), expected_dy=px("expected_dy_px"),
        )
        res = detect_substrates(rgb, params)
        annotated = annotate(res.pop("_bgr"), res.pop("_markers"), res["substrates"])
        out_w = min(1280, w)
        small = cv2.resize(annotated, (out_w, round(annotated.shape[0] * out_w / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        res["image"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        res["shown_scale"] = out_w / w
        res["warnings"] = []
        if res["unmatched_markers"]:
            res["warnings"].append(
                f"{len(res['unmatched_markers'])} red marker(s) were not paired. Enter the real marker spacing "
                "(pixels) to stop neighbors being paired by mistake.")
        if res["marker_count"] == 0:
            res["warnings"].append("No red markers found. Check lighting, focus and that the tape is in view.")
        if body.get("save", True):
            import json

            d, n = _next_scan_dir()
            cv2.imwrite(str(d / "original.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(d / "detected.jpg"), annotated)
            (d / "substrates.json").write_text(json.dumps({"scan_id": n, **{k2: v2 for k2, v2 in res.items() if k2 not in ("image",)}}, indent=2))
            res["scan_id"], res["saved_to"] = n, str(d)
        return res

    try:
        return JSONResponse(await asyncio.to_thread(work))
    except Exception as exc:                                     # noqa: BLE001
        return JSONResponse({"error": f"camera unavailable: {exc}"}, status_code=503)


@app.get("/validate")
def validate_page() -> FileResponse:
    return FileResponse(STATIC / "validate.html")


@app.get("/api/devices")
def devices() -> dict:
    """What this process is actually talking to, and whether it answers.

    The page shows this, so there is never any doubt about whether a reading
    came from hardware or from a mock.
    """
    reachable, url, detail = False, "", None
    try:
        reachable = hub.printer.ping()          # builds it, so describe() comes after
        url = getattr(hub.printer, "base_url", "")
    except HTTPException as exc:
        detail = str(exc.detail)
    out = {name: d.describe() for name, d in hub.devices.items()}
    out["printer"].update(reachable=reachable, url=url)
    out["camera"]["remote"] = os.environ.get("PAPA_CAMERA_URL", "").rstrip("/") or None
    if detail:
        out["printer"]["detail"] = detail
    return {"sim": cfgmod.sim_mode(), "devices": out,
            "moonraker_url": hub.machine.moonraker_url,
            "zcal": hub.machine.zcal, "speeds": hub.machine.speeds,
            "limits": hub.machine.limits}


# ---------- operation ----------

@app.post("/api/pattern")
async def upload_pattern(file: UploadFile) -> dict:
    hub.require_idle()
    tmp = STATE_DIR / "upload.dxf"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())
    try:
        strokes = dxf_loader.load(tmp)
    except dxf_loader.UnsupportedEntity as exc:
        raise HTTPException(400, str(exc)) from exc
    w, h = hub.cal.pattern_area
    result = fit.fit(strokes, w, h)
    hub.strokes = result.strokes
    hub.dxf_name = file.filename or "pattern.dxf"
    return {
        "name": hub.dxf_name,
        "strokes": len(result.strokes),
        "scale": result.scale,
        "source_size_mm": result.source_size,
        "fitted_size_mm": result.fitted_size,
        "area_mm": [w, h],
        "svg": fit.to_svg(result.strokes, w, h),
        "spacing": spacingmod.report(result.strokes, hub.cal.kerf_mm),
    }


@app.get("/api/scan")
def scan() -> dict:
    """One still frame, filtered for blue: which slots hold a film and which
    rack positions hold a swab."""
    hub.require_idle()
    from papa.vision import presence

    frame = hub.camera.frame()
    det = hub.cal.detection
    try:
        cmap = presence.CameraMap.from_cfg(det)
    except ValueError as exc:
        raise HTTPException(400, f"camera is not mapped to machine coordinates: {exc}") from exc

    film = presence.HsvBand.from_cfg(det.get("film_hsv", {}))
    stem = presence.HsvBand.from_cfg(det.get("stem_hsv", det.get("film_hsv", {})))
    w, h = hub.cal.pattern_area

    slots = []
    for slot in hub.cal.slots:
        d = presence.slide_present(
            frame, cmap, (slot.x, slot.y), w, h, film,
            min_fraction=float(det.get("slide_min_blue_fraction", 0.35)),
            misaligned_fraction=float(det.get("slide_misaligned_fraction", 0.12)),
        )
        slots.append({"id": slot.id, "present": d.present, "fraction": round(d.fraction, 3),
                      "misaligned": d.misaligned, "calibrated": slot.calibrated})

    swabs = []
    for s in hub.cal.swabs:
        d = presence.swab_present(
            frame, cmap, (s.x, s.y), float(det.get("swab_roi_mm", 4.0)), stem,
            min_fraction=float(det.get("swab_min_blue_fraction", 0.30)),
        )
        swabs.append({"id": s.id, "label": s.label, "present": d.present,
                      "fraction": round(d.fraction, 3), "calibrated": s.calibrated})
    return {"slots": slots, "swabs": swabs}


@app.post("/api/job")
def start_job(payload: dict) -> dict:
    hub.require_idle()
    if not hub.strokes:
        raise HTTPException(400, "upload a DXF first")
    slot_ids = [int(i) for i in payload.get("slot_ids", [])]
    if not slot_ids:
        raise HTTPException(400, "select at least one slide")
    per = int(payload.get("swabs_per_slide", hub.cal.swabs_per_slide))
    present = [s["label"] for s in scan()["swabs"] if s["present"]] if payload.get("rescan") \
        else [s.label for s in hub.cal.swabs]

    bl = blockmod.make_blocks(hub.strokes, hub.machine.toolpath.get("block_target_mm", 3.0))
    job = Job(
        dxf_name=hub.dxf_name, slot_ids=slot_ids, swabs_per_slide=per,
        pattern_size_mm=hub.cal.pattern_area[0], total_blocks=len(bl),
    )
    orch = Orchestrator(
        machine=hub.machine, cal=hub.cal, printer=hub.printer, loadcell=hub.loadcell,
        gripper=hub.gripper, rack=Rack.from_detection(hub.cal, present),
        blocks=bl, job=job, on_status=lambda s: setattr(hub, "status", s),
    )
    problems = orch.preflight()
    if problems:
        raise HTTPException(400, {"preflight": problems})

    hub.orch = orch
    hub.thread = threading.Thread(target=orch.run, daemon=True)
    hub.thread.start()
    return {"started": True, "blocks": len(bl), "slides": len(slot_ids)}


@app.get("/api/status")
def status() -> dict:
    s = hub.status
    job = hub.orch.job if hub.orch else Job.load()
    return {
        "busy": hub.busy, "sim": cfgmod.sim_mode(),
        "status": s.__dict__,
        "job": None if job is None else {
            "dxf": job.dxf_name, "state": job.state, "message": job.message,
            "fraction": round(job.fraction, 4),
            "slides": [sp.__dict__ for sp in job.slides],
        },
    }


@app.post("/api/control/{action}")
def control(action: str) -> dict:
    if hub.orch is None:
        raise HTTPException(400, "no job")
    match action:
        case "pause":
            hub.orch.pause()
        case "resume":
            hub.orch.resume()
        case "soft_stop":
            hub.orch.soft_stop()
        case "hard_stop":
            hub.orch.hard_stop()
        case _:
            raise HTTPException(404, f"unknown action {action}")
    return {"ok": True, "action": action}


# ---------- jog and test (refused while a job runs) ----------

@app.get("/api/machine/position")
def position() -> dict:
    p = hub.printer.position()
    return {"x": p.x, "y": p.y, "z": p.z, "homed": p.homed, "is_homed": p.is_homed}


@app.post("/api/machine/home")
def home(payload: dict | None = None) -> dict:
    hub.require_idle()
    axes = str((payload or {}).get("axes", "")).upper().replace(" ", "")
    if (not axes or "Z" in axes) and not hub.guard.allow_z_home:
        raise HTTPException(
            403,
            "Z homing is locked. Home X and Y here; home Z only after the Z "
            "switch and homing config are verified, then start the server with "
            "PAPA_ALLOW_Z_HOME=1.",
        )
    hub.printer.home(" ".join(axes))
    return position()


@app.post("/api/machine/jog")
def jog(payload: dict) -> dict:
    """Relative jog, clamped against limits_mm before anything is sent.

    Clamping here matters because the GUI repeats while a button is held.
    """
    hub.require_idle()
    p = hub.printer.position()
    speed = float(payload.get("speed_mm_s", hub.machine.speeds["travel"]))
    tx = hub.machine.clamp("x", p.x + float(payload.get("dx", 0.0)))
    ty = hub.machine.clamp("y", p.y + float(payload.get("dy", 0.0)))
    dz = float(payload.get("dz", 0.0))
    tz = hub.machine.clamp("z", p.z + dz) if dz else None
    hub.printer.move(x=tx, y=ty, z=tz, speed_mm_s=speed)
    return position()


@app.post("/api/machine/move")
def move(payload: dict) -> dict:
    hub.require_idle()
    x, y, z = (payload.get("x"), payload.get("y"), payload.get("z"))
    if not hub.machine.in_limits(
        float(x if x is not None else 0), float(y if y is not None else 0),
        float(z if z is not None else 0),
    ):
        raise HTTPException(400, "target is outside limits_mm")
    hub.printer.move(x=x, y=y, z=z, speed_mm_s=float(payload.get("speed_mm_s", 50.0)))
    return position()


@app.post("/api/machine/hold")
def hold_jog(payload: dict) -> dict:
    """Heartbeat of a held jog button: keeps one axis moving while they keep coming."""
    if hub.busy:
        raise HTTPException(409, "a job is running; the hardware is busy")
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy") and not payload.get("dz"):
        raise HTTPException(400, "home X and Y first")
    try:
        hub.hold.beat(payload.get("dx", 0), payload.get("dy", 0), payload.get("dz", 0),
                      payload.get("speed_mm_s", 0))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"moving": hub.hold.active, "why": hub.hold.stopped_because}


@app.post("/api/machine/hold/stop")
def hold_stop() -> dict:
    hub.hold.stop()
    why = hub.hold.stopped_because
    hub.hold.reset()
    p = hub.printer.position()
    return {"moving": hub.hold.active, "why": why,
            "x": p.x, "y": p.y, "z": p.z, "homed": p.homed, "is_homed": p.is_homed}


@app.get("/api/machine/hold")
def hold_state() -> dict:
    return {"moving": hub.hold.active, "why": hub.hold.stopped_because}


@app.post("/api/machine/estop")
def estop() -> dict:
    hub.hold.stop("emergency stop")
    hub.printer.emergency_stop()
    return {"ok": True}


@app.post("/api/machine/restart")
def restart() -> dict:
    hub.printer.firmware_restart()
    return {"ok": True}


def _patch_calibration(updates: dict) -> None:
    """Merge into calibration.yaml and reload.

    With no real calibration.yaml yet, start from the example file, never from
    the simulation file: invented slot and swab positions must not become the
    machine's calibration just because a validation value was saved.
    """
    import copy

    real = cfgmod.CONFIG_DIR / "calibration.yaml"
    base = hub.cal.raw if real.exists() else cfgmod._load(cfgmod.CONFIG_DIR / "calibration.example.yaml")
    raw = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k].update(v)
        else:
            raw[k] = v
    cfgmod.save_calibration(raw)
    hub.reload_calibration()


@app.get("/api/guard")
def guard_state() -> dict:
    return {"floor": hub.guard.floor, "allow_z_home": hub.guard.allow_z_home}


@app.post("/api/guard/floor")
def guard_floor(payload: dict) -> dict:
    z = payload.get("z")
    if z is None:
        hub.guard.clear()
    else:
        z = float(z)
        lo, hi = hub.machine.limits["z"]
        if not lo <= z <= hi:
            raise HTTPException(400, f"the floor must be within the Z limits {lo} to {hi}")
        hub.guard.set_floor(z)
    return guard_state()


def _grip_info() -> dict:
    g = hub.gripper
    cal = hub.cal.gripper
    return {
        "state": getattr(g, "state", "unknown"),
        "pulse_us": getattr(g, "pulse_us", None),
        "locked": {"open_us": cal["open_us"], "closed_us": cal["closed_us"]},
        "open_set": bool(cal.get("open_set")), "closed_set": bool(cal.get("closed_set")),
        "calibrated": bool(getattr(g, "calibrated", True)),
        "backend": getattr(g, "backend", "mock"), "backend_note": getattr(g, "backend_note", ""),
        "window": list(hub.servo_window),
        "min_us": MIN_US, "max_us": MAX_US,
    }


@app.get("/api/gripper")
def gripper_info() -> dict:
    return _grip_info()


@app.post("/api/gripper/window")
def gripper_window(payload: dict) -> dict:
    """The only range manual pulses may use. Starts narrow around centre."""
    hub.require_idle()
    lo, hi = int(payload.get("lo", 0)), int(payload.get("hi", 0))
    if not MIN_US <= lo < hi <= MAX_US:
        raise HTTPException(400, f"window must sit inside {MIN_US} to {MAX_US} us")
    hub.servo_window = [lo, hi]
    return _grip_info()


@app.post("/api/gripper/pulse")
def gripper_pulse(payload: dict) -> dict:
    hub.require_idle()
    us = int(payload.get("us", 0))
    lo, hi = hub.servo_window
    if not lo <= us <= hi:
        raise HTTPException(400, f"{us} us is outside the servo window {lo} to {hi} us; widen the window first")
    slew = payload.get("slew_us_s")
    slew = float(slew) if slew else None
    if slew is not None and not 1 <= slew <= 2000:
        raise HTTPException(400, "slew must be 1 to 2000 us per second")
    hub.gripper.pulse(us, slew_us_s=slew)
    return _grip_info()


@app.post("/api/gripper/lock")
def gripper_lock(payload: dict) -> dict:
    """Store the pulse width for open or closed and use it from now on."""
    hub.require_idle()
    key = {"open": "open_us", "closed": "closed_us"}.get(str(payload.get("which")))
    if key is None:
        raise HTTPException(400, "which must be open or closed")
    us = int(payload.get("us", 0))
    if not MIN_US <= us <= MAX_US:
        raise HTTPException(400, f"pulse width must be {MIN_US} to {MAX_US} us")
    g = dict(hub.cal.gripper)
    g[key] = us
    g[key.replace("_us", "_set")] = True
    if g["open_us"] == g["closed_us"]:
        raise HTTPException(400, "open and closed cannot be the same pulse width")
    _patch_calibration({"gripper": g})
    setattr(hub.gripper, key, us)
    if g.get("open_set") and g.get("closed_set") and hasattr(hub.gripper, "calibrated"):
        hub.gripper.calibrated = True
    out = _grip_info()
    if abs(g["open_us"] - g["closed_us"]) < 100:
        out["warning"] = "open and closed are less than 100 us apart; check both"
    return out


@app.post("/api/gripper/release")
def gripper_release() -> dict:
    hub.require_idle()
    hub.gripper.release()
    return _grip_info()


@app.post("/api/gripper/{action}")
def gripper(action: str, payload: dict | None = None) -> dict:
    hub.require_idle()
    slew = (payload or {}).get("slew_us_s")
    slew = float(slew) if slew else None
    if slew is not None and not 1 <= slew <= 2000:
        raise HTTPException(400, "slew must be 1 to 2000 us per second")
    try:
        if action == "open":
            hub.gripper.open(slew_us_s=slew)
        elif action == "close":
            hub.gripper.close(slew_us_s=slew)
        else:
            raise HTTPException(404, action)
    except GripperNotCalibrated as exc:
        raise HTTPException(409, str(exc)) from exc
    return _grip_info()


@app.get("/api/loadcell")
def loadcell() -> dict:
    """One filtered reading. grams is null until the scale has been calibrated."""
    lc = hub.loadcell
    scaled = bool(getattr(lc, "scaled", True))
    counts = lc.read_counts()
    off = getattr(lc, "offset", 0)
    sc = getattr(lc, "scale", 1.0) or 1.0
    return {"counts": counts, "offset": off, "scale": sc, "scaled": scaled,
            "grams": round(lc.grams(), 3) if scaled else None}


@app.get("/api/loadcell/live")
def loadcell_live(since: int = 0, limit: int = 2000) -> dict:
    """New raw samples since a sequence number. Calling it keeps the background
    sampler running; the sampler stops by itself when nobody asks."""
    lc = hub.loadcell
    sm = hub.sampler
    sm.touch()
    seq, items = sm.since(int(since), max(1, min(int(limit), 4000)))
    return {
        "seq": seq,
        "samples": [[round(t, 4), c] for t, c in items],
        "offset": getattr(lc, "offset", 0),
        "scale": getattr(lc, "scale", 1.0) or 1.0,
        "scaled": bool(getattr(lc, "scaled", True)),
        "error": sm.error,
        "real": cfgmod.is_real("loadcell"),
    }


@app.post("/api/loadcell/tare")
def tare() -> dict:
    hub.require_idle()
    hub.stop_sampler()
    return {"tare_counts": hub.loadcell.tare()}


@app.post("/api/loadcell/scale")
def scale(payload: dict) -> dict:
    hub.require_idle()
    hub.stop_sampler()
    g = float(payload.get("grams", 0))
    return {"scale_counts_per_g": hub.loadcell.calibrate_with_known_weight(g)}


@app.post("/api/loadcell/position")
def loadcell_position(payload: dict) -> dict:
    """Store where the load cell is. XY from the live position or typed; the Z is
    only the approximate height to start the stepped descent from."""
    hub.require_idle()
    p = hub.printer.position()
    lc: dict = {}
    if payload.get("capture_xy"):
        if not all(a in p.homed for a in "xy"):
            raise HTTPException(400, "home X and Y first")
        lc.update(x=round(p.x, 3), y=round(p.y, 3), xy_set=True)
    if payload.get("capture_z"):
        lc.update(z_safe=round(p.z, 3), z_safe_set=True)
    if "z_safe" in payload:
        lc.update(z_safe=float(payload["z_safe"]), z_safe_set=True)
    lo, hi = hub.machine.limits["z"]
    if "z_safe" in lc and not lo <= lc["z_safe"] <= hi:
        raise HTTPException(400, "probe start Z is outside the Z limits")
    if not lc:
        raise HTTPException(400, "nothing to save")
    _patch_calibration({"load_cell": lc})
    return {"load_cell": hub.cal.raw["load_cell"]}


@app.post("/api/loadcell/save_scale")
def loadcell_save_scale(payload: dict) -> dict:
    hub.require_idle()
    _patch_calibration({"load_cell": {
        "tare_counts": int(payload["tare_counts"]),
        "scale_counts_per_g": float(payload["scale_counts_per_g"]),
    }})
    return {"load_cell": hub.cal.raw["load_cell"]}


@app.post("/api/loadcell/probe")
def loadcell_probe(payload: dict | None = None) -> dict:
    """Run the same stepped Z calibration a job runs, once, on its own.

    The body may override, for this probe only: start_z (where the descent
    begins), speed_mm_s (the stepped descent speed) and step_mm.
    """
    hub.require_idle()
    hub.stop_sampler()
    body = payload or {}
    lc, z, sp = hub.cal.load_cell, hub.machine.zcal, hub.machine.speeds
    if hub.guard.floor is None:
        raise HTTPException(400, "Set the Z floor first")
    if not (lc.get("xy_set") and lc.get("z_safe_set")) and body.get("start_z") is None:
        raise HTTPException(400, "Capture the load cell XY and probe start Z first")
    if not lc.get("xy_set"):
        raise HTTPException(400, "Capture the load cell XY first")
    start_z = float(body["start_z"]) if body.get("start_z") is not None else float(lc["z_safe"])
    speed = float(body.get("speed_mm_s") or sp["z_approach"])
    step = float(body.get("step_mm") or z["step_mm"])
    zlo, zhi = hub.machine.limits["z"]
    if not zlo <= start_z <= zhi:
        raise HTTPException(400, f"start Z must be {zlo} to {zhi} mm")
    if not 0.05 <= speed <= 10:
        raise HTTPException(400, "descent speed must be 0.05 to 10 mm/s")
    if not 0.01 <= step <= 1:
        raise HTTPException(400, "step must be 0.01 to 1 mm")
    if start_z < hub.guard.floor:
        raise HTTPException(400, "probe start Z is below the Z floor")
    if not hub.printer.position().is_homed:
        raise HTTPException(400, "home the machine first")
    try:
        r = calibrate_z(
            hub.printer, hub.loadcell,
            cell_x=lc["x"], cell_y=lc["y"], z_safe=start_z,
            target_force_g=z["target_force_g"], abort_force_g=z["abort_force_g"],
            step_mm=step, max_travel_mm=z["max_travel_mm"],
            approach_speed_mm_s=speed, travel_speed_mm_s=sp["travel"],
            settle_readings=int(z.get("settle_readings", 3)),
            backoff_mm=z.get("backoff_mm", 0.5),
            z_speed_mm_s=min(float(sp.get("z_lift", 10.0)), 10.0),
        )
    except (GripFailed, OverForce) as exc:
        raise HTTPException(409, str(exc)) from exc
    hub.last_probe = {"contact_z": r.contact_z, "force_g": r.force_g, "steps": r.steps}
    return hub.last_probe


# ---------- calibration ----------

@app.get("/api/calibration")
def get_calibration() -> dict:
    return hub.cal.raw


@app.post("/api/calibration")
def put_calibration(payload: dict) -> dict:
    """Stage and write calibration.yaml, then reload it in place."""
    hub.require_idle()
    path = cfgmod.save_calibration(payload)
    hub.reload_calibration()
    return {"written": str(path)}


@app.post("/api/calibration/capture")
def capture(payload: dict) -> dict:
    """Store the live machine position for the target being taught, so nothing
    is typed by hand."""
    hub.require_idle()
    p = hub.printer.position()
    return {"target": payload.get("target"), "id": payload.get("id"),
            "x": p.x, "y": p.y, "z": p.z}


@app.post("/api/calibration/surface_offset")
def surface_offset(payload: dict) -> dict:
    """scribe Z = load cell contact Z + this offset. Usually negative: the cell
    sits above the glass."""
    hub.require_idle()
    off = float(payload["offset_mm"])
    if abs(off) > 100:
        raise HTTPException(400, "offset over 100 mm is almost certainly a mistake")
    _patch_calibration({"slide_surface_offset_mm": round(off, 4)})
    return {"slide_surface_offset_mm": hub.cal.surface_offset_mm}


def _slide_info(slot_id: int) -> dict:
    cal = hub.cal
    rows, cols = cal.slot_grid
    sub = cal.substrate
    slot = next((s for s in cal.slots if s.id == slot_id), None)
    out: dict = {
        "grid": [rows, cols], "substrate": sub, "fits": cal.pattern_fits(),
        "axes": hub.machine.axes,
        "slots": [{"id": s.id, "x": s.x, "y": s.y, "calibrated": s.calibrated} for s in cal.slots],
        "slot": None,
    }
    if slot is not None:
        def pts(w: float, h: float) -> list[list[float]]:
            return [list(to_machine(hub.machine.axes, slot.x, slot.y, lx, ly))
                    for lx, ly in square(w, h)]
        out["slot"] = {"id": slot.id, "x": slot.x, "y": slot.y, "calibrated": slot.calibrated,
                       "outline": pts(sub["width_mm"], sub["length_mm"]),
                       "pattern": pts(sub["pattern_size_mm"], sub["pattern_size_mm"])}
    return out


@app.get("/api/slide")
def slide_info(slot: int = 1) -> dict:
    return _slide_info(slot)


@app.post("/api/slide/size")
def slide_size(payload: dict) -> dict:
    hub.require_idle()
    w, l, p = (float(payload[k]) for k in ("width_mm", "length_mm", "pattern_size_mm"))
    if min(w, l, p) <= 0:
        raise HTTPException(400, "dimensions must be positive")
    if p > min(w, l):
        raise HTTPException(400, "the pattern square must fit inside the slide")
    _patch_calibration({"substrate": {"width_mm": w, "length_mm": l, "pattern_size_mm": p}})
    return _slide_info(int(payload.get("slot", 1)))


@app.post("/api/slide/corner")
def slide_corner(payload: dict) -> dict:
    """Store the live X/Y as the slide's top-left corner for this slot."""
    hub.require_idle()
    sid = int(payload["id"])
    rows, cols = hub.cal.slot_grid
    if not 1 <= sid <= rows * cols:
        raise HTTPException(400, f"slot must be 1 to {rows * cols}")
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy"):
        raise HTTPException(400, "home X and Y first")
    slots = [dict(id=s.id, x=s.x, y=s.y, calibrated=s.calibrated)
             for s in hub.cal.slots if s.id != sid]
    slots.append(dict(id=sid, x=round(p.x, 3), y=round(p.y, 3), calibrated=True))
    slots.sort(key=lambda s: s["id"])
    _patch_calibration({"slide_slots": {"slots": slots}})
    return _slide_info(sid)


@app.post("/api/slide/trace")
def slide_trace(payload: dict) -> dict:
    """Walk the corners of the pattern square or the slide outline, XY only, at
    whatever Z the toolhead is at. Proves the corner, size and axis mapping."""
    hub.require_idle()
    info = _slide_info(int(payload["id"]))
    slot = info["slot"]
    if slot is None or not slot["calibrated"]:
        raise HTTPException(400, "capture this slot's top-left corner first")
    which = payload.get("which", "pattern")
    if which not in ("pattern", "outline"):
        raise HTTPException(400, "which must be pattern or outline")
    speed = min(max(float(payload.get("speed_mm_s", 20.0)), 1.0), hub.machine.speeds["travel"])
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy"):
        raise HTTPException(400, "home X and Y first")
    pts = slot[which]
    for x, y in pts + [pts[0]]:
        if not hub.machine.in_limits(x, y, p.z):
            raise HTTPException(400, f"corner X{x:.1f} Y{y:.1f} is outside the machine limits")
    for x, y in pts + [pts[0]]:
        hub.printer.move(x=x, y=y, speed_mm_s=speed)
    return position()


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(HTTPException)
async def http_error(_req: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(ZFloorError)
async def floor_error(_req: Any, exc: ZFloorError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=400)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
