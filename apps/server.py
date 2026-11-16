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
from papa.ops import autonomous as automod
from papa.ops import scrub as scrubmod
from papa.ops import routine as routinemod
from papa.ops import alignment as alignmod
from papa.vision import correction as corrmod
from papa.vision.electrodes import ElectrodeParams
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
    A failure is retried at most every RETRY_S, so a device that was busy or
    unplugged comes back once it is free, without restarting the server.
    """

    RETRY_S = 5.0
    HINTS = {
        "camera": " The camera is probably in use by another program on the Pi (a camera script, "
                  "rpicam-hello/rpicam-still, or a second copy of this server). Close it; the page "
                  "retries by itself. If it stays, check the ribbon cable and run rpicam-hello --list-cameras.",
    }

    def __init__(self, name: str, factory: Callable[[], Any]) -> None:
        self.name = name
        self._factory = factory
        self._obj: Any = None
        self.error: str | None = None
        self._failed_at = 0.0

    @property
    def built(self) -> bool:
        return self._obj is not None

    def get(self) -> Any:
        if self._obj is None and (self.error is None or time.monotonic() - self._failed_at >= self.RETRY_S):
            try:
                self._obj = self._factory()
                self.error = None
            except Exception as exc:                     # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                self._failed_at = time.monotonic()
        if self.error is not None and self._obj is None:
            raise HTTPException(503, f"{self.name} is unavailable ({self.error}).{self.HINTS.get(self.name, '')}")
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
        self.auto = automod.AutoRunner(lambda: self.printer, lambda: self.gripper, self.machine)
        self.scrub = scrubmod.ScrubRunner(lambda: self.printer, self.machine)
        self.routine = routinemod.RoutineRunner(self.auto, self.scrub, self.machine)
        self.align_scan: dict[str, Any] | None = None   # last alignment scan, see /api/align

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

    IDLE_TIMEOUT_S = 14400

    def _keepalive(self) -> None:
        """Klipper turns the motors off after its idle timeout, which also makes it
        forget Z. Raise the timeout and note the Z while it is known."""
        while True:
            try:
                p = self.printer.position()
                if "z" in p.homed:
                    self.guard.remember_z(p.z)
                    self.printer.script(f"SET_IDLE_TIMEOUT TIMEOUT={self.IDLE_TIMEOUT_S}", wait=False)
            except Exception:                              # noqa: BLE001
                pass
            time.sleep(45)


hub = Hub()


@app.on_event("startup")
def _start_keepalive() -> None:
    threading.Thread(target=hub._keepalive, daemon=True, name="keepalive").start()


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


def _grab_jpeg(width: int, corrected: bool = False) -> bytes:
    rgb = hub.camera.frame()
    label = "REAL" if cfgmod.is_real("camera") else "MOCK"
    if corrected and corrmod.supported(rgb.shape[1::-1]):
        rgb = corrmod.get_correction(rgb.shape[1::-1]).correct_image(rgb)
        label += " CORRECTED"
    return encode_jpeg(rgb, width, label)


async def _probe_host(base: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(base + "/api/camera/snapshot", params={"width": 160})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"camera host {base} is unreachable ({type(exc).__name__})") from exc
    if r.status_code != 200:
        raise HTTPException(502, f"camera host answered {r.status_code}: {r.text[:200]}")


@app.get("/api/camera/snapshot")
async def camera_snapshot(width: int = 1280, corrected: int = 0) -> Response:
    width = min(max(width, 160), 4056)
    base = _camera_host()
    if base:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(base + "/api/camera/snapshot", params={"width": width, "corrected": corrected})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"camera host {base} is unreachable ({type(exc).__name__})") from exc
        if r.status_code != 200:
            raise HTTPException(502, f"camera host answered {r.status_code}")
        jpeg = r.content
    else:
        jpeg = await asyncio.to_thread(_grab_jpeg, width, bool(corrected))
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/camera/stream")
async def camera_stream(request: Request, width: int = 960, fps: float = 8.0,
                        frames: int = 0, corrected: int = 0) -> StreamingResponse:
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
                    jpeg = await asyncio.to_thread(_grab_jpeg, width, bool(corrected))
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
                                    params={"width": width, "fps": fps, "frames": frames,
                                            "corrected": corrected}) as r:
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


# ---------- capture and correct (cameraUpdates/capturedCorrection, in the GUI) ----------

def _captures_dir() -> Path:
    d = STATE_DIR / "captures"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.post("/api/camera/capture")
async def camera_capture(payload: dict | None = None) -> JSONResponse:
    """Take a photo, save the original and the corrected (straight-down) view,
    and optionally measure the 5 x 5 checkerboard. Full resolution by default."""
    import base64

    import cv2

    body = payload or {}
    remote = _camera_host()
    if remote:
        try:
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(f"{remote}/api/camera/capture", json=body)
            return JSONResponse(r.json(), status_code=r.status_code)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"camera host {remote} unreachable: {exc}"}, status_code=502)

    full = bool(body.get("full", True))

    def work() -> dict:
        rgb = hub.camera.still(corrmod.SENSOR_SIZE if full else None)
        size = rgb.shape[1::-1]
        if not corrmod.supported(size):
            raise HTTPException(400, f"no lens calibration for {size[0]} x {size[1]} photos")
        corrected = corrmod.get_correction(size).correct_image(rgb)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        d = _captures_dir()
        orig_name, corr_name = f"{stamp}_original.jpg", f"{stamp}_corrected.jpg"
        cv2.imwrite(str(d / orig_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(d / corr_name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out_w = min(1600, bgr.shape[1])
        small = cv2.resize(bgr, (out_w, round(bgr.shape[0] * out_w / bgr.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        res = {"photo_size": list(size), "corrected_size": list(corrected.shape[1::-1]),
               "original": orig_name, "corrected": corr_name, "saved_to": str(d),
               "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}
        if body.get("check"):
            res["check"] = corrmod.check_checkerboard(rgb)
        return res

    try:
        return JSONResponse(await asyncio.to_thread(work))
    except HTTPException:
        raise
    except Exception as exc:                                     # noqa: BLE001
        return JSONResponse({"error": f"camera unavailable: {exc}"}, status_code=503)


@app.get("/api/camera/captures/{name}")
def camera_capture_file(name: str) -> FileResponse:
    p = _captures_dir() / name
    if "/" in name or "\\" in name or not name.endswith(".jpg") or not p.is_file():
        raise HTTPException(404, "no such capture")
    return FileResponse(p, media_type="image/jpeg", filename=name)


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
    if "z" in p.homed:
        hub.guard.remember_z(p.z)
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
    hub.auto.stop()
    hub.scrub.stop()
    hub.routine.stop()
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
    return {"floor": hub.guard.floor, "ceiling": hub.guard.ceiling,
            "allow_z_home": hub.guard.allow_z_home, "last_z": hub.guard.last_z}


@app.post("/api/machine/set_z")
def set_z(payload: dict) -> dict:
    """Tell Klipper where Z is without moving, so Z can be jogged with no Z home.

    The number is a reference you choose, not a measured height, unless you
    measured it. Because every Z limit is relative to it, declaring a new one
    clears the floor and sets a ceiling `up_mm` above it.
    """
    hub.require_idle()
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy"):
        raise HTTPException(400, "Home X and Y first; Z is declared on top of a known X and Y")
    z = float(payload.get("z", 100.0))
    up = float(payload.get("up_mm", 50.0))
    lo, hi = hub.machine.limits["z"]
    if not lo <= z <= hi:
        raise HTTPException(400, f"Z must be {lo} to {hi}")
    if not 0 <= up <= hi - z:
        raise HTTPException(400, f"the room above must be 0 to {hi - z:g} mm")
    hub.printer.script(f"SET_KINEMATIC_POSITION X={p.x:.4f} Y={p.y:.4f} Z={z:.4f}", wait=False)
    hub.guard.reset_reference(z + up)
    return {**position(), **guard_state()}


@app.post("/api/machine/restore_z")
def restore_z() -> dict:
    """Re-declare the last known Z after Klipper forgot it. Keeps the floor and
    the taught positions, because the reference itself is unchanged.

    Only right if the head has not moved since: if the motors were off and the
    gantry sagged, jog Z to a known height and use Set Z instead.
    """
    hub.require_idle()
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy"):
        raise HTTPException(400, "Home X and Y first")
    if "z" in p.homed:
        raise HTTPException(400, "Z is already set; nothing to restore")
    z = hub.guard.last_z
    if z is None:
        raise HTTPException(400, "No saved Z yet. Use Set Z here once.")
    hub.printer.script(f"SET_KINEMATIC_POSITION X={p.x:.4f} Y={p.y:.4f} Z={z:.4f}", wait=False)
    return {**position(), **guard_state()}


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
    if hub.guard.ceiling is not None and start_z > hub.guard.ceiling:
        raise HTTPException(400, "probe start Z is above the Z ceiling")
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
    merged = dict(hub.cal.raw)
    merged.update(payload)
    path = cfgmod.save_calibration(merged)
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


# ---------- autonomous: one-button grip and dispose ----------

def _auto_settings() -> dict:
    return automod.merged(hub.cal.raw.get("autonomous"))


def _auto_info() -> dict:
    return {**automod.describe(_auto_settings(), hub.machine, hub.guard.ref), "status": hub.auto.status.as_dict()}


@app.get("/api/auto")
def auto_info() -> dict:
    return _auto_info()


@app.get("/api/auto/status")
def auto_status() -> dict:
    return hub.auto.status.as_dict()


@app.post("/api/auto/teach")
def auto_teach(payload: dict) -> dict:
    """Save where the tool is now as the pickup, the bin, or the safe Z."""
    hub.require_idle()
    target = payload.get("target")
    if target not in automod.TARGETS:
        raise HTTPException(400, f"target must be one of {', '.join(automod.TARGETS)}")
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy"):
        raise HTTPException(400, "Home X and Y first; a taught position means nothing without them")
    value: Any = round(p.z, 3) if target == "safe_z" else {
        "x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3)}
    refs = dict(_auto_settings().get("refs") or {})
    refs[target] = hub.guard.ref
    _patch_calibration({"autonomous": {target: value, "refs": refs}})
    return _auto_info()


@app.post("/api/auto/clear")
def auto_clear(payload: dict) -> dict:
    hub.require_idle()
    target = payload.get("target")
    if target not in automod.TARGETS:
        raise HTTPException(400, f"target must be one of {', '.join(automod.TARGETS)}")
    _patch_calibration({"autonomous": {target: None}})
    return _auto_info()


_AUTO_RANGES = {
    "lift_mm": (0.0, 200.0), "grip_shift_mm": (-10.0, 10.0), "grip_close_s": (0.2, 10.0), "grip_wait_s": (0.0, 10.0), "release_wait_s": (0.0, 10.0),
    "xy_speed_mm_s": (1.0, 150.0), "z_speed_mm_s": (0.5, 20.0), "skip_rise_xy_mm": (0.0, 50.0),
}


@app.post("/api/auto/settings")
def auto_settings(payload: dict) -> dict:
    hub.require_idle()
    updates: dict[str, float] = {}
    for key, (lo, hi) in _AUTO_RANGES.items():
        if key in payload:
            v = float(payload[key])
            if not lo <= v <= hi:
                raise HTTPException(400, f"{key} must be {lo:g} to {hi:g}")
            updates[key] = v
    if updates:
        _patch_calibration({"autonomous": updates})
    return _auto_info()


def _auto_start(action: str) -> dict:
    hub.require_idle()
    try:
        hub.thread = hub.auto.start(action, _auto_settings(), hub.guard.ref)
    except automod.AutoError as exc:
        raise HTTPException(400, str(exc)) from exc
    return hub.auto.status.as_dict()


@app.post("/api/auto/grip")
def auto_grip() -> dict:
    return _auto_start("grip")


@app.post("/api/auto/dispose")
def auto_dispose() -> dict:
    return _auto_start("dispose")


@app.post("/api/auto/stop")
def auto_stop() -> dict:
    hub.auto.stop()
    return hub.auto.status.as_dict()


def _scrub_settings() -> dict:
    return scrubmod.merged(hub.cal.raw.get("scrub"))


def _scrub_info() -> dict:
    st = _scrub_settings()
    info = scrubmod.describe(st, hub.machine, hub.guard.ref, hub.guard.floor)
    info["start"] = st["start"]
    info["status"] = hub.scrub.status.as_dict()
    return info


@app.get("/api/scrub")
def scrub_info() -> dict:
    return _scrub_info()


@app.get("/api/scrub/status")
def scrub_status() -> dict:
    return hub.scrub.status.as_dict()


@app.get("/api/scrub/plan")
def scrub_plan() -> dict:
    try:
        return scrubmod.plan(_scrub_settings())
    except scrubmod.ScrubError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/scrub/settings")
def scrub_settings(payload: dict) -> dict:
    hub.require_idle()
    updates: dict[str, float] = {}
    for key, (lo, hi) in scrubmod.RANGES.items():
        if key in payload:
            v = float(payload[key])
            if not lo <= v <= hi:
                raise HTTPException(400, f"{key} must be {lo:g} to {hi:g}")
            updates[key] = v
    if updates:
        _patch_calibration({"scrub": updates})
    return _scrub_info()


@app.post("/api/scrub/sketch")
def scrub_sketch(payload: dict) -> dict:
    hub.require_idle()
    sub = payload.get("substrate") or {}
    try:
        w, h = float(sub.get("w", 25.0)), float(sub.get("h", 25.0))
        bridges = scrubmod.clean_bridges(payload.get("bridges"))
    except (scrubmod.ScrubError, TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not (1.0 <= w <= 100.0 and 1.0 <= h <= 100.0):
        raise HTTPException(400, "the pattern area must be 1 to 100 mm on each side")
    _patch_calibration({"scrub": {"area": {"w": w, "h": h}, "bridges": bridges}})
    return _scrub_info()


@app.post("/api/scrub/start-here")
def scrub_start_here() -> dict:
    hub.require_idle()
    p = hub.printer.position()
    if not all(a in p.homed for a in "xyz"):
        raise HTTPException(400, "Home X and Y and set Z first.")
    start = {"x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3), "ref": hub.guard.ref}
    _patch_calibration({"scrub": {"start": start}})
    return _scrub_info()


@app.post("/api/scrub/clear-start")
def scrub_clear_start() -> dict:
    hub.require_idle()
    _patch_calibration({"scrub": {"start": None}})
    return _scrub_info()


@app.post("/api/scrub/run")
def scrub_run() -> dict:
    hub.require_idle()
    try:
        hub.thread = hub.scrub.start(_scrub_settings(), hub.guard.ref, hub.guard.floor)
    except scrubmod.ScrubError as exc:
        raise HTTPException(400, str(exc)) from exc
    return hub.scrub.status.as_dict()


@app.post("/api/scrub/stop")
def scrub_stop() -> dict:
    hub.scrub.stop()
    return hub.scrub.status.as_dict()


def _routine_info() -> dict:
    info = routinemod.describe(_auto_settings(), _scrub_settings(), hub.machine, hub.guard.ref, hub.guard.floor)
    info["status"] = hub.routine.status.as_dict()
    return info


@app.get("/api/routine1")
def routine_info() -> dict:
    return _routine_info()


@app.get("/api/routine1/status")
def routine_status() -> dict:
    return hub.routine.status.as_dict()


@app.post("/api/routine1/run")
def routine_run() -> dict:
    hub.require_idle()
    try:
        hub.thread = hub.routine.start(_auto_settings(), _scrub_settings(), hub.guard.ref, hub.guard.floor)
    except (routinemod.RoutineError, scrubmod.ScrubError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return hub.routine.status.as_dict()


@app.post("/api/routine1/stop")
def routine_stop() -> dict:
    hub.routine.stop()
    return hub.routine.status.as_dict()


# ---------- alignment: one scan, one placed toolpath per substrate ----------

def _align_settings() -> dict:
    return alignmod.merged(hub.cal.raw.get("alignment"))


def _camera_axes() -> tuple[corrmod.PrinterAxes | None, str]:
    try:
        return corrmod.PrinterAxes.from_detection(hub.cal.detection), ""
    except corrmod.AxesError as exc:
        return None, str(exc)


def _camera_points() -> dict:
    m = dict(hub.cal.detection.get("camera_to_machine") or {})
    return {"points_px": list(m.get("points_px") or []), "points_mm": list(m.get("points_mm") or []),
            "photo_size": list(m.get("photo_size") or corrmod.GUI_PHOTO_SIZE)}


def _scrub_tab_pattern() -> dict:
    st = _scrub_settings()
    return {"name": "Scrub tab sketch", "area": dict(st["area"]), "bridges": list(st["bridges"])}


def _substrate(sid: str) -> tuple[alignmod.Placement, dict]:
    if not hub.align_scan:
        raise HTTPException(400, "Scan the substrates first.")
    if hub.align_scan.get("result") is None:
        raise HTTPException(400, "Teach the camera points first, then scan again.")
    for p in hub.align_scan["result"].placements:
        if p.id == sid:
            pattern = _align_settings()["patterns"].get(sid) or _scrub_tab_pattern()
            return p, pattern
    raise HTTPException(404, f"no substrate {sid} in the last scan")


def _align_wording(problem: str) -> str:
    if problem.startswith("Jog to the top-left"):
        return ("Set the contact Z first: on the Scrub tab, jog the swab down to contact height and press "
                "Set start here. Only its Z is used; X and Y come from the camera.")
    if problem.startswith("The pattern area runs outside"):
        return "This substrate's toolpath runs outside the machine limits."
    return problem


def _substrate_scrub(sid: str, force: bool = False) -> dict:
    p, pattern = _substrate(sid)
    if p.warnings and not force:
        raise HTTPException(400, f"{sid}: " + " ".join(p.warnings))
    return alignmod.scrub_settings(_scrub_settings(), pattern, p)


def _align_info() -> dict:
    import numpy as np

    st = _align_settings()
    axes, axes_err = _camera_axes()
    pts = _camera_points()
    info: dict[str, Any] = {
        "axes": hub.machine.axes,
        "settings": {k: st[k] for k in ("size_tolerance_mm", "max_rotation_deg")},
        "patterns": {k: {"name": v.get("name", ""), "area": v["area"], "bridges": len(v.get("bridges") or [])}
                     for k, v in st["patterns"].items()},
        "scrub_tab": {"area": _scrub_settings()["area"], "bridges": len(_scrub_settings()["bridges"])},
        "camera": {"points": [{"px": px, "mm": mm,
                               "px_view": (corrmod.get_correction(pts["photo_size"]).to_corrected([px])[0].round(2).tolist()
                                           if corrmod.supported(pts["photo_size"]) else None)}
                              for px, mm in zip(pts["points_px"], pts["points_mm"])],
                   "photo_size": pts["photo_size"],
                   "fit": axes.report() if axes else None, "error": axes_err},
        "scan": None,
        "scrub_start": _scrub_settings()["start"],
        "scrub_status": hub.scrub.status.as_dict(),
        "routine_status": hub.routine.status.as_dict(),
    }
    if hub.align_scan and hub.align_scan.get("result") is None:
        sc = hub.align_scan
        subs = []
        for n, v in enumerate(sc["views"], 1):
            sid = f"S{n}"
            pat = st["patterns"].get(sid)
            subs.append({"id": sid, "size_px": [round(float(np.linalg.norm(v[1] - v[0])), 1),
                                                round(float(np.linalg.norm(v[3] - v[0])), 1)],
                         "crop": sc["crops"].get(sid),
                         "pattern_name": (pat or {}).get("name") or ("own pattern" if pat else "not sketched yet"),
                         "problems": ["Teach the camera points to place this substrate's toolpath on the printer."]})
        info["scan"] = {"pixel_only": True, "substrates": subs, "warnings": [] if subs else
                        ["No electrode arrays found. Check the lighting and focus on the Camera tab."],
                        "time": sc["time"], "scan_id": sc.get("scan_id"), "image": sc["image"],
                        "shown_scale": sc["shown_scale"], "stale": False}
    elif hub.align_scan:
        sc = hub.align_scan
        d = sc["result"].as_dict()
        for sub in d["substrates"]:
            sid = sub["id"]
            try:
                stt = _substrate_scrub(sid, force=True)
                bad = [_align_wording(b) for b in scrubmod.problems(stt, hub.machine, hub.guard.ref, hub.guard.floor)]
                plan = scrubmod.plan(stt)
                sub["plan"] = {k: plan[k] for k in ("rows", "hops", "scrub_mm", "est_s", "uncovered_pct")}
            except (scrubmod.ScrubError, HTTPException) as exc:
                bad = [str(getattr(exc, "detail", exc))]
            sub["problems"] = list(sub["warnings"]) + bad
            sub["crop"] = sc.get("crops", {}).get(sid)
            pat = st["patterns"].get(sid)
            sub["pattern_name"] = (pat or {}).get("name") or ("own pattern" if pat else "Scrub tab sketch")
        info["scan"] = {**d, "time": sc["time"], "scan_id": sc.get("scan_id"), "image": sc["image"],
                        "shown_scale": sc["shown_scale"], "stale": sc.get("stale", False)}
    return info


def _draw_alignment(img: Any, result: alignmod.ScanResult, box: tuple[float, float, float, float],
                    ppm: float, only: str | None = None, overview: bool = True) -> Any:
    """Draw on a BGR printer view (see alignmod.printer_view: right = +X, up = +Y):
    each array as measured (green, red if it does not match its pattern), its
    placed toolpath (white scrub, gold hops), its bridges (red), and in the
    overview the labels, a 10 mm printer grid and the +X / +Y arrows."""
    import cv2
    import numpy as np

    from papa.ops.geometry import to_machine

    x0, x1, y0, y1 = box
    H, W = img.shape[:2]
    t = max(1, round(W / 1500)) if overview else 1
    st_patterns = _align_settings()["patterns"]

    def at(mm: Any) -> Any:
        m = np.asarray(mm, dtype=np.float64).reshape(-1, 2)
        return np.c_[(m[:, 0] - x0) * ppm, (y1 - m[:, 1]) * ppm]

    def view(mm: Any) -> Any:
        return np.round(at(mm) * 8).astype(np.int32)                       # 3 bits subpixel

    def local(pts: list[tuple[float, float]], p: alignmod.Placement) -> Any:
        return [to_machine(hub.machine.axes, p.origin_mm[0], p.origin_mm[1], x, y, p.rotation_deg)
                for x, y in pts]

    if overview:                                    # printer grid: thin every 10 mm, labelled every 50
        for gx in np.arange(np.ceil(x0 / 10) * 10, x1, 10):
            u = int(round((gx - x0) * ppm))
            major = abs(gx % 50) < 1e-6
            cv2.line(img, (u, 0), (u, H), (90, 90, 90) if major else (55, 55, 55), 1)
            if major:
                cv2.putText(img, f"X{gx:g}", (u + 3, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (150, 150, 150), t)
        for gy in np.arange(np.ceil(y0 / 10) * 10, y1, 10):
            v = int(round((y1 - gy) * ppm))
            major = abs(gy % 50) < 1e-6
            cv2.line(img, (0, v), (W, v), (90, 90, 90) if major else (55, 55, 55), 1)
            if major:
                cv2.putText(img, f"Y{gy:g}", (3, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (150, 150, 150), t)

    for p in result.placements:
        if only is not None and p.id != only:
            continue
        pattern = st_patterns.get(p.id) or _scrub_tab_pattern()
        try:
            plan = scrubmod.plan(alignmod.scrub_settings(_scrub_settings(), pattern, p))
            for sg in plan["segments"]:
                colour = (60, 190, 255) if sg["kind"] == "hop" else (255, 255, 255)
                cv2.polylines(img, [view(local([(sg["x0"], sg["y0"]), (sg["x1"], sg["y1"])], p))],
                              False, colour, 1, cv2.LINE_AA, shift=3)
            for b in plan["bridges"]:
                bx0, bx1 = b["cx"] - b["l"] / 2, b["cx"] + b["l"] / 2
                by0, by1 = b["cy"] - b["w"] / 2, b["cy"] + b["w"] / 2
                ring = local([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], p)
                cv2.polylines(img, [view(ring)], True, (60, 60, 255), t, cv2.LINE_AA, shift=3)
        except scrubmod.ScrubError:
            pass
        colour = (80, 220, 80) if not p.warnings else (60, 60, 255)
        cv2.polylines(img, [view(p.corners_mm)], True, colour, t, cv2.LINE_AA, shift=3)
        o = at(p.corners_mm[0])[0]
        cv2.circle(img, (int(o[0]), int(o[1])), 3 * t, colour, -1, cv2.LINE_AA)     # pattern origin
        if overview:
            q = at(p.corners_mm)
            cv2.putText(img, f"{p.id}  {p.rotation_deg:+.2f} deg", (int(q[:, 0].min()), int(q[:, 1].min()) - 8 * t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * t, colour, t, cv2.LINE_AA)
    if overview:                                    # +X right and +Y up, exactly
        L = int(0.1 * W)
        ox, oy = int(0.04 * W), int(H - 0.06 * W)
        cv2.arrowedLine(img, (ox, oy), (ox + L, oy), (0, 0, 255), t + 1, tipLength=0.12)
        cv2.arrowedLine(img, (ox, oy), (ox, oy - L), (0, 200, 0), t + 1, tipLength=0.12)
        cv2.putText(img, "+X", (ox + L + 5 * t, oy + 5 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (0, 0, 255), t, cv2.LINE_AA)
        cv2.putText(img, "+Y", (ox - 8 * t, oy - L - 8 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (0, 200, 0), t, cv2.LINE_AA)
    return img


def _scan_views(rgb: Any, result: alignmod.ScanResult, axes_fit: corrmod.PrinterAxes,
                closeup_px: int = 360) -> tuple[Any, dict[str, Any]]:
    """The overview and one close-up per substrate, both square to the printer."""
    import cv2
    import numpy as np

    box = alignmod.field_mm(result.photo_size, axes_fit)
    ppm = min(2400 / (box[1] - box[0]), 2400 / (box[3] - box[2]),
              1 / (corrmod.get_correction(result.photo_size)._step * axes_fit.mm_per_plate_px))
    over = cv2.cvtColor(alignmod.printer_view(rgb, axes_fit, box, ppm), cv2.COLOR_RGB2BGR)
    over = _draw_alignment(over, result, box, ppm)
    crops = {}
    for p in result.placements:
        c = np.asarray(p.corners_mm)
        (cx0, cy0), (cx1, cy1) = c.min(0), c.max(0)
        m = 0.3 * max(cx1 - cx0, cy1 - cy0)
        b = (cx0 - m, cx1 + m, cy0 - m, cy1 + m)
        k = closeup_px / (b[1] - b[0])
        img = cv2.cvtColor(alignmod.printer_view(rgb, axes_fit, b, k), cv2.COLOR_RGB2BGR)
        crops[p.id] = _draw_alignment(img, result, b, k, only=p.id, overview=False)
    return over, crops


def _pixel_scan(rgb: Any, save: bool = True) -> dict:
    """A scan without camera points: the arrays are found and numbered S1, S2,
    ... in the corrected view so they can be sketched, but nothing is placed
    on the printer until the camera points are taught."""
    import base64
    import json

    import cv2
    import numpy as np

    params = ElectrodeParams.from_cfg(hub.cal.detection.get("electrodes"))
    try:
        views = alignmod.find(rgb, params)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    c = corrmod.get_correction(rgb.shape[1::-1])
    img = cv2.cvtColor(c.correct_image(rgb), cv2.COLOR_RGB2BGR)
    t = max(1, img.shape[1] // 1200)
    crops = {}
    for n, v in enumerate(views, 1):
        sid = f"S{n}"
        (x0, y0), (x1, y1) = v.min(0), v.max(0)
        m = 0.3 * max(x1 - x0, y1 - y0)
        X0, Y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
        X1, Y1 = int(min(img.shape[1], x1 + m)), int(min(img.shape[0], y1 + m))
        k = 360 / max(1, X1 - X0)
        crop = cv2.resize(img[Y0:Y1, X0:X1], (360, max(1, round((Y1 - Y0) * k))), interpolation=cv2.INTER_CUBIC)
        cv2.polylines(crop, [np.round((v - [X0, Y0]) * k * 8).astype(np.int32)], True, (80, 220, 80), 1,
                      cv2.LINE_AA, shift=3)
        ok, cb = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        crops[sid] = "data:image/jpeg;base64," + base64.b64encode(cb.tobytes()).decode()
        cv2.polylines(img, [np.round(v * 8).astype(np.int32)], True, (80, 220, 80), t, cv2.LINE_AA, shift=3)
        cv2.circle(img, tuple(np.round(v[0]).astype(int)), 3 * t, (80, 220, 80), -1, cv2.LINE_AA)
        cv2.putText(img, sid, (int(x0), int(y0) - 8 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, (80, 220, 80),
                    2 * t, cv2.LINE_AA)
    w = img.shape[1]
    out_w = min(1600, w)
    small = cv2.resize(img, (out_w, round(img.shape[0] * out_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    scan = {"result": None, "views": views, "rgb": rgb, "crops": crops, "time": time.strftime("%H:%M:%S"),
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
            "shown_scale": out_w / w}
    if save:
        d, n = _next_scan_dir()
        cv2.imwrite(str(d / "original.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / "found.jpg"), img)
        (d / "arrays.json").write_text(json.dumps({"scan_id": n, "corrected_view_corners": {
            f"S{i}": v.round(2).tolist() for i, v in enumerate(views, 1)}}, indent=2))
        scan["scan_id"] = n
    return scan


async def _grab_rgb() -> Any:
    """One full-size RGB frame, from this Pi's camera or the camera host."""
    import cv2
    import numpy as np

    base = _camera_host()
    if not base:
        return await asyncio.to_thread(lambda: hub.camera.still(corrmod.SENSOR_SIZE))
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(base + "/api/camera/snapshot", params={"width": 4056})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"camera host {base} is unreachable ({type(exc).__name__})") from exc
    if r.status_code != 200:
        raise HTTPException(502, f"camera host answered {r.status_code}")
    bgr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.get("/api/align")
def align_info() -> dict:
    return _align_info()


@app.post("/api/align/scan")
async def align_scan(payload: dict | None = None) -> dict:
    """Photo of the plate -> every electrode array -> its pattern placed on it."""
    import base64
    import json

    import cv2

    hub.require_idle()
    axes_fit, err = _camera_axes()
    rgb = await _grab_rgb()
    if axes_fit is None:
        hub.align_scan = await asyncio.to_thread(_pixel_scan, rgb, (payload or {}).get("save", True))
        return _align_info()

    def work() -> dict:
        params = ElectrodeParams.from_cfg(hub.cal.detection.get("electrodes"))
        try:
            result = alignmod.scan(rgb, hub.machine.axes, axes_fit, _align_settings(),
                                   _scrub_tab_pattern(), params)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        drawn, closeups = _scan_views(rgb, result, axes_fit)    # square to the printer's X and Y
        crops = {}
        for sid, crop in closeups.items():
            ok, cb = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            crops[sid] = "data:image/jpeg;base64," + base64.b64encode(cb.tobytes()).decode()
        w = drawn.shape[1]
        out_w = min(1600, w)
        small = cv2.resize(drawn, (out_w, round(drawn.shape[0] * out_w / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        scan = {"result": result, "time": time.strftime("%H:%M:%S"), "crops": crops, "rgb": rgb,
                "views": alignmod.find(rgb, params),
                "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                "shown_scale": out_w / w}
        if (payload or {}).get("save", True):
            d, n = _next_scan_dir()
            cv2.imwrite(str(d / "original.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(d / "aligned.jpg"), drawn)
            (d / "alignment.json").write_text(json.dumps({"scan_id": n, **result.as_dict()}, indent=2))
            scan["scan_id"] = n
        return scan

    hub.align_scan = await asyncio.to_thread(work)
    return _align_info()


@app.get("/api/align/substrate/{sid}/sketch")
def align_substrate_sketch(sid: str) -> dict:
    """One substrate from the last scan, cut out, straightened and zoomed for the
    bridge sketcher. Size: its own pattern's if it has one, else the measured
    array (needs camera points), else the last size used (the sketcher asks
    for the real size then)."""
    import base64

    import cv2
    import numpy as np

    sc = hub.align_scan
    if not sc:
        raise HTTPException(400, "Scan the substrates first.")
    ids = [f"S{n}" for n in range(1, len(sc["views"]) + 1)]
    if sid not in ids:
        raise HTTPException(404, f"no substrate {sid} in the last scan")
    st = _align_settings()
    own = st["patterns"].get(sid)
    bridges = list((own or {}).get("bridges") or [])
    axes_fit, _ = _camera_axes()
    if sc.get("result") is not None and axes_fit is not None:
        p = next(x for x in sc["result"].placements if x.id == sid)
        if own and own.get("drawn_on") != "camera view":
            w, h = float(own["area"]["w"]), float(own["area"]["h"])
            origin, rot = p.origin_mm, p.rotation_deg
        else:
            w, h = round(p.measured_mm[0], 2), round(p.measured_mm[1], 2)
            fit = alignmod.place(sid, hub.machine.axes, p.corners_mm, {"area": {"w": w, "h": h}}, {})
            origin, rot = fit.origin_mm, fit.rotation_deg
        ppm = min(40.0, 2400.0 / max(w, h))
        img = alignmod.rectify(sc["rgb"], hub.machine.axes, axes_fit, origin, rot, w, h, ppm)
        size_known = True
    else:
        view = sc["views"][ids.index(sid)]
        aspect = float(np.linalg.norm(view[1] - view[0]) / max(1e-6, np.linalg.norm(view[3] - view[0])))
        if own:
            w, h = float(own["area"]["w"]), float(own["area"]["h"])
        else:
            other = next((v for v in st["patterns"].values()), None)
            w = float(other["area"]["w"]) if other else 10.0
            h = round(w / aspect, 2)
        ppm = min(40.0, 2400.0 / max(w, h))
        img = alignmod.rectify_view(sc["rgb"], view, w, h, ppm)
        size_known = False
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"id": sid, "w": w, "h": h, "px_per_mm": ppm, "bridges": bridges, "size_known": size_known,
            "pattern_name": (own or {}).get("name") or "",
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}


@app.post("/api/align/settings")
def align_settings(payload: dict) -> dict:
    hub.require_idle()
    ranges = {"size_tolerance_mm": (0.01, 5.0), "max_rotation_deg": (0.1, 45.0)}
    updates = {}
    for key, (lo, hi) in ranges.items():
        if key in payload:
            v = float(payload[key])
            if not lo <= v <= hi:
                raise HTTPException(400, f"{key} must be {lo:g} to {hi:g}")
            updates[key] = v
    if updates:
        _patch_calibration({"alignment": updates})
    _refresh_scan()
    return _align_info()


def _refresh_scan() -> None:
    """Re-place the last scan's arrays after a pattern or setting change (no new photo)."""
    if not hub.align_scan or hub.align_scan.get("result") is None:
        return
    r: alignmod.ScanResult = hub.align_scan["result"]
    st = _align_settings()
    for i, p in enumerate(r.placements):
        own = st["patterns"].get(p.id)
        r.placements[i] = alignmod.place(p.id, hub.machine.axes, p.corners_mm, own or _scrub_tab_pattern(), st)
        r.pattern_source[p.id] = "own" if own else "scrub tab"
    hub.align_scan["stale"] = True      # the picture still shows the old toolpaths


@app.post("/api/align/pattern")
def align_pattern(payload: dict) -> dict:
    """Give substrates a pattern. Body: targets ["S1", ...] and one of
    sketch (a bridge-sketcher export), copy_from "S2", or clear true
    (back to the Scrub tab sketch)."""
    hub.require_idle()
    targets = [str(t) for t in payload.get("targets") or []]
    if not targets or not all(t.startswith("S") and t[1:].isdigit() for t in targets):
        raise HTTPException(400, "targets must be substrate names like S1, S2")
    patterns = dict(_align_settings()["patterns"])
    if payload.get("clear"):
        for t in targets:
            patterns.pop(t, None)
    else:
        if payload.get("copy_from"):
            src = patterns.get(str(payload["copy_from"]))
            pattern = dict(src) if src else _scrub_tab_pattern()
        elif payload.get("sketch"):
            pattern = alignmod.pattern_from_sketch(payload["sketch"], str(payload.get("name") or ""))
            if hub.align_scan and hub.align_scan.get("result") is None:
                pattern["drawn_on"] = "camera view"
        else:
            raise HTTPException(400, "send sketch, copy_from or clear")
        try:
            pattern["bridges"] = scrubmod.clean_bridges(pattern["bridges"])
        except scrubmod.ScrubError as exc:
            raise HTTPException(400, str(exc)) from exc
        w, h = pattern["area"]["w"], pattern["area"]["h"]
        if not (1.0 <= w <= 100.0 and 1.0 <= h <= 100.0):
            raise HTTPException(400, "the pattern area must be 1 to 100 mm on each side")
        for t in targets:
            patterns[t] = pattern
    raw = dict(hub.cal.raw.get("alignment") or {})
    raw["patterns"] = patterns
    _patch_calibration({"alignment": raw})
    _refresh_scan()
    return _align_info()


@app.post("/api/align/camera/teach")
def align_teach(payload: dict) -> dict:
    """Pair a clicked camera pixel with where the gantry is now (the tool tip
    jogged over the same mark)."""
    hub.require_idle()
    try:
        px = [float(payload["px"][0]), float(payload["px"][1])]
        if payload.get("view") == "corrected":
            # clicked on the corrected view: find which photo size makes a view that big
            want = (int(payload["image_size"][0]), int(payload["image_size"][1]))
            size = next((list(sz) for sz in corrmod.PHOTO_MODES
                         if corrmod.get_correction(sz).output_size == want), None)
            if size is None:
                raise HTTPException(400, f"no camera frame size gives a {want[0]} x {want[1]} corrected view")
            px = corrmod.get_correction(size).to_original([px])[0].tolist()
        else:
            size = [int(payload["photo_size"][0]), int(payload["photo_size"][1])]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise HTTPException(400, "send px [x, y] and photo_size [w, h]") from exc
    if not corrmod.supported(size):
        raise HTTPException(400, f"no lens calibration for {size[0]} x {size[1]} frames")
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy"):
        raise HTTPException(400, "Home X and Y first.")
    pts = _camera_points()
    if pts["points_px"] and list(pts["photo_size"]) != size:
        raise HTTPException(400, f"the saved points were clicked on {pts['photo_size'][0]} x "
                                 f"{pts['photo_size'][1]} frames; clear them to change size")
    pts["points_px"].append([round(px[0], 2), round(px[1], 2)])
    pts["points_mm"].append([round(p.x, 3), round(p.y, 3)])
    pts["photo_size"] = size
    _patch_calibration({"detection": {"camera_to_machine": pts}})
    return _align_info()


@app.post("/api/align/camera/delete")
def align_teach_delete(payload: dict) -> dict:
    hub.require_idle()
    pts = _camera_points()
    if payload.get("all"):
        pts["points_px"], pts["points_mm"] = [], []
    else:
        i = int(payload.get("index", -1))
        if not 0 <= i < len(pts["points_px"]):
            raise HTTPException(400, "no such camera point")
        del pts["points_px"][i]
        del pts["points_mm"][i]
    _patch_calibration({"detection": {"camera_to_machine": pts}})
    return _align_info()


@app.post("/api/align/scrub")
def align_scrub(payload: dict) -> dict:
    """Scrub one substrate with the swab already in the jaws."""
    hub.require_idle()
    st = _substrate_scrub(str(payload.get("id")), bool(payload.get("force")))
    try:
        hub.thread = hub.scrub.start(st, hub.guard.ref, hub.guard.floor)
    except scrubmod.ScrubError as exc:
        raise HTTPException(400, str(exc)) from exc
    return hub.scrub.status.as_dict()


@app.post("/api/align/routine")
def align_routine(payload: dict) -> dict:
    """Routine 1 (grip, scrub, dispose) on one substrate."""
    hub.require_idle()
    st = _substrate_scrub(str(payload.get("id")), bool(payload.get("force")))
    try:
        hub.thread = hub.routine.start(_auto_settings(), st, hub.guard.ref, hub.guard.floor)
    except (routinemod.RoutineError, scrubmod.ScrubError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return hub.routine.status.as_dict()


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
