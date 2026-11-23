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
from papa.ops import zhome as zhomemod
from papa.ops import alignment as alignmod
from papa.ops import reference as refmod
from papa.vision import correction as corrmod
from papa.vision.electrodes import ElectrodeParams
from papa.ops.guard import Guard, GuardedPrinter, ZFloorError
from papa.ops.jobs import STATE_DIR, Job
from papa.ops.machine import Orchestrator, Status
from papa.ops.swabs import Rack
from papa.ops.zcal import GripFailed, OverForce, calibrate_z
from papa.hw.camera import CameraStuck, LockedCamera
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
        stuck = getattr(self._obj, "stuck", None) if self._obj is not None else None
        return {
            "real": cfgmod.is_real(self.name),
            "built": self.built,
            "error": self.error or stuck,
            "stuck": bool(stuck),
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
        self.routine2 = routinemod.Routine2Runner(self.auto, self.scrub, self.machine)
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
    threading.Thread(target=_restore_scan, daemon=True, name="restore-scan").start()


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


# ---------- software update from GitHub, from the page ----------

REPO_DIR = Path(__file__).resolve().parent.parent


def _git(*args: str, timeout: float = 60, raw: bool = False) -> str:
    import subprocess

    r = subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise HTTPException(500, (r.stderr or r.stdout).strip() or f"git {args[0]} failed")
    return r.stdout.rstrip("\n") if raw else r.stdout.strip()


def _software_info(fetch: bool = False) -> dict:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    info: dict[str, Any] = {
        "branch": branch,
        "commit": _git("log", "-1", "--format=%h"),
        "subject": _git("log", "-1", "--format=%s"),
        "date": _git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"),
        "local_changes": [line[3:] for line in
                          _git("status", "--porcelain", "--untracked-files=no", raw=True).splitlines() if line],
        "service": bool(os.environ.get("INVOCATION_ID")),     # set by systemd
    }
    if fetch:
        _git("fetch", "--quiet", "origin", branch, timeout=120)
        new = _git("log", "--format=%h %s", f"HEAD..origin/{branch}")
        info["new"] = new.splitlines() if new else []
    return info


@app.get("/api/software")
def software_info() -> dict:
    return _software_info()


@app.post("/api/software/check")
def software_check() -> dict:
    """Ask GitHub what is new on this branch (nothing is changed)."""
    return _software_info(fetch=True)


@app.post("/api/software/restart")
def software_restart() -> dict:
    """Restart the GUI server (clears a stuck camera or any device that hung)."""
    hub.require_idle()
    if not _restart_soon():
        raise HTTPException(400, "This server is not running as the papa service, so it cannot restart itself.")
    return {"restarting": True, "message": "Restarting, the page reconnects by itself."}


@app.post("/api/software/update")
def software_update() -> dict:
    """Pull the new code from GitHub, install any new Python packages, and
    restart. Only moves forward along the branch: edits made on the Pi are
    never overwritten (the update stops and says which files)."""
    import subprocess
    import sys

    hub.require_idle()
    info = _software_info(fetch=True)
    if info["local_changes"]:
        raise HTTPException(409, "Files were edited on the Pi, so the update stopped to keep them: "
                                 + ", ".join(info["local_changes"]))
    if not info["new"]:
        return {**info, "updated": False, "message": "Already up to date."}
    before = _git("rev-parse", "HEAD")
    _git("merge", "--ff-only", f"origin/{info['branch']}")
    changed = _git("diff", "--name-only", before, "HEAD").splitlines()
    log = []
    for req in ("requirements.txt", "requirements-pi.txt"):
        if req in changed and (REPO_DIR / req).exists():
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / req)],
                               capture_output=True, text=True, timeout=900)
            log.append(f"{req}: " + ("installed" if r.returncode == 0 else (r.stderr or r.stdout)[-400:]))
    after = _software_info()
    restart = after["service"]
    if restart:
        # exit with an error code a moment after answering: systemd (Restart=on-failure)
        # starts the server again on the new code within a few seconds
        threading.Timer(1.5, lambda: os._exit(3)).start()
    return {**after, "updated": True, "applied": info["new"], "packages": log, "restarting": restart,
            "message": "Updated. Restarting, the page reconnects by itself." if restart else
                       "Updated. Restart the server to run the new code."}


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
    hub.routine2.stop()
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
            "grams": round(lc.grams(), 3) if scaled else None,
            "newtons": round(lc.newtons(), 3) if scaled else None}


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
    getattr(hub.loadcell, "settle", lambda: True)()
    counts = hub.loadcell.tare()
    _patch_calibration({"load_cell": {"tare_counts": int(counts)}})
    return {"tare_counts": counts, "saved": True}


@app.post("/api/loadcell/scale")
def scale(payload: dict) -> dict:
    hub.require_idle()
    hub.stop_sampler()
    g = float(payload.get("grams", 0))
    return {"scale_counts_per_g": hub.loadcell.calibrate_with_known_weight(g)}


@app.post("/api/loadcell/calibrate")
def loadcell_calibrate(payload: dict) -> dict:
    """Scale from a known mass pressing on the cell, saved at once together with
    the tare it was measured against. Nothing else to press, and it survives
    a restart."""
    hub.require_idle()
    hub.stop_sampler()
    g = float(payload.get("grams", 0))
    if not g > 0:
        raise HTTPException(400, "Enter the known mass in grams")
    lc = hub.loadcell
    sc = lc.calibrate_with_known_weight(g)
    if abs(sc) < 1e-6:
        raise HTTPException(400, "No change in the reading: is the mass on the cell, and was the cell tared empty first?")
    _patch_calibration({"load_cell": {"tare_counts": int(getattr(lc, "offset", 0)), "scale_counts_per_g": float(sc)}})
    return {"scale_counts_per_g": sc, "tare_counts": int(getattr(lc, "offset", 0)), "saved": True,
            "newtons_now": round(lc.newtons(), 3)}


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
            target_force_n=z["target_force_n"], abort_force_n=z["abort_force_n"],
            step_mm=step, max_travel_mm=z["max_travel_mm"],
            approach_speed_mm_s=speed, travel_speed_mm_s=sp["travel"],
            settle_readings=int(z.get("settle_readings", 3)),
            backoff_mm=z.get("backoff_mm", 0.5),
            z_speed_mm_s=min(float(sp.get("z_lift", 10.0)), 10.0),
        )
    except (GripFailed, OverForce) as exc:
        raise HTTPException(409, str(exc)) from exc
    hub.last_probe = {"contact_z": r.contact_z, "force_n": r.force_n, "steps": r.steps}
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


@app.post("/api/auto/pickup2")
def auto_pickup2() -> dict:
    """Save where the tool is now as swab position 2. Position 1 is the taught
    pickup; the step between them is what routine 2 uses for the second grip."""
    hub.require_idle()
    a = _auto_settings()
    if not a.get("pickup"):
        raise HTTPException(400, "Save the pickup (swab position 1) first")
    p = hub.printer.position()
    if not all(ax in p.homed for ax in "xy"):
        raise HTTPException(400, "Home X and Y first")
    step = {"x": round(p.x - float(a["pickup"]["x"]), 3), "y": round(p.y - float(a["pickup"]["y"]), 3)}
    _patch_calibration({"autonomous": {"pickup_step": step}})
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
    "pickup_step_x": (-50.0, 50.0), "pickup_step_y": (-50.0, 50.0),
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
    step = {k[-1]: updates.pop(k) for k in ("pickup_step_x", "pickup_step_y") if k in updates}
    if step:
        cur = dict(_auto_settings().get("pickup_step") or {"x": 0.0, "y": 0.0})
        cur.update(step)
        updates["pickup_step"] = cur
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
            return p, _pattern_of(sid)
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
    if hub.align_scan and hub.align_scan.get("result") is None and _camera_axes()[0] is not None:
        _render_scan(hub.align_scan)          # camera points were taught after this scan: place it now
    if hub.align_scan and hub.align_scan.get("result") is None:
        sc = hub.align_scan
        subs = []
        for sid, v in zip(_ids(sc), sc["views"]):
            pat = st["patterns"].get(sid)
            subs.append({"id": sid, "size_px": [round(float(np.linalg.norm(v[1] - v[0])), 1),
                                                round(float(np.linalg.norm(v[3] - v[0])), 1)],
                         "crop": sc["crops"].get(sid), "manual": sid in sc.get("manual", set()),
                         "slot": _sub_slot(sc, sid),
                         "pattern_name": (pat or {}).get("name") or ("own pattern" if pat else "not sketched yet"),
                         "problems": ["Teach the camera points to place this substrate's toolpath on the printer."]})
        info["scan"] = {"pixel_only": True, "substrates": subs, "timing": sc.get("timing"), "warnings": [] if subs else
                        ["No electrode arrays found. Check the lighting and focus on the Camera tab, or add the substrate by hand: Sketch → + Add a substrate, then click its 4 corners."],
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
            sub["manual"] = sid in sc.get("manual", set())
            sub["slot"] = _sub_slot(sc, sid)
            vm = _vs_master(sc, sid)
            sub["vs_master"] = None if vm is None else {k: vm[k] for k in
                                                        ("slot", "master_slot", "rotation_deg", "shift_mm", "step_mm", "spacing")}
            sub["bridges_from"] = _bridges_from(sid)
            mi = (sc.get("mount_info") or [None] * len(d["substrates"]))[_ids(sc).index(sid)]
            sub["mount"] = None if mi is None else {"source": mi["source"], **mi["match"].as_dict(),
                                                    "gcode_ready": _ref_ready()[0]}
            sub["gcode_from"] = ("reference G-code" if _ref_gcode() is not None and _ref_sketch() is None
                                 and _ref_ready()[0] and mi is not None
                                 else "plan")
            sub["glass_placed"] = sub.get("pattern_source") in ("reference", "glass")
            pat = st["patterns"].get(sid)
            sub["pattern_name"] = (((pat or {}).get("name") or "own pattern") + (" (on the glass)" if sub["glass_placed"] else "")
                                   ) if pat else ("Reference sketch (on the glass)" if sub["glass_placed"] else "Scrub tab sketch")
        info["scan"] = {**d, "time": sc["time"], "scan_id": sc.get("scan_id"), "image": sc["image"],
                        "timing": sc.get("timing"),
                        "shown_scale": sc["shown_scale"], "stale": sc.get("stale", False)}
    info["slot_map"] = _slot_map(hub.align_scan)
    return info


def _draw_alignment(img: Any, result: alignmod.ScanResult, box: tuple[float, float, float, float],
                    ppm: float, only: str | None = None, overview: bool = True,
                    slots: dict | None = None) -> Any:
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
        pattern = _pattern_of(p.id, result)
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
            sl = (slots or {}).get(p.id)
            cv2.putText(img, f"{p.id}{f' slot {sl}' if sl else ''}  {p.rotation_deg:+.2f} deg",
                        (int(q[:, 0].min()), int(q[:, 1].min()) - 8 * t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * t, colour, t, cv2.LINE_AA)
    if overview:                                    # +X right and +Y up, exactly
        L = int(0.1 * W)
        ox, oy = int(0.04 * W), int(H - 0.06 * W)
        cv2.arrowedLine(img, (ox, oy), (ox + L, oy), (0, 0, 255), t + 1, tipLength=0.12)
        cv2.arrowedLine(img, (ox, oy), (ox, oy - L), (0, 200, 0), t + 1, tipLength=0.12)
        cv2.putText(img, "+X", (ox + L + 5 * t, oy + 5 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (0, 0, 255), t, cv2.LINE_AA)
        cv2.putText(img, "+Y", (ox - 8 * t, oy - L - 8 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (0, 200, 0), t, cv2.LINE_AA)
    return img


def _draw_mounts(img: Any, sc: dict | None, axes_fit: corrmod.PrinterAxes, box: tuple, ppm: float,
                 only: str | None = None) -> None:
    """Each substrate's glass outline (cyan, red ring on the dot corner) and the
    reference G-code moved onto it (orange), on a printer view."""
    import cv2
    import numpy as np

    if not sc or not sc.get("mount_info"):
        return
    x0, x1, y0, y1 = box
    c = corrmod.get_correction(sc["rgb"].shape[1::-1])
    gcode = _ref_gcode()

    def at(mm: Any) -> Any:
        m = np.asarray(mm, dtype=np.float64).reshape(-1, 2)
        return np.c_[(m[:, 0] - x0) * ppm, (y1 - m[:, 1]) * ppm]

    for i, info in enumerate(sc["mount_info"]):
        sid = _sid_at(sc, i)
        if info is None or (only and sid != only):
            continue
        mm = axes_fit.plate_to_mm(c.corrected_to_plate(np.asarray(info["corners_view"])))
        t = 1 if only else max(1, img.shape[1] // 1500)
        if gcode:
            try:
                _, segs = refmod.transform_gcode(gcode, info["match"].R, info["match"].t)
                for sg in segs:
                    if sg["cut"]:
                        p = np.round(at([[sg["x0"], sg["y0"]], [sg["x1"], sg["y1"]]]) * 8).astype(np.int32)
                        cv2.line(img, tuple(p[0]), tuple(p[1]), (30, 140, 255), 1, cv2.LINE_AA, shift=3)
            except refmod.ReferenceError:
                pass
        cv2.polylines(img, [np.round(at(mm) * 8).astype(np.int32)], True, (255, 220, 60), t, cv2.LINE_AA, shift=3)
        if info.get("red_view") is not None:
            r = at(axes_fit.plate_to_mm(c.corrected_to_plate(np.asarray([info["red_view"]]))))[0]
            cv2.circle(img, (int(r[0]), int(r[1])), 8 * t, (0, 0, 255), t + 1, cv2.LINE_AA)


def _scan_views(rgb: Any, result: alignmod.ScanResult, axes_fit: corrmod.PrinterAxes,
                sc: dict | None = None, closeup_px: int = 360) -> tuple[Any, dict[str, Any]]:
    """The overview and one close-up per substrate, both square to the printer."""
    import cv2
    import numpy as np

    box = alignmod.field_mm(result.photo_size, axes_fit)
    ppm = min(2400 / (box[1] - box[0]), 2400 / (box[3] - box[2]),
              1 / (corrmod.get_correction(result.photo_size)._step * axes_fit.mm_per_plate_px))
    over = cv2.cvtColor(alignmod.printer_view(rgb, axes_fit, box, ppm), cv2.COLOR_RGB2BGR)
    if sc is not None:
        sc["draw_map"] = ("printer", box, ppm)
    slots = {sid: k for sid, k in zip(_ids(sc), sc.get("slots") or []) if sid != f"S{k}"} if sc else {}
    over = _draw_alignment(over, result, box, ppm, slots=slots)
    _draw_mounts(over, sc, axes_fit, box, ppm)
    crops = {}
    for p in result.placements:
        c = np.asarray(p.corners_mm)
        (cx0, cy0), (cx1, cy1) = c.min(0), c.max(0)
        m = 0.3 * max(cx1 - cx0, cy1 - cy0)
        b = (cx0 - m, cx1 + m, cy0 - m, cy1 + m)
        k = closeup_px / (b[1] - b[0])
        img = cv2.cvtColor(alignmod.printer_view(rgb, axes_fit, b, k), cv2.COLOR_RGB2BGR)
        crops[p.id] = _draw_alignment(img, result, b, k, only=p.id, overview=False)
        _draw_mounts(crops[p.id], sc, axes_fit, b, k, only=p.id)
    return over, crops


def _code_rev() -> str:
    """The commit this server process started from (not what is on disk now)."""
    global _BOOT_REV
    if _BOOT_REV is None:
        try:
            _BOOT_REV = _git("rev-parse", "--short", "HEAD", timeout=5).strip() or "?"
        except Exception:                                        # noqa: BLE001
            _BOOT_REV = "?"
    return _BOOT_REV


_BOOT_REV: str | None = None


def _scan_photo(rgb: Any, save: bool, t_photo: float) -> dict:
    """Find the arrays in one photo and draw the result. With camera points
    each substrate is placed on the printer; without them the arrays are
    still found and numbered S1, S2, ... so they can be sketched."""
    params = ElectrodeParams.from_cfg(hub.cal.detection.get("electrodes"))
    sc: dict[str, Any] = {"rgb": rgb, "manual": set(), "time": time.strftime("%H:%M:%S")}
    t = time.monotonic()
    if not corrmod.supported(rgb.shape[1::-1]):
        raise HTTPException(400, f"no lens calibration for {rgb.shape[1]} x {rgb.shape[0]} photos")
    view = _view_image(sc)
    t_corr = time.monotonic() - t
    t = time.monotonic()
    sc["views"] = alignmod.find(rgb, params, corrected=view)
    if not sc["views"] and params != ElectrodeParams():
        # detector settings saved on this machine found nothing: try the built-in ones
        sc["views"] = alignmod.find(rgb, ElectrodeParams(), corrected=view)
    sc["views"] = _by_slot(sc["views"])          # S1, S2, ... follow the slot numbers
    t_find = time.monotonic() - t
    t = time.monotonic()
    _render_scan(sc)
    t_draw = time.monotonic() - t
    t = time.monotonic()
    if save:
        _save_scan(sc)
    sc["timing"] = {"photo": round(t_photo, 1), "straighten": round(t_corr, 1), "find": round(t_find, 1),
                    "draw": round(t_draw, 1), "save": round(time.monotonic() - t, 1), "code": _code_rev(),
                    "size": f"{view.shape[1]}x{view.shape[0]}"}
    return sc


def _jpeg(bgr: Any, q: int = 85) -> str:
    import base64

    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


GLASS_MM = 25.0          # the glass substrate is 25 x 25 mm (the electrode array on it is smaller)


def _glass_mm() -> float:
    return float(((_ref_cfg() or {}).get("placement") or {}).get("size") or GLASS_MM)


def _on_glass(own: dict | None) -> bool:
    """An own pattern belongs on the glass if it was drawn on the glass, or
    if its area is the glass's size (a sketch of the whole glass square)."""
    if not own:
        return True
    if own.get("frame") == "glass":
        return True
    a, g = own.get("area") or {}, _glass_mm()
    return abs(float(a.get("w", 0)) - g) < 0.5 and abs(float(a.get("h", 0)) - g) < 0.5


def _glass_placements(sc: dict, result: alignmod.ScanResult) -> None:
    """Every substrate whose glass corners are known (set by hand, or
    estimated from its array through the reference) is sketched and run on
    its glass: the sketch covers the whole 25 x 25 mm glass with (0, 0) at
    its top-left corner, and the array is wherever it is on that glass.
    Placed by the glass-vs-reference match when there is a reference, else
    by the glass corners themselves."""
    import numpy as np

    from papa.ops.geometry import axis_matrix

    sk, r = _ref_sketch(), _ref_cfg()
    st = _align_settings()
    bx, by = axis_matrix(hub.machine.axes)
    axes_fit, _ = _camera_axes()
    G = _glass_mm()
    mounts = sc.get("mounts") or []
    corr_ = corrmod.get_correction(sc["rgb"].shape[1::-1])
    for i, p in enumerate(result.placements):
        mi = (sc.get("mount_info") or [None] * len(result.placements))[i]
        mv = mounts[i] if i < len(mounts) else None
        own = st["patterns"].get(p.id)
        if p.id not in sc.get("mount_manual", set()) or _master().get("array_mm"):
            continue                            # mounts are identical: the array, not the glass, places the path
        if (mi is None and mv is None) or not _on_glass(own):
            continue                            # no glass known, or an own sketch drawn on the array
        area = (own or sk or {"area": {"w": G, "h": G}})["area"]
        w, h = float(area["w"]), float(area["h"])
        if mi is not None and r.get("placement"):
            mt = mi["match"]
            g = refmod.glass_placement(r["placement"], mt, bx, by)
            c, origin, rot, rms = np.asarray(g["corners"]), g["origin"], g["rotation_deg"], mt.rms_mm
            warns = list(mt.warnings) + g["warnings"]
            lim = float(st["max_rotation_deg"])
            if abs(mt.rotation_deg) > lim and not mt.red_found:
                warns.append(f"Glass turned {mt.rotation_deg:+.1f} deg from the reference, more than {lim:g} deg.")
        elif axes_fit is not None:
            glass_mm = axes_fit.plate_to_mm(corr_.corrected_to_plate(np.asarray(mv, np.float64)))
            q = alignmod.place(p.id, hub.machine.axes, glass_mm, {"area": {"w": G, "h": G}}, st)
            c, origin, rot, rms = np.asarray(q.corners_mm), q.origin_mm, q.rotation_deg, 0.0
            warns = [w_.replace("pattern is", "the glass is") for w_ in q.warnings]
        else:
            continue
        mw = float((np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2)
        mh = float((np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2)
        result.placements[i] = alignmod.Placement(
            p.id, c.tolist(), c.mean(0).tolist(), origin, rot, [mw, mh], [w, h], [mw - w, mh - h], rms, warns)
        result.pattern_source[p.id] = "glass" if (own or sk is None) else "reference"


def _pattern_of(sid: str, result: alignmod.ScanResult | None = None) -> dict:
    """The pattern a placed substrate runs: its own, else the reference sketch
    (when placed on its glass), else the Scrub tab's."""
    own = _align_settings()["patterns"].get(sid)
    if own:
        return own
    res = result or (hub.align_scan or {}).get("result")
    if res is not None and res.pattern_source.get(sid) == "reference" and _ref_sketch() is not None:
        sk = _ref_sketch()
        return {"name": sk.get("name") or "reference sketch", "area": sk["area"], "bridges": sk["bridges"]}
    if res is not None and res.pattern_source.get(sid) == "glass":
        return {"name": "glass (no bridges sketched yet)", "area": {"w": _glass_mm(), "h": _glass_mm()},
                "bridges": [], "frame": "glass"}
    return _default_pattern()


# ---------- slots: fixed numbers for the holder positions on the plate ----------

def _slot_txt(sc: dict, sid: str) -> str:
    k = _sub_slot(sc, sid) if sc.get("views") is not None else None
    return f", slot {k}" if k else ""


def _slots_file() -> Path:
    return STATE_DIR / "slots.json"


def _slots() -> dict | None:
    """The holder grid: {rows, cols, corners} with corners the grid's outer
    corners (top-left, top-right, bottom-right, bottom-left) in the
    straightened camera view. The camera is fixed, so it is set once."""
    import json

    try:
        sl = json.loads(_slots_file().read_text())
    except (OSError, ValueError):
        return None
    if not sl.get("corners") or len(sl["corners"]) != 4:
        return None
    sl["corners"] = _untwist(sl["corners"])
    return sl


def _crosses(a: Any, b: Any, c: Any, d: Any) -> bool:
    """Segments ab and cd cross."""
    def side(p: Any, q: Any, r: Any) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _untwist(corners: Any) -> list:
    """Four grid corners as a simple quadrilateral in the clicked order's sense:
    if the last two were given the other way round (top-left, top-right,
    bottom-left, bottom-right, i.e. in reading order) they are swapped back."""
    c = [list(map(float, p)) for p in corners]
    if _crosses(c[1], c[2], c[3], c[0]):      # e.g. TL, TR, BL, BR (reading order): last two reversed
        c[2], c[3] = c[3], c[2]
    elif _crosses(c[0], c[1], c[2], c[3]):    # e.g. TL, BR, TR, BL: middle two reversed
        c[1], c[2] = c[2], c[1]
    return c


def _screen_order(pts: Any) -> Any:
    """Clicked corners in any order -> top-left, top-right, bottom-right,
    bottom-left as seen on the picture (image y grows downward)."""
    import numpy as np

    p = np.asarray(pts, np.float64)
    m = p.mean(0)
    ang = np.arctan2(p[:, 1] - m[1], p[:, 0] - m[0])       # clockwise on screen
    p = p[np.argsort(ang)]
    k = int(np.argmin(p[:, 0] + p[:, 1]))                   # top-left: smallest x + y
    return np.roll(p, -k, axis=0)


def _slot_of_views(views: list, sl: dict | None = None) -> list[int | None]:
    """Slot number (1 = top-left, reading left to right, then down) of each
    array's centre, or None outside the grid."""
    import cv2
    import numpy as np

    sl = sl if sl is not None else _slots()
    if not sl:
        return [None] * len(views)
    rows, cols = int(sl["rows"]), int(sl["cols"])
    H = cv2.getPerspectiveTransform(np.float32(sl["corners"]),
                                    np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]))
    out: list[int | None] = []
    for v in views:
        u, w = cv2.perspectiveTransform(np.float64([[np.asarray(v, np.float64).mean(0)]]), H)[0, 0]
        if -0.25 <= u <= cols + 0.25 and -0.25 <= w <= rows + 0.25:
            c, r = min(cols - 1, max(0, int(u))), min(rows - 1, max(0, int(w)))
            out.append(r * cols + c + 1)
        else:
            out.append(None)
    return out


def _by_slot(views: list) -> list:
    slots = _slot_of_views(views)
    if not any(s is not None for s in slots):
        return views
    order = sorted(range(len(views)), key=lambda i: (slots[i] is None, slots[i] or 0, i))
    return [views[i] for i in order]


def _ids(sc: dict) -> list[str]:
    """Each array's name. With a slot grid, a substrate is named after its
    slot (one substrate in slot 6 is S6); one outside the grid gets the next
    number after the grid's last slot. Without a grid: S1, S2, ... in order."""
    views = sc.get("views") or []
    sl = sc.get("slots")
    if sl is None or len(sl) != len(views):
        sl = _slot_of_views(views)
    if not any(k is not None for k in sl):
        return [f"S{i + 1}" for i in range(len(views))]
    g = _slots() or {"rows": 0, "cols": 0}
    nxt = max(int(g["rows"]) * int(g["cols"]), max(k for k in sl if k is not None))
    out, used = [], set()
    for k in sl:
        if k is None or k in used:
            nxt += 1
            k = nxt
        used.add(k)
        out.append(f"S{k}")
    return out


def _sid_at(sc: dict, i: int) -> str:
    return _ids(sc)[i]


def _sub_slot(sc: dict, sid: str) -> int | None:
    ids = _ids(sc)
    if sid not in ids:
        return None
    sl = sc.get("slots")
    i = ids.index(sid)
    return sl[i] if sl and 0 <= i < len(sl) else None


def _slot_cell_view(sc: dict, i: int) -> Any:
    """The 4 corners (view px) of the slot the i-th array sits in, or None."""
    import cv2
    import numpy as np

    sl, slots = _slots(), sc.get("slots") or []
    if not sl or i >= len(slots) or slots[i] is None:
        return None
    rows, cols = int(sl["rows"]), int(sl["cols"])
    r, c = divmod(slots[i] - 1, cols)
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]), np.float32(sl["corners"]))
    return cv2.perspectiveTransform(np.float64([[[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]]]), H)[0]


def _view_to_drawn(sc: dict, pts: Any) -> Any:
    """Straightened-view pixels -> pixels of the drawn scan picture."""
    import numpy as np

    pts = np.asarray(pts, np.float64)
    dm = sc.get("draw_map")
    if not dm or dm[0] == "view":
        return pts
    _, box, ppm = dm
    axes_fit, _ = _camera_axes()
    c = corrmod.get_correction(sc["rgb"].shape[1::-1])
    mm = axes_fit.plate_to_mm(c.corrected_to_plate(pts))
    return np.c_[(mm[:, 0] - box[0]) * ppm - 0.5, (box[3] - mm[:, 1]) * ppm - 0.5]


def _drawn_to_view(sc: dict, pts: Any) -> Any:
    import numpy as np

    pts = np.asarray(pts, np.float64)
    dm = sc.get("draw_map")
    if not dm or dm[0] == "view":
        return pts
    _, box, ppm = dm
    axes_fit, _ = _camera_axes()
    c = corrmod.get_correction(sc["rgb"].shape[1::-1])
    mm = np.c_[box[0] + (pts[:, 0] + 0.5) / ppm, box[3] - (pts[:, 1] + 0.5) / ppm]
    return (axes_fit.mm_to_plate(mm) - c._origin) / c._step


def _draw_slots(sc: dict, drawn: Any) -> None:
    """The holder grid and its slot numbers on the scan picture."""
    import cv2
    import numpy as np

    sl = _slots()
    if not sl:
        return
    rows, cols = int(sl["rows"]), int(sl["cols"])
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]),
                                    np.float32(sl["corners"]))

    def grid(uv: Any) -> Any:
        return _view_to_drawn(sc, cv2.perspectiveTransform(np.float64([uv]), H)[0])

    t = max(1, drawn.shape[1] // 1200)
    colour = (230, 200, 60)                      # BGR: light blue-cyan
    for r in range(rows + 1):
        p = grid([[0, r], [cols, r]])
        cv2.line(drawn, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), colour, t, cv2.LINE_AA)
    for c in range(cols + 1):
        p = grid([[c, 0], [c, rows]])
        cv2.line(drawn, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), colour, t, cv2.LINE_AA)
    cell = float(np.linalg.norm(grid([[0, 0], [1, 0]])[1] - grid([[0, 0], [1, 0]])[0]))
    fs = max(0.6, cell / 110)
    for r in range(rows):
        for c in range(cols):
            p = grid([[c + 0.12, r + 0.3]])[0]
            txt = str(r * cols + c + 1)
            cv2.putText(drawn, txt, tuple(np.round(p).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                        4 * t, cv2.LINE_AA)
            cv2.putText(drawn, txt, tuple(np.round(p).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, fs, colour,
                        2 * t, cv2.LINE_AA)


# ---------- master path: one calibrated automation, moved to every slot ----------

def _master_file() -> Path:
    return STATE_DIR / "master.json"


def _master() -> dict:
    import json

    try:
        return json.loads(_master_file().read_text())
    except (OSError, ValueError):
        return {}


def _save_master(m: dict) -> None:
    import json

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _master_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(m))
    tmp.replace(_master_file())


def _master_pattern() -> dict | None:
    sk = _master().get("sketch")
    if not sk or not sk.get("bridges"):
        return None
    return {"name": sk.get("name") or "master bridges", "area": dict(sk["area"]),
            "bridges": [dict(b) for b in sk["bridges"]], "master": True}


def _default_pattern() -> dict:
    """What a substrate runs unless its bridges were changed for it: the
    master bridges, else the Scrub tab sketch."""
    return _master_pattern() or _scrub_tab_pattern()


def _spacing(sc: dict | None) -> tuple[Any, Any, str]:
    """Distance from one mount to the next: (one column, one row) in printer mm,
    and where it comes from. Measured values entered for the master win;
    until then the camera's view of the slot grid."""
    sp = _master().get("spacing") or {}
    if sp.get("col") is not None or sp.get("row") is not None:
        return sp.get("col"), sp.get("row"), "entered"
    sm = _slot_map(sc) if sc else None
    if sm and ("col_step_mm" in sm):
        return sm.get("col_step_mm"), sm.get("row_step_mm"), "camera grid"
    return None, None, "none"


def _vs_master(sc: dict, sid: str) -> dict | None:
    """Where this substrate's array is compared with where the master array
    would be in this slot (the master's array moved by the mount spacing):
    its turn and X/Y shift, and the transform new = R (master + step) + t."""
    import numpy as np

    m = _master()
    if not m.get("array_mm") or m.get("slot") is None or sc is None or sc.get("result") is None:
        return None
    k = _sub_slot(sc, sid)
    sl = _slots()
    axes_fit, _ = _camera_axes()
    if k is None or not sl or axes_fit is None:
        return None
    col, row, src = _spacing(sc)
    cols = int(sl["cols"])
    r0, c0 = divmod(int(m["slot"]) - 1, cols)
    r1, c1 = divmod(k - 1, cols)
    if (c1 != c0 and col is None) or (r1 != r0 and row is None):
        return None
    step = (c1 - c0) * np.asarray(col or [0, 0], float) + (r1 - r0) * np.asarray(row or [0, 0], float)
    expected = np.asarray(m["array_mm"], float) + step
    c = corrmod.get_correction(sc["rgb"].shape[1::-1])
    view = np.asarray(sc["views"][_sid_index(sc, sid)], float)
    measured = alignmod.local_order(hub.machine.axes, axes_fit.plate_to_mm(c.corrected_to_plate(view)))
    side = float(np.linalg.norm(expected[1] - expected[0]))
    mt = refmod.match(expected, 0, measured, None, side, 1e9)
    shift = measured.mean(0) - expected.mean(0)
    return {"slot": k, "master_slot": int(m["slot"]), "step_mm": step.round(3).tolist(), "spacing": src,
            "rotation_deg": round(mt.rotation_deg, 3), "shift_mm": shift.round(3).tolist(),
            "expected_mm": expected.round(3).tolist(), "measured_mm": measured.round(3).tolist(),
            "R": mt.R, "t": mt.t + mt.R @ step}


def _bridges_from(sid: str) -> str:
    if _align_settings()["patterns"].get(sid):
        return "own"
    return "master" if _master_pattern() else "scrub tab"


@app.get("/api/master")
def master_get() -> dict:
    m = _master()
    col, row, src = _spacing(hub.align_scan)
    return {"slot": m.get("slot"), "sid": m.get("sid"), "time": m.get("time"),
            "array_mm": m.get("array_mm"), "sketch": m.get("sketch"),
            "gcode_name": m.get("gcode_name"), "gcode_lines": len((m.get("gcode") or "").splitlines()),
            "spacing": {"col": col, "row": row, "source": src},
            "ready": bool(m.get("array_mm") and (m.get("gcode") or m.get("sketch")))}


@app.post("/api/master")
def master_set(payload: dict) -> dict:
    """Set up the master path. One of:
    from: "S3"   the master automation was done on this substrate: record its
                 slot and array corners (from the last scan) and, if it has
                 its own bridges, those as the master bridges;
    sketch: {...} master bridges (bridge-sketcher file, mm from the array's top-left);
    gcode: "...", name   the calibrated G-code of the master automation;
    spacing: {col: [x, y], row: [x, y]}   measured distance between mounts (mm),
             or null to use the camera's slot grid;
    clear: "all" | "gcode" | "sketch" | "spacing"."""
    import numpy as np

    m = _master()
    if payload.get("clear"):
        what = payload["clear"]
        if what == "all":
            m = {}
        elif what in ("gcode", "sketch", "spacing"):
            m.pop(what, None)
            if what == "gcode":
                m.pop("gcode_name", None)
    elif payload.get("from"):
        sid = str(payload["from"])
        sc = _scan_or_400()
        i = _sid_index(sc, sid)
        axes_fit, _ = _camera_axes()
        if axes_fit is None:
            raise HTTPException(400, "Teach the camera points first: the master's array is recorded in printer mm.")
        k = _sub_slot(sc, sid)
        if k is None:
            raise HTTPException(400, f"{sid} is not in a slot: set the slot grid first.")
        c = corrmod.get_correction(sc["rgb"].shape[1::-1])
        arr = alignmod.local_order(hub.machine.axes,
                                   axes_fit.plate_to_mm(c.corrected_to_plate(np.asarray(sc["views"][i], float))))
        m.update({"slot": k, "sid": sid, "array_mm": arr.round(4).tolist(),
                  "time": time.strftime("%Y-%m-%d %H:%M")})
        own = _align_settings()["patterns"].get(sid)
        if own:
            m["sketch"] = {"name": own.get("name") or f"{sid} bridges", "area": dict(own["area"]),
                           "bridges": [dict(b) for b in own["bridges"]]}
    elif payload.get("sketch") is not None:
        try:
            sk = refmod.clean_sketch(payload["sketch"])
        except refmod.ReferenceError as exc:
            raise HTTPException(400, str(exc)) from exc
        m["sketch"] = {"name": str(payload.get("name") or sk.get("name") or "master bridges"),
                       "area": sk["area"], "bridges": sk["bridges"]}
    elif payload.get("gcode") is not None:
        text = str(payload["gcode"])
        try:
            refmod.transform_gcode(text)            # refuses files that cannot be moved safely
        except refmod.ReferenceError as exc:
            raise HTTPException(400, str(exc)) from exc
        m["gcode"], m["gcode_name"] = text, str(payload.get("name") or "master.gcode")
    elif "spacing" in payload:
        sp = payload["spacing"]
        if sp is None:
            m.pop("spacing", None)
        else:
            try:
                m["spacing"] = {k: (None if sp.get(k) in (None, "") else [float(sp[k][0]), float(sp[k][1])])
                                for k in ("col", "row")}
            except (TypeError, ValueError, IndexError, AttributeError) as exc:
                raise HTTPException(400, "spacing: col and row as [x, y] in mm") from exc
    else:
        raise HTTPException(400, "send from, sketch, gcode, spacing or clear")
    _save_master(m)
    if hub.align_scan:
        _render_scan(hub.align_scan)
    return {"master": master_get(), **_align_info()}


def _slot_map(sc: dict | None) -> dict | None:
    """The holder grid for the GUI's slot map: every slot with the substrate
    found in it (if any) and, with camera points, its centre on the printer
    and the X/Y step from one slot to the next."""
    import cv2
    import numpy as np

    sl = _slots()
    if not sl:
        return None
    rows, cols = int(sl["rows"]), int(sl["cols"])
    owner = {}
    if sc and sc.get("views") is not None:
        for i, k in enumerate(sc.get("slots") or []):
            if k is not None and k not in owner:
                owner[k] = _sid_at(sc, i)
    centres = None
    axes_fit, _ = _camera_axes()
    if axes_fit is not None and sc is not None:
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]),
                                        np.float32(sl["corners"]))
        uv = np.float64([[[c + 0.5, r + 0.5] for r in range(rows) for c in range(cols)]])
        view = cv2.perspectiveTransform(uv, H)[0]
        c = corrmod.get_correction(sc["rgb"].shape[1::-1])
        centres = axes_fit.plate_to_mm(c.corrected_to_plate(view))
    cells = []
    for r in range(rows):
        for cc in range(cols):
            k = r * cols + cc + 1
            cells.append({"slot": k, "row": r + 1, "col": cc + 1, "sid": owner.get(k),
                          "centre_mm": None if centres is None else [round(float(v), 3) for v in centres[k - 1]]})
    out = {"rows": rows, "cols": cols, "cells": cells}
    if centres is not None:
        g = centres.reshape(rows, cols, 2)
        out["col_step_mm"] = [round(float(v), 3) for v in (g[:, 1:] - g[:, :-1]).reshape(-1, 2).mean(0)] if cols > 1 else None
        out["row_step_mm"] = [round(float(v), 3) for v in (g[1:] - g[:-1]).reshape(-1, 2).mean(0)] if rows > 1 else None
    return out


@app.get("/api/align/slots/gcode")
def align_slots_gcode(source: str, targets: str, z: float | None = None) -> Response:
    """One substrate's toolpath (the template) repeated on other slots: its
    G-code shifted in X and Y by the distance from its slot to each target
    slot, run one slot after the other. targets: slot numbers, comma separated."""
    import numpy as np

    sc = _scan_or_400()
    sm = _slot_map(sc)
    if not sm:
        raise HTTPException(400, "Set the slot grid first (Slots → Set slot grid).")
    if "col_step_mm" not in sm:
        raise HTTPException(400, "Teach the camera points first: the slot shifts are in printer mm.")
    src_slot = _sub_slot(sc, source)
    if src_slot is None:
        raise HTTPException(400, f"{source} is not inside the slot grid.")
    try:
        want = [int(v) for v in targets.split(",") if v.strip()]
    except ValueError as exc:
        raise HTTPException(400, "targets: slot numbers like 1,2,5") from exc
    n = sm["rows"] * sm["cols"]
    if not want or any(not 1 <= k <= n for k in want):
        raise HTTPException(400, f"Slots must be 1 to {n}.")
    base = align_substrate_gcode(source, z).body.decode()
    centre = {c["slot"]: np.asarray(c["centre_mm"]) for c in sm["cells"]}
    parts = [f"; PAPA: template {source} (slot {src_slot}) repeated on slots {', '.join(map(str, want))}",
             f"; one column = X{sm['col_step_mm'][0]:+.3f} Y{sm['col_step_mm'][1]:+.3f} mm" if sm.get("col_step_mm") else "",
             f"; one row = X{sm['row_step_mm'][0]:+.3f} Y{sm['row_step_mm'][1]:+.3f} mm" if sm.get("row_step_mm") else ""]
    for k in want:
        d = centre[k] - centre[src_slot]
        head = f"\n; ===== slot {k}: template shifted X{d[0]:+.3f} Y{d[1]:+.3f} mm ====="
        try:
            text, _ = refmod.transform_gcode(base, None, d, header=head)
        except refmod.ReferenceError as exc:
            raise HTTPException(400, str(exc)) from exc
        parts.append(text)
    name = f"slots_{'-'.join(map(str, want))}_from_{source}.gcode"
    return Response("\n".join(p for p in parts if p) + "\n", media_type="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/align/slots")
def align_slots_get() -> dict:
    return {"slots": _slots()}


@app.post("/api/align/slots")
def align_slots_set(payload: dict) -> dict:
    """Set the holder grid once: rows, cols and its 4 outer corners clicked
    on the scan picture (as fractions of the shown picture, top-left,
    top-right, bottom-right, bottom-left), or {clear: true}."""
    import json

    import numpy as np

    if payload.get("clear"):
        _slots_file().unlink(missing_ok=True)
    else:
        rows, cols = int(payload.get("rows") or 0), int(payload.get("cols") or 0)
        if not (1 <= rows <= 20 and 1 <= cols <= 20):
            raise HTTPException(400, "Rows and columns must be 1 to 20.")
        old = _slots()
        if payload.get("points"):
            sc = _scan_or_400()
            pts = np.asarray(payload["points"], np.float64)
            if pts.shape != (4, 2):
                raise HTTPException(400, "Click the 4 outer corners of the holder grid.")
            dh, dw = sc["drawn"].shape[:2]
            corners = _drawn_to_view(sc, _screen_order(pts * [dw, dh])).round(2).tolist()
        elif old:
            corners = old["corners"]
        else:
            raise HTTPException(400, "Click the 4 outer corners of the holder grid on the scan first.")
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        _slots_file().write_text(json.dumps({"rows": rows, "cols": cols, "corners": corners}))
    if hub.align_scan:
        _render_scan(hub.align_scan)
    return {"slots": _slots(), **_align_info()}


def _render_scan(sc: dict) -> None:
    """(Re)place and (re)draw a scan from its photo and its arrays' corners,
    after a scan, a corner edit or an added substrate."""
    import cv2
    import numpy as np

    rgb, views = sc["rgb"], sc["views"]
    axes_fit, _ = _camera_axes()
    sc["stale"] = False
    sc["slots"] = _slot_of_views(views)
    _mounts(sc)
    if axes_fit is not None:
        result = alignmod.place_views(views, rgb.shape[1::-1], hub.machine.axes, axes_fit,
                                      _align_settings(), _default_pattern(), ids=_ids(sc))
        _glass_placements(sc, result)
        drawn, closeups = _scan_views(rgb, result, axes_fit, sc)    # square to the printer's X and Y
        sc["result"], sc["drawn"] = result, drawn
        sc["crops"] = {sid: _jpeg(img, 88) for sid, img in closeups.items()}
    else:
        drawn = cv2.cvtColor(_view_image(sc), cv2.COLOR_RGB2BGR)
        base = drawn.copy()
        t = max(1, drawn.shape[1] // 1200)
        crops = {}
        for n, (sid, v) in enumerate(zip(_ids(sc), views), 1):
            colour = (255, 170, 60) if sid in sc["manual"] else (80, 220, 80)
            (x0, y0), (x1, y1) = v.min(0), v.max(0)
            m = 0.3 * max(x1 - x0, y1 - y0)
            X0, Y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
            X1, Y1 = int(min(base.shape[1], x1 + m)), int(min(base.shape[0], y1 + m))
            k = 360 / max(1, X1 - X0)
            crop = cv2.resize(base[Y0:Y1, X0:X1], (360, max(1, round((Y1 - Y0) * k))), interpolation=cv2.INTER_CUBIC)
            cv2.polylines(crop, [np.round((v - [X0, Y0]) * k * 8).astype(np.int32)], True, colour, 1,
                          cv2.LINE_AA, shift=3)
            crops[sid] = _jpeg(crop, 88)
            cv2.polylines(drawn, [np.round(v * 8).astype(np.int32)], True, colour, t, cv2.LINE_AA, shift=3)
            cv2.circle(drawn, tuple(np.round(v[0]).astype(int)), 3 * t, colour, -1, cv2.LINE_AA)
            slot = (sc.get("slots") or [None] * n)[n - 1]
            cv2.putText(drawn, sid if slot is None or sid == f"S{slot}" else f"{sid} slot {slot}", (int(x0), int(y0) - 8 * t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, colour,
                        2 * t, cv2.LINE_AA)
        sc["result"], sc["drawn"], sc["crops"] = None, drawn, crops
        sc["draw_map"] = ("view",)
    _draw_slots(sc, sc["drawn"])
    w = sc["drawn"].shape[1]
    out_w = min(1600, w)
    small = cv2.resize(sc["drawn"], (out_w, round(sc["drawn"].shape[0] * out_w / w)), interpolation=cv2.INTER_AREA)
    sc["image"], sc["shown_scale"] = _jpeg(small), out_w / w
    try:
        _persist_scan(sc)
    except Exception as exc:                                     # noqa: BLE001
        print(f"could not keep the scan for a restart: {exc}", flush=True)


def _last_scan_dir() -> Path:
    return STATE_DIR / "scans" / "last"


def _persist_scan(sc: dict) -> None:
    """Keep the live scan (photo, arrays, hand-set corners) on disk, so a
    server restart or a software update does not lose it."""
    import json

    import cv2
    import numpy as np

    d = _last_scan_dir()
    d.mkdir(parents=True, exist_ok=True)
    if sc.get("_kept_photo") is not id(sc["rgb"]):
        tmp = d / "photo.tmp.jpg"
        cv2.imwrite(str(tmp), cv2.cvtColor(sc["rgb"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
        tmp.replace(d / "photo.jpg")
        sc["_kept_photo"] = id(sc["rgb"])
    state = {"views": [np.asarray(v).tolist() for v in sc["views"]], "manual": sorted(sc.get("manual", set())),
             "mounts": [None if m is None else np.asarray(m).tolist() for m in sc.get("mounts", [])],
             "mount_manual": sorted(sc.get("mount_manual", set())), "time": sc.get("time"),
             "scan_id": sc.get("scan_id"), "timing": sc.get("timing")}
    (d / "state.tmp.json").write_text(json.dumps(state))
    (d / "state.tmp.json").replace(d / "state.json")


def _restore_scan() -> None:
    """Bring back the last scan after a restart (see _persist_scan)."""
    import json

    import cv2
    import numpy as np

    d = _last_scan_dir()
    if hub.align_scan or not (d / "state.json").is_file() or not (d / "photo.jpg").is_file():
        return
    try:
        st = json.loads((d / "state.json").read_text())
        bgr = cv2.imread(str(d / "photo.jpg"))
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sc: dict[str, Any] = {"rgb": rgb, "_kept_photo": id(rgb), "manual": set(st.get("manual") or []),
                              "time": st.get("time") or "", "scan_id": st.get("scan_id"), "timing": st.get("timing"),
                              "views": [np.asarray(v, np.float64) for v in st.get("views") or []],
                              "mounts": [None if m is None else np.asarray(m, np.float64) for m in st.get("mounts") or []],
                              "mount_manual": set(st.get("mount_manual") or [])}
        _render_scan(sc)
        hub.align_scan = sc
        print(f"restored scan #{st.get('scan_id')} from before the restart", flush=True)
    except Exception as exc:                                     # noqa: BLE001
        print(f"could not restore the last scan: {exc}", flush=True)


def _save_scan(sc: dict) -> None:
    import json

    import cv2

    d, n = _next_scan_dir()
    cv2.imwrite(str(d / "original.jpg"), cv2.cvtColor(sc["rgb"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(d / "scan.jpg"), sc["drawn"], [cv2.IMWRITE_JPEG_QUALITY, 85])
    out = {"scan_id": n, "corrected_view_corners": {sid: v.round(2).tolist()
                                                     for sid, v in zip(_ids(sc), sc["views"])}}
    if sc.get("result") is not None:
        out.update(sc["result"].as_dict())
    (d / "alignment.json").write_text(json.dumps(out, indent=2))
    sc["scan_id"] = n


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
    hub.require_idle()
    t0 = time.monotonic()
    rgb = await _grab_rgb()
    t_photo = time.monotonic() - t0

    def work() -> dict:
        return _scan_photo(rgb, (payload or {}).get("save", True), t_photo)

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
    ids = _ids(sc)
    if sid not in ids:
        raise HTTPException(404, f"no substrate {sid} in the last scan")
    st = _align_settings()
    own = st["patterns"].get(sid)
    bridges = list((own or {}).get("bridges") or [])
    mp = _master_pattern()
    if not own and mp:
        bridges = [dict(b) for b in mp["bridges"]]          # the master bridges unless changed for this one
    axes_fit, _ = _camera_axes()
    frame = "array"
    i = ids.index(sid)
    view = np.asarray(sc["views"][i], np.float64)
    mounts = sc.get("mounts") or []
    glass_view = None                       # the glass corners in the sketch's order (TL, TR, BR, BL), view px
    if sc.get("result") is not None and axes_fit is not None:
        p = next(x for x in sc["result"].placements if x.id == sid)
        glass = sc["result"].pattern_source.get(sid) in ("reference", "glass")
        c_ = corrmod.get_correction(sc["rgb"].shape[1::-1])
        if glass:
            # the whole glass, turned the way the reference photo shows it; the
            # sketch covers the 25 x 25 mm glass with (0, 0) at its top-left corner
            frame = "glass"
            ref = _ref_sketch() or {}
            area = (own or ref or {"area": {"w": _glass_mm(), "h": _glass_mm()}})["area"]
            w, h = float(area["w"]), float(area["h"])
            origin, rot = p.origin_mm, p.rotation_deg
            if not own:
                bridges = [dict(b) for b in ref.get("bridges") or []]
            # the array in real mm on this glass (the camera points give true mm)
            from papa.ops.geometry import to_local
            arr_mm = axes_fit.plate_to_mm(c_.corrected_to_plate(view))
            local_arr = np.array([to_local(hub.machine.axes, origin[0], origin[1], x, y, rot) for x, y in arr_mm])
        elif own and own.get("drawn_on") != "camera view" and not _on_glass(own):
            w, h = float(own["area"]["w"]), float(own["area"]["h"])     # its own array-size pattern
            origin, rot = p.origin_mm, p.rotation_deg
        else:
            # no glass corners: the array at its measured size (never the 25 x 25 mm glass size)
            w, h = round(p.measured_mm[0], 2), round(p.measured_mm[1], 2)
            fit = alignmod.place(sid, hub.machine.axes, p.corners_mm, {"area": {"w": w, "h": h}}, {})
            origin, rot = fit.origin_mm, fit.rotation_deg
        ppm = min(40.0, 2400.0 / max(w, h))
        img = alignmod.rectify(sc["rgb"], hub.machine.axes, axes_fit, origin, rot, w, h, ppm)
        size_known = True
    elif i < len(mounts) and mounts[i] is not None:
        # no camera points, glass corners set: the glass gives the scale (25 x 25 mm)
        frame = "glass"
        w = h = _glass_mm()
        glass_view = alignmod.view_order(np.asarray(mounts[i], np.float64))
        ppm = min(40.0, 2400.0 / max(w, h))
        img = alignmod.rectify_view(sc["rgb"], glass_view, w, h, ppm)
        size_known = True
    else:
        aspect = float(np.linalg.norm(view[1] - view[0]) / max(1e-6, np.linalg.norm(view[3] - view[0])))
        if own and not _on_glass(own):
            w, h = float(own["area"]["w"]), float(own["area"]["h"])
        else:
            other = next((v for v in st["patterns"].values() if not _on_glass(v)), None)
            w = float(other["area"]["w"]) if other else 10.0
            h = round(w / aspect, 2)
        ppm = min(40.0, 2400.0 / max(w, h))
        img = alignmod.rectify_view(sc["rgb"], view, w, h, ppm)
        size_known = False
    array_mm, q = None, None
    if glass_view is not None:              # no camera points: the glass is the scale
        Hm = cv2.getPerspectiveTransform(np.float32(glass_view), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        q = cv2.perspectiveTransform(np.float64([view]), Hm)[0]
    elif frame == "glass":
        q = local_arr
    if q is not None:
        # the electrode array outlined where it sits on the glass
        array_mm = q.round(3).tolist()
        img = img.copy()
        t = max(1, img.shape[1] // 600)
        cv2.polylines(img, [np.round(q * ppm * 8).astype(np.int32)], True, (80, 220, 80), t, cv2.LINE_AA, shift=3)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"id": sid, "w": w, "h": h, "px_per_mm": ppm, "bridges": bridges, "size_known": size_known,
            "array_mm": array_mm, "glass_mm": _glass_mm(),
            "glass_by_hand": sid in sc.get("mount_manual", set()), "array_by_hand": sid in sc.get("manual", set()),
            "bridges_from": _bridges_from(sid), "master_name": (mp or {}).get("name"),
            "frame": frame, "from_reference": frame == "glass" and not own and _ref_sketch() is not None,
            "pattern_name": (own or {}).get("name") or "",
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}


# ---------- corners set by hand ----------

def _view_image(sc: dict) -> Any:
    if "view_rgb" not in sc:
        sc["view_rgb"] = corrmod.get_correction(sc["rgb"].shape[1::-1]).correct_image(sc["rgb"])
    return sc["view_rgb"]


def _scan_or_400() -> dict:
    if not hub.align_scan:
        raise HTTPException(400, "Scan the substrates first.")
    return hub.align_scan


def _sid_index(sc: dict, sid: str) -> int:
    ids = _ids(sc)
    if sid not in ids:
        raise HTTPException(404, f"no substrate {sid} in the last scan")
    return ids.index(sid)


def _check_corners(sc: dict, corners: Any) -> Any:
    import numpy as np

    try:
        c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "send corners as four [x, y] points") from exc
    h, w = _view_image(sc).shape[:2]
    if not (np.all(c >= -1) and np.all(c[:, 0] <= w + 1) and np.all(c[:, 1] <= h + 1)):
        raise HTTPException(400, "a corner is outside the photo")
    c = alignmod.view_order(c)
    x, y = c[:, 0], c[:, 1]
    a = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))   # shoelace
    if a < 100:
        raise HTTPException(400, "the four corners enclose almost no area")
    return c


@app.get("/api/align/photo")
def align_photo() -> Response:
    """The last scan's straightened photo at full size, as a JPEG download
    (what the array finder looked at)."""
    import cv2

    sc = _scan_or_400()
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(_view_image(sc), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    name = f"scan_{sc.get('scan_id') or 'last'}_full.jpg"
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/align/view")
def align_view(sid: str = "", margin: float = 1.5, what: str = "array") -> dict:
    """The corrected view of the last scan for picking corners: around one
    substrate (sid, with its current corners) or the whole photo (to add one).
    Coordinates in the reply: view px = picture px / scale + offset."""
    import cv2
    import numpy as np

    sc = _scan_or_400()
    view = _view_image(sc)
    H, W = view.shape[:2]
    corners = None
    context = []                            # outlines shown for reference while picking (array / glass)
    if sid:
        i = _sid_index(sc, sid)
        arr = np.asarray(sc["views"][i], np.float64)
        _mounts(sc)
        mv = sc["mounts"][i] if i < len(sc.get("mounts") or []) else None
        if what == "mount":                 # glass corners: hand-set or estimated, else start outside the array
            if mv is not None:
                v = np.asarray(mv, np.float64)
            else:
                m0 = arr.mean(0)
                v = m0 + (arr - m0) * 1.8
            context.append(arr)
        else:
            v = arr
            if mv is not None and sid in sc.get("mount_manual", set()):
                context.append(np.asarray(mv, np.float64))
        box = np.vstack([v] + context)
        cell = _slot_cell_view(sc, i)       # the whole slot (holder) when the slot grid is set
        if cell is not None:
            box = np.vstack([box, cell])
            pad = 0.08 * float(np.ptp(cell, axis=0).max())
        else:
            pad = margin * float(np.ptp(v, axis=0).max()) / 2
        (x0, y0), (x1, y1) = box.min(0), box.max(0)
        X0, Y0, X1, Y1 = int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(W, x1 + pad)), int(min(H, y1 + pad))
        corners = v
    else:
        X0, Y0, X1, Y1 = 0, 0, W, H
    k = min(1.0, 2400 / max(X1 - X0, Y1 - Y0))
    crop = cv2.resize(view[Y0:Y1, X0:X1], (max(1, round((X1 - X0) * k)), max(1, round((Y1 - Y0) * k))),
                      interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    others = [((np.asarray(o) - [X0, Y0]) * k).round(2).tolist() for i, o in enumerate(sc["views"])
              if not sid or i != _sid_index(sc, sid)]
    return {"image": _jpeg(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), 90), "offset": [X0, Y0], "scale": k,
            "context": [((o - [X0, Y0]) * k).round(2).tolist() for o in context],
            "corners": None if corners is None else ((corners - [X0, Y0]) * k).round(2).tolist(),
            "others": others}


@app.post("/api/align/substrate/{sid}/corners")
def align_set_corners(sid: str, payload: dict) -> dict:
    """Replace a substrate's corners (corrected-view px of the scan photo)."""
    hub.require_idle()
    sc = _scan_or_400()
    i = _sid_index(sc, sid)
    sc["views"][i] = _check_corners(sc, payload.get("corners"))
    sc.setdefault("manual", set()).add(sid)
    if sc.get("mounts") and sid not in sc.get("mount_manual", set()):
        sc["mounts"][i] = None              # estimated glass follows the corrected array
    _render_scan(sc)
    return _align_info()


@app.post("/api/align/substrate/add")
def align_add_substrate(payload: dict) -> dict:
    """A substrate the scan missed, from four corners picked by hand. It is
    numbered after the others."""
    hub.require_idle()
    sc = _scan_or_400()
    sc["views"].append(_check_corners(sc, payload.get("corners")))
    sc["slots"] = _slot_of_views(sc["views"])
    sid = _ids(sc)[-1]
    if sc.get("mounts") is not None:
        sc["mounts"].append(None)
    sc.setdefault("manual", set()).add(sid)
    _render_scan(sc)
    return {**_align_info(), "added": sid}


@app.post("/api/align/substrate/{sid}/remove")
def align_remove_substrate(sid: str) -> dict:
    """Drop a false find from the last scan. Later substrates move up one
    number, and so do their patterns, so each pattern stays on its substrate."""
    hub.require_idle()
    sc = _scan_or_400()
    i = _sid_index(sc, sid)
    old_ids = _ids(sc)
    slot_named = old_ids != [f"S{k + 1}" for k in range(len(old_ids))]
    manual_i = {old_ids.index(m) for m in sc.get("manual", set()) if m in old_ids and m != sid}
    mount_i = {old_ids.index(m) for m in sc.get("mount_manual", set()) if m in old_ids and m != sid}
    del sc["views"][i]
    if sc.get("mounts") is not None and i < len(sc["mounts"]):
        del sc["mounts"][i]
    sc["slots"] = _slot_of_views(sc["views"])
    new_ids = _ids(sc)
    sc["manual"] = {new_ids[j - (j > i)] for j in manual_i}
    sc["mount_manual"] = {new_ids[j - (j > i)] for j in mount_i}
    patterns = dict(_align_settings()["patterns"])
    patterns.pop(sid, None)
    if not slot_named:                      # S1, S2, ... by order: later ones move up one number
        for j in range(i + 1, len(old_ids)):
            if old_ids[j] in patterns:
                patterns[new_ids[j - 1]] = patterns.pop(old_ids[j])
    raw = dict(hub.cal.raw.get("alignment") or {})
    raw["patterns"] = patterns
    _patch_calibration({"alignment": raw})
    _render_scan(sc)
    return _align_info()


# ---------- reference substrate: known glass corners + known G-code toolpath ----------

def _ref_dir() -> Path:
    d = STATE_DIR / "reference"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ref_cfg() -> dict:
    return dict(hub.cal.raw.get("reference") or {})


def _ref_photo() -> Any:
    import cv2

    p = _ref_dir() / "photo.jpg"
    if not p.exists():
        return None
    bgr = cv2.imread(str(p))
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _ref_gcode() -> str | None:
    p = _ref_dir() / "toolpath.gcode"
    return p.read_text() if p.exists() else None


def _ref_sketch() -> dict | None:
    import json

    p = _ref_dir() / "sketch.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ref_ready() -> tuple[bool, list[str]]:
    r = _ref_cfg()
    missing = []
    if _ref_photo() is None:
        missing.append("a photo of the reference substrate")
    if not r.get("photo_corners"):
        missing.append("its 4 glass corners")
    elif r.get("red_corner") is None:
        missing.append("which corner has the red dot")
    if not r.get("placement"):
        missing.append("where that glass was on the printer for the G-code")
    if _ref_gcode() is None and _ref_sketch() is None:
        missing.append("the toolpath (a bridge sketch or a G-code file)")
    return not missing, missing


def _ref_info() -> dict:
    import cv2
    import numpy as np

    r = _ref_cfg()
    photo = _ref_photo()
    ready, missing = _ref_ready()
    info: dict[str, Any] = {"ready": ready, "missing": missing, "photo": None, "photo_size": None,
                            "corners": r.get("photo_corners"), "red_corner": r.get("red_corner"),
                            "placement": r.get("placement"), "gcode": None, "array_found": bool(r.get("array_photo")),
                            "array": r.get("array_photo")}
    sk = _ref_sketch()
    if sk is not None:
        info["sketch"] = {"name": sk.get("name") or "sketch", "w": sk["area"]["w"], "h": sk["area"]["h"],
                          "bridges": len(sk["bridges"]), "warning": None}
        if r.get("placement") and max(abs(sk["area"]["w"] - r["placement"]["size"]),
                                      abs(sk["area"]["h"] - r["placement"]["size"])) > 0.5:
            info["sketch"]["warning"] = (f"The sketch is {sk['area']['w']:g} x {sk['area']['h']:g} mm but the glass "
                                         f"is {r['placement']['size']:g} mm: the sketch should cover the whole glass, "
                                         "with (0, 0) at the photo's top-left glass corner.")
    else:
        info["sketch"] = None
    g = _ref_gcode()
    if g is not None:
        try:
            st = refmod.gcode_stats(g)
            info["gcode"] = {"name": r.get("gcode_name") or "toolpath.gcode", "moves": st.moves,
                             "x": [round(v, 3) for v in st.x], "y": [round(v, 3) for v in st.y], "z_cut": st.z_cut}
        except refmod.ReferenceError as exc:
            info["gcode"] = {"name": r.get("gcode_name"), "error": str(exc)}
    if photo is not None:
        h, w = photo.shape[:2]
        info["photo_size"] = [w, h]
        img = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
        t = max(1, w // 700)
        c = np.asarray(r["photo_corners"]) if r.get("photo_corners") else None
        if c is not None and r.get("placement") and g is not None and info["gcode"] and "error" not in info["gcode"]:
            Hinv = np.linalg.inv(refmod.photo_to_mm(c, r["placement"]))       # mm -> photo
            _, segs = refmod.transform_gcode(g)
            for sg in segs:
                pts = cv2.perspectiveTransform(np.float64([[[sg["x0"], sg["y0"]], [sg["x1"], sg["y1"]]]]), Hinv)[0]
                cv2.line(img, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)),
                         (30, 120, 255) if sg["cut"] else (200, 200, 200), 1, cv2.LINE_AA)
        if c is not None and sk is not None:
            # bridges in photo pixels: sketch (0,0) = top-left glass corner, x right, y down
            cc = refmod.image_order(c)
            ex, ey = (cc[1] - cc[0]) / sk["area"]["w"], (cc[3] - cc[0]) / sk["area"]["h"]
            for b in sk["bridges"]:
                pts = [cc[0] + (b["cx"] + dx * b["l"] / 2) * ex + (b["cy"] + dy * b["w"] / 2) * ey
                       for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (255, 0, 255), t + 1, cv2.LINE_AA)
        if r.get("array_photo"):
            cv2.polylines(img, [np.round(np.asarray(r["array_photo"])).astype(np.int32)], True, (80, 200, 80), t, cv2.LINE_AA)
        if c is not None:
            cv2.polylines(img, [np.round(c).astype(np.int32)], True, (255, 200, 0), t + 1, cv2.LINE_AA)
            for i, (x, y) in enumerate(c):
                cv2.putText(img, ["TL", "TR", "BR", "BL"][i], (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 * t, (255, 200, 0), t, cv2.LINE_AA)
            if r.get("red_corner") is not None:
                x, y = c[int(r["red_corner"])]
                cv2.circle(img, (int(x), int(y)), 14 * t, (0, 0, 255), 2 * t, cv2.LINE_AA)
        info["photo"] = _jpeg(img, 88)
    return info


@app.get("/api/reference")
def reference_info() -> dict:
    return _ref_info()


@app.post("/api/reference/photo")
def reference_photo(payload: dict) -> dict:
    """The reference substrate's photo, sent as a data URL (any camera)."""
    import base64

    import cv2
    import numpy as np

    hub.require_idle()
    try:
        data = base64.b64decode(str(payload["image"]).split(",", 1)[-1])
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, "send the photo as image: a data URL") from exc
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "that file is not a photo this can read (use JPEG or PNG)")
    scale = min(1.0, 2000 / max(bgr.shape[:2]))           # plenty for clicking corners
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(_ref_dir() / "photo.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
    found = alignmod.elec.detect_loose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    r = _ref_cfg()
    r.update({"photo_corners": None, "red_corner": None,
              "array_photo": alignmod.view_order(found[0].corners).round(2).tolist() if found else None})
    _patch_calibration({"reference": r})
    return _ref_info()


@app.post("/api/reference/corners")
def reference_corners(payload: dict) -> dict:
    """The reference glass's 4 corners on its photo (any order); the red dot
    is looked for next to them."""
    hub.require_idle()
    photo = _ref_photo()
    if photo is None:
        raise HTTPException(400, "Upload the reference photo first.")
    try:
        c = refmod.image_order(payload["corners"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "send corners as four [x, y] points") from exc
    r = _ref_cfg()
    r["photo_corners"] = c.round(2).tolist()
    r["red_corner"] = refmod.red_corner(photo, c)
    _patch_calibration({"reference": r})
    _refresh_scan()
    return _ref_info()


@app.post("/api/reference/array")
def reference_array(payload: dict) -> dict:
    """The reference electrode array's 4 corners on its photo, set by hand
    (when the automatic outline is off)."""
    hub.require_idle()
    if _ref_photo() is None:
        raise HTTPException(400, "Upload the reference photo first.")
    try:
        c = refmod.image_order(payload["corners"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "send corners as four [x, y] points") from exc
    r = _ref_cfg()
    r["array_photo"] = c.round(2).tolist()
    _patch_calibration({"reference": r})
    _refresh_scan()
    return _ref_info()


@app.post("/api/reference/red")
def reference_red(payload: dict) -> dict:
    hub.require_idle()
    i = int(payload.get("corner", -1))
    if i not in range(4):
        raise HTTPException(400, "corner is 0 (top-left), 1 (top-right), 2 (bottom-right) or 3 (bottom-left)")
    r = _ref_cfg()
    r["red_corner"] = i
    _patch_calibration({"reference": r})
    _refresh_scan()
    return _ref_info()


@app.post("/api/reference/placement")
def reference_placement(payload: dict) -> dict:
    """Where the reference glass was on the printer when the G-code was made:
    a square from (x_min, y_min), size mm a side, square to the printer."""
    hub.require_idle()
    try:
        pl = {k: float(payload[k]) for k in ("x_min", "y_min", "size")}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "send x_min, y_min and size in mm") from exc
    if not 1 <= pl["size"] <= 100:
        raise HTTPException(400, "size must be 1 to 100 mm")
    r = _ref_cfg()
    r["placement"] = pl
    _patch_calibration({"reference": r})
    _refresh_scan()
    return _ref_info()


@app.post("/api/reference/sketch")
def reference_sketch(payload: dict) -> dict:
    """The reference substrate's bridge sketch (a bridge-sketcher export),
    drawn over its whole glass: (0, 0) is the photo's top-left glass corner."""
    import json

    hub.require_idle()
    if payload.get("clear"):
        (_ref_dir() / "sketch.json").unlink(missing_ok=True)
    else:
        try:
            sk = refmod.clean_sketch(payload.get("sketch") or {})
        except refmod.ReferenceError as exc:
            raise HTTPException(400, str(exc)) from exc
        sk["name"] = str(payload.get("name") or "reference sketch")
        (_ref_dir() / "sketch.json").write_text(json.dumps(sk))
    _refresh_scan()
    return _ref_info()


@app.post("/api/reference/gcode")
def reference_gcode(payload: dict) -> dict:
    """The known toolpath for the reference substrate, as text."""
    hub.require_idle()
    if payload.get("clear"):
        (_ref_dir() / "toolpath.gcode").unlink(missing_ok=True)
        _refresh_scan()
        return _ref_info()
    text = str(payload.get("text") or "")
    try:
        refmod.gcode_stats(text)
    except refmod.ReferenceError as exc:
        raise HTTPException(400, str(exc)) from exc
    (_ref_dir() / "toolpath.gcode").write_text(text)
    r = _ref_cfg()
    r["gcode_name"] = str(payload.get("name") or "toolpath.gcode")
    _patch_calibration({"reference": r})
    _refresh_scan()
    return _ref_info()


def _mounts(sc: dict) -> None:
    """Each substrate's glass corners (estimated from its array through the
    reference, unless set by hand) and its match to the reference."""
    import numpy as np

    n = len(sc["views"])
    sc.setdefault("mounts", [None] * n)
    sc["mounts"] = (sc["mounts"] + [None] * n)[:n]
    sc.setdefault("mount_manual", set())
    sc["mount_info"] = [None] * n
    r = _ref_cfg()
    key = repr((r.get("array_photo"), r.get("photo_corners"), r.get("placement")))
    if sc.get("mount_ref_key") != key:   # reference changed: re-estimate the glass not set by hand
        sc["mounts"] = [m if _sid_at(sc, i) in sc["mount_manual"] else None for i, m in enumerate(sc["mounts"])]
        sc["mount_ref_key"] = key
    axes_fit, _ = _camera_axes()
    if axes_fit is None or not r.get("photo_corners") or not r.get("placement"):
        return
    size = sc["rgb"].shape[1::-1]
    c = corrmod.get_correction(size)

    def view_to_mm(v: Any) -> Any:
        return axes_fit.plate_to_mm(c.corrected_to_plate(np.asarray(v)))

    def mm_to_view(m: Any) -> Any:
        return (axes_fit.mm_to_plate(np.asarray(m)) - c._origin) / c._step

    ref_mm = refmod.reference_mm(r["placement"])
    H = refmod.photo_to_mm(r["photo_corners"], r["placement"])
    ref_array = None
    if r.get("array_photo"):
        import cv2
        ref_array = cv2.perspectiveTransform(np.float64([r["array_photo"]]), H)[0]
    view = _view_image(sc)
    for i, v in enumerate(sc["views"]):
        sid = _sid_at(sc, i)
        if sc["mounts"][i] is None and ref_array is not None:
            # same glass-to-array layout as the reference: move the reference glass with the array
            arr = view_to_mm(v)
            m = refmod.match(ref_array, 0, arr, None, float(np.linalg.norm(ref_array[1] - ref_array[0])), 1e9)
            sc["mounts"][i] = mm_to_view(ref_mm @ m.R.T + m.t)
        mv = sc["mounts"][i]
        if mv is None:
            continue
        mmc = view_to_mm(mv)
        red_i = refmod.red_corner(view, mv)
        try:
            mt = refmod.match(ref_mm, int(r["red_corner"]) if r.get("red_corner") is not None else 0, mmc,
                              None if red_i is None else mmc[red_i], float(r["placement"]["size"]))
        except refmod.ReferenceError:
            continue
        sc["mount_info"][i] = {"corners_view": np.asarray(mv).round(2).tolist(),
                               "source": "set by hand" if sid in sc["mount_manual"] else "estimated from the array",
                               "match": mt, "red_view": None if red_i is None else np.asarray(mv)[red_i].round(1).tolist()}


@app.post("/api/align/substrate/{sid}/mount")
def align_set_mount(sid: str, payload: dict) -> dict:
    """Set a substrate's glass corners by hand (corrected-view px of the scan photo)."""
    hub.require_idle()
    sc = _scan_or_400()
    i = _sid_index(sc, sid)
    _mounts(sc)
    sc["mounts"][i] = _check_corners(sc, payload.get("corners"))
    sc["mount_manual"].add(sid)
    _render_scan(sc)
    return _align_info()


def _ref_sketch_settings(z: float | None) -> tuple[dict, dict]:
    """Scrub settings for the reference sketch exactly where the reference glass
    sat (the known, unmodified toolpath), and the sketch."""
    import numpy as np

    from papa.ops.geometry import axis_matrix

    sk, r = _ref_sketch(), _ref_cfg()
    if sk is None or not r.get("placement") or not r.get("photo_corners"):
        raise HTTPException(400, "Finish the reference first: photo, glass corners, placement and bridge sketch.")
    same = refmod.Match(0.0, [0.0, 0.0], 0.0, float(r["placement"]["size"]), True, np.eye(2), np.zeros(2), [])
    g = refmod.glass_placement(r["placement"], same, *axis_matrix(hub.machine.axes))
    st = scrubmod.merged(_scrub_settings())
    st.update({"area": dict(sk["area"]), "bridges": [dict(b) for b in sk["bridges"]], "rotation_deg": g["rotation_deg"]})
    cz = (st.get("start") or {}).get("z")
    if cz is None:
        if z is None:
            raise HTTPException(400, "No contact Z yet: set it on the Scrub tab (Set start here) or give one.")
        cz = float(z)
    st["start"] = {"x": round(g["origin"][0], 4), "y": round(g["origin"][1], 4), "z": float(cz)}
    return st, sk


def _machine_segs(st: dict) -> list[list[float]]:
    """The contact moves of a scrub plan in printer mm: [[x0, y0, x1, y1], ...]."""
    from papa.ops.geometry import to_machine

    s, rot, out = st["start"], float(st.get("rotation_deg") or 0.0), []
    for seg in scrubmod.plan(st)["segments"]:
        if seg["kind"] == "hop":
            continue
        a = to_machine(hub.machine.axes, s["x"], s["y"], seg["x0"], seg["y0"], rot)
        b = to_machine(hub.machine.axes, s["x"], s["y"], seg["x1"], seg["y1"], rot)
        out.append([round(a[0], 3), round(a[1], 3), round(b[0], 3), round(b[1], 3)])
    return out


@app.get("/api/reference/gcode")
def reference_gcode_download(z: float | None = None) -> Response:
    """The known toolpath: the reference bridge sketch, unmodified, at the
    reference glass's place on the printer. Each substrate's G-code is this one
    turned and shifted onto its glass."""
    st, sk = _ref_sketch_settings(z)
    r = _ref_cfg()["placement"]
    head = "\n".join([
        f"; PAPA: REFERENCE toolpath from {sk.get('name') or 'the reference sketch'} (unmodified)",
        f"; reference glass: left X{r['x_min']:g}, front Y{r['y_min']:g}, {r['size']:g} mm square to the printer",
        f"; bridges: {len(sk['bridges'])}, sketch {sk['area']['w']:g} x {sk['area']['h']:g} mm",
    ])
    try:
        text = scrubmod.to_gcode(st, hub.machine.axes, header=head)
    except scrubmod.ScrubError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(text, media_type="text/plain",
                    headers={"Content-Disposition": 'attachment; filename="REFERENCE_scrub.gcode"'})


def _compare(sid: str, z: float | None = 1.0) -> dict:
    """Reference toolpath vs the one moved onto this substrate's glass."""
    from papa.ops.geometry import to_machine

    sc = _scan_or_400()
    i = _sid_index(sc, sid)
    src = sc["result"].pattern_source.get(sid) if sc.get("result") is not None else None
    if src not in ("reference", "glass"):
        raise HTTPException(400, f"{sid} is not placed on its glass (no glass match yet, or its own sketch was drawn on the "
                                 "array): open it in the sketcher and save it again to compare it to the reference.")
    ref_st, sk = _ref_sketch_settings(z)
    p, pattern = _substrate(sid)
    new_st = alignmod.scrub_settings(_scrub_settings(), pattern, p)
    new_st["start"] = {"x": p.origin_mm[0], "y": p.origin_mm[1], "z": ref_st["start"]["z"]}
    mt = sc["mount_info"][i]["match"]
    def at(st_x: float, st_y: float, rot: float, b: dict) -> tuple[float, float]:
        return to_machine(hub.machine.axes, st_x, st_y, b["cx"], b["cy"], rot)

    # pair each of this substrate's bridges with a reference bridge: same id
    # when it is close, else the nearest unpaired one within 3 mm
    ref_b, own_b = list(sk["bridges"]), list(pattern["bridges"])
    pairs, free = {}, set(range(len(ref_b)))
    for j, b in enumerate(own_b):
        d = [(((b["cx"] - r_["cx"]) ** 2 + (b["cy"] - r_["cy"]) ** 2) ** 0.5, k) for k, r_ in enumerate(ref_b) if k in free]
        same = next((k for k in free if ref_b[k]["id"] == b.get("id")), None)
        k = same if same is not None and next(x for x, kk in d if kk == same) <= 3.0 else min(d, default=(9e9, None))[1]
        if k is not None and next(x for x, kk in d if kk == k) <= 3.0:
            pairs[j] = k
            free.discard(k)
    bridges = []
    for j, b in enumerate(own_b):
        n = at(p.origin_mm[0], p.origin_mm[1], p.rotation_deg, b)
        row = {"id": b.get("id") or f"B{j + 1}", "new": [round(n[0], 3), round(n[1], 3)], "size": [b["l"], b["w"]]}
        if j in pairs:
            rb = ref_b[pairs[j]]
            a = at(ref_st["start"]["x"], ref_st["start"]["y"], ref_st["rotation_deg"], rb)
            # where the reference bridge lands on this glass with no edit: the glass move alone
            g = at(p.origin_mm[0], p.origin_mm[1], p.rotation_deg, rb)
            d = (n[0] - a[0], n[1] - a[1])
            e = (b["cx"] - rb["cx"], b["cy"] - rb["cy"])
            row.update({"ref_id": rb["id"], "ref": [round(a[0], 3), round(a[1], 3)],
                        "moved": [round(d[0], 3), round(d[1], 3)], "dist": round((d[0] ** 2 + d[1] ** 2) ** 0.5, 3),
                        "glass_only": [round(g[0], 3), round(g[1], 3)],
                        "edit": [round(e[0], 3), round(e[1], 3)], "edit_mm": round((e[0] ** 2 + e[1] ** 2) ** 0.5, 3),
                        "resized": [round(b["l"] - rb["l"], 3), round(b["w"] - rb["w"], 3)]})
        else:
            row.update({"ref_id": None, "added": True})
        bridges.append(row)
    removed = []
    for k in sorted(free):
        rb = ref_b[k]
        a = at(ref_st["start"]["x"], ref_st["start"]["y"], ref_st["rotation_deg"], rb)
        removed.append({"id": rb["id"], "ref": [round(a[0], 3), round(a[1], 3)]})
    rc = _ref_cfg()
    r = rc["placement"]
    ref_glass = refmod.reference_mm(r)
    on_glass = _array_on_glass(sc, i, rc, ref_glass, mt)
    edited = [b for b in bridges if b.get("edit_mm", 0) > 0.0005 or any(abs(v) > 0.0005 for v in b.get("resized", [0, 0]))]
    return {"id": sid, "source": src, "pattern_name": pattern.get("name") or "",
            "rotation_deg": round(mt.rotation_deg, 4), "shift_mm": [round(v, 4) for v in mt.shift_mm],
            "red_found": mt.red_found, "warnings": list(mt.warnings), "bridges": bridges, "removed": removed,
            "edited": len(edited), "added": sum(1 for b in bridges if b.get("added")),
            "max_edit_mm": max((b.get("edit_mm", 0.0) for b in bridges), default=0.0),
            "max_move_mm": max((b.get("dist", 0.0) for b in bridges), default=0.0),
            "ref_bridges": [[*at(ref_st["start"]["x"], ref_st["start"]["y"], ref_st["rotation_deg"], rb)] for rb in ref_b],
            "ref_glass": [[round(x, 3) for x in c] for c in ref_glass.tolist()],
            "new_glass": [[round(x, 3) for x in c] for c in p.corners_mm],
            "ref_segs": _machine_segs(ref_st), "new_segs": _machine_segs(new_st), "on_glass": on_glass}


def _on_glass_html(c: dict) -> str:
    """Report section: the array's turn and offset on its glass vs the reference."""
    g = c.get("on_glass")
    if not g:
        return ""

    def lr(v: float) -> str:
        return f"{v:.2f} mm right" if v >= 0 else f"{-v:.2f} mm left"

    def ud(v: float) -> str:
        return f"{v:.2f} mm up" if v >= 0 else f"{-v:.2f} mm down"

    o, rf, nf = g["offset_mm"], g["ref_from_glass_centre_mm"], g["new_from_glass_centre_mm"]
    side = (f'<p class="w">⚠ The array is on the other side of the glass ({" and ".join(g["swapped"])} swapped): '
            "the substrate is flipped or turned relative to the reference.</p>") if g["swapped"] else \
        "<p>Same side of the glass as the reference.</p>"
    est = "" if g.get("measured") else ('<p class="w">⚠ Glass corners estimated from the array: this layout equals '
                                        "the reference by definition. Set the glass corners by hand to measure it.</p>")
    return (f'<h2>Array on its glass vs the reference (JOST) substrate</h2><div class="card">'
            f'<div class="k"><div><span>Turned on the glass</span><b>{g["rotation_deg"]:+.2f}°</b></div>'
            f'<div><span>Moved left/right</span><b>{lr(o[0])}</b></div><div><span>Moved up/down</span><b>{ud(o[1])}</b></div></div>'
            f'<p class="sub">Directions as in the reference photo. Reference array centre: {lr(rf[0])}, {ud(rf[1])} of the '
            f'glass centre; this substrate: {lr(nf[0])}, {ud(nf[1])}.</p>{side}{est}</div>\n')


def _array_on_glass(sc: dict, i: int, rc: dict, ref_glass: Any, mt: Any) -> dict | None:
    """The array's turn and offset on its glass vs the reference's (see
    refmod.array_on_glass), or None without a reference array outline. Only
    meaningful when the glass corners were set by hand: a glass estimated from
    the array has the reference layout by construction."""
    import cv2
    import numpy as np

    if not rc.get("array_photo") or not rc.get("photo_corners"):
        return None
    axes_fit, _ = _camera_axes()
    if axes_fit is None:
        return None
    c = corrmod.get_correction(sc["rgb"].shape[1::-1])
    arr_mm = axes_fit.plate_to_mm(c.corrected_to_plate(np.asarray(sc["views"][i], np.float64)))
    H = refmod.photo_to_mm(rc["photo_corners"], rc["placement"])
    ref_arr = cv2.perspectiveTransform(np.float64([rc["array_photo"]]), H)[0]
    out = refmod.array_on_glass(ref_glass, ref_arr, arr_mm, mt)
    out["glass_source"] = sc["mount_info"][i].get("source")
    out["measured"] = _sid_at(sc, i) in sc.get("mount_manual", set())
    out["ref_array"] = ref_arr.round(3).tolist()
    return out


def _cmp_svg(c: dict) -> str:
    """The reference vs modified toolpath picture of a comparison, as SVG."""
    pts = [*c["ref_glass"], *c["new_glass"], *c["ref_bridges"]]
    for sg in c["ref_segs"] + c["new_segs"]:
        pts += [sg[:2], sg[2:]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs) - 3, max(xs) + 3, min(ys) - 3, max(ys) + 3
    W, H = x1 - x0, y1 - y0
    k = max(W, H) / 400

    def P(x: float, y: float) -> str:
        return f"{x - x0:.3f},{y1 - y:.3f}"

    def poly(c4: list, col: str, dash: bool) -> str:
        d = f' stroke-dasharray="{4 * k:.3f}"' if dash else ""
        return f'<polygon points="{" ".join(P(*q) for q in c4)}" fill="none" stroke="{col}" stroke-width="{2 * k:.3f}"{d}/>'

    def segs(ss: list, col: str, dash: bool) -> str:
        d = f' stroke-dasharray="{3 * k:.3f}" opacity="0.7"' if dash else ' opacity="0.95"'
        return "".join(f'<line x1="{a - x0:.3f}" y1="{y1 - b:.3f}" x2="{e - x0:.3f}" y2="{y1 - f:.3f}" stroke="{col}" '
                       f'stroke-width="{1.2 * k:.3f}"{d}/>' for a, b, e, f in ss)

    def sq(q: list, col: str) -> str:
        return (f'<rect x="{q[0] - x0 - 0.5:.3f}" y="{y1 - q[1] - 0.5:.3f}" width="1" height="1" fill="none" '
                f'stroke="{col}" stroke-width="{1.5 * k:.3f}"/>')

    out = [poly(c["ref_glass"], "#8a8a8a", True), segs(c["ref_segs"], "#9a9a9a", True)]
    out += [sq(q, "#bbbbbb") for q in c["ref_bridges"]]
    out += [f'<path d="M{b["ref"][0] - x0 - 0.8:.3f},{y1 - b["ref"][1] - 0.8:.3f}l1.6,1.6m0,-1.6l-1.6,1.6" stroke="#ff4d4d" '
            f'stroke-width="{1.5 * k:.3f}"/>' for b in c["removed"]]
    out += [poly(c["new_glass"], "#3fd0e0", False), segs(c["new_segs"], "#ff8a2a", False)]
    out += [sq(b["new"], "#4dff88" if b.get("added") else "#ffffff") for b in c["bridges"]]
    return (f'<svg viewBox="0 0 {W:.3f} {H:.3f}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;background:#0b0b0b;border-radius:6px">{"".join(out)}</svg>')


@app.get("/api/align/substrate/{sid}/report")
def align_substrate_report(sid: str, z: float | None = None) -> Response:
    """A self-contained page for showing the comparison anywhere: the original
    (bridge sketch on the reference glass) next to this calibrated substrate,
    both toolpaths overlaid, the per-bridge table, the calibration quality and
    both G-code files."""
    import base64
    import html as h

    import cv2
    import numpy as np

    sc = _scan_or_400()
    zz = z if z is not None else ((_scrub_settings().get("start") or {}).get("z"))
    if zz is None:
        raise HTTPException(400, "No contact Z yet: set it on the Scrub tab (Set start here) or give one.")
    c = _compare(sid, float(zz))
    ref = _ref_info()
    sk = _ref_sketch() or {}
    p, pattern = _substrate(sid)

    # the new substrate's glass, turned like the reference, with its bridges drawn on
    view = align_substrate_sketch(sid)
    raw = base64.b64decode(view["image"].split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    ppm = float(view["px_per_mm"])
    t = max(1, int(round(img.shape[1] / 300)))
    for b in pattern["bridges"]:
        x0_, y0_ = (b["cx"] - b["l"] / 2) * ppm, (b["cy"] - b["w"] / 2) * ppm
        cv2.rectangle(img, (int(x0_), int(y0_)), (int(x0_ + b["l"] * ppm), int(y0_ + b["w"] * ppm)), (255, 0, 255), t + 1)
    new_img = _jpeg(img, 88)

    ref_gc = reference_gcode_download(float(zz)).body.decode()
    new_gc = align_substrate_gcode(sid, float(zz)).body.decode()
    fit = (_align_info().get("camera") or {}).get("fit") or {}
    errs = fit.get("errors_mm") or []

    def dl(text: str, name: str) -> str:
        u = "data:text/plain;base64," + base64.b64encode(text.encode()).decode()
        return f'<a class="btn" download="{h.escape(name)}" href="{u}">{h.escape(name)} ↓</a>'

    def xy(v: list | None) -> str:
        return "–" if not v else f"{v[0]:.2f}, {v[1]:.2f}"

    rows = []
    for b in c["bridges"]:
        if b.get("added"):
            rows.append(f"<tr><td>{h.escape(b['id'])}</td><td>–</td><td>{xy(b['new'])}</td><td class=g>added</td><td class=g>new</td></tr>")
            continue
        ed = f"{b['edit'][0]:+.2f}, {b['edit'][1]:+.2f}" if b["edit_mm"] > 0.0005 else "–"
        if any(abs(v) > 0.0005 for v in b["resized"]):
            ed += f" · size {b['resized'][0]:+.2f} × {b['resized'][1]:+.2f}"
        rows.append(f"<tr><td>{h.escape(b['id'])}</td><td>{xy(b['ref'])}</td><td>{xy(b['new'])}</td>"
                    f"<td>{b['moved'][0]:+.2f}, {b['moved'][1]:+.2f} ({b['dist']:.2f})</td><td>{ed}</td></tr>")
    rows += [f"<tr><td>{h.escape(b['id'])}</td><td>{xy(b['ref'])}</td><td>–</td><td class=r>removed</td><td class=r>deleted</td></tr>"
             for b in c["removed"]]
    own = c["source"] == "glass"
    mi = (sc.get("mount_info") or [None] * (_sid_index(sc, sid) + 1))[_sid_index(sc, sid)]
    glass_src = (mi or {}).get("source") or "-"
    head_lines = [ln for ln in new_gc.splitlines() if ln.startswith(";")]
    warn = c["warnings"] + p.warnings
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAPA · reference vs calibrated substrate {h.escape(sid)}</title>
<style>
body{{margin:0;background:#000;color:#e8e8e8;font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 24px 60px}}
h1{{font-size:26px;letter-spacing:.06em;margin:0 0 4px}} h2{{font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:#bbb;margin:34px 0 12px;border-top:1px solid #333;padding-top:18px}}
.sub{{color:#999}} .grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px}} .grid>*{{min-width:0}} @media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:16px}} .card img{{width:100%;border-radius:6px;display:block}}
.k{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} @media(max-width:820px){{.k{{grid-template-columns:repeat(2,1fr)}}}}
.k div{{background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:12px}} .k b{{display:block;font-size:22px;color:#fff}} .k span{{color:#999;font-size:12px;letter-spacing:.08em;text-transform:uppercase}}
table{{width:100%;border-collapse:collapse;font:13px/1.4 ui-monospace,Menlo,monospace}} th,td{{padding:7px 8px;border-bottom:1px solid #222;text-align:left}} th{{color:#999;font-weight:500;font-family:inherit}}
.g{{color:#4dff88}} .r{{color:#ff6b6b}} .w{{color:#ffb35c}} pre{{background:#0b0b0b;border:1px solid #222;border-radius:6px;padding:12px;overflow:auto;max-height:360px;font:12px/1.45 ui-monospace,Menlo,monospace}}
.btn{{display:inline-block;border:1px solid #d9b45b;color:#d9b45b;padding:7px 12px;border-radius:4px;text-decoration:none;margin:4px 8px 4px 0;font-size:13px}}
.leg span{{display:inline-block;margin-right:14px;font-size:12.5px;color:#aaa}} .leg i{{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}}
</style></head><body><main>
<h1>Reference vs calibrated substrate {h.escape(sid)}{h.escape(_slot_txt(hub.align_scan or {}, sid).replace(",", " ·"))}</h1>
<div class="sub">PAPA substrate alignment · scan #{h.escape(str(sc.get('scan_id', '-')))} at {h.escape(str(sc['time']))} · toolpath from {h.escape(sk.get('name') or 'the bridge sketch')}</div>

<div class="k" style="margin-top:22px">
<div><span>Glass turned</span><b>{c['rotation_deg']:+.3f}°</b></div>
<div><span>Glass moved</span><b>{c['shift_mm'][0]:+.2f}, {c['shift_mm'][1]:+.2f} mm</b></div>
<div><span>Bridge edits</span><b>{c['edited'] if own else 0} changed · {c['added']} added · {len(c['removed'])} removed</b></div>
<div><span>Red dot</span><b class="{'' if c['red_found'] else 'w'}">{'found' if c['red_found'] else 'not found'}</b></div>
</div>
{"".join(f'<p class="w">⚠ {h.escape(x)}</p>' for x in dict.fromkeys(warn))}

<h2>1 · Original and calibrated substrate</h2>
<div class="grid">
<div class="card"><b>Original: reference substrate</b><p class="sub" style="margin:4px 0 10px">Your photo with the bridge sketch ({len(sk.get('bridges') or [])} bridges, {sk.get('area', {}).get('w', 0):g} × {sk.get('area', {}).get('h', 0):g} mm) drawn in pink; gold = glass corners, red ring = red dot.</p><img src="{ref.get('photo') or ''}" alt="Reference substrate"></div>
<div class="card"><b>Calibrated: {h.escape(sid)} from the camera</b><p class="sub" style="margin:4px 0 10px">Its glass cut from the scan, straightened in printer millimetres and turned like the reference; the bridges it runs in pink ({'its own edited sketch' if own else 'the reference bridges, unchanged'}).</p><img src="{new_img}" alt="Calibrated substrate"></div>
</div>

{_on_glass_html(c)}
<h2>2 · Toolpaths on the printer (mm, seen from above)</h2>
<div class="card">{_cmp_svg(c)}
<div class="leg" style="margin-top:10px"><span><i style="background:#9a9a9a"></i>reference glass + toolpath (original)</span><span><i style="background:#3fd0e0"></i>{h.escape(sid)}'s glass (measured)</span><span><i style="background:#ff8a2a"></i>modified toolpath</span><span>□ bridges · <span class="g">green</span> added · <span class="r">×</span> removed</span></div></div>

<h2>3 · Bridge by bridge (printer X, Y mm)</h2>
<div class="card"><table><thead><tr><th>Bridge</th><th>Original</th><th>Calibrated</th><th>Moved (total)</th><th>Sketch edit</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p class="sub" style="font-size:12.5px">Moved = original → calibrated position. Sketch edit = the part that comes from changing the bridges for this substrate; the rest is the glass being turned and shifted (measured by the camera).</p></div>

<h2>4 · How it was measured</h2>
<div class="card"><p style="margin-top:0">Camera → printer calibration: <b>{fit.get('points', 0)}</b> camera points, largest error <b>{(max(errs) if errs else 0):.2f} mm</b>{', X/Y scale differ ' + format(fit.get('xy_scale_diff_pct', 0), '.2f') + ' %' if fit.get('xy_scale_diff_pct') is not None else ''}. Glass corners: {h.escape(glass_src)}.</p>
<p class="sub">Every pixel of the fixed camera is turned into printer X/Y with those points; the substrate's glass corners are measured in each scan and matched to the reference glass (the red dot fixes which way round). The bridges are placed from that measured corner, so the toolpath follows the glass.</p></div>

<h2>5 · G-code</h2>
<div class="grid">
<div class="card"><b>Original (reference)</b><div>{dl(ref_gc, 'REFERENCE_scrub.gcode')}</div><pre>{h.escape(chr(10).join(ref_gc.splitlines()[:40]))}{chr(10) + '…' if len(ref_gc.splitlines()) > 40 else ''}</pre></div>
<div class="card"><b>Calibrated ({h.escape(sid)})</b><div>{dl(new_gc, f'{sid}_scrub.gcode')}</div><pre>{h.escape(chr(10).join(head_lines + [ln for ln in new_gc.splitlines() if not ln.startswith(';')][:40 - min(len(head_lines), 30)]))}{chr(10) + '…'}</pre></div>
</div>
</main></body></html>"""
    return Response(page, media_type="text/html",
                    headers={"Content-Disposition": f'attachment; filename="PAPA_{sid}_vs_reference.html"'})


@app.get("/api/align/substrate/{sid}/compare")
def align_substrate_compare(sid: str) -> dict:
    return _compare(sid)


@app.get("/api/align/substrate/{sid}/gcode")
def align_substrate_gcode(sid: str, z: float | None = None) -> Response:
    """This substrate's toolpath as G-code: the reference G-code file moved onto
    its glass when there is one, otherwise its scrub plan (the reference sketch
    on its glass, or its own pattern on its array) written out. z: contact Z
    when the Scrub tab has none yet."""
    sc = _scan_or_400()
    i = _sid_index(sc, sid)
    if _camera_axes()[0] is None or sc.get("result") is None:
        raise HTTPException(400, "Teach the camera points first.")
    info = (sc.get("mount_info") or [None] * (i + 1))[i]
    name = _ref_cfg().get("gcode_name") or "toolpath.gcode"
    vm, mst = _vs_master(sc, sid), _master()
    if mst.get("gcode") and vm is not None and _bridges_from(sid) != "own":
        # the master automation, turned and shifted from the master's array onto this one
        head = "\n".join([
            f"; PAPA: master path ({mst.get('gcode_name') or 'master G-code'}) moved onto {sid}" + _slot_txt(sc, sid),
            f"; scan {sc.get('scan_id', '-')} at {sc['time']}; master recorded on slot {vm['master_slot']}",
            f"; slot step X{vm['step_mm'][0]:+.3f} Y{vm['step_mm'][1]:+.3f} mm (mount spacing: {vm['spacing']})",
            f"; vs master: array turned {vm['rotation_deg']:+.3f} deg, shifted X{vm['shift_mm'][0]:+.3f} "
            f"Y{vm['shift_mm'][1]:+.3f} mm from where the master array would be"])
        try:
            text, _ = refmod.transform_gcode(mst["gcode"], vm["R"], vm["t"], header=head)
        except refmod.ReferenceError as exc:
            raise HTTPException(400, str(exc)) from exc
        return Response(text, media_type="text/plain",
                        headers={"Content-Disposition": f'attachment; filename="{sid}_{mst.get("gcode_name") or "master.gcode"}"'})
    # the bridge sketch wins: the G-code file is used only when there is no sketch
    if _ref_gcode() is not None and _ref_sketch() is None and _ref_ready()[0]:
        if info is None:
            raise HTTPException(400, f"{sid} has no glass corners yet: set them in the sketcher.")
        mt = info["match"]
        head = "\n".join([
            f"; PAPA: reference toolpath moved onto substrate {sid}" + _slot_txt(sc, sid),
            f"; scan {sc.get('scan_id', '-')} at {sc['time']}; glass corners {info['source']}",
            f"; turned {mt.rotation_deg:+.3f} deg, centre moved {mt.shift_mm[0]:+.3f}, {mt.shift_mm[1]:+.3f} mm; "
            f"fit {mt.rms_mm:.3f} mm; red dot {'found' if mt.red_found else 'NOT found'}",
        ] + [f"; WARNING: {w}" for w in mt.warnings])
        try:
            text, _ = refmod.transform_gcode(_ref_gcode() or "", mt.R, mt.t, header=head)
        except refmod.ReferenceError as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        p, pattern = _substrate(sid)
        st = alignmod.scrub_settings(_scrub_settings(), pattern, p)
        if not st.get("start") or st["start"].get("z") is None:
            if z is None:
                raise HTTPException(400, "No contact Z yet: set it on the Scrub tab (Set start here) or give one.")
            st["start"] = {"x": round(p.origin_mm[0], 4), "y": round(p.origin_mm[1], 4), "z": float(z)}
        src = sc["result"].pattern_source.get(sid)
        lines = [
            f"; PAPA: {pattern.get('name') or 'pattern'} on substrate {sid}" + _slot_txt(sc, sid)
            + {"reference": " (the REFERENCE toolpath, modified for this glass)",
               "glass": " (the reference toolpath with this substrate's own bridge edits, on its glass)"}.get(
                src, " (placed on its array)"),
            f"; scan {sc.get('scan_id', '-')} at {sc['time']}; origin X{p.origin_mm[0]:.3f} Y{p.origin_mm[1]:.3f}, "
            f"turned {p.rotation_deg:+.3f} deg"]
        if src in ("reference", "glass"):
            cmp_ = _compare(sid, st["start"]["z"])
            lines.append(f"; vs reference: glass turned {cmp_['rotation_deg']:+.3f} deg, centre moved "
                         f"{cmp_['shift_mm'][0]:+.3f}, {cmp_['shift_mm'][1]:+.3f} mm; red dot "
                         f"{'found' if cmp_['red_found'] else 'NOT found'}")
            if src == "glass":
                lines.append(f"; bridge edits vs reference: {cmp_['edited']} changed (up to {cmp_['max_edit_mm']:.3f} mm), "
                             f"{cmp_['added']} added, {len(cmp_['removed'])} removed")
            for b in cmp_["bridges"]:
                if b.get("added"):
                    lines.append(f";   {b['id']}: NEW (not in the reference) at X{b['new'][0]:.3f} Y{b['new'][1]:.3f}")
                    continue
                ed = (f"; edited {b['edit'][0]:+.3f}, {b['edit'][1]:+.3f} mm on the glass" if b["edit_mm"] > 0.0005 else "")
                rs = (f"; resized {b['resized'][0]:+.3f} x {b['resized'][1]:+.3f}" if any(abs(v) > 0.0005 for v in b["resized"]) else "")
                lines.append(f";   {b['id']}: X{b['ref'][0]:.3f} Y{b['ref'][1]:.3f} -> X{b['new'][0]:.3f} Y{b['new'][1]:.3f} "
                             f"(moved {b['moved'][0]:+.3f}, {b['moved'][1]:+.3f}; {b['dist']:.3f} mm{ed}{rs})")
            lines += [f";   {b['id']}: REMOVED (was X{b['ref'][0]:.3f} Y{b['ref'][1]:.3f})" for b in cmp_["removed"]]
        head = "\n".join(lines + [f"; WARNING: {w}" for w in p.warnings])
        try:
            text = scrubmod.to_gcode(st, hub.machine.axes, header=head)
        except scrubmod.ScrubError as exc:
            raise HTTPException(400, str(exc)) from exc
        name = "scrub.gcode"
    return Response(text, media_type="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{sid}_{name}"'})


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
    """Re-place and redraw the last scan after a pattern or setting change (no new photo)."""
    if hub.align_scan:
        _render_scan(hub.align_scan)


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
            if payload.get("frame") == "glass":
                pattern["frame"] = "glass"      # drawn over the whole glass, like the reference sketch
            elif hub.align_scan and hub.align_scan.get("result") is None:
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


# ---------- run every filled slot, in slot order; empty slots are skipped ----------

def _run_plan(sc: dict) -> list[dict]:
    """Slot 1, 2, ... in order: run the substrate found there, or skip an
    empty slot (and say why a filled one cannot run)."""
    sm = _slot_map(sc)
    if not sm:
        raise HTTPException(400, "Set the slot grid first (Slots → Set slot grid).")
    subs = {s["id"]: s for s in ((_align_info().get("scan") or {}).get("substrates") or [])}
    plan = []
    for c in sm["cells"]:
        sid = c["sid"]
        if not sid:
            plan.append({"slot": c["slot"], "sid": None, "run": False, "why": "empty"})
            continue
        s = subs.get(sid) or {}
        probs = list(s.get("problems") or [])
        plan.append({"slot": c["slot"], "sid": sid, "run": True, "problems": probs,
                     "bridges_from": s.get("bridges_from"), "vs_master": s.get("vs_master")})
    return plan


@app.get("/api/align/run-plan")
def align_run_plan() -> dict:
    sc = _scan_or_400()
    plan = _run_plan(sc)
    return {"plan": plan, "runs": [p["slot"] for p in plan if p["run"]],
            "skips": [p["slot"] for p in plan if not p["run"]], "batch": _BATCH.as_dict()}


@app.get("/api/align/run-all/gcode")
def align_run_all_gcode(z: float | None = None) -> Response:
    """One G-code file for the whole mount: each filled slot's own G-code
    (master path moved onto its array, or its own bridges) in slot order;
    empty slots are skipped."""
    sc = _scan_or_400()
    plan = _run_plan(sc)
    runs = [p for p in plan if p["run"]]
    if not runs:
        raise HTTPException(400, "No substrate in any slot.")
    parts = [f"; PAPA: all filled slots, in slot order · scan {sc.get('scan_id', '-')} at {sc['time']}",
             f"; runs slots {', '.join(str(p['slot']) for p in runs)}; empty (skipped): "
             + (", ".join(str(p['slot']) for p in plan if not p['run']) or "none")]
    for p in plan:
        if not p["run"]:
            parts.append(f"\n; ===== slot {p['slot']}: empty, skipped =====")
            continue
        parts.append(f"\n; ===== slot {p['slot']}: {p['sid']} =====")
        parts.append(align_substrate_gcode(p["sid"], z).body.decode())
    return Response("\n".join(parts) + "\n", media_type="text/plain",
                    headers={"Content-Disposition": 'attachment; filename="all_slots.gcode"'})


class _Batch:
    def __init__(self) -> None:
        self.running = False
        self.mode = ""
        self.queue: list[dict] = []
        self.current: dict | None = None
        self.done: list[dict] = []
        self.error: str | None = None
        self.stop = False

    def as_dict(self) -> dict:
        return {"running": self.running, "mode": self.mode, "current": self.current,
                "queue": [{"slot": q["slot"], "sid": q["sid"]} for q in self.queue],
                "done": self.done, "error": self.error, "stopping": self.stop and self.running}


_BATCH = _Batch()


@app.get("/api/align/run-all")
def align_run_all_status() -> dict:
    return _BATCH.as_dict()


@app.post("/api/align/run-all")
def align_run_all(payload: dict) -> dict:
    """Run every filled slot one after the other, in slot order, on the
    printer: mode "routine" (Routine 1: grip a swab, scrub, dispose) or
    "scrub" (scrub with the swab already in the jaws). Empty slots are
    skipped. Every slot is checked before anything moves; a failure stops
    the rest."""
    hub.require_idle()
    sc = _scan_or_400()
    mode = str(payload.get("mode") or "routine")
    if mode not in ("routine", "scrub"):
        raise HTTPException(400, "mode: routine or scrub")
    force = bool(payload.get("force"))
    plan = [p for p in _run_plan(sc) if p["run"]]
    only = payload.get("slots")
    if only:
        plan = [p for p in plan if p["slot"] in {int(v) for v in only}]
    if not plan:
        raise HTTPException(400, "No substrate in any slot to run.")
    jobs = []
    for p in plan:                              # check them all before moving anything
        try:
            st = _substrate_scrub(p["sid"], force)
        except HTTPException as exc:
            raise HTTPException(exc.status_code, f"Slot {p['slot']} ({p['sid']}): {exc.detail}") from exc
        jobs.append({"slot": p["slot"], "sid": p["sid"], "st": st})
    b = _BATCH
    b.__init__()
    b.running, b.mode, b.queue = True, mode, jobs[:]

    def run() -> None:
        try:
            while b.queue and not b.stop:
                job = b.queue.pop(0)
                b.current = {"slot": job["slot"], "sid": job["sid"]}
                if mode == "routine":
                    t = hub.routine.start(_auto_settings(), job["st"], hub.guard.ref, hub.guard.floor)
                    t.join()
                    err = hub.routine.status.error
                else:
                    t = hub.scrub.start(job["st"], hub.guard.ref, hub.guard.floor)
                    t.join()
                    err = getattr(hub.scrub.status, "error", None)
                err = err or None
                b.done.append({"slot": job["slot"], "sid": job["sid"], "error": err})
                if err:
                    b.error = f"Slot {job['slot']} ({job['sid']}): {err}"
                    break
            if b.stop and not b.error:
                b.error = "Stopped" + (f" before slot {b.queue[0]['slot']}" if b.queue else "")
        except Exception as exc:                 # noqa: BLE001
            b.error = str(exc) or type(exc).__name__
        finally:
            b.current, b.running = None, False

    hub.thread = threading.Thread(target=run, daemon=True, name="run-all-slots")
    hub.thread.start()
    return b.as_dict()


@app.post("/api/align/run-all/stop")
def align_run_all_stop(payload: dict | None = None) -> dict:
    """Stop the run: now (stops the slot being worked on, like Stop) or, with
    {after: true}, once the current slot is finished."""
    _BATCH.stop = True
    if not (payload or {}).get("after"):
        for r in (hub.routine, hub.scrub):
            try:
                r.stop()
            except Exception:                    # noqa: BLE001
                pass
    return _BATCH.as_dict()


# ---------- Z home on the load cell, heights, park, routine 2 ----------

def _zhome_settings() -> dict:
    return zhomemod.merged(hub.cal.raw.get("zhome"))


def _zhome_info() -> dict:
    st = _zhome_settings()
    return {"settings": st, "problems": zhomemod.problems(st, hub.guard.ref),
            "floor_would_be": zhomemod.floor_z(st), "target_force_n": hub.machine.zcal["target_force_n"],
            "abs": {k: zhomemod.absolute(st, k) for k in zhomemod.HEIGHTS},
            "positions": _setup_positions(), "loadcell": _lc_status()}


def _setup_positions() -> dict:
    a = _auto_settings()
    lc = hub.cal.load_cell
    slots = {x.id: x for x in hub.cal.slots if x.calibrated}

    def xy(o) -> dict | None:
        return None if o is None else {"x": round(float(o["x"]), 2), "y": round(float(o["y"]), 2)}

    pick = a.get("pickup")
    step = a.get("pickup_step") or {"x": 0.0, "y": 0.0}
    set2 = bool(step.get("x") or step.get("y"))
    return {
        "slide1": xy({"x": slots[1].x, "y": slots[1].y}) if 1 in slots else None,
        "slide2": xy({"x": slots[2].x, "y": slots[2].y}) if 2 in slots else None,
        "bin": xy(a.get("bin")),
        "swab1": xy(pick),
        "swab2": xy({"x": pick["x"] + step["x"], "y": pick["y"] + step["y"]}) if pick and set2 else None,
        "cell": xy(lc) if lc.get("xy_set") else None,
    }


def _lc_status() -> dict:
    lc = hub.loadcell
    return {"scaled": bool(getattr(lc, "scaled", True)),
            "scale": float(getattr(lc, "scale", 1.0) or 1.0),
            "saved": bool(hub.cal.load_cell.get("scale_counts_per_g"))}


@app.get("/api/zhome")
def zhome_info() -> dict:
    return _zhome_info()


_ZH_RANGES = {"ref_z": (10.0, 300.0), "probe_start_mm": (0.5, 20.0), "floor_margin_mm": (0.0, 5.0),
              "up_mm": (5.0, 290.0), "slide_mm": (-100.0, 200.0), "pickup_mm": (-100.0, 200.0),
              "bin_mm": (-100.0, 200.0), "safe_mm": (-100.0, 250.0)}


@app.post("/api/zhome/heights")
def zhome_heights(payload: dict) -> dict:
    """Heights are mm above the load cell contact. If Z is already homed on the
    cell, the taught Zs and the floor follow at once."""
    hub.require_idle()
    upd: dict[str, float] = {}
    for k, (lo, hi) in _ZH_RANGES.items():
        if payload.get(k) is not None:
            v = float(payload[k])
            if not lo <= v <= hi:
                raise HTTPException(400, f"{k} must be {lo:g} to {hi:g}")
            upd[k] = v
    if "ref_z" in upd and _zhome_settings().get("ref") is not None:
        raise HTTPException(400, "ref_z is fixed by the last Z home; home Z again to change it")
    if upd:
        _patch_calibration({"zhome": upd})
    st = _zhome_settings()
    if st.get("ref") is not None and st["ref"] == hub.guard.ref:
        _patch_calibration(zhomemod.patch_for(st, hub.cal.raw.get("autonomous"), hub.cal.raw.get("scrub"), hub.guard.ref))
        hub.guard.set_floor(zhomemod.floor_z(st))
    return _zhome_info()


def _wait_z(target: float, timeout: float = 3.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if abs(hub.printer.position().z - target) < 0.01:
            return
        time.sleep(0.05)


def _loadcell_alive() -> None:
    """Read the cell once before any motion, so a dead wire stops the run at the
    start, not after the head has travelled to the cell."""
    try:
        getattr(hub.loadcell, "wake", hub.loadcell.read_counts)()
        settle = getattr(hub.loadcell, "settle", None)
        steady = settle() if settle else True
    except Exception as exc:                               # noqa: BLE001
        raise HTTPException(409, f"The load cell is not answering ({exc}). Check its four wires and power, then try again. Nothing moved.") from exc
    if not steady:
        raise HTTPException(409, "The load cell is not settling (readings keep jumping). Check its wires and power, then try again. Nothing moved.")


@app.post("/api/zhome/run")
def zhome_run(payload: dict | None = None) -> dict:
    """Probe the cell to the target force, call that Z `ref_z`, re-derive every
    taught Z and the floor from the typed heights, and rise to the safe Z."""
    hub.require_idle()
    hub.stop_sampler()
    _loadcell_alive()
    lc, zc, sp = hub.cal.load_cell, hub.machine.zcal, hub.machine.speeds
    if hub.guard.floor is None:
        raise HTTPException(400, "Set a Z floor first (a few mm below the load cell) so the probe may descend")
    st = _zhome_settings()
    framed = st.get("ref") is not None and st["ref"] == hub.guard.ref
    if not lc.get("xy_set"):
        raise HTTPException(400, "Set the load cell position first (Set here on the Load cell row)")
    if not framed and not lc.get("z_safe_set"):
        raise HTTPException(400, "Set the load cell position first, with the tip 3 to 5 mm above the cell (Set here on the Load cell row)")
    if not getattr(hub.loadcell, "scaled", True):
        raise HTTPException(400, "Scale the load cell first (known mass, once)")
    p = hub.printer.position()
    if not all(a in p.homed for a in "xyz"):
        raise HTTPException(400, "Home X and Y and set Z first")
    if framed:
        # the Z frame from the last Z home is still the one in use (also after a restart
        # and a Z restore), so start a fixed distance above the cell contact
        start_z = float(st["ref_z"]) + float(st["probe_start_mm"])
    else:
        start_z = float(lc["z_safe"])
    if start_z < hub.guard.floor or (hub.guard.ceiling is not None and start_z > hub.guard.ceiling):
        raise HTTPException(400, "the probe start Z is outside the Z floor / ceiling")
    try:
        r = zhomemod.probe(hub.printer, hub.loadcell, lc, zc, sp, start_z=start_z)
    except (GripFailed, OverForce) as exc:
        raise HTTPException(409, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:                               # noqa: BLE001
        import logging, traceback
        logging.getLogger("papa").error("Z probe failed: %s", traceback.format_exc())
        raise HTTPException(500, f"Z probe crashed: {type(exc).__name__}: {exc}") from exc
    new_z = zhomemod.declare(hub.printer._inner if hasattr(hub.printer, "_inner") else hub.printer,
                             float(st["ref_z"]), r.stepped_z, r.contact_z)
    _wait_z(new_z)
    hub.guard.reset_reference(float(st["ref_z"]) + float(st["up_mm"]))
    ref = hub.guard.ref
    _patch_calibration({"zhome": {"ref": ref, "homed_at": time.time(), "last_check": None,
                                  "first_contact_z_before": round(r.contact_z, 4)}})
    st = _zhome_settings()
    hub.guard.set_floor(zhomemod.floor_z(st))
    _patch_calibration(zhomemod.patch_for(st, hub.cal.raw.get("autonomous"), hub.cal.raw.get("scrub"), ref))
    safe = zhomemod.absolute(st, "safe_mm") or (float(st["ref_z"]) + 10.0)
    hub.printer.move(z=safe, speed_mm_s=min(float(sp.get("z_lift", 10.0)), 10.0))
    return {"contact_z_before": round(r.contact_z, 4), "force_n": round(r.force_n, 3), "steps": r.steps,
            "ref_z": st["ref_z"], "floor": hub.guard.floor, **_zhome_info()}


@app.post("/api/zhome/check")
def zhome_check() -> dict:
    """Probe the cell with the swab in the jaws and report the drift. Z is not changed."""
    hub.require_idle()
    hub.stop_sampler()
    _loadcell_alive()
    st = _zhome_settings()
    if st.get("ref") is None or st["ref"] != hub.guard.ref:
        raise HTTPException(400, "Home Z on the load cell first")
    safe = zhomemod.absolute(st, "safe_mm")
    if safe is None:
        raise HTTPException(400, "Set the safe height first")
    lc = hub.cal.load_cell
    if not lc.get("xy_set"):
        raise HTTPException(400, "Capture the load cell XY first")
    try:
        r = zhomemod.check(hub.printer, hub.loadcell, lc, hub.machine.zcal, hub.machine.speeds, st, safe)
    except (GripFailed, OverForce) as exc:
        raise HTTPException(409, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:                               # noqa: BLE001
        import logging, traceback
        logging.getLogger("papa").error("Z probe failed: %s", traceback.format_exc())
        raise HTTPException(500, f"Z probe crashed: {type(exc).__name__}: {exc}") from exc
    _patch_calibration({"zhome": {"last_check": r}})
    return r


@app.post("/api/park")
def park() -> dict:
    hub.require_idle()
    st = _zhome_settings()
    safe = _auto_settings().get("safe_z")
    if safe is None:
        raise HTTPException(400, "Save the safe Z first")
    if not zhomemod.park(hub.printer, float(safe)):
        raise HTTPException(400, "Z is not set")
    return position()


@app.on_event("shutdown")
def _park_on_exit() -> None:
    """Leave the head at the safe Z, so the next Z home does not start low."""
    try:
        if hub.busy or not hub.devices["printer"].built:
            return
        safe = _auto_settings().get("safe_z")
        if safe is not None:
            zhomemod.park(hub.printer, float(safe))
    except Exception:                                      # noqa: BLE001
        pass


def _routine2_slides(payload: dict) -> list[dict]:
    ids = payload.get("slides") or ["1", "2"]
    if len(ids) != 2:
        raise HTTPException(400, "Routine 2 needs exactly two slides, e.g. S1 and S2")
    st = _zhome_settings()
    z = zhomemod.absolute(st, "slide_mm")
    out = []
    for sid in ids:
        if str(sid).strip().isdigit():
            # a slot corner captured by hand on the Slide tab: the corner is the pattern's start
            n = int(str(sid).strip())
            slot = next((x for x in hub.cal.slots if x.id == n), None)
            if slot is None or not slot.calibrated:
                raise HTTPException(400, f"Slide {n}: capture its top-left corner on the Slide tab first")
            if z is None:
                raise HTTPException(400, "Set the slide height first")
            s = dict(_scrub_settings())
            s["start"] = {"x": slot.x, "y": slot.y, "z": z, "ref": hub.guard.ref}
            s["rotation_deg"] = 0.0
        else:
            s = _substrate_scrub(str(sid), bool(payload.get("force")))
            if z is not None and s.get("start"):
                s["start"] = {**s["start"], "z": z, "ref": hub.guard.ref}
        out.append(s)
    return out


def _routine2_extra() -> list[str]:
    st = _zhome_settings()
    bad = zhomemod.problems(st, hub.guard.ref)
    lc = hub.cal.load_cell
    if not lc.get("xy_set"):
        bad.append("Capture the load cell XY.")
    if not getattr(hub.loadcell, "scaled", True):
        bad.append("Scale the load cell (known mass, once).")
    return bad


@app.get("/api/routine2")
def routine2_info() -> dict:
    bad = _routine2_extra()
    return {"problems": bad, "ready": not bad, "status": hub.routine2.status.as_dict(),
            "last_check": hub.routine2.last_check or _zhome_settings().get("last_check")}


@app.get("/api/routine2/status")
def routine2_status() -> dict:
    return {**hub.routine2.status.as_dict(), "last_check": hub.routine2.last_check}


@app.post("/api/routine2/run")
def routine2_run(payload: dict | None = None) -> dict:
    hub.require_idle()
    hub.stop_sampler()
    _loadcell_alive()
    body = payload or {}
    slides = _routine2_slides(body)
    st = _zhome_settings()
    safe = float(zhomemod.absolute(st, "safe_mm") or _auto_settings().get("safe_z") or 0)
    lc, zc, sp = hub.cal.load_cell, hub.machine.zcal, hub.machine.speeds

    def check_fn(stop):
        r = zhomemod.check(hub.printer, hub.loadcell, lc, zc, sp, _zhome_settings(), safe, stop)
        _patch_calibration({"zhome": {"last_check": r}})
        return r

    def park_fn():
        zhomemod.park(hub.printer, safe)

    try:
        hub.thread = hub.routine2.start2(_auto_settings(), slides, hub.guard.ref, hub.guard.floor,
                                         check_fn, park_fn, int(body.get("swab_index", 0)), _routine2_extra())
    except (routinemod.RoutineError, scrubmod.ScrubError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return hub.routine2.status.as_dict()


# ---------- Routine 3 x 3: Routine 2 over the whole mount ----------

# Mount-to-mount spacing from the caliper photos (IMG_5118-5122): reading minus
# the 0.205 in mount-edge-to-pocket offset. Columns go +X, rows go down (-Y).
GRID_COL_MM = (1.995 * 25.4, 0.0)        # 50.673 mm
GRID_ROW_MM = (0.0, -2.011 * 25.4)       # -51.079 mm
GRID_ANCHOR_SLOT = 2                     # Routine 2's slide A (Slide 1 corner) is this grid slot


def _grid_spacing() -> tuple[list[float], list[float], str]:
    """Entered mount spacing (Master path panel) if set, else the caliper values."""
    sp = _master().get("spacing") or {}
    col = sp.get("col") or list(GRID_COL_MM)
    row = sp.get("row") or list(GRID_ROW_MM)
    src = "entered" if (sp.get("col") or sp.get("row")) else "caliper photos (1.995 in x 2.011 in)"
    return [float(v) for v in col], [float(v) for v in row], src


def _grid_plan(payload: dict | None = None) -> dict:
    """Each slot of the 3 x 3 mount: its top-left corner (Slide 1's corner, taken
    as the anchor slot, moved by the mount spacing) and whether it runs: a slot
    runs if the last scan found a substrate in it; empty slots are skipped."""
    body = payload or {}
    anchor = int(body.get("anchor_slot") or _master().get("grid_anchor") or GRID_ANCHOR_SLOT)
    rows = cols = 3
    col, row, src = _grid_spacing()
    s1 = next((x for x in hub.cal.slots if x.id == 1), None)
    corner = (s1.x, s1.y) if s1 is not None and s1.calibrated else None
    ar, ac = divmod(anchor - 1, cols)
    filled, why_unknown = None, None
    if body.get("slots") is not None:
        filled = {int(v) for v in body["slots"]}
    else:
        sc = hub.align_scan
        sm = _slot_map(sc) if sc else None
        if sm is None:
            why_unknown = "Scan the substrates with the slot grid set, so empty slots can be skipped."
        else:
            filled = {c["slot"] for c in sm["cells"] if c["sid"]}
    cells = []
    sh = _master().get("grid_shift") or [0, 0]          # whole grid moved by whole mounts (columns, rows)
    sc_, sr_ = int(sh[0]), int(sh[1])
    for k in range(1, rows * cols + 1):
        r, c = divmod(k - 1, cols)
        dx = (c - ac + sc_) * col[0] + (r - ar + sr_) * row[0]
        dy = (c - ac + sc_) * col[1] + (r - ar + sr_) * row[1]
        nx, ny = (_master().get("grid_nudge") or [0.0, 0.0])
        xy = None if corner is None else [round(corner[0] + dx + nx, 3), round(corner[1] + dy + ny, 3)]
        own = (_master().get("slot_corners") or {}).get(str(k))
        if own is not None:                       # set by hand for this slot: used as it is
            xy = [round(float(own[0]), 3), round(float(own[1]), 3)]
        run = filled is not None and k in filled
        cells.append({"slot": k, "corner": xy, "own": own is not None, "offset_mm": [round(dx, 3), round(dy, 3)], "run": run,
                      "why": None if run else ("empty" if filled is not None else "unknown")})
    problems = []
    if corner is None and not all(c["own"] for c in cells if c["run"]):
        problems.append("Routine 2's first slide is not saved yet: set Slide 1 · top-left in Where everything is "
                        "(it is grid slot %d)." % anchor)
    if why_unknown:
        problems.append(why_unknown)
    runs = [c["slot"] for c in cells if c["run"]]
    if filled is not None and not runs:
        problems.append("No substrate in any slot.")
    return {"anchor_slot": anchor, "shift": [sc_, sr_], "nudge": _master().get("grid_nudge") or [0.0, 0.0], "spacing": {"col": col, "row": row, "source": src},
            "cells": cells, "runs": runs, "skips": [c["slot"] for c in cells if not c["run"]],
            "problems": problems}


@app.post("/api/routine3x3/slot-corner")
def routine3x3_slot_corner(payload: dict) -> dict:
    """Set one slot's array corner by hand: {slot, here: true} takes the tip's
    X and Y now; {slot, x, y} takes typed values; {slot, clear: true} goes back
    to the calculated grid; {clear_all: true} for every slot. Saved."""
    m = _master()
    own = dict(m.get("slot_corners") or {})
    if payload.get("clear_all"):
        own = {}
    else:
        try:
            k = int(payload.get("slot"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "slot: 1 to 9") from exc
        if not 1 <= k <= 9:
            raise HTTPException(400, "slot: 1 to 9")
        if payload.get("clear"):
            own.pop(str(k), None)
        else:
            if payload.get("here"):
                p = hub.printer.position()
                if not all(a in p.homed for a in "xy"):
                    raise HTTPException(400, "Home X and Y first.")
                x, y = p.x, p.y
            else:
                try:
                    x, y = float(payload["x"]), float(payload["y"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise HTTPException(400, "x and y in mm") from exc
            if not hub.machine.in_limits(x, y, hub.machine.limits["z"][1]):
                raise HTTPException(400, f"X {x:.2f} Y {y:.2f} is outside the machine limits.")
            own[str(k)] = [round(x, 3), round(y, 3)]
    m["slot_corners"] = own
    _save_master(m)
    return routine3x3_info()


@app.post("/api/routine3x3/correct")
def routine3x3_correct(payload: dict) -> dict:
    """The tip is exactly on this slot's array corner: {slot: k}. The difference
    between where the tip is and where the grid put that corner is applied to
    every slot (the mounts are identical). {reset: true} removes it."""
    m = _master()
    if payload.get("reset"):
        m.pop("grid_nudge", None)
    else:
        try:
            k = int(payload.get("slot"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "slot: 1 to 9") from exc
        p = hub.printer.position()
        if not all(a in p.homed for a in "xy"):
            raise HTTPException(400, "Home X and Y first.")
        cell = next((c for c in _grid_plan()["cells"] if c["slot"] == k), None)
        if cell is None or cell["corner"] is None:
            raise HTTPException(400, "Save Routine 2's first slide (Slide 1 · top-left) first.")
        old = m.get("grid_nudge") or [0.0, 0.0]
        dx, dy = p.x - cell["corner"][0], p.y - cell["corner"][1]
        if abs(dx) > 15 or abs(dy) > 15:
            raise HTTPException(400, f"The tip is {dx:+.2f}, {dy:+.2f} mm from slot {k}'s corner: more than 15 mm. "
                                     "Is it on the right mount? Use the shift buttons for whole mounts.")
        m["grid_nudge"] = [round(old[0] + dx, 3), round(old[1] + dy, 3)]
    _save_master(m)
    return routine3x3_info()


@app.post("/api/routine3x3/shift")
def routine3x3_shift(payload: dict) -> dict:
    """Move every slot's corner by whole mounts: {cols: +1 | -1} (toward slot
    3 / toward slot 1), {rows: +1 | -1} (toward slot 7 / toward slot 1), or
    {reset: true}. Saved."""
    m = _master()
    sh = list(m.get("grid_shift") or [0, 0])
    if payload.get("reset"):
        sh = [0, 0]
    else:
        sh[0] += int(payload.get("cols") or 0)
        sh[1] += int(payload.get("rows") or 0)
    if abs(sh[0]) > 3 or abs(sh[1]) > 3:
        raise HTTPException(400, "That moves the grid more than 3 mounts.")
    m["grid_shift"] = sh
    _save_master(m)
    return routine3x3_info()


@app.post("/api/routine3x3/anchor")
def routine3x3_anchor(payload: dict) -> dict:
    """Which grid slot Routine 2's first slide (Slide 1's saved corner) sits in."""
    try:
        k = int(payload.get("slot"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "slot: 1 to 9") from exc
    if not 1 <= k <= 9:
        raise HTTPException(400, "slot: 1 to 9")
    m = _master()
    m["grid_anchor"] = k
    _save_master(m)
    return routine3x3_info()


@app.post("/api/routine3x3/flip")
def routine3x3_flip(payload: dict) -> dict:
    """Turn round which way the grid's columns ({axis: "col"}) or rows
    ({axis: "row"}) run on the printer. Saved as the mount spacing."""
    col, row, _ = _grid_spacing()
    ax = payload.get("axis")
    if ax == "col":
        col = [-col[0], -col[1]]
    elif ax == "row":
        row = [-row[0], -row[1]]
    else:
        raise HTTPException(400, "axis: col or row")
    m = _master()
    m["spacing"] = {"col": [round(v, 4) for v in col], "row": [round(v, 4) for v in row]}
    _save_master(m)
    return routine3x3_info()


@app.get("/api/routine3x3")
def routine3x3_info() -> dict:
    plan = _grid_plan()
    bad = plan["problems"] + _routine2_extra()
    if _zhome_settings().get("slide_mm") is None and "Set the slide height." not in bad:
        bad.append("Set the slide height.")
    return {**plan, "problems": bad, "ready": not bad, "status": hub.routine2.status.as_dict()}


@app.post("/api/routine3x3/run")
def routine3x3_run(payload: dict | None = None) -> dict:
    """Routine 2 on every filled slot of the 3 x 3 mount, slot 1 to 9: check Z,
    scrub, dispose, grip the next swab; park at the end. Each slot scrubs the
    Scrub tab pattern from its corner at the slide height, like Routine 2's
    slides. Empty slots are skipped."""
    hub.require_idle()
    hub.stop_sampler()
    _loadcell_alive()
    body = payload or {}
    plan = _grid_plan(body)
    if plan["problems"]:
        raise HTTPException(400, " ".join(plan["problems"]))
    st = _zhome_settings()
    z = zhomemod.absolute(st, "slide_mm")
    if z is None:
        raise HTTPException(400, "Set the slide height first")
    slides = []
    for c in plan["cells"]:
        if not c["run"]:
            continue
        s = dict(_scrub_settings())
        s["start"] = {"x": c["corner"][0], "y": c["corner"][1], "z": z, "ref": hub.guard.ref}
        s["rotation_deg"] = 0.0
        slides.append((f"slot {c['slot']}", s))
    safe = float(zhomemod.absolute(st, "safe_mm") or _auto_settings().get("safe_z") or 0)
    lc, zc, sp = hub.cal.load_cell, hub.machine.zcal, hub.machine.speeds

    def check_fn(stop):
        r = zhomemod.check(hub.printer, hub.loadcell, lc, zc, sp, _zhome_settings(), safe, stop)
        _patch_calibration({"zhome": {"last_check": r}})
        return r

    def park_fn():
        zhomemod.park(hub.printer, safe)

    try:
        hub.thread = hub.routine2.start_many(_auto_settings(), slides, hub.guard.ref, hub.guard.floor,
                                             check_fn, park_fn, int(body.get("swab_index", 0)),
                                             _routine2_extra(), action="routine3x3")
    except (routinemod.RoutineError, scrubmod.ScrubError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**hub.routine2.status.as_dict(), "runs": plan["runs"], "skips": plan["skips"]}


class _GridCheck:
    def __init__(self) -> None:
        self.running = False
        self.stop = False
        self.at: dict | None = None
        self.done: list[int] = []
        self.error: str | None = None
        self.z: float | None = None

    def as_dict(self) -> dict:
        return {"running": self.running, "at": self.at, "done": self.done, "error": self.error, "z": self.z}


_GCHECK = _GridCheck()


@app.get("/api/routine3x3/check")
def routine3x3_check_status() -> dict:
    return _GCHECK.as_dict()


@app.post("/api/routine3x3/check")
def routine3x3_check(payload: dict | None = None) -> dict:
    """Dry run of the mount's X/Y: visit each slot's corner high above the
    slides (never lowering, no scrub, no grip) and pause there, optionally
    tracing the pattern's outline. slots: list, or "filled" (default) or
    "all"; height_mm above the slide height (default 30); dwell_s; outline."""
    hub.require_idle()
    p = hub.printer.position()
    if not all(a in p.homed for a in "xy"):
        raise HTTPException(400, "Home X and Y first.")
    body = payload or {}
    plan = _grid_plan({k: v for k, v in body.items() if k == "anchor_slot"})
    if any("Slide 1" in m for m in plan["problems"]):
        raise HTTPException(400, plan["problems"][0])
    which = body.get("slots", "filled")
    if which == "all":
        slots = [c["slot"] for c in plan["cells"]]
    elif which == "filled":
        if not plan["runs"]:
            raise HTTPException(400, " ".join(plan["problems"]) or "No filled slot: tour all 9 instead.")
        slots = plan["runs"]
    else:
        slots = [int(v) for v in which]
    cells = {c["slot"]: c for c in plan["cells"]}
    if any(k not in cells for k in slots):
        raise HTTPException(400, "Slots are 1 to 9.")
    st = _zhome_settings()
    slide = zhomemod.absolute(st, "slide_mm")
    safe = zhomemod.absolute(st, "safe_mm") or _auto_settings().get("safe_z")
    height = float(body.get("height_mm", 30.0))
    z_known = "z" in p.homed
    if not z_known and not body.get("z_unknown_ok"):
        raise HTTPException(409, "Z is not known, so it cannot rise to the check height. Raise Z with Z+ until "
                                 "the tip is well clear, then confirm to tour at the current height.")
    if not z_known:
        z = None                                      # Z untouched: X and Y moves only
    else:
        z = slide + height if slide is not None else p.z
        if safe is not None:
            z = max(z, float(safe))
        z = max(z, p.z)                               # never lower
        z = min(z, float(hub.machine.limits["z"][1]))
    dwell = max(0.0, min(30.0, float(body.get("dwell_s", 3.0))))
    outline = bool(body.get("outline"))
    area = scrubmod.merged(_scrub_settings())["area"]
    w, h = float(area["w"]), float(area["h"])
    pts: list[tuple[int, list[tuple[float, float]]]] = []
    from papa.ops.geometry import to_machine
    for k in slots:
        x0, y0 = cells[k]["corner"]
        path = [(x0, y0)]
        if outline:
            path += [to_machine(hub.machine.axes, x0, y0, lx, ly) for lx, ly in ((w, 0), (w, h), (0, h), (0, 0))]
        for x, y in path:
            if not hub.machine.in_limits(x, y, z if z is not None else hub.machine.limits["z"][1]):
                raise HTTPException(400, f"Slot {k}: X {x:.2f} Y {y:.2f} is outside the machine limits.")
        pts.append((k, path))
    speed = float(body.get("speed_mm_s", 40.0))
    g = _GCHECK
    g.__init__()
    g.running, g.z = True, (round(z, 3) if z is not None else None)

    def run() -> None:
        try:
            if z is not None and hub.printer.position().z < z - 1e-6:
                hub.printer.move(z=z, speed_mm_s=min(float(hub.machine.speeds.get("z_lift", 10.0)), 10.0))
            for k, path in pts:
                for i, (x, y) in enumerate(path):
                    if g.stop:
                        raise RuntimeError("Stopped")
                    hub.printer.move(x=x, y=y, speed_mm_s=speed)
                    if i == 0:
                        g.at = {"slot": k, "x": round(x, 3), "y": round(y, 3)}
                        t0 = time.monotonic()
                        while time.monotonic() - t0 < dwell and not g.stop:
                            time.sleep(0.05)
                g.done.append(k)
        except Exception as exc:                 # noqa: BLE001
            g.error = str(exc) or type(exc).__name__
        finally:
            g.running = False

    hub.thread = threading.Thread(target=run, daemon=True, name="grid-check")
    hub.thread.start()
    return g.as_dict()


@app.post("/api/routine3x3/check/stop")
def routine3x3_check_stop() -> dict:
    _GCHECK.stop = True
    return _GCHECK.as_dict()


@app.post("/api/routine2/stop")
def routine2_stop() -> dict:
    hub.routine2.stop()
    return hub.routine2.status.as_dict()


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(HTTPException)
async def http_error(_req: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def _restart_soon(delay: float = 1.5) -> bool:
    """Exit with an error code so systemd (Restart=on-failure) starts the
    server again. Only under systemd; returns whether a restart is coming."""
    if not os.environ.get("INVOCATION_ID"):
        return False
    threading.Timer(delay, lambda: os._exit(3)).start()
    return True


@app.exception_handler(CameraStuck)
async def camera_stuck(_req: Any, exc: CameraStuck) -> JSONResponse:
    # A hung capture can only be cleared by a fresh process. Restart by itself
    # when nothing is running; otherwise say so and leave the job alone.
    mark = STATE_DIR / "camera_auto_restart"
    recent = mark.exists() and time.time() - mark.stat().st_mtime < 120
    restarting = not hub.busy and not recent and _restart_soon()
    if restarting:
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.touch()
    elif recent:
        exc = CameraStuck(f"{exc} (again right after a restart, so it was not restarted automatically: "
                          "check the camera ribbon cable at both ends, then press Restart server)")
    msg = f"Camera: {exc}. " + ("Restarting the server to reset it; the page reconnects in about 10 s."
                                if restarting else
                                "Restart the server (Software panel) once the job has stopped. "
                                "If it keeps happening, check the camera ribbon cable.")
    return JSONResponse({"error": msg, "restarting": restarting}, status_code=503)


@app.exception_handler(ZFloorError)
async def floor_error(_req: Any, exc: ZFloorError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=400)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
