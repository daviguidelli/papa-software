"""Per-device real/mock selection, and what happens when a device is absent."""
import importlib

import pytest
from fastapi.testclient import TestClient

from papa import config as cfgmod


def reload_env(monkeypatch, sim=None, real=None):
    monkeypatch.delenv("PAPA_SIM", raising=False)
    monkeypatch.delenv("PAPA_REAL", raising=False)
    if sim is not None:
        monkeypatch.setenv("PAPA_SIM", sim)
    if real is not None:
        monkeypatch.setenv("PAPA_REAL", real)


def test_sim_mocks_everything(monkeypatch):
    reload_env(monkeypatch, sim="1")
    assert not any(cfgmod.is_real(d) for d in cfgmod.DEVICES)


def test_no_sim_means_everything_real(monkeypatch):
    reload_env(monkeypatch)
    assert all(cfgmod.is_real(d) for d in cfgmod.DEVICES)


def test_laptop_combination_is_real_printer_only(monkeypatch):
    # the whole point: drive a real printer from a laptop that has no GPIO
    reload_env(monkeypatch, sim="1", real="printer")
    assert cfgmod.is_real("printer")
    assert not cfgmod.is_real("loadcell")
    assert not cfgmod.is_real("gripper")
    assert not cfgmod.is_real("camera")


def test_several_devices_can_be_forced_real(monkeypatch):
    reload_env(monkeypatch, sim="1", real="printer, gripper")
    assert cfgmod.is_real("printer") and cfgmod.is_real("gripper")
    assert not cfgmod.is_real("camera")


def test_a_typo_in_papa_real_fails_loudly(monkeypatch):
    reload_env(monkeypatch, sim="1", real="printr")
    with pytest.raises(ValueError, match="printr"):
        cfgmod.is_real("printer")


def test_unknown_device_name_is_rejected(monkeypatch):
    reload_env(monkeypatch, sim="1")
    with pytest.raises(ValueError, match="toolhead"):
        cfgmod.is_real("toolhead")


def test_a_missing_device_does_not_block_the_rest(monkeypatch):
    """A laptop has no HX711. Jogging must still work."""
    reload_env(monkeypatch, sim="1")
    server = importlib.reload(importlib.import_module("apps.server"))

    def boom():
        raise RuntimeError("no lgpio on this machine")

    server.hub.devices["loadcell"] = server.Device("loadcell", boom)
    client = TestClient(server.app)

    assert client.get("/api/machine/position").status_code == 200
    r = client.get("/api/loadcell")
    assert r.status_code == 503
    assert "no lgpio" in r.json()["error"]
    # and the failure is reported rather than hidden
    assert client.get("/api/devices").json()["devices"]["loadcell"]["error"]


def test_devices_endpoint_reports_what_it_is_talking_to(monkeypatch):
    reload_env(monkeypatch, sim="1")
    server = importlib.reload(importlib.import_module("apps.server"))
    body = TestClient(server.app).get("/api/devices").json()
    assert body["sim"] is True
    assert body["devices"]["printer"]["real"] is False
    assert body["devices"]["printer"]["reachable"] is True
    assert "mock" in body["devices"]["printer"]["url"]
