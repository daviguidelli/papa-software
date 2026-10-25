"""The bench scripts are what runs first on real hardware, so they must at least
run clean against the mocks."""
import builtins
import importlib
import sys

import pytest


def run(mod, argv, monkeypatch):
    monkeypatch.setenv("PAPA_SIM", "1")
    monkeypatch.setattr(builtins, "input", lambda *_: "y")
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    return importlib.import_module(f"scripts.bench.{mod}").run()


def test_gantry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parents[1]))
    assert run("b1_gantry", ["--repeats", "2"], monkeypatch) == 0


def test_loadcell_flags_a_100g_cell(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parents[1]))
    assert run("b2_loadcell", ["--drift-s", "0.1", "--rating-g", "100"], monkeypatch) == 1
    assert run("b2_loadcell", ["--drift-s", "0.1", "--rating-g", "500"], monkeypatch) == 1  # abort 20 g < working
