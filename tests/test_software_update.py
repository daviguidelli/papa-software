"""Updating the Pi from the page: fetch, fast-forward, never overwrite local edits."""
import subprocess

import pytest

from tests.test_validate import api  # noqa: F401  (fixture)


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    remote, pi, dev = tmp_path / "remote.git", tmp_path / "pi", tmp_path / "dev"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(dev)], check=True)
    for d in (dev,):
        git(d, "config", "user.email", "t@example.com")
        git(d, "config", "user.name", "t")
    (dev / "a.txt").write_text("one\n")
    git(dev, "add", "a.txt")
    git(dev, "commit", "-q", "-m", "first")
    git(dev, "push", "-q", "origin", "HEAD:main")
    subprocess.run(["git", "clone", "-q", str(remote), str(pi)], check=True)
    (dev / "a.txt").write_text("two\n")
    git(dev, "commit", "-q", "-am", "second")
    git(dev, "push", "-q", "origin", "HEAD:main")
    return pi, dev


def test_check_then_update_fast_forwards(api, repos, monkeypatch):
    pi, _ = repos
    monkeypatch.setattr(api.server, "REPO_DIR", pi)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    j = api.get("/api/software").json()
    assert j["branch"] == "main" and j["subject"] == "first"
    j = api.post("/api/software/check").json()
    assert [x.split(" ", 1)[1] for x in j["new"]] == ["second"]
    assert (pi / "a.txt").read_text() == "one\n"                  # checking changes nothing
    j = api.post("/api/software/update").json()
    assert j["updated"] and not j["restarting"] and j["subject"] == "second"
    assert (pi / "a.txt").read_text() == "two\n"
    assert api.post("/api/software/update").json()["message"] == "Already up to date."


def test_update_never_overwrites_edits_on_the_pi(api, repos, monkeypatch):
    pi, _ = repos
    monkeypatch.setattr(api.server, "REPO_DIR", pi)
    (pi / "a.txt").write_text("edited on the pi\n")
    r = api.post("/api/software/update")
    assert r.status_code == 409 and "a.txt" in r.json()["error"]
    assert (pi / "a.txt").read_text() == "edited on the pi\n"
