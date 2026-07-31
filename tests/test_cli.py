import subprocess
from types import SimpleNamespace

from gpu_queue import cli


def _fake_show(linger: str):
    """Stand-in for `loginctl show-user ... --property=Linger`."""
    return lambda *a, **kw: SimpleNamespace(stdout=f"Linger={linger}\n", returncode=0)


def test_ensure_linger_noop_when_already_enabled(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_show("yes"))
    monkeypatch.setattr(subprocess, "call", lambda *a, **kw: calls.append(a) or 0)

    cli._ensure_linger()

    assert calls == []  # must not shell out to enable-linger
    assert capsys.readouterr().err == ""


def test_ensure_linger_enables_when_off(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_show("no"))
    monkeypatch.setattr(subprocess, "call", lambda *a, **kw: calls.append(a[0]) or 0)

    cli._ensure_linger()

    assert calls and calls[0][:2] == ["loginctl", "enable-linger"]
    assert capsys.readouterr().err == ""


def test_ensure_linger_warns_when_it_cannot(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", _fake_show("no"))
    monkeypatch.setattr(subprocess, "call", lambda *a, **kw: 1)

    cli._ensure_linger()

    err = capsys.readouterr().err
    assert "could not enable linger" in err
    assert "sudo loginctl enable-linger" in err


def test_ensure_linger_survives_missing_loginctl(monkeypatch, capsys):
    def boom(*a, **kw):
        raise FileNotFoundError("loginctl")

    monkeypatch.setattr(subprocess, "run", boom)

    cli._ensure_linger()  # must not raise on boxes without systemd-logind

    assert "could not enable linger" in capsys.readouterr().err
