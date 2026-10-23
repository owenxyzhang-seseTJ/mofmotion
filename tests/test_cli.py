import json

import pytest

from mofmotion.cli import main

PAIR = [("C1", "C", 0.95, 0.25, 0.25, 1), ("O1", "O", 0.07, 0.25, 0.25, 1)]


def test_clean_then_audit(write, tmp_path, capsys):
    source = write(PAIR)
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    assert main(["clean", str(source), str(tmp_path / "clean.cif")]) == 0
    assert "LF" in capsys.readouterr().out
    assert main(["audit", str(tmp_path / "clean.cif"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sites"] == 2


def test_audit_several_states(write, capsys):
    paths = [str(write(PAIR, "open.cif")), str(write(PAIR[:1], "closed.cif", ops=("x,y",)))]
    assert main(["audit", *paths]) == 1
    out = capsys.readouterr().out
    assert "open.cif" in out and "错误" in out


ZNO3 = [
    ("Zn1", "Zn", 0, 0, 0, 1),
    ("O1", "O", 0.5, 0, 0, 1),
    ("O2", "O", 0, 0.5, 0, 1),
    ("O3", "O", 0, 0, 0.5, 1),
]


def test_compare_writes_html(write, tmp_path, capsys):
    a = write(ZNO3, "a.cif", cell=(4, 4, 4, 90, 90, 90))
    b = write(ZNO3, "b.cif", cell=(4.2, 3.8, 4, 90, 90, 90))
    html = tmp_path / "report.html"
    assert main(["compare", str(a), str(b), "--probe", "0.5", "--html", str(html)]) == 0
    assert "呼吸" in capsys.readouterr().out
    text = html.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and "孔道变化归因" in text
    with pytest.raises(SystemExit):
        main(["compare", str(a), str(b), "--html", str(b)])
