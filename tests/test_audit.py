import json

import pytest
from conftest import make_cif

from mofmotion import audit_cif

PAIR = [("C1", "C", 0.95, 0.25, 0.25, 1), ("O1", "O", 0.07, 0.25, 0.25, 1)]
INVERSION = ("x,y,z", "-x,-y,-z")


def test_minimal_p1(write):
    path = write(PAIR)
    before = path.read_bytes()
    audit = audit_cif(path)
    assert audit.ok and not audit.warnings and not audit.text_fixes
    assert audit.flags == {}
    assert (audit.rows, audit.sites, audit.symop_count) == (2, 2, 1)
    assert audit.min_distance == pytest.approx(1.2)
    assert audit.clashes == []
    assert audit.spacegroup == "Pmm2 (25)"
    assert len(audit.structure) == 2
    data = json.loads(audit.to_json())
    assert "structure" not in data and data["formula"] == "C1 O1"
    assert path.read_bytes() == before


def test_partial_occupancy_kept(write):
    audit = audit_cif(write([("C1", "C", 0.1, 0.2, 0.3, 0.5)]))
    assert audit.ok
    assert audit.flags == {"partial_occupancy": ["C1"]}
    assert not audit.structure.is_ordered
    assert audit.spacegroup is None


def test_disorder_group(write, tmp_path):
    text = make_cif([("C1", "C", 0.1, 0.2, 0.3, 1), ("C2", "C", 0.5, 0.5, 0.5, 1)])
    text = text.replace(
        "_atom_site_occupancy\n", "_atom_site_occupancy\n_atom_site_disorder_group\n"
    )
    text = text.replace("0.3 1\n", "0.3 1 .\n").replace("0.5 1\n", "0.5 1 2\n")
    path = tmp_path / "d.cif"
    path.write_text(text)
    assert audit_cif(path).flags == {"disorder_group": ["C2"]}


def test_symmetry_expansion_and_special_position(write):
    audit = audit_cif(
        write([("C1", "C", 0.1, 0.2, 0.3, 1), ("O1", "O", 0, 0, 0, 1)], ops=INVERSION)
    )
    assert audit.ok
    assert audit.sites == 3
    assert audit.flags == {"special_positions": ["O1"]}


def test_preexpanded_symmetry_copy(write):
    sites = [("C1", "C", 0.1, 0.2, 0.3, 1), ("C2", "C", 0.9, 0.8, 0.7, 1)]
    audit = audit_cif(write(sites, ops=INVERSION))
    assert audit.flags["symmetry_duplicates"] == ["C1", "C2"]


def test_boundary_image_is_periodic_duplicate(write):
    sites = [("C1", "C", 0.1, 0.2, 0.3, 1), ("C2", "C", 1.1, 0.2, 0.3, 1)]
    audit = audit_cif(write(sites))
    assert audit.flags == {"outside_cell": ["C2"], "periodic_duplicates": ["C1", "C2"]}


def test_short_contacts(write):
    audit = audit_cif(write([("C1", "C", 0.10, 0.2, 0.3, 1), ("C2", "C", 0.15, 0.2, 0.3, 1)]))
    assert [(c.labels, c.distance) for c in audit.clashes] == [(("C1", "C2"), 0.5)]
    selfimage = audit_cif(write([("C1", "C", 0, 0, 0, 1)], cell=(0.9, 11, 12, 90, 90, 90)))
    assert selfimage.clashes[0].site_1 == selfimage.clashes[0].site_2 == 0
    assert len(selfimage.clashes) == 1


def test_skewed_cell_distance(write):
    cell = (10, 10, 10, 90, 90, 60)
    audit = audit_cif(write([("C1", "C", 0, 0, 0, 1), ("C2", "C", 0.9, 0.1, 0, 1)], cell=cell))
    assert audit.min_distance == pytest.approx(1.0)


def test_dirty_text_is_cleaned_in_memory(tmp_path):
    text = make_cif(PAIR).replace("0.07", "\u22120.93").replace("\n", "\r\n")
    path = tmp_path / "dirty.cif"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode())
    audit = audit_cif(path)
    assert audit.ok and len(audit.text_fixes) == 3
    assert audit.min_distance == pytest.approx(1.2)


def test_block_selection(tmp_path):
    path = tmp_path / "multi.cif"
    path.write_text(make_cif(PAIR, name="open") + make_cif(PAIR[:1], name="closed"))
    assert "block" in audit_cif(path).errors[0]
    assert audit_cif(path, block="closed").sites == 1
    path.write_text("data_global\n_journal_year 2024\n" + make_cif(PAIR, name="open"))
    assert audit_cif(path).block == "open"


def test_errors_are_reported_not_raised(write, tmp_path):
    assert audit_cif(tmp_path / "missing.cif").errors
    assert "对称操作" in audit_cif(write(PAIR, ops=("x,y",))).errors[0]
    missing = audit_cif(write([("C1", "C", "?", 0.2, 0.3, 1)]))
    assert missing.flags["missing_data"] == ["C1"]
