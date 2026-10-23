import numpy as np
import pytest
from pymatgen.core import Element, Lattice, Structure

from mofmotion import compare, pores


def _reo3(abc, water=False):
    s = Structure(Lattice.orthorhombic(*abc), ["Zn", "O", "O", "O"], np.eye(4, 3, -1) / 2)
    if water:
        center = np.array(abc) / 2
        for symbol, d in (("O", (0, 0, 0)), ("H", (0.76, 0.59, 0)), ("H", (-0.76, 0.59, 0))):
            s.append(symbol, center + d, coords_are_cartesian=True)
    return s


def test_simple_cubic_matches_geometry():
    s = Structure(Lattice.cubic(6), ["Ar"], [[0, 0, 0]])
    p = pores(s, keep=np.ones(1, bool), probe=1.0)
    r = Element("Ar").van_der_waals_radius
    assert p.lcd == pytest.approx(2 * (3 * 3**0.5 - r), abs=0.1)
    assert p.pld == pytest.approx(2 * (3 * 2**0.5 - r), abs=0.1)
    assert p.dimension == 3


def test_guests_removed_by_default():
    loaded = _reo3((4, 4, 4), water=True)
    p = pores(loaded, probe=0.5)
    assert p.removed == "H2O"
    assert p.lcd == pytest.approx(pores(_reo3((4, 4, 4)), probe=0.5).lcd)
    assert pores(loaded, probe=0.5, keep=np.ones(len(loaded), bool)).lcd < p.lcd - 1


def test_compare_breathing_with_guest():
    c = compare(_reo3((4, 4, 4)), _reo3((4.2, 3.8, 4), water=True), probe=0.5)
    labels = {f.label for f in c.findings}
    assert {"breathing", "guest"} <= labels
    assert np.sort(c.motion.strain.principal) == pytest.approx([-0.05, 0, 0.05], abs=1e-6)
    assert c.pores_b.removed == "H2O"
    effect = c.attribution.effects[0]
    assert effect.factor == "晶格"
    assert effect.forward["lcd"] == pytest.approx(c.attribution.b["lcd"] - c.attribution.a["lcd"])
