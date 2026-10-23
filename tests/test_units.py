from collections import Counter

import numpy as np
from pymatgen.core import Lattice, Structure

from mofmotion import bond_graph, structural_units


def _structure(sites, abc):
    return Structure(
        Lattice.orthorhombic(*abc),
        [symbol for symbol, _ in sites],
        [xyz for _, xyz in sites],
        coords_are_cartesian=True,
    )


def test_chain_units():
    sites = [
        ("Zn", (5, 5, 5)),
        ("O", (7, 5, 5)),
        ("C", (8.3, 5, 5)),
        ("C", (9.7, 5, 5)),
        ("C", (11.1, 5, 5)),
        ("O", (12.4, 5, 5)),
        ("O", (5, 7, 5)),
        ("H", (5.76, 7.59, 5)),
        ("H", (4.24, 7.59, 5)),
        ("C", (5, 3, 5)),
        ("O", (5, 1.87, 5)),
        ("O", (5, 15, 5)),
        ("H", (5.76, 15.59, 5)),
        ("H", (4.24, 15.59, 5)),
        ("Fe", (5, 15, 15)),
        ("C", (5, 15, 13)),
        ("C", (5, 15, 17)),
        ("O", (5, 15, 11.87)),
        ("O", (5, 15, 18.13)),
    ]
    s = _structure(sites, (9.4, 20, 20))
    units = structural_units(bond_graph(s))
    kinds = {(u.kind, u.formula, u.dimension) for u in units.units}
    assert kinds == {
        ("node", "H2OZn", 0),
        ("linker", "C3O2", 0),
        ("terminal", "CO", 0),
        ("free", "H2O", 0),
        ("free", "C2FeO2", 0),
    }
    assert Counter(units.net_dimensions) == {1: 1, 0: 2}

    for u, unit in enumerate(units.units):
        xyz = dict(zip(unit.sites, units.coords(u), strict=True))
        for b in units.graph.bonds:
            if units.unit_of[b.i] == units.unit_of[b.j] == u:
                assert abs(np.linalg.norm(xyz[b.i] - xyz[b.j]) - b.length) < 1e-3


def test_infinite_inorganic_rod_is_node():
    s = _structure([("Zn", (0, 5, 5)), ("O", (1.9, 5, 5))], (3.8, 20, 20))
    (unit,) = structural_units(bond_graph(s)).units
    assert (unit.kind, unit.dimension) == ("node", 1)
