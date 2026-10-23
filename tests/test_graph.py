from pymatgen.core import Lattice, Structure

from mofmotion import bond_graph
from mofmotion.cli import main


def test_bond_across_boundary_records_image():
    s = Structure(
        Lattice.orthorhombic(10, 11, 12), ["C", "O"], [[0.95, 0.25, 0.25], [0.07, 0.25, 0.25]]
    )
    (bond,) = bond_graph(s).bonds
    assert (bond.i, bond.j, bond.image, bond.kind) == (0, 1, (1, 0, 0), "covalent")
    assert abs(bond.length - 1.2) < 1e-6
    graph = bond_graph(s)
    assert graph.neighbors(1) == [(0, (-1, 0, 0), bond)]


def test_self_image_bond_kept_once():
    s = Structure(Lattice.orthorhombic(1.5, 20, 20), ["C"], [[0, 0, 0]])
    (bond,) = bond_graph(s).bonds
    assert (bond.i, bond.j, bond.image) == (0, 0, (1, 0, 0))
    assert sorted(img for _, img, _ in bond_graph(s).neighbors(0)) == [(-1, 0, 0), (1, 0, 0)]


def test_edge_classes():
    sites = {
        "Zn1": ("Zn", (2, 2, 2)),
        "O1": ("O", (4, 2, 2)),
        "C1": ("C", (2, 4.6, 2)),
        "O2": ("O", (12, 2, 2)),
        "H2": ("H", (12.97, 2, 2)),
        "O3": ("O", (14.77, 2, 2)),
        "Cu1": ("Cu", (2, 12, 2)),
        "Cu2": ("Cu", (4.64, 12, 2)),
        "Zn2": ("Zn", (12, 12, 2)),
        "Zn3": ("Zn", (15.2, 12, 2)),
        "C2": ({"C": 0.5}, (22, 2, 2)),
        "O4": ("O", (23.4, 2, 2)),
        "H3": ("H", (22, 12, 2)),
        "H4": ("H", (22.5, 12, 2)),
    }
    labels = list(sites)
    s = Structure(
        Lattice.cubic(30),
        [species for species, _ in sites.values()],
        [xyz for _, xyz in sites.values()],
        coords_are_cartesian=True,
        labels=labels,
    )
    edges = {(labels[b.i], labels[b.j]): (b.kind, b.reason) for b in bond_graph(s).bonds}
    assert edges == {
        ("Zn1", "O1"): ("coordination", None),
        ("Zn1", "C1"): ("uncertain", "distance"),
        ("O2", "H2"): ("covalent", None),
        ("Cu1", "Cu2"): ("uncertain", "metal-metal"),
        ("C2", "O4"): ("uncertain", "disorder"),
        ("H3", "H4"): ("uncertain", "short"),
    }


def test_bonds_command(write, capsys):
    path = write([("C1", "C", 0.95, 0.25, 0.25, 1), ("O1", "O", 0.07, 0.25, 0.25, 1)])
    assert main(["bonds", str(path)]) == 0
    assert "共价 1  配位 0  不确定 0" in capsys.readouterr().out
