from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from mofmotion.graph import BondGraph, _symbols

_RELIABLE = {None, "disorder", "metal-metal"}
_KINDS = {"node": "节点", "linker": "连接体", "terminal": "端基", "free": "游离"}


@dataclass
class Unit:
    kind: str
    sites: np.ndarray
    shifts: np.ndarray
    dimension: int
    formula: str
    net: int


@dataclass
class StructuralUnits:
    graph: BondGraph = field(repr=False)
    units: list[Unit]
    net_dimensions: list[int]
    unit_of: np.ndarray = field(repr=False)

    def coords(self, u: int) -> np.ndarray:
        unit, structure = self.units[u], self.graph.structure
        return (structure.frac_coords[unit.sites] + unit.shifts) @ structure.lattice.matrix

    def summary(self) -> str:
        nets = Counter(self.net_dimensions)
        lines = [
            f"{len(self.units)} 单元  网络 "
            + "，".join(f"{d} 维 ×{n}" for d, n in sorted(nets.items(), reverse=True))
        ]
        groups = Counter((u.kind, u.formula, u.dimension) for u in self.units)
        order = list(_KINDS)
        for (kind, formula, dim), n in sorted(groups.items(), key=lambda g: order.index(g[0][0])):
            extent = f"（{dim} 维无限）" if dim else ""
            lines.append(f"  {_KINDS[kind]} {formula}{extent} ×{n}")
        skipped = Counter(b.reason for b in self.graph.bonds if b.reason not in _RELIABLE)
        if skipped:
            lines.append(
                "  未用于划分的不确定边："
                + "，".join(f"{reason} {n}" for reason, n in sorted(skipped.items()))
            )
        return "\n".join(lines)


def _components(n: int, edges: list[tuple[int, int, np.ndarray]]):
    adjacency = defaultdict(list)
    for i, j, image in edges:
        adjacency[i].append((j, image))
        adjacency[j].append((i, -image))
    label = np.full(n, -1)
    shift = np.zeros((n, 3), dtype=int)
    loops: list[list[np.ndarray]] = []
    for root in range(n):
        if label[root] >= 0:
            continue
        label[root], stack, found = len(loops), [root], []
        while stack:
            a = stack.pop()
            for b, image in adjacency[a]:
                if label[b] < 0:
                    label[b], shift[b] = label[root], shift[a] + image
                    stack.append(b)
                elif (loop := shift[a] + image - shift[b]).any():
                    found.append(loop)
        loops.append(found)
    dims = [int(np.linalg.matrix_rank(np.array(f))) if f else 0 for f in loops]
    return label, shift, dims


def structural_units(graph: BondGraph) -> StructuralUnits:
    from pymatgen.core import Composition, Element

    structure = graph.structure
    n = len(structure)
    symbols = _symbols(structure)
    metal = np.array([Element(s).is_metal for s in symbols])
    heavy = np.array([s not in {"H", "C"} for s in symbols], dtype=float)
    carbon = np.array([s == "C" for s in symbols], dtype=float)
    bonds = [b for b in graph.bonds if b.reason in _RELIABLE]
    edges = [(b.i, b.j, np.array(b.image)) for b in bonds]

    node_metal = np.zeros(n, dtype=bool)
    for b in bonds:
        node_metal[b.i] |= metal[b.i] and not carbon[b.j]
        node_metal[b.j] |= metal[b.j] and not carbon[b.i]
    metal = node_metal

    molecule, _, _ = _components(n, [e for e in edges if not metal[e[0]] and not metal[e[1]]])
    touches = np.zeros(n)
    for b in bonds:
        if metal[b.i] != metal[b.j]:
            touches[b.j if metal[b.i] else b.i] = 1
    m = molecule.max() + 1
    small = (np.bincount(molecule, heavy, m) <= 1) & (np.bincount(molecule, carbon, m) == 0)
    in_node = metal | (small & np.bincount(molecule, touches, m).astype(bool))[molecule]

    label, shift, dims = _components(n, [e for e in edges if in_node[e[0]] == in_node[e[1]]])
    net, _, net_dims = _components(n, edges)
    count = label.max() + 1
    for u in range(count):
        if not dims[u]:
            members = label == u
            mean = (structure.frac_coords[members] + shift[members]).mean(axis=0)
            shift[members] -= np.floor(mean).astype(int)

    partners = defaultdict(set)
    for i, j, image in edges:
        if label[i] != label[j]:
            t = tuple(shift[i] + image - shift[j])
            partners[label[i]].add((label[j], t))
            partners[label[j]].add((label[i], tuple(-x for x in t)))

    units = []
    for u in range(count):
        sites, links = np.flatnonzero(label == u), len(partners[u])
        if in_node[sites[0]]:
            kind = "node" if links or dims[u] else "free"
        else:
            kind = ("free", "terminal", "linker")[min(links, 2)]
        formula = sum((structure[k].species for k in sites), Composition()).hill_formula
        units.append(
            Unit(kind, sites, shift[sites], dims[u], formula.replace(" ", ""), int(net[sites[0]]))
        )
    return StructuralUnits(graph, units, net_dims, label)
