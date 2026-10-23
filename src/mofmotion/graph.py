from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from mofmotion.audit import _CLASH, _periodic_pairs

COVALENT = (1.15, 1.3)
COORDINATION = (1.2, 1.35)


@dataclass(frozen=True)
class Bond:
    i: int
    j: int
    image: tuple[int, int, int]
    length: float
    kind: str
    reason: str | None = None


@dataclass
class BondGraph:
    structure: object = field(repr=False)
    bonds: list[Bond]

    @cached_property
    def _adjacency(self) -> dict[int, list[tuple[int, tuple[int, int, int], Bond]]]:
        table = defaultdict(list)
        for bond in self.bonds:
            table[bond.i].append((bond.j, bond.image, bond))
            table[bond.j].append((bond.i, tuple(-x for x in bond.image), bond))
        return table

    def neighbors(self, i: int) -> list[tuple[int, tuple[int, int, int], Bond]]:
        return self._adjacency.get(i, [])

    def summary(self) -> str:
        counts = Counter(b.kind for b in self.bonds)
        lines = [
            f"{len(self.structure)} 位点  共价 {counts['covalent']}  "
            f"配位 {counts['coordination']}  不确定 {counts['uncertain']}"
        ]
        labels = [site.label for site in self.structure]
        groups = defaultdict(list)
        for b in self.bonds:
            if b.kind == "uncertain":
                groups[b.reason, *sorted((labels[b.i], labels[b.j]))].append(b.length)
        for (reason, a, b), lengths in sorted(groups.items()):
            span = (
                f"{min(lengths):.3f}–{max(lengths):.3f}"
                if len(lengths) > 1
                else f"{lengths[0]:.3f}"
            )
            lines.append(f"  不确定 {reason}：{a}–{b} ×{len(lengths)}，{span} Å")
        return "\n".join(lines)


def _symbols(structure) -> list[str]:
    symbols = []
    for site in structure:
        element = max(site.species.items(), key=lambda item: item[1])[0]
        symbol = getattr(element, "element", element).symbol
        symbols.append("H" if symbol in {"D", "T"} else symbol)
    return symbols


def bond_graph(structure) -> BondGraph:
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    from pymatgen.core import Element

    symbols = _symbols(structure)
    radii = np.array([CovalentRadius.radius[s] for s in symbols])
    metal = np.array([Element(s).is_metal for s in symbols])
    hydrogen = np.array([s == "H" for s in symbols])
    ordered = np.array([site.is_ordered for site in structure])

    cutoff = 2 * radii.max() * max(COVALENT[1], COORDINATION[1])
    i, j, images, d = _periodic_pairs(structure.lattice, structure.frac_coords, cutoff)
    q = d / (radii[i] + radii[j])
    coordination, metal_metal = metal[i] ^ metal[j], metal[i] & metal[j]
    sure = np.where(coordination, COORDINATION[0], COVALENT[0])
    loose = np.where(coordination, COORDINATION[1], np.where(metal_metal, sure, COVALENT[1]))
    keep = q <= loose
    i, j, images, d, q = i[keep], j[keep], images[keep], d[keep], q[keep]
    coordination, metal_metal, sure = coordination[keep], metal_metal[keep], sure[keep]

    short = d < np.vectorize(_CLASH.get)(hydrogen[i].astype(int) + hydrogen[j])
    reasons = np.select(
        [short, q > sure, ~(ordered[i] & ordered[j]), metal_metal],
        ["short", "distance", "disorder", "metal-metal"],
        default="",
    )
    kinds = np.where(reasons != "", "uncertain", np.where(coordination, "coordination", "covalent"))
    order = np.lexsort((d, j, i))
    bonds = [
        Bond(
            int(i[k]),
            int(j[k]),
            tuple(int(x) for x in images[k]),
            round(float(d[k]), 4),
            str(kinds[k]),
            str(reasons[k]) or None,
        )
        for k in order
    ]
    return BondGraph(structure, bonds)
