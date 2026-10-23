from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from mofmotion.graph import _symbols, bond_graph
from mofmotion.match import CUTOFF, Match, match
from mofmotion.units import _RELIABLE, StructuralUnits, structural_units


@dataclass
class Strain:
    cell_a: tuple[float, ...]
    cell_b: tuple[float, ...]
    volume: float
    principal: np.ndarray
    directions: np.ndarray
    axes: list[tuple[str, float]]

    @property
    def magnitude(self) -> float:
        return float(np.abs(self.principal).max())


@dataclass
class UnitMotion:
    unit: int
    matched: float
    translation: np.ndarray
    angle: float
    axis: np.ndarray
    mode: str
    rmsd: float
    displacement: float


@dataclass
class Internals:
    kind: np.ndarray
    sites: np.ndarray
    names: list[str]
    a: np.ndarray
    b: np.ndarray
    junction: np.ndarray

    @property
    def delta(self) -> np.ndarray:
        d = self.b - self.a
        torsion = self.kind == "torsion"
        d[torsion] = (d[torsion] + 180) % 360 - 180
        return d

    def top(self, kind: str, n: int = 10, junction: bool | None = None) -> np.ndarray:
        mask = self.kind == kind
        if junction is not None:
            mask &= self.junction == junction
        index = np.flatnonzero(mask)
        seen, out = set(), []
        for k in index[np.argsort(-np.abs(self.delta[index]))]:
            key = (self.names[k], round(abs(self.a[k]), 1), round(abs(self.b[k]), 1))
            if key not in seen:
                seen.add(key)
                out.append(k)
                if len(out) == n:
                    break
        return np.array(out, dtype=int)


@dataclass
class BondChange:
    kind: str
    names: str
    length_a: float
    length_b: float


@dataclass
class Motion:
    match: Match = field(repr=False)
    units_a: StructuralUnits = field(repr=False)
    units_b: StructuralUnits = field(repr=False)
    strain: Strain
    units: list[UnitMotion]
    nets: dict[int, np.ndarray]
    internals: Internals = field(repr=False)
    bonds: list[BondChange]
    coordination: list[tuple[str, int, int]]


def _strain(m: Match) -> Strain:
    values, vectors = np.linalg.eigh(m.stretch)
    lattice = m.a.lattice.matrix
    basis = lattice / np.linalg.norm(lattice, axis=1)[:, None]
    axes = []
    for v in vectors.T:
        cos = np.abs(basis @ v)
        k = int(cos.argmax())
        axes.append(("abc"[k], float(np.degrees(np.arccos(min(cos[k], 1.0))))))

    def cell(s):
        return tuple(round(float(x), 4) for x in (*s.lattice.abc, *s.lattice.angles))

    volume = float(np.linalg.det(m.stretch) - 1)
    return Strain(cell(m.a), cell(m.b), volume, values - 1, vectors.T, axes)


def _rigid(x: np.ndarray, y: np.ndarray, finite: bool):
    from scipy.spatial.transform import Rotation

    translation = (y - x).mean(axis=0)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    rmsd = float(np.sqrt(((yc - xc) ** 2).sum(axis=1).mean()))
    angle, axis, mode = 0.0, np.zeros(3), ""
    if not finite or len(x) < 3:
        return translation, angle, axis, mode, rmsd
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    spread = s / np.sqrt(len(x))
    if spread[1] < 0.3:
        return translation, angle, axis, mode, rmsd
    rotation, _ = Rotation.align_vectors(yc, xc)
    rmsd = float(np.sqrt(((yc - rotation.apply(xc)) ** 2).sum(axis=1).mean()))
    vector = rotation.as_rotvec()
    angle = float(np.degrees(np.linalg.norm(vector)))
    if angle > 1e-6:
        axis = vector / np.linalg.norm(vector)
        if spread[0] > 1.5 * spread[1]:
            mode = "spin" if abs(axis @ vt[0]) > np.cos(np.radians(30)) else "tilt"
        else:
            mode = "rotate"
    return translation, angle, axis, mode, rmsd


def _unit_motions(m: Match, ua: StructuralUnits, heavy: np.ndarray):
    lb = m.b.lattice.matrix
    fa, fb = m.a.frac_coords, m.positions()
    paired = ~np.isnan(fb[:, 0])
    motions, nets = [], defaultdict(list)
    for index, unit in enumerate(ua.units):
        ok = paired[unit.sites]
        use = ok & heavy[unit.sites]
        use = use if use.any() else ok
        if not use.any():
            nan = np.full(3, np.nan)
            motions.append(UnitMotion(index, 0.0, nan, np.nan, nan, "", np.nan, np.nan))
            continue
        x = (fa[unit.sites[use]] + unit.shifts[use]) @ lb
        y = (fb[unit.sites[use]] + unit.shifts[use]) @ lb
        if unit.kind != "free":
            nets[unit.net].append(y - x)
        moved = _rigid(x, y, unit.dimension == 0)
        largest = float(np.linalg.norm(y - x, axis=1).max())
        motions.append(UnitMotion(index, float(ok.mean()), *moved, largest))
    return motions, {k: np.concatenate(v).mean(axis=0) for k, v in sorted(nets.items())}


def _length(p, q) -> np.ndarray:
    return np.linalg.norm(q - p, axis=-1)


def _angle(p, q, r) -> np.ndarray:
    u, v = p - q, r - q
    cos = (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _dihedral(p0, p1, p2, p3) -> np.ndarray:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def _internals(m: Match, ua: StructuralUnits, heavy: np.ndarray) -> Internals:
    fa, fb = m.a.frac_coords, m.positions()
    ok = heavy & ~np.isnan(fb[:, 0])
    neighbors, bonds = defaultdict(list), []
    for bond in ua.graph.bonds:
        if bond.reason in _RELIABLE and ok[bond.i] and ok[bond.j]:
            t = np.array(bond.image)
            neighbors[bond.i].append((bond.j, t))
            neighbors[bond.j].append((bond.i, -t))
            bonds.append((bond.i, bond.j, t))

    chains: dict[str, list] = {"bond": [], "angle": [], "torsion": []}
    zero = np.zeros(3, dtype=int)
    for i, j, t in bonds:
        chains["bond"].append([(i, zero), (j, t)])
    for c, around in neighbors.items():
        for p in range(len(around)):
            for q in range(p + 1, len(around)):
                chains["angle"].append([around[p], (c, zero), around[q]])
    for i, j, t in bonds:
        left = [(h, th) for h, th in neighbors[i] if not (h == j and (th == t).all())]
        right = [(k, t + tk) for k, tk in neighbors[j] if not (k == i and (tk == -t).all())]
        if left and right:
            h, k = min(left, key=lambda x: (x[0], tuple(x[1]))), min(right, key=lambda x: x[0])
            if not (h[0] == k[0] and (h[1] == k[1]).all()):
                chains["torsion"].append([h, (i, zero), (j, t), k])

    labels = [site.label for site in m.a]
    kinds, sites, names, va, vb, junction = [], [], [], [], [], []
    measure = {"bond": _length, "angle": _angle, "torsion": _dihedral}
    for kind, items in chains.items():
        if not items:
            continue
        idx = np.array([[s for s, _ in chain] for chain in items])
        img = np.array([[t for _, t in chain] for chain in items])
        pa = (fa[idx] + img) @ m.a.lattice.matrix
        pb = (fb[idx] + img) @ m.b.lattice.matrix
        a, b = (measure[kind](*p.transpose(1, 0, 2)) for p in (pa, pb))
        keep = np.ones(len(idx), bool)
        if kind == "torsion":
            for p in (pa, pb):
                keep &= _angle(p[:, 0], p[:, 1], p[:, 2]) < 170
                keep &= _angle(p[:, 1], p[:, 2], p[:, 3]) < 170
        for row, x, y in zip(idx[keep], a[keep], b[keep], strict=True):
            kinds.append(kind)
            sites.append(np.pad(row, (0, 4 - len(row)), constant_values=-1))
            names.append("–".join(labels[s] for s in row))
            va.append(x)
            vb.append(y)
            junction.append(len(set(ua.unit_of[row])) > 1)
    return Internals(
        np.array(kinds, dtype=str),
        np.array(sites, dtype=int).reshape(-1, 4),
        names,
        np.array(va, dtype=float),
        np.array(vb, dtype=float),
        np.array(junction, dtype=bool),
    )


def _key(i: int, j: int, image) -> tuple:
    image = tuple(int(x) for x in image)
    if (j, tuple(-x for x in image)) < (i, image):
        return j, i, tuple(-x for x in image)
    return i, j, image


def _bond_changes(m: Match, ua: StructuralUnits):
    from pymatgen.core import Element

    gb = bond_graph(m.b)
    sym_a, sym_b = _symbols(m.a), _symbols(m.b)
    a_of = np.full(len(m.b), -1)
    a_of[m.pairs[:, 1]] = m.pairs[:, 0]
    b_of = np.full(len(m.a), -1)
    b_of[m.pairs[:, 0]] = m.pairs[:, 1]
    n = np.zeros((len(m.b), 3), dtype=int)
    n[m.pairs[:, 1]] = np.rint(m.target - m.b.frac_coords[m.pairs[:, 1]]).astype(int)
    labels_a, labels_b = [s.label for s in m.a], [s.label for s in m.b]

    found_a, found_b, changes = {}, {}, []
    for bond in ua.graph.bonds:
        if sym_a[bond.i] == "H" or sym_a[bond.j] == "H":
            continue
        if b_of[bond.i] >= 0 and b_of[bond.j] >= 0:
            found_a[_key(bond.i, bond.j, bond.image)] = bond
        elif bond.reason in _RELIABLE and (b_of[bond.i] >= 0) != (b_of[bond.j] >= 0):
            names = f"{labels_a[bond.i]}–{labels_a[bond.j]}"
            changes.append(BondChange("released", names, bond.length, np.nan))
    for bond in gb.bonds:
        if sym_b[bond.i] == "H" or sym_b[bond.j] == "H":
            continue
        p, q = a_of[bond.i], a_of[bond.j]
        if p >= 0 and q >= 0:
            image = np.array(bond.image) + n[bond.i] - n[bond.j]
            found_b[_key(p, q, image)] = bond
        elif bond.reason in _RELIABLE and (p >= 0) != (q >= 0):
            names = f"{labels_b[bond.i]}–{labels_b[bond.j]}"
            changes.append(BondChange("captured", names, np.nan, bond.length))
    for key, bond in found_a.items():
        if bond.reason in _RELIABLE and key not in found_b:
            names = f"{labels_a[key[0]]}–{labels_a[key[1]]}"
            changes.append(BondChange("broken", names, bond.length, np.nan))
    for key, bond in found_b.items():
        if bond.reason in _RELIABLE and key not in found_a:
            names = f"{labels_a[key[0]]}–{labels_a[key[1]]}"
            changes.append(BondChange("formed", names, np.nan, bond.length))

    metal_a = np.array([Element(s).is_metal for s in sym_a])
    metal_b = np.array([Element(s).is_metal for s in sym_b])
    cn_a, cn_b = np.zeros(len(m.a), int), np.zeros(len(m.b), int)
    for graph, metal, cn in ((ua.graph, metal_a, cn_a), (gb, metal_b, cn_b)):
        for bond in graph.bonds:
            if bond.reason in _RELIABLE and metal[bond.i] != metal[bond.j]:
                cn[bond.i if metal[bond.i] else bond.j] += 1
    coordination = [
        (labels_a[k], int(cn_a[k]), int(cn_b[b_of[k]]))
        for k in np.flatnonzero(metal_a & (b_of >= 0) & (cn_a > 0))
    ]
    unique = {f"{c.kind} {c.names} {c.length_a:.2f} {c.length_b:.2f}": c for c in changes}
    return list(unique.values()), list(dict.fromkeys(coordination))


def motion(a, b, *, cutoff: float = CUTOFF) -> Motion:
    ua, ub = structural_units(bond_graph(a)), structural_units(bond_graph(b))

    def free(units):
        return np.array([units.units[u].kind == "free" for u in units.unit_of])

    m = match(a, b, free_a=free(ua), free_b=free(ub), cutoff=cutoff)
    if len(m.a) != len(a):
        ua = structural_units(bond_graph(m.a))
    if len(m.b) != len(b):
        ub = structural_units(bond_graph(m.b))
    heavy = np.array([s != "H" for s in _symbols(m.a)])
    units, nets = _unit_motions(m, ua, heavy)
    changes, coordination = _bond_changes(m, ua)
    return Motion(
        m, ua, ub, _strain(m), units, nets, _internals(m, ua, heavy), changes, coordination
    )
