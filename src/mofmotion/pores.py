from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from pymatgen.core import Composition

from mofmotion.graph import _symbols
from mofmotion.units import _components

PROBE = 1.65
SPACING = 0.2
_MIN_VOXELS = 8


@dataclass
class Channel:
    label: int
    dimension: int
    fraction: float
    lcd: float


@dataclass
class Pores:
    structure: object = field(repr=False)
    probe: float
    shape: tuple[int, int, int]
    distance: np.ndarray = field(repr=False)
    nearest: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    channels: list[Channel]
    lcd: float
    pld: float
    dimension: int
    bottleneck: np.ndarray
    window: np.ndarray
    void: float
    accessible: float
    volume: float
    lining: np.ndarray
    removed: str

    def exposure(self) -> dict[str, float]:
        counts = Counter()
        for symbol, n in zip(_symbols(self.structure), self.lining, strict=True):
            counts[symbol] += int(n)
        total = sum(counts.values()) or 1
        return {s: n / total for s, n in sorted(counts.items(), key=lambda x: -x[1]) if n}

    def summary(self) -> str:
        labels = [site.label for site in self.structure]
        channels = Counter(c.dimension for c in self.channels)
        kinds = "，".join(
            f"{d} 维孔道 ×{n}" if d else f"封闭孔穴 ×{n}" for d, n in sorted(channels.items())
        )
        lines = [
            f"LCD {self.lcd:.2f} Å  PLD {self.pld:.2f} Å  探针 {self.probe:.2f} Å"
            f"（{'可通过，' + str(self.dimension) + ' 维' if self.dimension else '不可通过'}）",
            f"  孔隙率 {self.void:.3f}  可及 {self.accessible:.3f}（{self.volume:.3f} cm³/g）"
            + (f"  {kinds}" if kinds else ""),
            "  瓶颈窗口：" + (" ".join(labels[k] for k in self.window) or "—"),
        ]
        if self.removed:
            lines.insert(1, f"  已去除客体 {self.removed} 后计算")
        exposure = self.exposure()
        if exposure:
            lines.append(
                "  孔壁暴露：" + " ".join(f"{s} {f:.0%}" for s, f in list(exposure.items())[:6])
            )
        return "\n".join(lines)


def _radii(symbols: list[str]) -> np.ndarray:
    from pymatgen.core import Element

    out = []
    for s in symbols:
        r = Element(s).van_der_waals_radius
        out.append(float(r) if r else 2.0)
    return np.array(out)


def _images(lattice: np.ndarray, reach: float) -> np.ndarray:
    volume = abs(np.linalg.det(lattice))
    heights = [volume / np.linalg.norm(np.cross(lattice[i - 2], lattice[i - 1])) for i in range(3)]
    n = [int(np.ceil(reach / h)) + 1 for h in heights]
    return np.array(list(product(*(range(-k, k + 1) for k in n))))


def _field(frac, lattice, radii, shape):
    from scipy.spatial import cKDTree

    axes = [np.arange(n) / n for n in shape]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) @ lattice
    reach = 0.5 * np.linalg.norm(lattice, axis=1).sum() + radii.max()
    images = _images(lattice, reach)
    d = np.full(len(points), np.inf)
    nearest = np.zeros(len(points), dtype=int)
    for r in np.unique(radii):
        members = np.flatnonzero(radii == r)
        tiled = ((frac[members] % 1)[None] + images[:, None]).reshape(-1, 3) @ lattice
        dist, k = cKDTree(tiled).query(points, workers=-1)
        dist -= r
        better = dist < d
        d[better] = dist[better]
        nearest[better] = members[k[better] % len(members)]
    return d.reshape(shape), nearest.reshape(shape)


def _label(mask: np.ndarray):
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if not n:
        return np.full(mask.shape, -1), []
    edges = []
    for axis in range(3):
        hi, lo = np.take(labels, -1, axis=axis), np.take(labels, 0, axis=axis)
        both = (hi > 0) & (lo > 0)
        image = np.eye(3, dtype=int)[axis]
        for p, q in np.unique(np.stack([hi[both], lo[both]], 1), axis=0):
            edges.append((int(p) - 1, int(q) - 1, image))
    component, _, dims = _components(n, edges)
    return np.where(labels > 0, component[labels - 1], -1), dims


def _percolates(d: np.ndarray, t: float) -> bool:
    _, dims = _label(d >= t)
    return any(dims)


def _occupiable(d, centers, lattice, probe):
    from scipy.spatial import cKDTree

    shape = d.shape
    occupied = centers.copy()
    if not centers.any():
        return occupied
    grid = np.stack(np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij"), -1)
    source = grid[centers]
    volume = abs(np.linalg.det(lattice))
    margin = np.array(
        [
            probe * np.linalg.norm(np.cross(lattice[i - 2], lattice[i - 1])) / volume
            for i in range(3)
        ]
    )
    tiled = []
    for shift in product((-1, 0, 1), repeat=3):
        moved = source + shift
        inside = ((moved > -margin) & (moved < 1 + margin)).all(axis=1)
        tiled.append(moved[inside])
    tree = cKDTree(np.concatenate(tiled) @ lattice)
    candidate = (d > 0) & ~centers
    dist, _ = tree.query(grid[candidate] @ lattice, distance_upper_bound=probe, workers=-1)
    occupied[candidate] = np.isfinite(dist)
    return occupied


def pores(
    structure,
    *,
    probe: float = PROBE,
    spacing: float = SPACING,
    keep: np.ndarray | None = None,
    shape: tuple[int, int, int] | None = None,
) -> Pores:
    if keep is None:
        from mofmotion.graph import bond_graph
        from mofmotion.units import structural_units

        units = structural_units(bond_graph(structure))
        keep = np.array([units.units[u].kind != "free" for u in units.unit_of])
    keep = np.asarray(keep, bool)
    sites = np.flatnonzero(keep)
    lattice = structure.lattice.matrix
    if shape is None:
        shape = tuple(max(int(np.ceil(x / spacing)), 4) for x in structure.lattice.abc)
    radii = _radii([s for s, k in zip(_symbols(structure), keep, strict=True) if k])
    d, nearest = _field(structure.frac_coords[sites], lattice, radii, shape)
    nearest = sites[nearest]

    values = np.unique(d[d > 0])
    pld, bottleneck, window = 0.0, np.full(3, np.nan), np.zeros(0, int)
    if len(values) and _percolates(d, values[0]):
        lo, hi = 0, len(values) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if _percolates(d, values[mid]) else (lo, mid - 1)
        critical = values[lo]
        pld = 2 * float(critical)
        labels, dims = _label(d >= critical)
        through = np.isin(labels, np.flatnonzero(np.array(dims) > 0)) & (d == critical)
        voxel = np.argwhere(through)[0]
        bottleneck = voxel / np.array(shape)
        window = _window(structure, sites, radii, bottleneck, critical + spacing)

    labels, dims = _label(d >= probe)
    channels = []
    for c, dim in enumerate(dims):
        members = labels == c
        if members.sum() >= _MIN_VOXELS or dim:
            channels.append(Channel(c, dim, float(members.mean()), 2 * float(d[members].max())))
    channels.sort(key=lambda ch: (-ch.dimension, -ch.fraction))
    through = np.isin(labels, [c.label for c in channels if c.dimension])
    dimension = max((c.dimension for c in channels), default=0)
    occupied = _occupiable(d, through, lattice, probe)

    lining = np.bincount(nearest[(labels >= 0) & (d < probe + spacing)], minlength=len(structure))
    mass = sum(structure[k].species.weight for k in sites)
    accessible = float(occupied.mean())
    volume = accessible * structure.volume * 0.602214076 / mass if mass else 0.0
    guests = sum((structure[k].species for k in np.flatnonzero(~keep)), Composition())
    return Pores(
        structure,
        probe,
        tuple(shape),
        d,
        nearest,
        labels,
        channels,
        2 * float(d.max()),
        pld,
        dimension,
        bottleneck,
        window,
        float((d > 0).mean()),
        accessible,
        float(volume),
        lining,
        guests.hill_formula.replace(" ", "") if guests else "",
    )


def _window(structure, sites, radii, point, limit) -> np.ndarray:
    lattice = structure.lattice.matrix
    delta = structure.frac_coords[sites] - point
    delta -= np.rint(delta)
    shifts = np.array(list(product((-1, 0, 1), repeat=3)))
    dist = np.linalg.norm((delta[:, None] + shifts[None]) @ lattice, axis=2).min(axis=1) - radii
    close = np.flatnonzero(dist <= limit)
    return sites[close[np.argsort(dist[close])]]


@dataclass
class PoreEvent:
    kind: str
    a: list[int]
    b: list[int]
    dimension_a: list[int]
    dimension_b: list[int]
    fraction_a: float
    fraction_b: float


def pore_events(pa: Pores, pb: Pores, overlap: float = 0.05) -> list[PoreEvent]:
    if pa.shape != pb.shape:
        raise ValueError("两态孔道网格形状不同，无法逐点对应")
    ca = {c.label: c for c in pa.channels}
    cb = {c.label: c for c in pb.channels}
    la, lb = pa.labels.ravel(), pb.labels.ravel()
    both = np.isin(la, list(ca)) & np.isin(lb, list(cb))
    edges = []
    if both.any():
        pairs, counts = np.unique(np.stack([la[both], lb[both]], 1), axis=0, return_counts=True)
        size_a = {k: np.count_nonzero(la == k) for k in ca}
        size_b = {k: np.count_nonzero(lb == k) for k in cb}
        for (p, q), n in zip(pairs, counts, strict=True):
            if n >= max(_MIN_VOXELS, overlap * min(size_a[p], size_b[q])):
                edges.append((int(p), int(q)))

    nodes = [("a", k) for k in ca] + [("b", k) for k in cb]
    index = {node: i for i, node in enumerate(nodes)}
    component, _, _ = _components(
        len(nodes), [(index["a", p], index["b", q], np.zeros(3, int)) for p, q in edges]
    )
    events = []
    for c in range(component.max() + 1 if len(nodes) else 0):
        members = [nodes[i] for i in np.flatnonzero(component == c)]
        a = sorted(k for side, k in members if side == "a")
        b = sorted(k for side, k in members if side == "b")
        if not b:
            kind = "closed"
        elif not a:
            kind = "opened"
        elif len(a) == 1 and len(b) == 1:
            kind = "kept"
        elif len(a) == 1:
            kind = "split"
        elif len(b) == 1:
            kind = "merged"
        else:
            kind = "rearranged"
        events.append(
            PoreEvent(
                kind,
                a,
                b,
                [ca[k].dimension for k in a],
                [cb[k].dimension for k in b],
                sum(ca[k].fraction for k in a),
                sum(cb[k].fraction for k in b),
            )
        )
    events.sort(key=lambda e: -(e.fraction_a + e.fraction_b))
    return events
