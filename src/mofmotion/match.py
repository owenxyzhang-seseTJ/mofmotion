from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from mofmotion.graph import _symbols

CUTOFF = 3.0
_SHIFTS = np.array(list(product((-1, 0, 1), repeat=3)))


@dataclass
class Match:
    a: object = field(repr=False)
    b: object = field(repr=False)
    transform: np.ndarray
    rotation: np.ndarray
    stretch: np.ndarray
    pairs: np.ndarray
    target: np.ndarray
    distance: np.ndarray

    @property
    def unmatched(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.setdiff1d(np.arange(len(self.a)), self.pairs[:, 0]),
            np.setdiff1d(np.arange(len(self.b)), self.pairs[:, 1]),
        )

    def positions(self) -> np.ndarray:
        out = np.full((len(self.a), 3), np.nan)
        out[self.pairs[:, 0]] = self.target
        return out


def _settings(la: np.ndarray, lb: np.ndarray, keep: int = 6) -> list[np.ndarray]:
    from pymatgen.core import Lattice

    ra = Lattice(la).get_niggli_reduced_lattice().matrix
    rb = Lattice(lb).get_niggli_reduced_lattice().matrix
    pa, pb = np.rint(ra @ np.linalg.inv(la)), np.rint(rb @ np.linalg.inv(lb))
    ms = np.array(list(product((-1, 0, 1), repeat=9)), dtype=float).reshape(-1, 3, 3)
    ms = ms[np.isclose(np.linalg.det(ms), np.linalg.det(pa) / np.linalg.det(pb))]
    g = np.linalg.inv(ra) @ ms @ rb
    strain = np.linalg.norm(g @ g.transpose(0, 2, 1) - np.eye(3), axis=(1, 2))
    order = np.argsort(strain)
    order = order[strain[order] <= strain[order[0]] + 0.05][:keep]
    return [np.rint(np.linalg.inv(pa) @ ms[k] @ pb).astype(int) for k in order]


def _assign(fa: np.ndarray, fb: np.ndarray, lattice: np.ndarray, cutoff: float):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    empty = np.zeros(0, dtype=int)
    if not len(fa) or not len(fb):
        return empty, empty, np.zeros(0), np.zeros((0, 3))
    wa, wb = fa % 1, fb % 1
    images = (wb[None] + _SHIFTS[:, None]).reshape(-1, 3)
    hits = cKDTree(images @ lattice).query_ball_point(wa @ lattice, cutoff)
    i = np.repeat(np.arange(len(fa)), [len(h) for h in hits])
    k = np.concatenate([np.asarray(h, dtype=int) for h in hits]) if len(i) else empty
    if not len(i):
        return empty, empty, np.zeros(0), np.zeros((0, 3))
    j, d = k % len(fb), np.linalg.norm((images[k] - wa[i]) @ lattice, axis=1)
    order = np.argsort(d)
    i, j, k, d = i[order], j[order], k[order], d[order]
    _, first = np.unique(i * len(fb) + j, return_index=True)
    i, j, k, d = i[first], j[first], k[first], d[first]
    cost = np.full((len(fa), len(fb)), 10 * cutoff)
    cost[i, j] = d
    image = np.zeros((len(fa), len(fb)), dtype=int)
    image[i, j] = k // len(fb)
    ra, rb = linear_sum_assignment(cost)
    ok = cost[ra, rb] <= cutoff
    ra, rb = ra[ok], rb[ok]
    target = wb[rb] + _SHIFTS[image[ra, rb]] + np.floor(fa[ra])
    return ra, rb, cost[ra, rb], target


def _pairing(fa, fb, lattice, groups, cutoff):
    cost, parts = 0.0, []
    for ia, ib in groups:
        ra, rb, d, t = _assign(fa[ia], fb[ib], lattice, cutoff)
        cost += d.sum() + cutoff * (len(ia) + len(ib) - 2 * len(d)) / 2
        parts.append((ia[ra], ib[rb], d, t))
    a, b, d, t = (np.concatenate(x) for x in zip(*parts, strict=True)) if parts else ([],) * 4
    return cost, np.asarray(a, dtype=int), np.asarray(b, dtype=int), np.asarray(d), t


def match(a, b, *, free_a=None, free_b=None, cutoff: float = CUTOFF) -> Match:
    from pymatgen.core import Lattice, Structure
    from scipy.linalg import polar

    sym_a, sym_b = np.array(_symbols(a)), np.array(_symbols(b))
    free_a = np.zeros(len(a), bool) if free_a is None else np.asarray(free_a, bool)
    free_b = np.zeros(len(b), bool) if free_b is None else np.asarray(free_b, bool)
    keys = sorted(set(zip(sym_a, free_a, strict=True)) & set(zip(sym_b, free_b, strict=True)))

    def groups(heavy_framework: bool):
        out = []
        for symbol, free in keys:
            if heavy_framework and (free or symbol == "H"):
                continue
            ia = np.flatnonzero((sym_a == symbol) & (free_a == free))
            ib = np.flatnonzero((sym_b == symbol) & (free_b == free))
            out.append((ia, ib))
        return out

    search = groups(True) or groups(False)
    ia0, ib0 = min(search, key=lambda g: (len(g[0]), -len(g[1])))
    la, fa = a.lattice.matrix, a.frac_coords
    best = None
    for t in _settings(la, b.lattice.matrix):
        fb = b.frac_coords @ np.linalg.inv(t)
        for origin in fa[ia0[0]] - fb[ib0]:
            cost, pa, _, _, target = _pairing(fa, fb + origin, la, search, cutoff)
            if best is None or cost < best[0] - 1e-9:
                drift = (target - fa[pa]).mean(axis=0) if len(pa) else 0
                best = (cost, t, fb, origin - drift)
    _, t, fb, origin = best
    fb = fb + origin
    _, pa, pb, d, target = _pairing(fa, fb, la, groups(False), cutoff)

    rotation, stretch = polar(np.linalg.inv(la) @ t @ b.lattice.matrix, side="left")
    mapped = Structure(
        Lattice(la @ stretch),
        [site.species for site in b],
        fb % 1,
        labels=[site.label for site in b],
    )
    order = np.argsort(pa)
    return Match(
        a, mapped, t, rotation, stretch, np.stack([pa, pb], 1)[order], target[order], d[order]
    )
