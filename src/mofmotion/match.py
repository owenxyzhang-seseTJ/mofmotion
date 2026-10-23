from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from mofmotion.graph import _symbols

CUTOFF = 3.0
COVERAGE = 0.95
MARGIN = 0.05
_KEEP = 6
_SHIFTS = np.array(list(product((-1, 0, 1), repeat=3)))
_EYE = np.eye(3, dtype=int)
_UNIMODULAR = np.array(list(product((-1, 0, 1), repeat=9)), dtype=float).reshape(-1, 3, 3)
_UNIMODULAR = _UNIMODULAR[np.isclose(np.abs(np.linalg.det(_UNIMODULAR)), 1)]


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
    supercell: tuple[np.ndarray, np.ndarray]
    coverage: float
    margin: float

    @property
    def unmatched(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.setdiff1d(np.arange(len(self.a)), self.pairs[:, 0]),
            np.setdiff1d(np.arange(len(self.b)), self.pairs[:, 1]),
        )

    @property
    def reliable(self) -> bool:
        return self.coverage >= COVERAGE and self.margin >= MARGIN

    def positions(self) -> np.ndarray:
        out = np.full((len(self.a), 3), np.nan)
        out[self.pairs[:, 0]] = self.target
        return out


def _niggli(lattice: np.ndarray):
    from pymatgen.core import Lattice

    reduced = Lattice(lattice).get_niggli_reduced_lattice().matrix
    return reduced, np.rint(reduced @ np.linalg.inv(lattice))


def _settings(la: np.ndarray, lb: np.ndarray):
    ra, pa = _niggli(la)
    rb, pb = _niggli(lb)
    ms = _UNIMODULAR[np.isclose(np.linalg.det(_UNIMODULAR), np.linalg.det(pa) / np.linalg.det(pb))]
    g = np.linalg.inv(ra) @ ms @ rb
    strain = np.linalg.norm(g @ g.transpose(0, 2, 1) - np.eye(3), axis=(1, 2))
    return strain, np.rint(np.linalg.inv(pa) @ ms @ pb).astype(int)


def _supercells(n: int) -> list[np.ndarray]:
    out = []
    for p, q in product(range(1, n + 1), repeat=2):
        if n % (p * q):
            continue
        r = n // (p * q)
        for x, y, z in product(range(q), range(r), range(r)):
            out.append(np.array([[p, x, y], [0, q, z], [0, 0, r]]))
    return out


def _ratio(sym_a, free_a, sym_b, free_b, keys) -> tuple[int, int]:
    heavy = [k for k in keys if not k[1] and k[0] != "H"] or keys

    def count(sym, free, key):
        return np.count_nonzero((sym == key[0]) & (free == key[1]))

    key = min(heavy, key=lambda k: count(sym_a, free_a, k))
    na, nb = count(sym_a, free_a, key), count(sym_b, free_b, key)
    if nb % na == 0:
        return nb // na, 1
    if na % nb == 0:
        return 1, na // nb
    return 1, 1


def _groups(sym_a, free_a, sym_b, free_b, heavy_framework: bool):
    keys = sorted(set(zip(sym_a, free_a, strict=True)) & set(zip(sym_b, free_b, strict=True)))
    out = []
    for symbol, free in keys:
        if heavy_framework and (free or symbol == "H"):
            continue
        ia = np.flatnonzero((sym_a == symbol) & (free_a == free))
        ib = np.flatnonzero((sym_b == symbol) & (free_b == free))
        out.append((ia, ib))
    return out


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


def _same(x: np.ndarray, y: np.ndarray, tol: float = 0.1) -> bool:
    return len(x) == len(y) and bool(np.allclose(x, y, atol=tol))


def _expand(s, free: np.ndarray, h: np.ndarray):
    if (h == _EYE).all():
        return s, free
    return s * h, np.repeat(free, round(abs(np.linalg.det(h))))


def match(a, b, *, free_a=None, free_b=None, cutoff: float = CUTOFF) -> Match:
    from pymatgen.core import Lattice, Structure
    from scipy.linalg import polar

    sym_a, sym_b = np.array(_symbols(a)), np.array(_symbols(b))
    free_a = np.zeros(len(a), bool) if free_a is None else np.asarray(free_a, bool)
    free_b = np.zeros(len(b), bool) if free_b is None else np.asarray(free_b, bool)
    keys = sorted(set(zip(sym_a, free_a, strict=True)) & set(zip(sym_b, free_b, strict=True)))
    ka, kb = _ratio(sym_a, free_a, sym_b, free_b, keys)
    if ka > 1:
        cells = [(h, _EYE) for h in _supercells(ka)]
    else:
        cells = [(_EYE, h) for h in _supercells(kb)]

    options = []
    for c, (ha, hb) in enumerate(cells):
        strain, ts = _settings(ha @ a.lattice.matrix, hb @ b.lattice.matrix)
        options += [(s, c, t) for s, t in zip(strain, ts, strict=True)]
    options.sort(key=lambda o: o[0])
    options = [o for o in options if o[0] <= options[0][0] + 0.05][: _KEEP * len(cells)]

    built, trials = {}, []
    for _, c, t in options:
        if c not in built:
            sa, fra = _expand(a, free_a, cells[c][0])
            sb, frb = _expand(b, free_b, cells[c][1])
            sya, syb = np.array(_symbols(sa)), np.array(_symbols(sb))
            search = _groups(sya, fra, syb, frb, True) or _groups(sya, fra, syb, frb, False)
            built[c] = (sa, sb, fra, frb, sya, syb, search)
        sa, sb, *_, search = built[c]
        ia0, ib0 = min(search, key=lambda g: (len(g[0]), -len(g[1])))
        la, fa = sa.lattice.matrix, sa.frac_coords
        fb = sb.frac_coords @ np.linalg.inv(t)
        for origin in fa[ia0[0]] - fb[ib0]:
            cost, pa, _, d, target = _pairing(fa, fb + origin, la, search, cutoff)
            drift = (target - fa[pa]).mean(axis=0) if len(pa) else 0
            trials.append((cost, c, t, fb, origin - drift, np.sort(d)))

    cost, c, t, fb, origin, sig = min(trials, key=lambda x: x[0])
    rivals = [x[0] for x in trials if not _same(x[5], sig)]
    margin = (min(rivals) - cost) / max(len(sig), 1) if rivals else np.inf

    sa, sb, fra, frb, sya, syb, _ = built[c]
    la, fa = sa.lattice.matrix, sa.frac_coords
    fb = fb + origin
    _, pa, pb, d, target = _pairing(fa, fb, la, _groups(sya, fra, syb, frb, False), cutoff)
    bound = sum(min(len(ia), len(ib)) for ia, ib in _groups(sya, fra, syb, frb, True))
    matched = np.count_nonzero(~fra[pa] & (sya[pa] != "H"))
    coverage = matched / bound if bound else 0.0

    rotation, stretch = polar(np.linalg.inv(la) @ t @ sb.lattice.matrix, side="left")
    mapped = Structure(
        Lattice(la @ stretch),
        [site.species for site in sb],
        fb % 1,
        labels=[site.label for site in sb],
    )
    order = np.argsort(pa)
    return Match(
        sa,
        mapped,
        t,
        rotation,
        stretch,
        np.stack([pa, pb], 1)[order],
        target[order],
        d[order],
        cells[c],
        coverage,
        margin,
    )
