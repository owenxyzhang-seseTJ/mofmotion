from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mofmotion.motion import Motion
from mofmotion.pores import PROBE, SPACING, pores

METRICS = ("pld", "lcd", "accessible")
_MOVED = 0.1


@dataclass
class Effect:
    factor: str
    forward: dict[str, float]
    backward: dict[str, float]


@dataclass
class Attribution:
    a: dict[str, float]
    b: dict[str, float]
    effects: list[Effect]
    interaction: dict[str, float]

    def summary(self) -> str:
        lines = ["孔道变化归因（正向 / 反向，单位 Å，可及分数为体积比）"]
        for metric in METRICS:
            total = self.b[metric] - self.a[metric]
            if abs(total) < 0.005 and all(
                abs(e.forward[metric]) < 0.005 and abs(e.backward[metric]) < 0.005
                for e in self.effects
            ):
                continue
            parts = [
                f"{e.factor} {e.forward[metric]:+.2f}/{e.backward[metric]:+.2f}"
                for e in self.effects
            ]
            lines.append(
                f"  {metric.upper() if metric != 'accessible' else '可及'} "
                f"{self.a[metric]:.2f} → {self.b[metric]:.2f}（{total:+.2f}）："
                + "；".join(parts)
                + f"；耦合 {self.interaction[metric]:+.2f}"
            )
        return "\n".join(lines)


def _metrics(p) -> dict[str, float]:
    return {"pld": p.pld, "lcd": p.lcd, "accessible": p.accessible}


def attribute(mo: Motion, *, probe: float = PROBE, spacing: float = SPACING) -> Attribution:
    from pymatgen.core import Structure

    from mofmotion.units import _KINDS

    m, ua = mo.match, mo.units_a
    keep = np.array([ua.units[u].kind != "free" for u in ua.unit_of])
    fa, fb = m.a.frac_coords, m.positions()
    groups: dict[tuple[str, str], list[int]] = {}
    for um in mo.units:
        unit = ua.units[um.unit]
        if unit.kind != "free" and um.displacement >= _MOVED:
            groups.setdefault((unit.kind, unit.formula), []).append(um.unit)
    factors = [("晶格", None)] + [
        (f"{_KINDS[kind]} {formula} ×{len(us)}", np.isin(ua.unit_of, us))
        for (kind, formula), us in groups.items()
    ]
    shape = tuple(max(int(np.ceil(x / spacing)), 4) for x in m.a.lattice.abc)
    species = [site.species for site in m.a]
    labels = [site.label for site in m.a]
    cache: dict[frozenset, dict[str, float]] = {}

    def evaluate(chosen: frozenset) -> dict[str, float]:
        if chosen not in cache:
            frac = fa.copy()
            for k in chosen - {0}:
                moved = factors[k][1] & ~np.isnan(fb[:, 0])
                frac[moved] = fb[moved]
            lattice = m.b.lattice if 0 in chosen else m.a.lattice
            hybrid = Structure(lattice, species, frac, labels=labels)
            cache[chosen] = _metrics(
                pores(hybrid, probe=probe, spacing=spacing, keep=keep, shape=shape)
            )
        return cache[chosen]

    every = frozenset(range(len(factors)))
    base, full = evaluate(frozenset()), evaluate(every)
    effects = []
    for k, (name, _) in enumerate(factors):
        forward, backward = evaluate(frozenset({k})), evaluate(every - {k})
        effects.append(
            Effect(
                name,
                {x: forward[x] - base[x] for x in METRICS},
                {x: full[x] - backward[x] for x in METRICS},
            )
        )
    interaction = {x: full[x] - base[x] - sum(e.forward[x] for e in effects) for x in METRICS}
    return Attribution(base, full, effects, interaction)
