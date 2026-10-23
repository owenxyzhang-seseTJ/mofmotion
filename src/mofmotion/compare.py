from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mofmotion.graph import _symbols
from mofmotion.match import CUTOFF
from mofmotion.motion import Motion, motion
from mofmotion.pores import PROBE, SPACING, PoreEvent, Pores, pore_events, pores

THRESHOLDS = {
    "strain": 0.02,
    "volume": 0.03,
    "translation": 0.3,
    "rotation": 5.0,
    "rmsd": 0.15,
    "angle": 10.0,
    "torsion": 15.0,
    "sliding": 0.3,
    "bond": 0.1,
    "pore": 0.2,
    "exposure": 0.10,
}
LABELS = {
    "breathing": "呼吸（各向异性形变）",
    "swelling": "整体膨胀",
    "shrinkage": "整体收缩",
    "sliding": "网络滑移",
    "rotation": "单元转动",
    "translation": "单元平移",
    "deformation": "单元内部变形",
    "hinge": "节点铰链",
    "torsion": "二面角扭转",
    "coordination": "配位变化",
    "guest": "客体变化",
    "gate-opening": "门控打开",
    "gate-closing": "门控关闭",
    "pore": "孔道变化",
    "environment": "局域环境变化",
    "rigid": "无显著运动",
}
_MODES = {"spin": "自旋", "tilt": "摆动", "rotate": "转动", "": ""}
_GATE = {"opens": "打开", "closes": "关闭", "open": "始终可通过", "closed": "始终不可通过"}
_BOND = {"formed": "成键", "broken": "断键", "released": "配体离去", "captured": "配体结合"}
_EVENT = {
    "kept": "孔道延续",
    "closed": "孔道消失",
    "opened": "新孔道",
    "split": "孔道分裂",
    "merged": "孔道合并",
    "rearranged": "孔道重排",
}


@dataclass
class Finding:
    label: str
    evidence: str


@dataclass
class Gate:
    state: str
    pld_a: float
    pld_b: float
    window_a: list[str]
    window_b: list[str]
    units: list[str]


@dataclass
class Comparison:
    name_a: str
    name_b: str
    motion: Motion = field(repr=False)
    pores_a: Pores = field(repr=False)
    pores_b: Pores = field(repr=False)
    events: list[PoreEvent]
    gate: Gate
    findings: list[Finding]
    attribution: object | None = None

    def summary(self) -> str:
        mo = self.motion
        s = mo.strain
        lines = [
            f"{self.name_a} → {self.name_b}",
            "结论：" + "、".join(dict.fromkeys(LABELS[f.label] for f in self.findings)),
        ]
        lines += [f"  · {LABELS[f.label]}：{f.evidence}" for f in self.findings]
        lines.append(
            f"晶格：体积 {s.volume:+.1%}，主应变 "
            + " ".join(
                f"{e:+.1%}{_direction(*axis)}" for e, axis in zip(s.principal, s.axes, strict=True)
            )
        )
        lines.append(f"孔道 {self.name_a}：{self.pores_a.summary()}")
        lines.append(f"孔道 {self.name_b}：{self.pores_b.summary()}")
        g = self.gate
        lines.append(
            f"门控：{_GATE[g.state]}（PLD {g.pld_a:.2f} → {g.pld_b:.2f} Å，"
            f"探针直径 {2 * self.pores_a.probe:.2f} Å）"
        )
        if self.attribution is not None:
            lines.append(self.attribution.summary())
        return "\n".join(lines)


def _direction(axis: str, angle: float) -> str:
    return f"∥{axis}" if angle < 20 else f"(与 {axis} 成 {angle:.0f}°)"


def _name(unit, units) -> str:
    from mofmotion.units import _KINDS

    u = units.units[unit]
    return f"{_KINDS[u.kind]} {u.formula}"


def _groups(mo: Motion):
    out: dict[str, list] = {}
    for um in mo.units:
        if np.isnan(um.rmsd) or mo.units_a.units[um.unit].kind == "free":
            continue
        out.setdefault(_name(um.unit, mo.units_a), []).append(um)
    return out


def _gate(mo: Motion, pa: Pores, pb: Pores) -> Gate:
    m, ua = mo.match, mo.units_a
    a_of = np.full(len(m.b), -1)
    a_of[m.pairs[:, 1]] = m.pairs[:, 0]
    labels_a, labels_b = [s.label for s in m.a], [s.label for s in m.b]
    window_b = [labels_a[a_of[k]] if a_of[k] >= 0 else labels_b[k] for k in pb.window]
    units = {_name(ua.unit_of[k], ua) for k in pa.window}
    units |= {_name(ua.unit_of[a_of[k]], ua) for k in pb.window if a_of[k] >= 0}
    state = {
        (False, True): "opens",
        (True, False): "closes",
        (True, True): "open",
        (False, False): "closed",
    }[(pa.dimension > 0, pb.dimension > 0)]
    return Gate(state, pa.pld, pb.pld, [labels_a[k] for k in pa.window], window_b, sorted(units))


def classify(mo: Motion, pa: Pores, pb: Pores, events, gate: Gate) -> list[Finding]:
    from pymatgen.core import Element

    t = THRESHOLDS
    out: list[Finding] = []
    s = mo.strain
    big = s.principal[np.abs(s.principal) >= t["strain"]]
    if len(big) and (big > 0).any() and (big < 0).any():
        axes = " ".join(
            f"{e:+.1%}{_direction(*axis)}"
            for e, axis in zip(s.principal, s.axes, strict=True)
            if abs(e) >= t["strain"]
        )
        out.append(Finding("breathing", f"主应变 {axes}，体积 {s.volume:+.1%}"))
    elif abs(s.volume) >= t["volume"] or len(big):
        label = "swelling" if s.volume > 0 else "shrinkage"
        out.append(Finding(label, f"体积 {s.volume:+.1%}，最大主应变 {s.magnitude:.1%}"))

    nets = list(mo.nets.values())
    for p in range(len(nets)):
        for q in range(p + 1, len(nets)):
            shift = float(np.linalg.norm(nets[p] - nets[q]))
            if shift >= t["sliding"]:
                out.append(Finding("sliding", f"网络 {p} 与 {q} 相对平移 {shift:.2f} Å"))

    for name, members in _groups(mo).items():
        n = len(members)
        rot = [u for u in members if u.angle >= t["rotation"]]
        if rot:
            top = max(rot, key=lambda u: u.angle)
            modes = "/".join(sorted({_MODES[u.mode] for u in rot} - {""}))
            out.append(
                Finding("rotation", f"{name} {len(rot)}/{n} 个转动最大 {top.angle:.1f}°（{modes}）")
            )
        moved = [float(np.linalg.norm(u.translation)) for u in members]
        if max(moved) >= t["translation"]:
            out.append(Finding("translation", f"{name} 平移最大 {max(moved):.2f} Å（{n} 个）"))
        bent = [u.rmsd for u in members if mo.units_a.units[u.unit].dimension == 0]
        if bent and max(bent) >= t["rmsd"]:
            out.append(Finding("deformation", f"{name} 刚体拟合残差最大 {max(bent):.2f} Å"))

    ic = mo.internals
    delta = ic.delta
    for kind, junction, label, limit in (
        ("angle", True, "hinge", t["angle"]),
        ("torsion", True, "hinge", t["torsion"]),
        ("torsion", False, "torsion", t["torsion"]),
    ):
        hits = [k for k in ic.top(kind, 3, junction) if abs(delta[k]) >= limit]
        if hits:
            text = "；".join(f"{ic.names[k]} {ic.a[k]:.0f}° → {ic.b[k]:.0f}°" for k in hits)
            out.append(Finding(label, text))

    coordination = [f"{c.names} {_BOND[c.kind]}" for c in mo.bonds]
    coordination += [f"{name} CN {x} → {y}" for name, x, y in mo.coordination if x != y]
    metal = np.array([Element(x).is_metal for x in _symbols(mo.match.a)])
    for k in ic.top("bond", 5):
        if metal[ic.sites[k, :2]].any() and abs(delta[k]) >= t["bond"]:
            coordination.append(f"{ic.names[k]} {ic.a[k]:.2f} → {ic.b[k]:.2f} Å")
    if coordination:
        out.append(Finding("coordination", "；".join(coordination[:6])))

    guests = _guests(mo)
    if guests:
        out.append(Finding("guest", guests))

    if gate.state in {"opens", "closes"}:
        label = "gate-opening" if gate.state == "opens" else "gate-closing"
        out.append(
            Finding(
                label,
                f"PLD {gate.pld_a:.2f} → {gate.pld_b:.2f} Å 跨过探针直径 {2 * pa.probe:.2f} Å；"
                f"窗口：{'、'.join(gate.units) or '—'}",
            )
        )
    pore = []
    if abs(pb.pld - pa.pld) >= t["pore"]:
        pore.append(f"PLD {pa.pld:.2f} → {pb.pld:.2f} Å")
    if abs(pb.lcd - pa.lcd) >= t["pore"]:
        pore.append(f"LCD {pa.lcd:.2f} → {pb.lcd:.2f} Å")
    pore += [
        f"{_EVENT[e.kind]}（维度 {e.dimension_a} → {e.dimension_b}）"
        for e in events
        if e.kind != "kept" or e.dimension_a != e.dimension_b
    ]
    if pore:
        out.append(Finding("pore", "；".join(pore)))

    ea, eb = pa.exposure(), pb.exposure()
    shifts = [
        f"{x} {ea.get(x, 0):.0%} → {eb.get(x, 0):.0%}"
        for x in sorted(set(ea) | set(eb))
        if abs(eb.get(x, 0) - ea.get(x, 0)) >= t["exposure"]
        or (Element(x).is_metal and (ea.get(x, 0) > 0) != (eb.get(x, 0) > 0))
    ]
    if shifts:
        out.append(Finding("environment", "孔壁暴露 " + "；".join(shifts)))
    return out or [Finding("rigid", "所有指标低于阈值")]


def _guests(mo: Motion) -> str:
    def formulas(units):
        return Counter(u.formula for u in units.units if u.kind == "free")

    fa, fb = formulas(mo.units_a), formulas(mo.units_b)
    if fa == fb:
        return ""
    out = [f"{f} {fa[f]} → {fb[f]}" for f in sorted(set(fa) | set(fb)) if fa[f] != fb[f]]
    return "游离组分 " + "；".join(out)


def _load(state, block, encoding):
    if isinstance(state, (str, Path)):
        from mofmotion.audit import audit_cif

        report = audit_cif(state, block=block, encoding=encoding)
        if report.structure is None:
            raise ValueError(report.summary())
        return report.structure, str(state)
    return state, state.composition.reduced_formula


def compare(
    a,
    b,
    *,
    probe: float = PROBE,
    spacing: float = SPACING,
    attribute: bool = True,
    cutoff: float = CUTOFF,
    block: str | None = None,
    encoding: str | None = None,
) -> Comparison:
    (sa, name_a), (sb, name_b) = _load(a, block, encoding), _load(b, block, encoding)
    mo = motion(sa, sb, cutoff=cutoff)
    keep_a = np.array([mo.units_a.units[u].kind != "free" for u in mo.units_a.unit_of])
    keep_b = np.array([mo.units_b.units[u].kind != "free" for u in mo.units_b.unit_of])
    pa = pores(mo.match.a, probe=probe, spacing=spacing, keep=keep_a)
    pb = pores(mo.match.b, probe=probe, spacing=spacing, keep=keep_b, shape=pa.shape)
    events = pore_events(pa, pb)
    gate = _gate(mo, pa, pb)
    findings = classify(mo, pa, pb, events, gate)
    result = None
    if attribute:
        from mofmotion.attribution import attribute as run

        result = run(mo, probe=probe, spacing=spacing)
    return Comparison(name_a, name_b, mo, pa, pb, events, gate, findings, result)


def compare_states(states, **options) -> list[Comparison]:
    if len(states) < 2:
        raise ValueError("至少需要两个状态")
    return [compare(x, y, **options) for x, y in zip(states, states[1:], strict=False)]
