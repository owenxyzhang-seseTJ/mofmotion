from __future__ import annotations

from html import escape

import numpy as np

from mofmotion.compare import _BOND, _EVENT, _GATE, _MODES, LABELS, Comparison, _direction, _groups

_CSS = """
body{font:14px/1.5 -apple-system,"PingFang SC","Noto Sans CJK SC",sans-serif;color:#222;
max-width:1100px;margin:2em auto;padding:0 1em}
h1{font-size:1.5em}h2{margin-top:2em;border-bottom:2px solid #444}h3{margin:1.2em 0 .4em}
table{border-collapse:collapse;margin:.3em 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #ccc;padding:.25em .6em;text-align:left;vertical-align:top}
th{background:#f2f2f2}.tag{display:inline-block;background:#e8eef7;border-radius:3px;
padding:0 .5em;margin:0 .3em .3em 0}.note{color:#666}
"""
_METRICS = {"pld": "PLD (Å)", "lcd": "LCD (Å)", "accessible": "可及分数"}


def _f(x, n: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def _table(head, rows) -> str:
    if not rows:
        return '<p class="note">无</p>'
    h = "".join(f"<th>{escape(str(x))}</th>" for x in head)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _section(title: str, *parts: str) -> str:
    return f"<section><h3>{escape(title)}</h3>{''.join(parts)}</section>"


def _lattice(c: Comparison) -> str:
    s, m = c.motion.strain, c.motion.match
    names = ("a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)")
    rows = [
        (n, _f(x), _f(y), f"{y - x:+.2f}")
        for n, x, y in zip(names, s.cell_a, s.cell_b, strict=True)
    ]
    rows.append(
        ("V (Å³)", _f(m.a.volume, 1), _f(m.a.volume * (1 + s.volume), 1), f"{s.volume:+.1%}")
    )
    strain = "主应变：" + "  ".join(
        f"{e:+.1%}{_direction(*axis)}" for e, axis in zip(s.principal, s.axes, strict=True)
    )
    note = '<p class="note">B 的晶胞已换到 A 的设定并扣除整体转动。</p>'
    return _section(
        "晶格", _table(("参数", "A", "B", "变化"), rows), f"<p>{escape(strain)}</p>", note
    )


def _units(c: Comparison) -> str:
    rows = []
    for name, members in _groups(c.motion).items():
        modes = "/".join(sorted({_MODES[u.mode] for u in members} - {""})) or "—"
        rows.append(
            (
                name,
                len(members),
                _f(max(float(np.linalg.norm(u.translation)) for u in members)),
                _f(max(u.angle for u in members), 1),
                modes,
                _f(max(u.rmsd for u in members)),
                _f(max(u.displacement for u in members)),
            )
        )
    head = ("单元", "数量", "最大平移 Å", "最大转角 °", "转动模式", "拟合残差 Å", "非仿射位移 Å")
    parts = [_table(head, rows)]
    nets = c.motion.nets
    if len(nets) > 1:
        parts.append(
            _table(
                ("网络", "平均平移 Å"), [(k, _f(float(np.linalg.norm(v)))) for k, v in nets.items()]
            )
        )
    return _section("结构单元运动（同类同式合并，取最大值）", *parts)


def _internals(c: Comparison) -> str:
    ic, delta = c.motion.internals, c.motion.internals.delta
    rows = []
    for kind, name, n in (
        ("bond", "键长 Å", 2),
        ("angle", "键角 °", 1),
        ("torsion", "二面角 °", 1),
    ):
        for k in ic.top(kind, 5):
            junction = "是" if ic.junction[k] else ""
            rows.append(
                (name, ic.names[k], _f(ic.a[k], n), _f(ic.b[k], n), f"{delta[k]:+.{n}f}", junction)
            )
    return _section(
        "内坐标变化（各类前 5）", _table(("类型", "原子", "A", "B", "Δ", "跨单元"), rows)
    )


def _coordination(c: Comparison) -> str:
    mo = c.motion
    rows = [(b.names, _BOND[b.kind], _f(b.length_a), _f(b.length_b)) for b in mo.bonds]
    rows += [(name, "配位数", x, y) for name, x, y in mo.coordination if x != y]
    return _section("配位变化", _table(("位点", "变化", "A", "B"), rows))


def _pores(c: Comparison) -> str:
    pa, pb = c.pores_a, c.pores_b
    labels_a = [s.label for s in pa.structure]
    labels_b = [s.label for s in pb.structure]

    def exposure(p):
        return " ".join(f"{s} {f:.0%}" for s, f in list(p.exposure().items())[:6]) or "—"

    rows = [
        ("LCD (Å)", _f(pa.lcd), _f(pb.lcd)),
        ("PLD (Å)", _f(pa.pld), _f(pb.pld)),
        ("探针可通过维度", pa.dimension, pb.dimension),
        ("孔隙率（点探针）", _f(pa.void, 3), _f(pb.void, 3)),
        ("可及分数", _f(pa.accessible, 3), _f(pb.accessible, 3)),
        ("可及孔体积 (cm³/g)", _f(pa.volume, 3), _f(pb.volume, 3)),
        ("孔道/孔穴数", len(pa.channels), len(pb.channels)),
        (
            "瓶颈窗口",
            " ".join(labels_a[k] for k in pa.window) or "—",
            " ".join(labels_b[k] for k in pb.window) or "—",
        ),
        ("孔壁暴露", exposure(pa), exposure(pb)),
        ("计算前去除的客体", pa.removed or "—", pb.removed or "—"),
    ]
    events = [
        (
            _EVENT[e.kind],
            f"{e.dimension_a} → {e.dimension_b}",
            _f(e.fraction_a, 3),
            _f(e.fraction_b, 3),
        )
        for e in c.events
    ]
    g = c.gate
    gate = (
        f"门控：{_GATE[g.state]}（PLD {g.pld_a:.2f} → {g.pld_b:.2f} Å，"
        f"探针直径 {2 * pa.probe:.2f} Å）；窗口单元：{'、'.join(g.units) or '—'}"
    )
    note = f'<p class="note">探针半径 {pa.probe:.2f} Å；B 在 A 的网格上计算，体素逐点对应。</p>'
    return _section(
        "孔道",
        _table(("指标", "A", "B"), rows),
        note,
        "<h3>孔道对应</h3>",
        _table(("事件", "维度", "A 体积分数", "B 体积分数"), events),
        f"<p>{escape(gate)}</p>",
    )


def _attribution(c: Comparison) -> str:
    at = c.attribution
    if at is None:
        return ""
    rows = []
    for metric, name in _METRICS.items():
        total = at.b[metric] - at.a[metric]
        rows.append((name, "总变化", f"{total:+.3f}", ""))
        rows += [
            (name, e.factor, f"{e.forward[metric]:+.3f}", f"{e.backward[metric]:+.3f}")
            for e in at.effects
        ]
        rows.append((name, "耦合项", f"{at.interaction[metric]:+.3f}", ""))
    note = (
        '<p class="note">正向 = 只施加该运动的效应；反向 = 从 B 中撤掉该运动失去的部分；'
        "两者相近且耦合项小，说明该运动可单独解释这部分孔道变化。</p>"
    )
    return _section(
        "孔道变化归因（敏感性分析）", _table(("指标", "因子", "正向", "反向"), rows), note
    )


def _comparison(c: Comparison) -> str:
    tags = "".join(
        f'<span class="tag">{escape(LABELS[x])}</span>'
        for x in dict.fromkeys(f.label for f in c.findings)
    )
    findings = _table(("结论", "证据"), [(LABELS[f.label], f.evidence) for f in c.findings])
    return (
        f"<h2>{escape(c.name_a)} → {escape(c.name_b)}</h2><p>{tags}</p>{findings}"
        + _lattice(c)
        + _units(c)
        + _internals(c)
        + _coordination(c)
        + _pores(c)
        + _attribution(c)
    )


def html_report(comparisons, title: str = "mofmotion 柔性分析报告") -> str:
    if isinstance(comparisons, Comparison):
        comparisons = [comparisons]
    body = "".join(_comparison(c) for c in comparisons)
    return (
        f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><h1>{escape(title)}</h1>{body}</body></html>\n"
    )
