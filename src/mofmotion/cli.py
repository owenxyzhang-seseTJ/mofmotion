from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from mofmotion.audit import audit_cif
from mofmotion.cif import clean_cif
from mofmotion.graph import bond_graph
from mofmotion.pores import PROBE, SPACING, pores
from mofmotion.units import structural_units


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mofmotion", description="柔性 MOF 多态结构分析")
    sub = parser.add_subparsers(dest="command", required=True)

    clean = sub.add_parser("clean", help="清洗 CIF 文本：编码、换行、排版字符")
    clean.add_argument("source", type=Path)
    clean.add_argument("destination", type=Path)
    clean.add_argument("--encoding", help="原文件编码，如 gb18030；默认 UTF-8")

    audit = sub.add_parser("audit", help="审计 CIF 并标记输入形态，可一次传入多个状态")
    audit.add_argument("cif", type=Path, nargs="+")
    audit.add_argument("--block")
    audit.add_argument("--encoding")
    audit.add_argument("--tol", type=float, default=0.01, help="位点重合的分数坐标容差")
    audit.add_argument("--json", action="store_true", help="输出 JSON 而不是摘要")

    for name, text in (
        ("bonds", "构建周期成键图，统计共价、配位与不确定边"),
        ("units", "划分结构单元：节点、连接体、端基与游离组分"),
    ):
        command = sub.add_parser(name, help=text)
        command.add_argument("cif", type=Path, nargs="+")
        command.add_argument("--block")
        command.add_argument("--encoding")

    pore = sub.add_parser("pores", help="孔道几何：LCD、PLD、孔道维度、可及体积与孔壁原子")
    compare = sub.add_parser("compare", help="比较两个或多个状态：运动、孔道、门控与归因")
    pore.add_argument("cif", type=Path, nargs="+")
    pore.add_argument("--keep-guests", action="store_true", help="保留游离客体参与成孔")
    compare.add_argument("cif", type=Path, nargs="+", help="按顺序给出各状态，依次比较相邻两态")
    compare.add_argument("--html", type=Path, help="写出 HTML 报告")
    compare.add_argument("--no-attribution", action="store_true", help="跳过归因（更快）")
    for command in (pore, compare):
        command.add_argument("--probe", type=float, default=PROBE, help="探针半径 Å，默认 CO2")
        command.add_argument("--spacing", type=float, default=SPACING, help="网格间距 Å")
        command.add_argument("--block")
        command.add_argument("--encoding")

    args = parser.parse_args(argv)
    if args.command == "compare":
        return _compare(parser, args)
    if args.command == "clean":
        try:
            notes = clean_cif(args.source, args.destination, args.encoding)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print("\n".join(notes) or "原文已经干净，内容未改动")
        return 0

    if args.command in {"bonds", "units", "pores"}:
        failed = False
        for path in args.cif:
            report = audit_cif(path, block=args.block, encoding=args.encoding)
            if report.structure is None:
                print(report.summary())
                failed = True
                continue
            structure = report.structure
            if args.command == "pores":
                keep = np.ones(len(structure), bool) if args.keep_guests else None
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = pores(structure, probe=args.probe, spacing=args.spacing, keep=keep)
            else:
                graph = bond_graph(structure)
                result = graph if args.command == "bonds" else structural_units(graph)
            print(f"{path}\n{result.summary()}")
        return int(failed)

    reports = [
        audit_cif(path, block=args.block, encoding=args.encoding, tol=args.tol) for path in args.cif
    ]
    if args.json:
        data = [r.to_dict() for r in reports]
        print(json.dumps(data[0] if len(data) == 1 else data, ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(r.summary() for r in reports))
    return 0 if all(r.ok for r in reports) else 1


def _compare(parser: argparse.ArgumentParser, args) -> int:
    from mofmotion.compare import compare_states
    from mofmotion.report import html_report

    if len(args.cif) < 2:
        parser.error("compare 至少需要两个状态")
    if args.html and any(args.html.resolve() == p.resolve() for p in args.cif):
        parser.error("HTML 输出不能覆盖输入文件")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = compare_states(
                [str(p) for p in args.cif],
                probe=args.probe,
                spacing=args.spacing,
                attribute=not args.no_attribution,
                block=args.block,
                encoding=args.encoding,
            )
    except ValueError as exc:
        print(exc)
        return 1
    print("\n\n".join(c.summary() for c in results))
    if args.html:
        args.html.write_text(html_report(results), encoding="utf-8")
        print(f"\nHTML 报告：{args.html}")
    return 0
