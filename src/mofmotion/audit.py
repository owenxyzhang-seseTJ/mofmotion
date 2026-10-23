from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from mofmotion.cif import clean_cif_text

_CELL_KEYS = [f"_cell_length_{x}" for x in "abc"] + [
    f"_cell_angle_{x}" for x in ("alpha", "beta", "gamma")
]
_SYMOP_KEYS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
_SYMOP_TEXT = re.compile(r"[xyzXYZ0-9+\-*/.\s]+(,[xyzXYZ0-9+\-*/.\s]+){2}")
_CLASH = {2: 1.0, 1: 0.75, 0: 1.0}


@dataclass
class Contact:
    site_1: int
    site_2: int
    labels: tuple[str, str]
    distance: float


@dataclass
class CifAudit:
    path: str
    block: str | None = None
    text_fixes: list[str] = field(default_factory=list)
    declared_space_group: str | None = None
    symop_count: int = 0
    rows: int = 0
    sites: int | None = None
    formula: str | None = None
    cell: list[float] | None = None
    spacegroup: str | None = None
    flags: dict[str, list[str]] = field(default_factory=dict)
    min_distance: float | None = None
    clashes: list[Contact] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structure: object = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k != "structure"}
        data["clashes"] = [asdict(c) for c in self.clashes]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    def summary(self) -> str:
        lines = [f"{self.path} [{self.block}]"]
        lines += [f"  文本修正：{note}" for note in self.text_fixes]
        if self.sites is not None:
            lines.append(
                f"  {self.rows} 行 → {self.sites} 位点  {self.formula}  "
                f"声明 {self.declared_space_group}  识别 {self.spacegroup}"
            )
        lines += [f"  标记 {name}：{', '.join(rows)}" for name, rows in self.flags.items()]
        if self.min_distance is not None:
            lines.append(f"  最短距离 {self.min_distance:.3f} Å，可疑短接触 {len(self.clashes)} 对")
        lines += [f"  错误：{msg}" for msg in self.errors]
        lines += [f"  警告：{msg}" for msg in self.warnings]
        return "\n".join(lines)


def _number(value) -> float:
    try:
        return float(re.sub(r"\(\d*\)", "", str(value)))
    except ValueError:
        return float("nan")


def _element(symbol: str) -> str:
    from pymatgen.core import Element

    match = re.match(r"([A-Za-z])([a-z]?)", symbol.strip())
    if not match:
        return "X"
    pair = (match[1].upper() + match[2]).strip()
    return pair if Element.is_valid_symbol(pair) else match[1].upper()


def _select_block(text: str, block: str | None):
    from pymatgen.io.cif import CifFile

    blocks = CifFile.from_str(text).data
    if block is None:
        with_sites = [name for name, b in blocks.items() if "_atom_site_fract_x" in b.data]
        if len(with_sites) != 1:
            raise ValueError(f"无法自动选择含原子坐标的 block，请指定：{list(blocks)}")
        block = with_sites[0]
    if block not in blocks:
        raise ValueError(f"找不到 block {block!r}；可用：{list(blocks)}")
    return block, blocks[block]


def _column(data: dict, key: str, n: int, default=None) -> list:
    value = data.get(key, [default] * n)
    return value if isinstance(value, list) else [value]


def _symops(texts: list[str]):
    from pymatgen.core.operations import SymmOp

    ops = []
    for text in texts:
        if not _SYMOP_TEXT.fullmatch(text.strip()):
            raise ValueError(f"无法解析对称操作 {text!r}")
        op = SymmOp.from_xyz_str(text.strip())
        if not np.allclose(op.rotation_matrix, np.round(op.rotation_matrix)) or not np.isclose(
            abs(np.linalg.det(op.rotation_matrix)), 1
        ):
            raise ValueError(f"对称操作 {text!r} 不是晶体学操作")
        ops.append(op)
    return ops


def _pairs_within(lattice, frac: np.ndarray, centers: np.ndarray, r: float):
    from pymatgen.optimization.neighbors import find_points_in_spheres

    m = np.ascontiguousarray(lattice.matrix, dtype=float)
    return find_points_in_spheres(
        np.ascontiguousarray(frac @ m),
        np.ascontiguousarray(centers @ m),
        r=r,
        pbc=np.ones(3, dtype=np.int64),
        lattice=m,
        tol=1e-8,
    )


def _flag_rows(data: dict, audit: CifAudit, tol: float):
    from pymatgen.core import Lattice

    labels = data.get("_atom_site_label")
    labels = labels if isinstance(labels, list) else [labels]
    n = audit.rows = len(labels)
    symbols = (
        _column(data, "_atom_site_type_symbol", n) if "_atom_site_type_symbol" in data else labels
    )
    elements = np.array([_element(str(s)) for s in symbols])
    frac = np.array(
        [[_number(v) for v in _column(data, f"_atom_site_fract_{x}", n)] for x in "xyz"]
    ).T
    occ = np.array([_number(v) for v in _column(data, "_atom_site_occupancy", n, 1)])
    groups = [_column(data, k, n) for k in data if k.startswith("_atom_site_disorder_")]
    disordered = [any(str(g[i]) not in {".", "?", "None"} for g in groups) for i in range(n)]

    flags: dict[str, set[int]] = {}
    expected = None

    def flag(name: str, rows) -> None:
        rows = {int(i) for i in rows}
        if rows:
            flags.setdefault(name, set()).update(rows)

    complete = np.isfinite(frac).all(axis=1)
    flag("missing_data", np.flatnonzero(~complete | ~np.isfinite(occ) | (occ <= 0) | (occ > 1.001)))
    flag("outside_cell", np.flatnonzero(complete & ((frac < -tol) | (frac >= 1 + tol)).any(axis=1)))
    flag("partial_occupancy", np.flatnonzero(np.isfinite(occ) & (occ < 1 - 1e-3)))
    flag("disorder_group", np.flatnonzero(disordered))
    dupes = [label for label, count in Counter(labels).items() if count > 1]
    if dupes:
        audit.warnings.append(f"重复的原子标签：{dupes}")

    cell = [_number(data.get(key)) for key in _CELL_KEYS]
    ops_text = next((data[k] for k in _SYMOP_KEYS if k in data), None)
    if ops_text is None:
        audit.warnings.append("未列出对称操作，跳过对称等价检查")
    elif np.isfinite(cell).all() and complete.any():
        ops_text = ops_text if isinstance(ops_text, list) else [ops_text]
        try:
            ops = _symops(ops_text)
        except ValueError as exc:
            audit.errors.append(str(exc))
        else:
            lattice, idx = Lattice.from_parameters(*cell), np.flatnonzero(complete)
            for op in ops:
                identity = np.allclose(op.affine_matrix, np.eye(4))
                moved = frac[idx] @ op.rotation_matrix.T + op.translation_vector
                distance_tol = tol * max(cell[:3])
                for c, p, _, _ in zip(
                    *_pairs_within(lattice, frac[idx], moved, distance_tol), strict=True
                ):
                    i, j = idx[c], idx[p]
                    if elements[i] != elements[j]:
                        continue
                    if i != j:
                        flag("periodic_duplicates" if identity else "symmetry_duplicates", (i, j))
                    elif not identity:
                        flag("special_positions", (i,))
            audit.symop_count = len(ops)
            usable = idx[np.isfinite(occ[idx])]
            expected = _expansion(
                lattice, frac[usable], elements[usable], occ[usable], ops, 1e-3 * max(cell[:3])
            )

    audit.flags = {name: [str(labels[i]) for i in sorted(rows)] for name, rows in flags.items()}
    return expected, cell


def _expansion(lattice, frac, elements, occ, ops, radius: float):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    images = np.concatenate([frac @ op.rotation_matrix.T + op.translation_vector for op in ops])
    rows = np.tile(np.arange(len(frac)), len(ops))
    c, p, _, _ = _pairs_within(lattice, images, images, radius)
    graph = coo_matrix((np.ones(len(c)), (c, p)), shape=(len(images), len(images)))
    count, component = connected_components(graph, directed=False)
    composition: Counter = Counter()
    for row, _ in set(zip(rows.tolist(), component.tolist(), strict=True)):
        composition[elements[row]] += occ[row]
    return count, composition


def _compare(structure, expected, cell, audit: CifAudit) -> None:
    count, composition = expected
    if count != len(structure):
        audit.errors.append(f"位点数不一致：pymatgen {len(structure)}，独立展开 {count}")
    reference = structure.composition.get_el_amt_dict()
    if any(
        abs(reference.get(e, 0) - composition.get(e, 0)) > 1e-3 for e in {*reference, *composition}
    ):
        audit.errors.append(f"组成不一致：pymatgen {dict(reference)}，独立展开 {dict(composition)}")
    if not np.allclose(cell, structure.lattice.parameters, atol=1e-4):
        audit.errors.append("pymatgen 改变了晶胞参数")


def _periodic_pairs(lattice, frac: np.ndarray, r: float):
    centers, points, images, distances = _pairs_within(lattice, frac, frac, r)
    first = images[np.arange(len(images)), (images != 0).argmax(axis=1)]
    keep = (centers < points) | ((centers == points) & (first > 0))
    return centers[keep], points[keep], images[keep].astype(int), distances[keep]


def _contacts(structure, audit: CifAudit, r: float = 2.0) -> None:
    centers, points, _, distances = _periodic_pairs(structure.lattice, structure.frac_coords, r)
    if not len(distances):
        return
    audit.min_distance = round(float(distances.min()), 4)

    hydrogens = np.array([site.species.elements[0].symbol in {"H", "D"} for site in structure])
    limits = np.vectorize(_CLASH.get)(hydrogens[centers].astype(int) + hydrogens[points])
    labels = [site.label for site in structure]
    audit.clashes = [
        Contact(
            int(centers[k]),
            int(points[k]),
            (labels[centers[k]], labels[points[k]]),
            round(float(distances[k]), 4),
        )
        for k in np.flatnonzero(distances < limits)[np.argsort(distances[distances < limits])]
    ]


def audit_cif(
    path: str | Path, *, block: str | None = None, encoding: str | None = None, tol: float = 0.01
) -> CifAudit:
    path = Path(path)
    audit = CifAudit(path=str(path))
    try:
        text, audit.text_fixes = clean_cif_text(path.read_bytes(), encoding)
        audit.block, raw = _select_block(text, block)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        audit.errors.append(str(exc))
        return audit

    data = {key.lower(): value for key, value in raw.data.items()}
    audit.declared_space_group = data.get("_space_group_name_h-m_alt") or data.get(
        "_symmetry_space_group_name_h-m"
    )
    expected, cell = _flag_rows(data, audit, tol)

    from pymatgen.io.cif import CifParser

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            structure = CifParser.from_str(str(raw)).parse_structures(primitive=False)[0]
        except Exception as exc:
            audit.errors.append(f"pymatgen 无法生成结构：{exc}")
            return audit
        finally:
            audit.warnings += sorted({str(w.message) for w in caught if w.category is UserWarning})

    audit.structure, audit.sites = structure, len(structure)
    audit.formula = structure.composition.formula
    audit.cell = [round(float(v), 6) for v in structure.lattice.parameters]
    if expected:
        _compare(structure, expected, cell, audit)
    _contacts(structure, audit)
    if structure.is_ordered:
        import spglib

        cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            audit.spacegroup = spglib.get_spacegroup(cell, symprec=0.01)
    return audit
