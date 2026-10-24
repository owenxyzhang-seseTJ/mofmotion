# mofmotion

Structural motion analysis for flexible metal–organic frameworks. Given crystal structures of the same framework in two or more states, mofmotion identifies which structural units move, how they move, by how much, and how the motion changes pores and local chemical environments.

## Install

```bash
pip install mofmotion
```

Requires Python ≥ 3.11 with numpy, scipy, pymatgen and spglib.

For development or the unreleased version:

```bash
git clone https://github.com/owenxyzhang-seseTJ/mofmotion
cd mofmotion
pip install -e '.[dev]'
```

Inside iCloud Drive, `.pth` files are hidden and editable installs break; install a copy instead:

```bash
uv pip install --python .venv/bin/python --no-deps --no-build-isolation --reinstall .
```

## Command line

```bash
mofmotion compare closed.cif open.cif --html report.html
```

| Command | Purpose |
| --- | --- |
| `clean SRC DST` | Repair the CIF text: encoding, BOM, line endings, typographic and control characters |
| `audit CIF...` | Read-only audit: input flags, symmetry expansion check, short contacts, space group |
| `bonds CIF...` | Periodic bond graph with covalent, coordination and uncertain edges |
| `units CIF...` | Structural units: nodes, linkers, terminal ligands, free components |
| `pores CIF...` | LCD, PLD, channel dimensionality, accessible volume, bottleneck window, pore lining |
| `compare CIF CIF...` | Motion, pores, gating, classification and attribution for consecutive states |

`clean` never overwrites its source; pass `--encoding gb18030` when detection fails. `audit` cleans the text in memory, accepts `--json`, `--block` and `--tol` (fractional tolerance for coincident sites, default 0.01), and exits with 1 if any report has errors. `pores` and `compare` take `--probe` (probe radius, default 1.65 Å, CO2) and `--spacing` (grid spacing, default 0.2 Å). Free guests are removed before pore calculation and listed in the result; `pores --keep-guests` keeps them. `compare --no-attribution` skips the slower attribution step. Summaries and reports are written in Chinese.

## Python

```python
from pathlib import Path

from mofmotion import audit_cif, bond_graph, compare, html_report, structural_units

report = audit_cif("open.cif")
print(report.summary())
if report.ok:
    graph = bond_graph(report.structure)
    units = structural_units(graph)
    for k, unit in enumerate(units.units):
        print(unit.kind, unit.formula, unit.dimension, units.coords(k).shape)

c = compare("closed.cif", "open.cif")
print(c.summary())
for f in c.findings:
    print(f.label, f.evidence)
Path("report.html").write_text(html_report(c), encoding="utf-8")
```

`compare` accepts paths or pymatgen `Structure` objects; `compare_states` chains more than two states.

## Method

Audit. The CIF is parsed after text cleaning. Symmetry operations are expanded independently and checked against pymatgen in site count, weighted composition and cell. Periodic short contacts below 1.0 Å (H–H), 0.75 Å (with H) or 1.0 Å (heavy atoms) are listed as clashes. The space group is determined with spglib (symprec 0.01) for ordered structures only. Raw `atom_site` rows are flagged as `missing_data`, `outside_cell`, `partial_occupancy`, `disorder_group`, `special_positions`, `symmetry_duplicates` or `periodic_duplicates`. Flags are evidence; removing guests or choosing a disorder component is left to the caller.

Bond graph. Each edge `Bond(i, j, image, length, kind, reason)` joins site i to site j translated by the integer lattice vector `image`. With q = d / (r_i + r_j) and Cordero covalent radii:

| Pair | Certain | Uncertain |
| --- | --- | --- |
| non-metal–non-metal | q ≤ 1.15, `covalent` | 1.15 < q ≤ 1.30 |
| metal–non-metal | q ≤ 1.20, `coordination` | 1.20 < q ≤ 1.35 |
| metal–metal | — | q ≤ 1.15 |

Uncertain edges carry a `reason`: `short`, `distance`, `disorder` or `metal-metal`. Hydrogen bonds are not edges.

Structural units. Units are built from certain edges plus edges uncertain only because of `disorder` or `metal-metal`. A node is a metal together with directly coordinated carbon-free species with a single non-H atom (μ-O, OH, H2O, halide); metals bonded only to carbon belong to their ligand. Every other covalent molecule is a linker, terminal ligand or free component, depending on whether it connects to two or more, one or no other units. Coordinated solvent therefore belongs to the framework. Finite units are unwrapped across cell boundaries and placed with their centroid in the cell; `dimension` is the rank of periodic cycles. Occupancies are kept, so a half-occupied guest reads as `C0.5H2O0.5`. A free component is not necessarily a removable solvent.

Motion. If one state is a supercell of the other (by the count of its rarest framework element), the smaller cell is expanded over all Hermite normal form matrices of that index. For each candidate cell the lowest-strain integer settings are kept, and the pairing cost — not the strain — chooses among them, since the lowest strain need not be the correct correspondence. The report gives the fraction of pairable heavy framework atoms that were paired and the cost gap per atom to the next distinct pairing; below 95 % or 0.05 Å the result is flagged as unreliable. Inputs that fail the audit are rejected. Global rotation is removed by polar decomposition G = UR, and U gives principal strains and volume change. Each unit is fitted as a rigid body, giving translation, rotation angle, rotation mode (spin, tilt, rotate), residual and non-affine displacement. Bond lengths, angles and torsions are compared on the A topology; those spanning more than one unit are hinges. Bond formation, cleavage and coordination number changes are listed separately.

Pores. On a fractional grid, each point stores its distance to the nearest van der Waals surface. Points reachable by the probe centre are joined into channels under periodic boundaries, and the rank of their cycles gives the dimensionality. LCD = 2 max d. PLD is the largest threshold that still percolates, found by bisection; the voxels at that threshold form the bottleneck and the surrounding atoms the window. Both states are evaluated on the same grid, so channels correspond by voxel overlap (kept, closed, opened, split, merged). Gating is decided by whether the probe percolates in each state. Accuracy is about one grid spacing.

Classification and attribution. `classify` turns motion and pore changes into findings (breathing, swelling or shrinking, network sliding, unit rotation, translation or deformation, node hinge, torsion, coordination change, guest change, gating, pore change, local environment), each with the values that triggered it. Thresholds live in `THRESHOLDS`. Attribution splits A → B into the lattice and each group of moving units, builds hybrid structures, and reports for each factor a forward effect (applying it alone), a backward effect (removing it from B) and an interaction term. This is a geometric sensitivity analysis, not a unique decomposition.

## Limits

Two end-point structures give a geometric difference, not a pathway, barrier or gate-opening pressure. Atom mapping can be ambiguous when the framework has automorphisms. Moving fragments independently breaks shared bonds, so pore changes have no unique share per motion.

## Development

```bash
.venv/bin/ruff format src tests && .venv/bin/ruff check . && .venv/bin/python -I -m pytest -q
```

Module layout is described in [docs/architecture.md](docs/architecture.md). Released under the MIT license.
