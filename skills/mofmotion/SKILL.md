---
name: mofmotion
description: Analyze how a flexible MOF moves between two or more crystal states — which structural units move, by how much, and what that does to pores and local environments. Use for breathing/swelling/gate-opening questions on CIF pairs, comparing activated vs loaded structures, or auditing a CIF before structural analysis.
---

# mofmotion

Given two states of the same framework, it answers: what moved, how, how much, and what changed in the pores. It compares geometry only — no pathway, barrier or pressure.

## Install and verify

```bash
pip install mofmotion
python -c "import mofmotion; print(mofmotion.__version__)"
```

## Golden path

```python
from mofmotion import audit_cif, compare, html_report

report = audit_cif("closed.cif")
if not report.ok:
    raise SystemExit(report.summary())          # do not analyze an unaudited structure

c = compare("closed.cif", "open.cif")           # paths or pymatgen Structure objects
print(c.summary())                              # Chinese, human-readable
print(c.motion.match.reliable, c.motion.match.coverage, c.motion.match.margin)

open("report.html", "w", encoding="utf-8").write(html_report(c))
```

`compare_states([cif1, cif2, cif3])` chains consecutive pairs. CLI equivalents:
`mofmotion compare a.cif b.cif --html r.html [--no-attribution] [--probe 1.65] [--spacing 0.2]`,
plus `audit`, `clean`, `bonds`, `units`, `pores`. `compare` exits 1 when an input
fails the audit or has nothing to pair.

## Answer questions from these fields

| Question | Read |
| --- | --- |
| Which units moved | `c.motion.units` (list of `UnitMotion`), name via `c.motion.units_a.units[u.unit]` → `.kind`, `.formula` |
| How much it moved | `u.translation` (Å, Cartesian), `u.angle` (°), `u.mode` (`spin`/`tilt`/`rotate`), `u.displacement` (max atom shift, Å) |
| Deformed internally | `u.rmsd` (Å, rigid-body residual) |
| Whole-cell change | `c.motion.strain.volume`, `.principal`, `.axes` |
| Network sliding | `c.motion.nets` (dict of net → mean shift vector) |
| Hinge / torsion | `c.motion.internals.top("torsion", 5, junction=True)`, also `"angle"`, `"bond"` |
| Bonds breaking/forming | `c.motion.bonds` (`formed`, `broken`, `released`, `captured`) |
| Coordination change | `c.motion.coordination` → `(label, CN_a, CN_b)` |
| Pore change | `c.pores_a`, `c.pores_b` (`.lcd`, `.pld`, `.dimension`, `.void`, `.accessible`, `.exposure()`), `c.events` |
| Gating | `c.gate.state` (`opens`/`closes`/`open`/`closed`), `.pld_a`, `.pld_b`, `.units` |
| Who caused the pore change | `c.attribution.effects` (forward/backward per factor), `.interaction` |
| Findings with evidence | `c.findings` (`label`, `evidence`); label names in `from mofmotion.compare import LABELS` |

Units are keyed by index; `u.unit` indexes `units_a.units`. Free guests are
excluded from pore calculation by default (`Pores.removed` records them).
A `nan` in `UnitMotion` means the unit had no paired atoms and must be reported
as such, not as "no motion".

## Confidence — always check before quoting numbers

- `audit.ok` false: refuse. The report explains why (composition, site count, cell).
- `match.reliable` false (coverage < 0.95 or margin < 0.05 Å per atom): say the
  atom pairing is ambiguous and the numbers are provisional. `match.coverage`
  is the fraction of pairable heavy framework atoms actually paired; `margin`
  is the gap to the next distinct pairing (∞ when unique).
- Supercells are detected automatically from the framework count ratio;
  `match.supercell` shows the matrices applied to each side, and `match.a`/`match.b`
  are the expanded structures used everywhere downstream.

## Pitfalls

- Coordinated solvent counts as framework; only genuinely free species are guests.
- `compare(..., attribute=False)` skips the slower hybrid-structure attribution —
  use it for large cells or when only motion is needed (attribution adds a few
  grid evaluations per factor).
- Pore numbers are grid-limited to about one `spacing` (default 0.2 Å).
- Structures without a bonded heavy framework (missing H, wrong coordinates) raise
  `ValueError` from `compare`; fix the input rather than lowering thresholds.
- Do not read `Match.unmatched` as a chemical result — it is the pairing residue.

## Reporting

Lead with the classification (`c.findings`), then the numbers that triggered it,
then pore/gating effects. State the confidence verdict and anything the audit
flagged. Never present a low-confidence pairing as a structural conclusion.
