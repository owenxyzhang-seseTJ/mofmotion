# Architecture

```text
src/mofmotion/
  cif.py          CIF text cleaning
  audit.py        read-only audit and structure loading
  graph.py        periodic bond graph
  units.py        structural units
  match.py        setting transformation and atom pairing
  motion.py       strain, rigid-body fits, internal coordinates, bond changes
  pores.py        grid distance field, percolation, channel correspondence
  compare.py      gating, classification, multi-state chaining
  attribution.py  hybrid-structure sensitivity analysis
  report.py       self-contained HTML report
  cli.py          command line
```

Data flows in one direction, in memory, and source files are never written:

bytes → `clean_cif_text` → `audit_cif` → `bond_graph` → `structural_units` → `match` → `motion` → `pores` → `compare` → `html_report`

Uncertainty is returned to the caller as `flags`, `errors`, `warnings` and uncertain bonds. Chemical decisions such as removing guests or picking a disorder component stay with the caller, except that free components are left out of pore calculation by default.
