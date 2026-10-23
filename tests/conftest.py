import pytest


def make_cif(sites, *, ops=("x,y,z",), cell=(10, 11, 12, 90, 90, 90), name="test", occ=True):
    a, b, c, alpha, beta, gamma = cell
    lines = [
        f"data_{name}",
        f"_cell_length_a {a}",
        f"_cell_length_b {b}",
        f"_cell_length_c {c}",
        f"_cell_angle_alpha {alpha}",
        f"_cell_angle_beta {beta}",
        f"_cell_angle_gamma {gamma}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        *(f"'{op}'" for op in ops),
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        *(["_atom_site_occupancy"] if occ else []),
    ]
    for label, symbol, x, y, z, o in sites:
        lines.append(f"{label} {symbol} {x} {y} {z}" + (f" {o}" if occ else ""))
    return "\n".join(lines) + "\n"


@pytest.fixture
def write(tmp_path):
    def _write(sites, filename="s.cif", **kwargs):
        path = tmp_path / filename
        path.write_text(make_cif(sites, **kwargs))
        return path

    return _write
