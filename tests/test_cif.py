import pytest
from conftest import make_cif
from pymatgen.io.cif import CifParser

from mofmotion import clean_cif, clean_cif_text

CIF = make_cif([("C1", "C", 0.1, 0.2, 0.3, 1), ("O1", "O", 0.5, 0.5, 0.5, 1)])


def test_clean_text_is_untouched():
    assert clean_cif_text(CIF) == (CIF, [])


def test_bom_crlf_and_final_newline():
    raw = b"\xef\xbb\xbf" + CIF.rstrip("\n").replace("\n", "\r\n").encode()
    text, notes = clean_cif_text(raw)
    assert text == CIF
    assert notes == ["去除 utf-8-sig BOM", "换行统一为 LF"]


@pytest.mark.parametrize("codec", ["utf-16", "utf-32"])
def test_utf16_and_utf32_with_bom(codec):
    assert clean_cif_text(CIF.encode(codec))[0] == CIF


def test_explicit_encoding_and_latin1_fallback():
    raw = ("# 作者注释\n" + CIF).encode("gb18030")
    assert clean_cif_text(raw, "gb18030")[0].startswith("# 作者注释")
    text, notes = clean_cif_text(raw)
    assert text.endswith(CIF)
    assert "latin-1" in notes[0]


def test_typographic_characters_become_parsable():
    dirty = CIF.replace("0.3", "\u22120.3").replace("0.5 0.5", "\uff10.5\u00a00.5")
    dirty = dirty.replace("'P 1'", "\u2018P 1\u2019") + "\u200b"
    text, notes = clean_cif_text(dirty)
    assert text == CIF.replace("0.3", "-0.3")
    assert notes[0].startswith("替换排版字符")
    structure = CifParser.from_str(text).parse_structures(primitive=False)[0]
    assert structure[0].frac_coords[2] == pytest.approx(0.7)


def test_control_characters_and_replacement_char():
    text, notes = clean_cif_text(CIF.replace("C1", "C\x001") + "# \ufffd\n")
    assert "C\x001" not in text and "C1" in text
    assert any("U+FFFD" in note for note in notes)


def test_clean_cif_writes_utf8_and_refuses_self(tmp_path):
    source = tmp_path / "a.cif"
    source.write_bytes(CIF.encode("utf-16"))
    notes = clean_cif(source, tmp_path / "b.cif")
    assert (tmp_path / "b.cif").read_text(encoding="utf-8") == CIF
    assert notes == ["去除 utf-16 BOM"]
    with pytest.raises(ValueError):
        clean_cif(source, source)
    assert clean_cif_text((tmp_path / "b.cif").read_bytes()) == (CIF, [])
