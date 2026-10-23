from __future__ import annotations

import codecs
import re
from collections import Counter
from pathlib import Path

_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

_REPLACEMENTS = {
    **{chr(code): chr(code - 0xFEE0) for code in range(0xFF01, 0xFF5F)},
    **dict.fromkeys("\u00a0\u2002\u2003\u2009\u202f\u3000", " "),
    **dict.fromkeys("\u200b\u200c\u200d\u2060\ufeff", ""),
    **dict.fromkeys("\u2018\u2019\u2032", "'"),
    **dict.fromkeys("\u201c\u201d\u2033", '"'),
    **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2212\ufe63", "-"),
}
_REPLACE = re.compile("[" + re.escape("".join(_REPLACEMENTS)) + "]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _decode(data: bytes, encoding: str | None) -> tuple[str, list[str]]:
    if encoding:
        return data.decode(encoding), [f"按指定编码 {encoding} 解码"]
    for bom, codec in _BOMS:
        if data.startswith(bom):
            return data.decode(codec), [f"去除 {codec} BOM"]
    try:
        return data.decode("utf-8"), []
    except UnicodeDecodeError as exc:
        note = f"非 UTF-8 字节（偏移 {exc.start}），按 latin-1 解码；如有中文注释请指定 encoding"
        return data.decode("latin-1"), [note]


def clean_cif_text(data: bytes | str, encoding: str | None = None) -> tuple[str, list[str]]:
    text, notes = (data, []) if isinstance(data, str) else _decode(data, encoding)

    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        notes.append("换行统一为 LF")
    found = Counter(_REPLACE.findall(text))
    if found:
        text = _REPLACE.sub(lambda m: _REPLACEMENTS[m.group()], text)
        notes.append("替换排版字符：" + ", ".join(f"U+{ord(c):04X}×{n}" for c, n in found.items()))
    controls = len(_CONTROL.findall(text))
    if controls:
        text = _CONTROL.sub("", text)
        notes.append(f"删除控制字符 {controls} 个")
    if "\ufffd" in text:
        notes.append("文本含 U+FFFD，原始字节在此前的转换中已丢失，请核对原始文件")
    if text and not text.endswith("\n"):
        text += "\n"
    if text.lstrip().startswith("#\\#CIF_2.0"):
        notes.append("CIF 2.0 文件，当前解析器按 CIF 1.1 读取，可能不兼容")
    return text, notes


def clean_cif(
    source: str | Path, destination: str | Path, encoding: str | None = None
) -> list[str]:
    source, destination = Path(source), Path(destination)
    if destination.resolve() == source.resolve():
        raise ValueError("destination 不能与 source 相同")
    text, notes = clean_cif_text(source.read_bytes(), encoding)
    destination.write_text(text, encoding="utf-8", newline="\n")
    return notes
