"""Glyphless font used to carry the invisible OCR text layer.

The font shipped with this package (``pdf.ttf``, inherited from Tesseract) contains
two glyphs, and the traditional way of using it is a ``/CIDToGIDMap`` stream that
sends every one of the 65536 character codes to the same glyph. Nothing is ever
painted - the text is drawn in rendering mode 3 - so a single glyph is all the
*rendering* needs.

It is not all that *editing* needs. When a PDF is rewritten by a producer that
subsets embedded fonts, the subsetter works in glyph space: codes that share a
glyph are indistinguishable to it, so they collapse to one code, and the ToUnicode
CMap is rebuilt for that one survivor. Every character of the OCR layer then decodes
to the same value. Apple's PDFKit (Preview, Safari, and anything using Quartz to
save a PDF, including annotation and redaction tools) does exactly this, which turns
the whole text layer into a run of U+0001 the moment the file is re-saved.

Giving each character code a glyph of its own avoids the collapse: the codes stay
distinct through a subsetting round trip, and the ToUnicode CMap keeps mapping them
to the right characters. So the font is expanded here to the full 65535 glyphs the
TrueType format allows, all of them empty, and used with ``/CIDToGIDMap /Identity``.
The added glyphs are empty and the added metrics are zero, so the expansion is a
long run of zero bytes that costs about 700 bytes once the stream is compressed.
"""

from __future__ import annotations

import struct

# TrueType stores the glyph count in a uint16, so this is the largest font we can
# build - and, conveniently, one glyph for every code Identity-H can address.
NUM_GLYPHS = 0xFFFF

_HEAD_CHECKSUM_MAGIC = 0xB1B0AFBA


def _checksum(data: bytes) -> int:
    """Sum `data` as big-endian uint32 words, zero padded, modulo 2**32."""
    padded = data + b"\x00" * (-len(data) % 4)
    return sum(struct.unpack(f">{len(padded) // 4}I", padded)) & 0xFFFFFFFF


def _read_tables(font: bytes) -> tuple[bytes, dict[str, bytes]]:
    """Split an sfnt into its version tag and its tables, keyed by tag."""
    (num_tables,) = struct.unpack(">H", font[4:6])
    tables = {}
    for i in range(num_tables):
        record = 12 + 16 * i
        tag = font[record : record + 4].decode("latin-1")
        _, offset, length = struct.unpack(">III", font[record + 4 : record + 16])
        tables[tag] = font[offset : offset + length]
    return font[:4], tables


def _build_font(version: bytes, tables: dict[str, bytes]) -> bytes:
    """Reassemble an sfnt from its tables, fixing up offsets and checksums."""
    tags = sorted(tables)
    num_tables = len(tags)
    # The binary search hints in the offset table: entry_selector = floor(log2(n)).
    entry_selector = max(num_tables.bit_length() - 1, 0)
    search_range = 16 * (1 << entry_selector)
    header = struct.pack(
        ">4sHHHH",
        version,
        num_tables,
        search_range,
        entry_selector,
        16 * num_tables - search_range,
    )

    records = bytearray()
    body = bytearray()
    offset = len(header) + 16 * num_tables
    for tag in tags:
        table = tables[tag]
        records += struct.pack(
            ">4sIII", tag.encode("latin-1"), _checksum(table), offset, len(table)
        )
        padded = table + b"\x00" * (-len(table) % 4)
        body += padded
        offset += len(padded)

    font = bytearray(header + records + body)

    # head.checkSumAdjustment makes the checksum of the whole file come out at a
    # fixed value. It is computed with the field itself zeroed, which it already is
    # at this point, because the caller zeroes it before handing over the table.
    head_offset = len(header) + 16 * tags.index("head")
    (head_start,) = struct.unpack(">I", font[head_offset + 8 : head_offset + 12])
    adjustment = (_HEAD_CHECKSUM_MAGIC - _checksum(bytes(font))) & 0xFFFFFFFF
    struct.pack_into(">I", font, head_start + 8, adjustment)
    return bytes(font)


def expand_glyphless_font(font: bytes, num_glyphs: int = NUM_GLYPHS) -> bytes:
    """Return `font` grown to `num_glyphs` glyphs by appending empty ones.

    Only the tables whose size is tied to the glyph count are touched: `maxp` gets
    the new count, `loca` gets one more entry per glyph - all pointing at the end of
    the existing outlines, which is how an empty glyph is spelled - and `hmtx` gets a
    zero left side bearing per glyph. The outlines in `glyf` and every other table
    are carried over unchanged.
    """
    version, tables = _read_tables(font)

    (old_num_glyphs,) = struct.unpack(">H", tables["maxp"][4:6])
    if old_num_glyphs >= num_glyphs:
        return font
    (index_to_loc_format,) = struct.unpack(">h", tables["head"][50:52])
    (num_h_metrics,) = struct.unpack(">H", tables["hhea"][34:36])

    maxp = bytearray(tables["maxp"])
    struct.pack_into(">H", maxp, 4, num_glyphs)
    tables["maxp"] = bytes(maxp)

    # loca has one entry per glyph plus a terminator. Repeating the terminator gives
    # each added glyph a zero-length outline.
    entry_size = 2 if index_to_loc_format == 0 else 4
    loca = tables["loca"]
    tables["loca"] = loca + loca[-entry_size:] * (num_glyphs - old_num_glyphs)

    # hmtx is numberOfHMetrics advance/bearing pairs followed by one bearing per
    # remaining glyph. The added glyphs are empty, so their bearings are zero.
    tables["hmtx"] += b"\x00\x00" * (num_glyphs - max(old_num_glyphs, num_h_metrics))

    # Zeroed before the file checksum is computed over it; _build_font fills it in.
    head = bytearray(tables["head"])
    struct.pack_into(">I", head, 8, 0)
    tables["head"] = bytes(head)

    return _build_font(version, tables)
