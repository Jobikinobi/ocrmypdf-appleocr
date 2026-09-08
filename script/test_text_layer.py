#!/usr/bin/env python3
"""Check that the OCR text layer survives being re-saved by Apple's PDFKit.

Preview, Safari and every other Quartz-based tool rewrite a PDF when they save
it, and rewriting subsets the embedded fonts. A glyphless font whose character
codes all share one glyph does not survive that: the codes collapse onto the one
survivor and every character of the text layer decodes to the same value. This
builds a small text layer, pushes it through PDFKit, and reads it back.

Usage: script/test_text_layer.py
"""

import sys
import tempfile
from pathlib import Path

from Foundation import NSURL
from Quartz import PDFDocument

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocrmypdf_appleocr.common import BoundingBox, Point, Textbox  # noqa: E402
from ocrmypdf_appleocr.pdf import generate_pdf  # noqa: E402

LINES = [
    "Latin ASCII line 0123456789",
    "Accents: Ünïcödé — «guillemets»",
    "日本語の横書きテキスト",
    "Русский текст и Ελληνικά",
]


def textbox(text: str, y: float) -> Textbox:
    x, width, height = 100.0, 500.0, 34.0
    bb = BoundingBox(
        ul=Point(x, y),
        ur=Point(x + width, y),
        ll=Point(x, y + height),
        lr=Point(x + width, y + height),
    )
    return Textbox(text=text, bb=bb, confidence=90, is_vertical=False, children=None)


def pdfkit_text(path: Path) -> str:
    doc = PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(str(path)))
    if doc is None:
        raise SystemExit(f"PDFKit could not open {path}")
    return doc.pageAtIndex_(0).string() or ""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        original = Path(tmp) / "text_layer.pdf"
        resaved = Path(tmp) / "text_layer_resaved.pdf"

        results = [textbox(line, 100.0 + 60.0 * i) for i, line in enumerate(LINES)]
        generate_pdf((200.0, 200.0), 1700, 2200, 1.0, results, original, False)

        doc = PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(str(original)))
        if not doc.writeToURL_(NSURL.fileURLWithPath_(str(resaved))):
            raise SystemExit("PDFKit failed to re-save the PDF")

        failures = 0
        for label, path in (("as generated", original), ("after PDFKit re-save", resaved)):
            text = pdfkit_text(path)
            missing = [line for line in LINES if line not in text]
            if missing:
                failures += 1
                print(f"FAIL {label}: {len(missing)} of {len(LINES)} line(s) not found")
                print(f"     extracted: {text!r}")
            else:
                print(f"ok   {label}: all {len(LINES)} lines read back")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
