"""OCR the 2015 refund-rules gazette (29 scanned pages) into corpus/parsed/."""
import subprocess, sys
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
work = ROOT / "corpus" / "parsed" / "ocr_work"
work.mkdir(parents=True, exist_ok=True)
doc = pymupdf.open(ROOT / "corpus" / "raw" / "refund_rules_2015_gazette.pdf")
pages = []
for i, page in enumerate(doc, 1):
    png = work / f"p{i}.png"
    page.get_pixmap(dpi=200).save(png)
    out = work / f"p{i}"
    subprocess.run(["tesseract", str(png), str(out)],
                   capture_output=True, check=True)
    text = (work / f"p{i}.txt").read_text()
    pages.append(f"\n\n===== PAGE {i} =====\n{text}")
    print(f"  page {i}/29: {len(text.split())} words", flush=True)
combined = ROOT / "corpus" / "parsed" / "refund_rules_2015_ocr.txt"
combined.write_text("".join(pages))
print(f"\nwrote {combined} ({sum(len(p.split()) for p in pages):,} words)")
