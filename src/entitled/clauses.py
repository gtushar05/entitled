"""Clause store: segment parsed sources into citable, regime-tagged clauses.

Every retrievable unit carries (id, source, regime, text) so answers can cite
"2015/rule-6" or "jan2026/6(4)(b)" — clause-level citations, not page blobs.

Segmentation is pragmatic, not perfect: the 2015 gazette text is OCR output,
so rule boundaries are detected by numbered-heading patterns and the result
is spot-verified by tests on the clauses the calculator depends on. The
calculator itself NEVER reads these strings — its constants live in code
with source comments; clauses exist to explain and cite, not to compute.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "corpus" / "raw"
PARSED = ROOT / "corpus" / "parsed"


def _strip_html(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_2015_gazette() -> list[dict]:
    """Segment the OCR'd 2015 gazette by rule-number headings."""
    text = (PARSED / "refund_rules_2015_ocr.txt").read_text()
    text = re.sub(r"===== PAGE \d+ =====", " ", text)
    text = re.sub(r"\s+", " ", text)
    # rule headings in this gazette look like " 6. Cancellation of tickets..."
    parts = re.split(r"(?=\s(\d{1,2})\.\s+[A-Z][a-z])", text)
    clauses, current_num, buf = [], None, []
    for part in parts:
        if part is None:
            continue
        m = re.match(r"^\s?(\d{1,2})$", part)
        if m:
            current_num = m.group(1)
            continue
        if current_num is not None and len(part.split()) > 15:
            clauses.append({
                "id": f"2015/rule-{current_num}",
                "source": "Refund Rules 2015 (G.S.R. 836(E)) — OCR of gazette",
                "regime": "2015",
                "text": part.strip()[:2500],
            })
            current_num = None
        else:
            buf.append(part)
    # the gazette prints rules in Hindi and English sections, so rule numbers
    # repeat — keep the longest text per rule id (the fuller English body)
    best: dict[str, dict] = {}
    for c in clauses:
        if c["id"] not in best or len(c["text"]) > len(best[c["id"]]["text"]):
            best[c["id"]] = c
    clauses = list(best.values())
    # preamble as its own clause
    head = text[: text.find(" 1. ")][:1500]
    if len(head.split()) > 20:
        clauses.insert(0, {
            "id": "2015/preamble",
            "source": "Refund Rules 2015 (G.S.R. 836(E)) — OCR of gazette",
            "regime": "2015",
            "text": head.strip(),
        })
    return clauses


def parse_cc08_2026() -> list[dict]:
    """The Jan-2026 amendment: extract the English notification clauses."""
    import pymupdf
    doc = pymupdf.open(RAW / "cc_08_2026_gsr41e.pdf")
    text = re.sub(r"\s+", " ", "".join(p.get_text() for p in doc))
    src = "CC 08/2026 carrying G.S.R. 41(E) dated 16.01.2026 (primary, text layer)"
    out = []
    # the three tier sub-clauses (a)/(b)/(c) of new rule 6(4)
    for tag, pat in [
        ("6(4)(a)", r"\(a\) if the ticket is presented for cancellation more than seventy-two hours.{0,220}?fare\."),
        ("6(4)(b)", r"\(b\) if the ticket is presented for cancellation between seventy-two hours.{0,220}?fare\."),
        ("6(4)(c)", r"\(c\) if the ticket is presented for cancellation in less than eight hours.{0,120}?granted\."),
        ("6(5)", r"\(5\) Notwithstanding anything contained in these rules.{0,260}?applicable\."),
    ]:
        m = re.search(pat, text)
        if m:
            out.append({"id": f"jan2026/{tag}", "source": src,
                        "regime": "jan2026-vb-sleeper", "text": m.group(0)})
    # the covering circular's operative paragraph
    m = re.search(r"rationalised refund rule for Vande Bharat Sleeper.{0,400}?egazette", text)
    if m:
        out.append({"id": "jan2026/circular-cover", "source": src,
                    "regime": "jan2026-vb-sleeper", "text": m.group(0)})
    return out


def parse_irctc_page() -> list[dict]:
    """IRCTC's displayed rules — cached with hash; tagged as the stale exhibit."""
    text = _strip_html((RAW / "irctc_eticket_cancel.html").read_text(errors="ignore"))
    src = ("IRCTC eticketCancel.html as cached (see MANIFEST fetch date) — "
           "NOTE: displays superseded 2015 tiers; kept as the stale-official exhibit")
    # split on sentence groups of decent size
    chunks, cur = [], []
    for sent in re.split(r"(?<=\.)\s+", text):
        cur.append(sent)
        if sum(len(s.split()) for s in cur) > 60:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        chunks.append(" ".join(cur))
    return [{"id": f"irctc-stale/para-{i+1}", "source": src,
             "regime": "2015-as-displayed-STALE", "text": c[:2000]}
            for i, c in enumerate(chunks) if len(c.split()) > 25]


def build_store() -> dict:
    clauses = parse_2015_gazette() + parse_cc08_2026() + parse_irctc_page()
    ids = [c["id"] for c in clauses]
    assert len(ids) == len(set(ids)), "clause ids must be unique"
    store = {"n_clauses": len(clauses), "clauses": clauses}
    (PARSED / "clauses.json").write_text(json.dumps(store, indent=1))
    return store
