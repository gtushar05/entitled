"""Corpus acquisition with provenance.

Every document is fetched once, cached raw, and recorded in a MANIFEST with
URL, fetch date, HTTP status, size, and SHA256. The recon that scoped this
project proved the sources drift (IRCTC's own rules page lags a major 2026
reform by months, and two of its canonical PDF links serve zero bytes) —
so the corpus itself needs the same provenance discipline as everything else.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "corpus" / "raw"
MANIFEST = ROOT / "corpus" / "MANIFEST.json"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")

# Verified by the 3-scout recon (2026-08-24). Keys are stable local names.
SOURCES = {
    "refund_rules_2015_gazette.pdf": {
        "url": "https://refunds.indianrail.gov.in/ref_rules/REFUND_RULES_CIRCULARS/RefundRules.pdf",
        "what": "Railway Passengers (Cancellation of Ticket and Refund of Fare) Rules 2015 — G.S.R. 836(E), 29-page scanned gazette (needs OCR)",
        "regime": "2015",
    },
    "cc_65_2015.pdf": {
        "url": "https://indianrailways.gov.in/railwayboard/uploads/directorate/traffic_comm/Comm-Cir-2015/CC_65_15.pdf",
        "what": "Railway Board Commercial Circular 65/2015 — dissemination of the 2015 refund rules",
        "regime": "2015",
    },
    "irctc_eticket_cancel.html": {
        "url": "https://contents.irctc.co.in/en/eticketCancel.html",
        "what": "IRCTC official e-ticket cancellation/refund policy (static HTML). NOTE: displayed stale 48/12/4 tiers as of 2026-08-24 — kept as the 'official sources disagree' exhibit",
        "regime": "2015-as-displayed",
    },
    "irctc_etkt_faq.pdf": {
        "url": "https://contents.irctc.co.in/en/etktfaq.pdf",
        "what": "IRCTC consolidated e-ticket FAQ (text PDF)",
        "regime": "mixed",
    },
    "indianrail_refund_rules.html": {
        "url": "https://www.indianrail.gov.in/enquiry/StaticPages/StaticEnquiry.jsp?StaticPage=refund_Rules.html&locale=en",
        "what": "CRIS passenger-enquiry portal refund rules summary (HTML)",
        "regime": "check-on-parse",
    },
    "commercial_manual_v1_index.htm": {
        "url": "https://indianrailways.gov.in/railwayboard/uploads/codesmanual/CommManual-I/ComercialManual_index.htm",
        "what": "Indian Railway Commercial Manual Vol I index (refund procedure = Chapter III; chapter pages fetched on parse day)",
        "regime": "procedural",
    },
    "wecrs_home.html": {
        "url": "https://refunds.indianrail.gov.in/",
        "what": "WECRS refunds portal homepage — public hrefs to ~14 refund-rules primary documents (1998-2020)",
        "regime": "index",
    },
}


def _fetch(url: str, timeout: int = 60) -> tuple[bytes, int, str]:
    """Fetch with a browser UA; fall back to unverified SSL for gov hosts
    with broken cert chains (recorded in the manifest when it happens)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.status, "verified-ssl"
    except ssl.SSLError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read(), r.status, "UNVERIFIED-ssl (gov cert chain)"


def fetch_all() -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {"fetch_date": date.today().isoformat(), "documents": {}}
    for name, meta in SOURCES.items():
        entry = dict(meta)
        try:
            body, status, ssl_mode = _fetch(meta["url"])
            (RAW / name).write_bytes(body)
            entry.update({
                "status": status,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "ssl": ssl_mode,
                "ok": status == 200 and len(body) > 500,
            })
        except Exception as e:
            entry.update({"ok": False, "error": repr(e)[:200]})
        manifest["documents"][name] = entry
        flag = "OK " if entry.get("ok") else "FAIL"
        print(f"  [{flag}] {name:<36} {entry.get('bytes', 0):>10,}B  {meta['url'][:62]}")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest
