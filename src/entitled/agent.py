"""The deterministic agent spine: case in -> computed, cited answer out.

Pipeline: parse (parser.py) -> compute (calculator.py) -> bind citations
to clause text (retrieval.by_id) -> compose the answer from a fixed
template. No LLM anywhere on this path — this IS the kill-switch path
the Day-7 LLM explainer degrades to, so it must be complete and correct
on its own. The LLM layer may rephrase; it may never re-derive.

Answer outcomes extend the calculator's three with one agent-level state:
  COMPUTED | NO_REFUND | ESCALATE | NEEDS_INFO
NEEDS_INFO is a parsing outcome (the intake was incomplete/ambiguous);
ESCALATE is a rules outcome (the intake was fine, the verified rules
don't cover it). Conflating them would hide which side needs the human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .calculator import TicketCase, Result, compute_refund, compute_party
from .parser import parse_case
from .retrieval import ClauseRetriever

# leading clause-store id in a calculator citation string, e.g.
# "2015/rule-6 (G.S.R. 836(E)) — ..." -> "2015/rule-6";
# "jan2026/6(4)(a)" -> itself. Symbolic refs ("IRCTC Tatkal rules: ...")
# deliberately don't match and pass through unbound.
CLAUSE_ID_RE = re.compile(r"^([a-z0-9-]+/[A-Za-z0-9()./-]+)")
EXCERPT_CHARS = 200


@dataclass
class BoundCitation:
    ref: str                     # the calculator's citation string, verbatim
    clause_id: str | None = None
    source: str | None = None
    text: str | None = None      # excerpt of the clause's actual text
    bound: bool = False


@dataclass
class Answer:
    outcome: str                 # COMPUTED | NO_REFUND | ESCALATE | NEEDS_INFO
    refund: float | None
    charge: float | None
    regime: str
    verified: bool
    citations: list = field(default_factory=list)   # list[BoundCitation]
    notes: list = field(default_factory=list)
    questions: list = field(default_factory=list)   # NEEDS_INFO only
    explanation: str = ""
    trace: list = field(default_factory=list)

    @property
    def needs_human(self) -> bool:
        return self.outcome == "ESCALATE"


_RETRIEVER: ClauseRetriever | None = None


def _default_retriever() -> ClauseRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        # by_id needs no ranking; BM25-only keeps the spine dependency-light
        _RETRIEVER = ClauseRetriever(use_dense=False)
    return _RETRIEVER


def bind_citations(refs: list[str], retriever: ClauseRetriever) -> list[BoundCitation]:
    out = []
    for ref in refs:
        m = CLAUSE_ID_RE.match(ref)
        clause = retriever.by_id(m.group(1)) if m else None
        if clause:
            excerpt = clause["text"][:EXCERPT_CHARS]
            if len(clause["text"]) > EXCERPT_CHARS:
                excerpt += "…"
            out.append(BoundCitation(ref, clause["id"], clause["source"],
                                     excerpt, True))
        else:
            out.append(BoundCitation(ref))
    return out


def _money(x: float) -> str:
    # half-up, matching calculator._r — round() is half-to-even and could
    # display a figure that disagrees with the computed reconciliation
    return f"₹{int(x + 0.5):,}"


def _compose(case: TicketCase | None, ans: Answer) -> str:
    """The fixed answer template. Every number here is a calculator output
    (or the fare the user themselves supplied) — by construction."""
    lines: list[str] = []
    if ans.outcome == "NEEDS_INFO":
        lines.append("I need a few details before I can compute this refund:")
        lines += [f"  • {q}" for q in ans.questions]
        lines.append("Nothing is computed until these are resolved — the "
                     "charge tiers are sensitive to exactly these fields.")
        return "\n".join(lines)

    if ans.outcome == "ESCALATE":
        lines.append("This case cannot be answered confidently from the "
                     "verified rules — it needs manual review (TDR/helpdesk).")
        lines += [f"  • {n}" for n in ans.notes]
    elif ans.outcome == "NO_REFUND":
        lines.append(f"No refund is due. The fare of {_money(case.fare)} "
                     "is forfeited as the cancellation charge.")
    else:  # COMPUTED
        lines.append(f"Refund due: {_money(ans.refund)} of the "
                     f"{_money(case.fare)} fare "
                     f"(cancellation charge: {_money(ans.charge)}).")
        if case.disruption == "NONE":
            hours = (case.departure - case.cancellation).total_seconds() / 3600
            lines.append(f"Cancelled {hours:.1f} hours before scheduled "
                         f"departure, computed under the {ans.regime} rules.")
        else:
            lines.append(f"Disruption case ({case.disruption}): the "
                         "time-before-departure tiers do not apply.")
    if ans.outcome != "ESCALATE" and not ans.verified:
        lines.append("CAUTION: this branch relies on a rule change whose "
                     "primary circular is still pending — treat as provisional.")

    if ans.citations:
        lines.append("Basis:")
        for c in ans.citations:
            if c.bound:
                lines.append(f"  [{c.clause_id}] ({c.source}) “{c.text}”")
            else:
                lines.append(f"  [unbound reference] {c.ref}")
    if ans.outcome != "ESCALATE" and ans.notes:
        lines += [f"  note: {n.removeprefix('note: ')}" for n in ans.notes]
    return "\n".join(lines)


def answer_case(case: TicketCase,
                retriever: ClauseRetriever | None = None) -> Answer:
    """TicketCase -> Answer. Total: invalid field values become an
    ESCALATE answer (naming the offending field), never an exception —
    an agent that crashes on odd input escalates nothing."""
    retriever = retriever or _default_retriever()
    try:
        r: Result = compute_refund(case)
    except Exception as e:  # totality: TypeError/OverflowError on bad fields
        ans = Answer("ESCALATE", None, None, "UNDETERMINED", False,
                     notes=[f"input rejected by the calculator: {e} — this "
                            "should have been caught at intake; routing to "
                            "manual review"],
                     trace=[f"calculator: {type(e).__name__}"])
        ans.explanation = _compose(case, ans)
        return ans
    ans = Answer(r.outcome, r.refund, r.charge, r.regime, r.verified,
                 citations=bind_citations(r.citations, retriever),
                 notes=list(r.notes))
    bound = sum(1 for c in ans.citations if c.bound)
    ans.trace = [f"calculator: {r.outcome} regime={r.regime} "
                 f"refund={r.refund} charge={r.charge}",
                 f"citations: {bound} bound to clause text, "
                 f"{len(ans.citations) - bound} symbolic"]
    ans.explanation = _compose(case, ans)
    return ans


def _answer_parsed(parsed, retriever) -> Answer:
    """parse result -> Answer, threading parse warnings into the notes
    (shared by the single-ticket and party paths so neither drops them)."""
    ans = answer_case(parsed.case, retriever)
    if parsed.warnings:
        ans.notes = list(parsed.warnings) + ans.notes
        ans.explanation = _compose(parsed.case, ans)
    ans.trace.insert(0, f"parse: ok ({len(parsed.warnings)} warning(s))")
    return ans


def answer_payload(payload: dict,
                   retriever: ClauseRetriever | None = None) -> Answer:
    """Raw dict (LLM extractor / form output) -> Answer, via the parser."""
    parsed = parse_case(payload)
    if not parsed.ok:
        ans = Answer("NEEDS_INFO", None, None, "UNDETERMINED", False,
                     questions=list(parsed.problems),
                     notes=list(parsed.warnings),
                     trace=[f"parse: {len(parsed.problems)} blocking problem(s)"])
        ans.explanation = _compose(None, ans)
        return ans
    return _answer_parsed(parsed, retriever)


def _party_dict(outcome, answers=(), total_refund=None, escalated=(),
                questions=(), notes=(), explanation="") -> dict:
    """One shape for every branch — callers never probe for missing keys."""
    return {"outcome": outcome, "answers": list(answers),
            "total_refund": total_refund,
            "escalated_passengers": list(escalated),
            "questions": list(questions), "notes": list(notes),
            "explanation": explanation}


def answer_party(payloads: list[dict],
                 retriever: ClauseRetriever | None = None) -> dict:
    """Party ticket / partial cancellation: parse all passengers first
    (one consolidated NEEDS_INFO if any fail), then compute_party."""
    retriever = retriever or _default_retriever()
    if not isinstance(payloads, (list, tuple)) or not payloads:
        return _party_dict("NEEDS_INFO",
                           questions=["expected a non-empty list of one "
                                      "payload per passenger"],
                           explanation="I need one set of ticket fields per "
                                       "passenger to compute a party refund.")
    parsed = [parse_case(p) for p in payloads]
    bad = [(i, pr) for i, pr in enumerate(parsed) if not pr.ok]
    if bad:
        questions = [f"passenger {i + 1}: {q}" for i, pr in bad
                     for q in pr.problems]
        return _party_dict("NEEDS_INFO", questions=questions,
                           explanation=_compose(None, Answer(
                               "NEEDS_INFO", None, None, "UNDETERMINED",
                               False, questions=questions)))
    party = compute_party([pr.case for pr in parsed])
    if not party["results"]:  # mixed-status post-chart party: escalated whole
        return _party_dict("ESCALATE", notes=party["notes"],
                           explanation="This party ticket needs manual review:\n" +
                                       "\n".join(f"  • {n}" for n in party["notes"]))
    answers = [_answer_parsed(pr, retriever) for pr in parsed]
    header = (f"Party of {len(answers)}: total refund "
              f"{_money(party['total_refund'])}."
              if party["outcome"] == "COMPUTED" else
              f"Party of {len(answers)}: passengers "
              f"{[i + 1 for i in party['escalated_passengers']]} need manual "
              f"review; computable refunds total {_money(party['total_refund'])}.")
    blocks = [f"— Passenger {i + 1} —\n{a.explanation}"
              for i, a in enumerate(answers)]
    notes = party["notes"]
    return _party_dict(party["outcome"], answers=answers,
                       total_refund=party["total_refund"],
                       escalated=party["escalated_passengers"], notes=notes,
                       explanation="\n\n".join(
                           [header] + blocks +
                           ([f"note: {n}" for n in notes] if notes else [])))
