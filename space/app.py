"""Entitled — Gradio Space (thin UI over entitled.demo).

Three tabs: a live deterministic calculator (no model needed), a
model-gated natural-language parser (request-capped, cached-trace
fallback), and a gallery of the frozen trap cases. All rupee figures come
from the deterministic calculator; the LLM, when present, only rephrases
and is checked by a numeric-faithfulness gate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from entitled.demo import (RequestBudget, answer_structured, answer_nl,
                           trap_gallery, provider_available)

BUDGET = RequestBudget(max_calls=40)   # shared across the Space process

INTRO = """# 🚆 Entitled — a rules-grounded Indian Railways refund agent

Ask about a train-ticket cancellation and get the **exact refund**, the
**cancellation charge**, and the **rule clauses** it rests on. The refund
arithmetic is done by a **deterministic, version-aware calculator** — the
language model *never* computes a rupee figure, and when it rephrases an
answer a numeric-faithfulness gate verifies every number first.

Version-aware across three rule regimes (2015 gazette · Jan-2026
Vande-Bharat rules · the pending Apr-2026 reform), and it **escalates
instead of guessing** when the verified rules don't determine the answer.
"""


def structured_fn(fare, cls, quota, status, channel, train, dep, canc,
                  disruption, chart):
    payload = {"fare": fare, "cls": cls, "quota": quota, "status": status,
               "channel": channel, "train_type": train, "departure": dep,
               "cancellation": canc, "disruption": disruption,
               "chart_prepared": chart}
    return answer_structured(payload)


def nl_fn(question):
    body, status = answer_nl(question, BUDGET)
    return body, f"status: {status}"


with gr.Blocks(title="Entitled — railway refund agent") as app:
    gr.Markdown(INTRO)

    with gr.Tab("Calculator (live, no model)"):
        gr.Markdown("Runs the verified calculator directly — no language "
                    "model, no cost, no cold start.")
        with gr.Row():
            with gr.Column():
                fare = gr.Number(label="Fare paid (₹, per passenger)", value=1500)
                cls = gr.Dropdown(["1A", "EC", "2A", "FC", "3A", "CC", "3E", "SL", "2S"],
                                  value="3A", label="Class")
                quota = gr.Dropdown(["GN", "TQ", "PT"], value="GN",
                                    label="Quota (General / Tatkal / Premium Tatkal)")
                status = gr.Dropdown(["CNF", "RAC", "WL"], value="CNF", label="Status")
                channel = gr.Dropdown(["E", "C"], value="E",
                                      label="Channel (E-ticket / Counter)")
            with gr.Column():
                train = gr.Dropdown(["REG", "VBS", "AB2"], value="REG",
                                    label="Train (Regular / Vande Bharat Sleeper / Amrit Bharat 2.0)")
                dep = gr.Textbox(label="Scheduled departure (YYYY-MM-DD HH:MM)",
                                 value="2026-03-20 18:00")
                canc = gr.Textbox(label="Cancellation time (YYYY-MM-DD HH:MM)",
                                  value="2026-03-19 18:00")
                disruption = gr.Dropdown(["NONE", "TRAIN_CANCELLED", "DELAY_GT_3H"],
                                         value="NONE", label="Disruption")
                chart = gr.Checkbox(label="Chart already prepared", value=False)
        go = gr.Button("Compute refund", variant="primary")
        out = gr.Markdown()
        go.click(structured_fn,
                 [fare, cls, quota, status, channel, train, dep, canc,
                  disruption, chart], out)

    with gr.Tab("Ask in words"):
        avail = ("A language model is configured — natural-language parsing "
                 f"is live (shared budget: {BUDGET.max_calls} requests)."
                 if provider_available() else
                 "⚠️ No language model is configured on this demo, so this "
                 "tab shows **cached traces**. The calculator tab runs live.")
        gr.Markdown(f"Describe your situation in plain English. {avail}")
        q = gr.Textbox(label="Your situation", lines=3,
                       placeholder="I have a 3AC ticket for ₹1500 on a regular "
                                   "train leaving 20 March 6 PM, cancelling now.")
        ask = gr.Button("Get my refund", variant="primary")
        nlout = gr.Markdown()
        nlstatus = gr.Markdown()
        ask.click(nl_fn, q, [nlout, nlstatus])

    with gr.Tab("Trap gallery"):
        gr.Markdown("The cases a naive implementation (or a pure-LLM agent) "
                    "gets wrong — tier boundaries, flat-minimum binding, "
                    "escalation discipline. Every answer here is the verified "
                    "calculator's, frozen in the test suite.")
        for t in trap_gallery():
            with gr.Accordion(f"🔹 {t['question']}", open=False):
                gr.Markdown(t["markdown"] + f"\n\n> **Why it's a trap:** {t['rationale']}")

    gr.Markdown("---\nCode & methodology: **github.com/gtushar05/entitled** · "
                "70-case frozen golden set, 123 tests, adversarial faithfulness gate.")

if __name__ == "__main__":
    app.launch()
