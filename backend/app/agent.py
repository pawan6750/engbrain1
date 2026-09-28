"""Deterministic evidence-only agent (MockProvider style). An LLM provider can rephrase `facts`, never add to them."""
from .memory import MemoryService

NONE = "No matching engineering memory was found."
INTENTS = [("CHANGE_HISTORY", ["last time", "changed", "modif", "before modifying"]),
           ("INCIDENT_SEARCH", ["seen", "before", "incident", "error", "exhausted", "duplicate", "problem"]),
           ("HISTORICAL_CONTEXT", ["why", "exist", "introduced", "workaround"])]

def detect_intent(q: str) -> str:
    ql = q.lower()
    return next((n for n, ks in INTENTS if any(k in ql for k in ks)), "GENERAL_ENGINEERING_QUERY")

def detect_service(q: str, mem: MemoryService) -> str | None:
    return next((s for s in {r["service"] for r in mem.b.all()} if s in q.lower()), None)

def ask(q: str, mem: MemoryService) -> dict:
    intent, svc = detect_intent(q), detect_service(q, mem)
    ctx = mem.get_context(q, svc)
    if not ctx["similar"]:
        return {"intent": intent, "answer": NONE, "facts": [], "inference": [], "evidence": []}
    top = ctx["similar"][0]
    chain = sorted([top] + mem.search_related(top["id"]), key=lambda r: r["date"])
    if intent == "CHANGE_HISTORY":
        chain = [r for r in chain if r["service"] == top["service"] and r["type"] in ("pull_request", "deployment", "incident")]
    facts = [f'{r["date"]} {r["id"]}: {r["title"]} — {r["text"]}' for r in chain]
    inference = []
    if any(r["type"] == "deployment" and "rolled back" in r["text"] for r in chain):
        inference.append("Possible: review connection-management changes before a similar modification (not certain).")
    if any("idempotency" in r["text"] for r in chain) and any("retr" in r["text"] for r in chain):
        inference.append("Likely: retry changes need idempotency protection, based on INC-103.")
    return {"intent": intent, "answer": facts[0] if len(facts) == 1 else "Historical context found; see facts in chronological order.",
            "facts": facts, "inference": inference,
            "evidence": [{k: r[k] for k in ("id", "type", "service", "date", "title", "text")} for r in chain]}

RISKY = {"retry": ["INC-103", "PR-247"], "timeout": ["INC-101", "INC-103"], "connection": ["INC-105", "INC-142", "PR-318"], "pool": ["INC-105"]}

def review(diff: str, service: str, mem: MemoryService) -> dict:
    d = diff.lower(); warnings = []
    for pat, ids in RISKY.items():
        recs = [r for i in ids if (r := mem.get(i)) and r["service"] == service]
        if pat in d and recs:
            warnings.append({"pattern": pat, "reason": f'Change touches "{pat}", which relates to earlier history in {service}.',
                             "related": [{"id": r["id"], "title": r["title"], "text": r["text"]} for r in recs]})
    return {"warnings": warnings, "verdict": "Historical Pattern Detected" if warnings else "No historical evidence found."}
