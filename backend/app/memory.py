"""MemoryService: the single place storage and Hindsight are touched."""
import logging
import re
from typing import Protocol

log = logging.getLogger("engbrain.memory")
STOP = set("why the use does have was what how and for with this that are our did when who last time before seen we in of to a is it exist changed happened".split())


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9\-\.]+", s.lower()) if len(t) > 2 and t not in STOP}


class MemoryBackend(Protocol):
    def add(self, rec: dict) -> None: ...
    def all(self) -> list[dict]: ...


class InMemoryBackend:
    def __init__(self):
        self._d: dict[str, dict] = {}

    def add(self, rec): self._d[rec["id"]] = rec
    def all(self): return list(self._d.values())


class MemoryService:
    def __init__(self, backend: MemoryBackend | None = None, hindsight=None):
        self.b = backend or InMemoryBackend()
        self.hs = hindsight
        self.last_note = ""
        self.hindsight_error = ""
        self.version = 0

    def remember(self, rec: dict) -> None:
        self.remember_many([rec])

    def remember_many(self, recs: list[dict]) -> None:
        if not recs:
            return
        for rec in recs:
            self.b.add(rec)
        self.version += 1
        if self.hs:
            try:
                self.hs.retain(recs)
                self.hindsight_error = ""
            except Exception as e:  # noqa: BLE001 - keep local memory available
                self.hindsight_error = str(e)
                log.warning("Hindsight retain failed; records remain in local memory: %s", e)

    def get(self, rid: str) -> dict | None:
        return next((r for r in self.b.all() if r["id"] == rid), None)

    def _local(self, query: str, service: str | None, k: int) -> list[dict]:
        q, out = tokens(query), []
        for r in self.b.all():
            if service and r["service"] != service:
                continue
            title, body = tokens(r["title"]), tokens(f'{r["text"]} {r["id"]}')
            s = sum(2 if t in title else 1 for t in q if t in title or t in body)
            if s:
                out.append({**r, "score": s})
        return sorted(out, key=lambda r: -r["score"])[:k]

    def recall(self, query: str, service: str | None = None, k: int = 5) -> list[dict]:
        """Hybrid: local keyword score + Hindsight semantic hits; local-only if Hindsight fails."""
        self.last_note = ""
        local = self._local(query, service, 8)
        if not self.hs:
            return local[:k]
        try:
            ids = self.hs.recall(query, [r["id"] for r in self.b.all()])
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            log.warning("Hindsight recall failed: %s", e)
            self.last_note = f"Hindsight unavailable ({e}); used local memory only."
            return local[:k]
        merged = {r["id"]: r for r in local}
        for i, rid in enumerate(ids):
            r = self.get(rid)
            if not r or (service and r["service"] != service):
                continue
            base = merged.get(rid, {**r, "score": 0})
            merged[rid] = {**base, "score": base["score"] + 3 - min(i, 2) * 0.5}
        return sorted(merged.values(), key=lambda r: -r["score"])[:k]

    def sync(self) -> int:
        if not self.hs:
            raise ValueError("Hindsight is not configured (set HINDSIGHT_URL in .env)")
        recs = self.b.all()
        try:
            self.hs.retain(recs)
            self.hindsight_error = ""
        except Exception as e:
            self.hindsight_error = str(e)
            raise
        return len(recs)

    def search_related(self, rid: str, depth: int = 2) -> list[dict]:
        idx = {r["id"]: r for r in self.b.all()}
        seen, frontier = {rid}, [rid]
        for _ in range(depth):
            nxt = []
            for i in frontier:
                for j in (idx.get(i) or {"rel": []})["rel"]:
                    if j not in seen and j in idx:
                        seen.add(j)
                        nxt.append(j)
            frontier = nxt
        return [idx[i] for i in seen if i != rid]

    def get_timeline(self, service: str | None = None) -> list[dict]:
        rs = [r for r in self.b.all() if not service or r["service"] == service]
        return sorted(rs, key=lambda r: r["date"])

    def get_context(self, query: str, service: str | None = None) -> dict:
        hits = self.recall(query, service)
        rel = {}
        for h in hits[:2]:
            for r in self.search_related(h["id"]):
                rel[r["id"]] = r
        hit_ids = {h["id"] for h in hits}
        return {"similar": hits, "connected": [r for r in rel.values() if r["id"] not in hit_ids]}
