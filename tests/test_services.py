import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app import github_service as gh_mod, hindsight as hs_mod, llm as llm_mod
from app.agent import ask, review
from app.demo import RECORDS
from app.github_service import GitHubService, verify_webhook_signature
from app.http_util import UpstreamError
from app.memory import MemoryService
from app.store import SQLiteBackend


def seeded(**kw):
    m = MemoryService(SQLiteBackend(":memory:"), **kw)
    for r in RECORDS: m.remember(r)
    return m


def ids(r): return {e["id"] for e in r["evidence"]}


class Workflows(unittest.TestCase):
    def setUp(self): self.m = seeded()
    def test_why_code_exists(self):
        r = ask("Why does the retry mechanism exist in payment-service?", self.m)
        self.assertLessEqual({"INC-101", "PR-247", "DEC-01", "INC-103"}, ids(r))
    def test_seen_before(self):
        r = ask("Have we seen duplicate payment problems before?", self.m)
        self.assertIn("INC-103", ids(r))
    def test_last_change(self):
        r = ask("What happened the last time we changed the database connection handling in payment-service?", self.m)
        self.assertLessEqual({"PR-318", "DP-v3.1.0", "INC-142"}, ids(r)); self.assertTrue(r["inference"])
    def test_no_evidence(self):
        r = ask("Why do we use kubernetes operators?", self.m)
        self.assertEqual(r["answer"], "No matching engineering memory was found."); self.assertFalse(r["evidence"])
    def test_review_and_isolation(self):
        self.assertEqual(review("+ retry on timeout", "payment-service", self.m)["verdict"], "Historical Pattern Detected")
        self.assertEqual(review("+ retry", "auth-service", self.m)["warnings"], [])
    def test_timeline_sorted(self):
        d = [e["date"] for e in self.m.get_timeline("payment-service")]; self.assertEqual(d, sorted(d))


class Persistence(unittest.TestCase):
    def test_survives_restart(self):
        p = os.path.join(tempfile.mkdtemp(), "t.db")
        a = MemoryService(SQLiteBackend(p)); a.remember(RECORDS[0])
        self.assertEqual(MemoryService(SQLiteBackend(p)).get("INC-101")["id"], "INC-101")

    def test_remember_retain_and_increment_version(self):
        class Capture:
            def __init__(self): self.retained = []
            def retain(self, recs): self.retained.extend(recs)
        hindsight = Capture()
        memory = MemoryService(SQLiteBackend(":memory:"), hindsight)
        memory.remember_many(RECORDS[:2])
        self.assertEqual(len(hindsight.retained), 2)
        self.assertEqual(memory.version, 1)


class GitHub(unittest.TestCase):
    def test_url_validation(self):
        self.assertEqual(GitHubService.parse_url("https://github.com/o/r.git"), ("o", "r"))
        for bad in ("http://github.com/o/r", "https://evil.com/o/r", "https://github.com/o"):
            with self.assertRaises(ValueError): GitHubService.parse_url(bad)

    def test_webhook_signature_and_event_mapping(self):
        import hashlib, hmac
        secret, body = "test-secret", b'{"action":"opened"}'
        digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        self.assertTrue(verify_webhook_signature(secret, "sha256=" + digest, body))
        self.assertFalse(verify_webhook_signature(secret, "sha256=bad", body))
        recs = GitHubService.webhook_records("pull_request", {
            "action": "opened", "repository": {"name": "payments", "html_url": "https://github.com/acme/payments"},
            "pull_request": {"number": 17, "title": "Fix retries", "body": "Protect charges", "state": "open",
                             "created_at": "2026-09-28T12:00:00Z", "html_url": "https://github.com/acme/payments/pull/17"}})
        self.assertEqual(recs[0]["id"], "PR-17")
        self.assertEqual(recs[0]["service"], "payments")
        self.assertEqual(recs[0]["url"], "https://github.com/acme/payments/pull/17")
    def test_ingest_links_records(self):
        def fake(m, url, headers=None, body=None, timeout=0):
            if "/pulls" in url: return [{"number": 5, "title": "Add retry (fixes #7)", "body": "", "merged_at": "2026-01-02T00:00:00Z", "created_at": "2026-01-01T00:00:00Z"}]
            if "/commits" in url: return [{"sha": "abcdef123", "commit": {"message": "retry\n\nPR #5", "author": {"date": "2026-01-01T00:00:00Z"}}}]
            if "/issues" in url: return [{"number": 7, "title": "Outage", "body": "", "labels": [], "created_at": "2026-01-01T00:00:00Z"}]
            return [{"tag_name": "v1.0", "body": "fixes #5", "published_at": "2026-01-03T00:00:00Z", "created_at": "2026-01-03T00:00:00Z"}]
        gh_mod.http_json = fake
        recs = {r["id"]: r for r in GitHubService().ingest("https://github.com/o/r")}
        self.assertEqual(recs["INC-7"]["type"], "incident"); self.assertIn("INC-7", recs["PR-5"]["rel"])
        self.assertIn("PR-5", recs["abcdef1"]["rel"]); self.assertEqual(recs["REL-v1.0"]["type"], "deployment")


class Adapters(unittest.TestCase):
    def test_hindsight_hybrid_and_fallback(self):
        hs_mod.http_json = lambda *a, **k: {"results": [{"text": "duplicates came from retries (INC-103)"}]}
        m = seeded(hindsight=hs_mod.HindsightClient("http://h", "b"))
        self.assertIn("INC-103", {r["id"] for r in m.recall("charges twice")}); self.assertEqual(m.last_note, "")
        def boom(*a, **k): raise UpstreamError("down")
        hs_mod.http_json = boom
        self.assertTrue(m.recall("retry")); self.assertIn("unavailable", m.last_note)
    def test_llm_citations_and_provider(self):
        ev = [{"id": "INC-103", "date": "d", "type": "incident", "title": "t", "text": "x"}]
        self.assertEqual(llm_mod.unknown_citations("ok [INC-103] and [PR-999]", ev), ["PR-999"])
        llm_mod.http_json = lambda *a, **k: {"choices": [{"message": {"content": " Answer [INC-103] "}}]}
        self.assertEqual(llm_mod.OpenAIProvider("k", "m", "http://x").summarize("q", ev), "Answer [INC-103]")
        llm_mod.http_json = lambda *a, **k: {"candidates": [{"content": {"parts": [{"text": " Answer [INC-103] "}]}}]}
        self.assertEqual(llm_mod.GeminiProvider("k", "m", "http://x").summarize("q", ev), "Answer [INC-103]")
        self.assertIsNone(llm_mod.MockProvider().summarize("q", ev))


@unittest.skipUnless(__import__("importlib").util.find_spec("fastapi"), "fastapi not installed")
class Api(unittest.TestCase):
    def test_endpoints(self):
        os.environ["ENGBRAIN_DB"] = os.path.join(tempfile.mkdtemp(), "api.db")
        from fastapi.testclient import TestClient
        from app.main import app
        c = TestClient(app)
        self.assertEqual(c.post("/api/demo/load").json()["loaded"], len(RECORDS))
        r = c.post("/api/chat", json={"question": "Why does the retry mechanism exist in payment-service?"}).json()
        self.assertIn("PR-247", {e["id"] for e in r["evidence"]})
        self.assertEqual(c.post("/api/repository/connect", json={"url": "https://evil.com/x/yyyy"}).status_code, 400)
        self.assertEqual(c.get("/api/records/NOPE").status_code, 404)

    def test_signed_webhook_endpoint(self):
        import hashlib, hmac
        from dataclasses import replace
        import app.main as main
        from fastapi.testclient import TestClient
        original = main.cfg
        main.cfg = replace(original, github_webhook_secret="test-secret")
        payload = {"action": "opened", "repository": {"name": "payments"},
                   "pull_request": {"number": 18, "title": "Fix retry handling", "body": "", "state": "open",
                                    "created_at": "2026-09-28T12:00:00Z"}}
        body = json.dumps(payload).encode()
        signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
        try:
            with TestClient(main.app) as c:
                path = "/api/webhooks/github"
                headers = {"x-github-event": "pull_request"}
                self.assertEqual(c.post(path, content=body, headers=headers).status_code, 401)
                accepted = c.post(path, content=body, headers={**headers, "x-hub-signature-256": signature})
                self.assertEqual(accepted.status_code, 200)
                self.assertEqual(c.get("/api/records/PR-18").json()["title"], "Fix retry handling")
        finally:
            main.cfg = original

if __name__ == "__main__": unittest.main()
