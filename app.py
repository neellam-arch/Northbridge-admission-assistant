"""Web interface for the admissions agent.

Run (with the venv active, from this folder):  python app.py
Then open http://127.0.0.1:8000 in your browser.

Uses only Python's built-in web server, so there is no web framework to install.
Enquiries are saved in enquiries.json next to this file (DATA_FILE to change it).

Settings (environment variables or .env):
  GOOGLE_API_KEY          required
  GEMINI_MODEL            main model; GEMINI_BACKUP_MODELS = comma-separated fallbacks
  PORT / HOST             set automatically by most hosts (HOST defaults to 0.0.0.0 when PORT is set)
  DEMO_MODE=1             public demo: banner, usage limits, sample data on first start
  DAILY_LIMIT             max AI drafts per day across all visitors (demo mode, default 40)
  VISITOR_LIMIT_PER_HOUR  max questions per visitor per hour (demo mode, default 5)
"""
import ast
import collections
import datetime
import json
import os
import re
import shutil
import threading
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import agent

# Some models ignore temperature and warn about it; that's harmless.
warnings.filterwarnings("ignore", message=".*sampling parameter.*")

# Tried in order: the model in .env first, then these backups.
# Set GEMINI_BACKUP_MODELS in .env (comma separated) to change the list.
BACKUP_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_BACKUP_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
).split(",") if m.strip()]
MODELS = list(dict.fromkeys([agent.MODEL_NAME, *BACKUP_MODELS]))

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
HERE = Path(__file__).parent
PAGE = HERE / "web" / "index.html"
DATA = Path(os.getenv("DATA_FILE") or HERE / "enquiries.json")
SEED = HERE / "demo_seed.json"
LOCK = threading.Lock()

DEMO_MODE = os.getenv("DEMO_MODE", "0").lower() in ("1", "true", "yes")
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "40")) if DEMO_MODE else 0
VISITOR_LIMIT = int(os.getenv("VISITOR_LIMIT_PER_HOUR", "5")) if DEMO_MODE else 0


# ---------- Demo usage limits (kept in memory; reset when the server restarts) ----------
class Limits:
    def __init__(self):
        self.lock = threading.Lock()
        self.day, self.used = datetime.date.today(), 0
        self.visits = collections.defaultdict(collections.deque)

    def remaining_today(self):
        if not DAILY_LIMIT:
            return None
        with self.lock:
            if self.day != datetime.date.today():
                self.day, self.used = datetime.date.today(), 0
            return max(0, DAILY_LIMIT - self.used)

    def take(self, visitor: str):
        """Use one AI draft. Returns an error message if a limit is reached, else None."""
        if not DEMO_MODE:
            return None
        with self.lock:
            if self.day != datetime.date.today():
                self.day, self.used = datetime.date.today(), 0
            if DAILY_LIMIT and self.used >= DAILY_LIMIT:
                return ("This demo has reached its limit of AI replies for today (it runs on a free "
                        "Gemini allowance). Please try again tomorrow, or explore the staff dashboard.")
            hits = self.visits[visitor]
            while hits and hits[0] < time.time() - 3600:
                hits.popleft()
            if VISITOR_LIMIT and len(hits) >= VISITOR_LIMIT:
                return f"You've sent {VISITOR_LIMIT} questions in the last hour. Please wait a little before sending more."
            hits.append(time.time())
            self.used += 1
            return None


LIMITS = Limits()


# ---------- Simple JSON storage ----------
def load() -> list:
    if DATA.exists():
        return json.loads(DATA.read_text(encoding="utf-8"))
    return []


def save(items: list) -> None:
    DATA.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def update(eid: str, **fields):
    with LOCK:
        items = load()
        for item in items:
            if item["id"] == eid:
                item.update(fields)
                save(items)
                return item
    return None


# ---------- Running the agent ----------
def reply_text(draft) -> str:
    """Newer Gemini models return a list of parts; keep only the text."""
    if isinstance(draft, list):
        draft = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in draft)
    return plain(str(draft)).strip()


def plain(text: str) -> str:
    """Emails are plain text: drop markdown bold and turn '* ' bullets into '• '."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return re.sub(r"(?m)^(\s*)[*-] ", r"\1• ", text)


# ---------- Triage: topic, priority and a suggested action for staff ----------
TOPIC_WORDS = [
    ("complaint", ("complain", "unhappy", "appeal", "unfair")),
    ("refunds", ("refund", "withdraw", "money back", "deposit back", "cancel")),
    ("fees", ("fee", "cost", "price", "how much", "£", "pay", "tuition", "bursary", "scholarship")),
    ("entry_requirements", ("join", "entry", "requirement", "ucas", "qualif", "ielts", "gcse", "a-level", "eligible")),
    ("application", ("apply", "application", "offer", "deadline", "interview", "enrol")),
]
LEVELS = ["low", "medium", "high"]


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def topic(question: str, tools_used: list) -> str:
    q = question.lower()
    for name, words in TOPIC_WORDS[:2]:  # complaints and refunds win over everything
        if any(w in q for w in words):
            return name
    if "check_entry_requirements" in tools_used:
        return "entry_requirements"
    for name, words in TOPIC_WORDS[2:]:
        if any(w in q for w in words):
            return name
    return "general"


def parse_tool_result(content):
    if isinstance(content, dict):
        return content
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(content)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return None


def assess(question: str, draft: str, tools_used: list, messages=None) -> dict:
    """Rule-based triage, so staff see the riskiest enquiries first."""
    q = question.lower()
    verdicts, summary = [], ""
    for m in messages or []:
        if m.type == "tool" and getattr(m, "name", "") == "check_entry_requirements":
            result = parse_tool_result(m.content)
            if result:
                verdicts.append(result)
        elif m.type == "ai" and m.content and not getattr(m, "tool_calls", None):
            summary = reply_text(m.content)  # the agent's summary of the facts it found

    category = topic(question, tools_used)
    state = {"priority": "low"}
    reasons = []

    def flag(level: str, why: str) -> None:
        if LEVELS.index(level) > LEVELS.index(state["priority"]):
            state["priority"] = level
        reasons.append(why)

    if category == "complaint":
        flag("high", "Complaint: needs a personal response from staff")
    if category == "refunds":
        flag("high", "Refund question: only the Finance Office can decide refunds")

    rec = None
    for v in verdicts:
        course, verdict = v.get("course", "the course"), v.get("verdict", "")
        why = "; ".join(v.get("reasons", []))
        detail = f": {why}" if why else ""
        if verdict == "does not meet requirements":
            flag("high", f"Doesn't meet entry requirements for {course}{detail}")
            rec = rec or "Confirm the shortfall, consider pointing them to a course they do qualify for, then approve."
        elif verdict.startswith("possible"):
            flag("high", f"{course}: possible, but an interview is needed{detail}")
            rec = rec or "Arrange an interview, then approve the reply."
        elif verdict == "unknown course":
            flag("medium", f"Asked about a course the college doesn't list ({course})")
        elif verdict == "meets requirements":
            reasons.append(f"Meets entry requirements for {course}")
            rec = rec or "Check the reply, approve it and invite them to apply."

    if any(w in q for w in ("visa", "international", "ielts", "overseas")):
        flag("medium", "International applicant: check visa and IELTS details")
    if not tools_used:
        flag("medium", "The assistant didn't check any college documents")
    if "follow up" in draft.lower() and state["priority"] == "low":
        flag("medium", "The reply promises a follow-up from staff")

    if category == "complaint":
        rec = "Acknowledge within 2 working days and pass to the Student Services Manager (complaints procedure)."
    elif category == "refunds":
        rec = "Forward to the Finance Office and make sure the reply doesn't promise a refund."
    rec = rec or "Check the facts against the college documents, then approve."

    return {"category": category, "priority": state["priority"], "reasons": reasons,
            "recommendation": rec, "summary": summary}


def is_model_problem(err: Exception) -> bool:
    """Quota used up, model busy or slow, or model retired: worth trying the next model."""
    text = str(err)
    return any(k in text for k in (
        "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED",
        "404", "NOT_FOUND", "timed out", "Timeout",
    ))


def friendly(err: Exception) -> str:
    text = str(err)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "The daily free Gemini allowance is used up for every model we tried. Try again later (it resets daily), or add billing to the API key."
    if "503" in text or "UNAVAILABLE" in text or "504" in text or "DEADLINE" in text:
        return "Gemini is very busy right now. Wait a few minutes and press Try again."
    if "API key" in text or "PERMISSION_DENIED" in text or "401" in text or "403" in text:
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in the .env file."
    return text[:300]


def short_reason(err: Exception) -> str:
    text = str(err)
    for keys, label in ((("429", "RESOURCE_EXHAUSTED"), "daily quota used up"),
                        (("503", "UNAVAILABLE"), "too busy"),
                        (("504", "DEADLINE", "timed out", "Timeout"), "too slow to answer"),
                        (("404", "NOT_FOUND"), "model retired")):
        if any(k in text for k in keys):
            return label
    return "error"


def clip(text, n=600) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return text if len(text) <= n else text[:n].rstrip() + " …"


def build_trace(messages, model: str, fallbacks: list, triage: dict) -> list:
    """The real sequence of steps the agent took, for the 'watch the agent think' replay."""
    steps = [{"kind": "start", "title": "Enquiry received", "detail": "Sent to the LangGraph agent node"}]
    for f in fallbacks:
        steps.append({"kind": "fallback", "title": f"{f['model']} unavailable ({f['reason']})",
                      "detail": "Switched to the next model in the fallback list"})
    steps.append({"kind": "model", "title": f"Model: {model}", "detail": ""})
    for m in messages:
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                steps.append({"kind": "call", "title": f"Agent decided to call {tc['name']}",
                              "detail": clip(tc.get("args", {}), 400)})
        elif m.type == "tool":
            steps.append({"kind": "result", "title": f"{getattr(m, 'name', 'tool')} returned",
                          "detail": clip(m.content)})
        elif m.type == "ai" and m.content:
            steps.append({"kind": "think", "title": "Agent decided it had enough facts",
                          "detail": clip(reply_text(m.content))})
    steps.append({"kind": "draft", "title": "write_reply node drafted the email", "detail": "Uses only the facts gathered above"})
    steps.append({"kind": "triage", "title": f"Triage: {triage['priority']} priority, {triage['category'].replace('_', ' ')}",
                  "detail": "; ".join(triage["reasons"]) or "No risk flags"})
    steps.append({"kind": "wait", "title": "Paused for a human", "detail": "Nothing is sent until a member of staff approves"})
    return steps


def draft_once(question: str):
    """Run the agent, falling back through MODELS. Returns (fields, None) or (None, error)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    last_err, fallbacks = None, []
    for model in MODELS:
        try:
            llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=1, timeout=60)
            out = agent.run(question, graph=agent.build_graph(llm))
            draft = reply_text(out["draft_reply"])
            triage = assess(question, draft, out["tools_used"], out["messages"])
            return {"status": "review", "tools_used": out["tools_used"], "draft": draft,
                    "model": model, "error": "", "drafted_at": now(),
                    "trace": build_trace(out["messages"], model, fallbacks, triage), **triage}, None
        except Exception as e:
            last_err = e
            if not is_model_problem(e):
                break  # a real bug or bad key: switching models won't help
            fallbacks.append({"model": model, "reason": short_reason(e)})
            print(f"{model} unavailable ({short_reason(e)}), trying the next model...")
    return None, friendly(last_err)


def draft_in_background(eid: str, question: str) -> None:
    fields, error = draft_once(question)
    update(eid, **fields) if fields else update(eid, status="error", error=error)


def start_draft(eid: str, question: str) -> None:
    threading.Thread(target=draft_in_background, args=(eid, question), daemon=True).start()


# ---------- HTTP handler ----------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":  # HEAD requests (uptime monitors) get headers only
            self.wfile.write(body)

    def _json(self, code: int, data) -> None:
        self._send(code, json.dumps(data).encode("utf-8"), "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _visitor(self) -> str:
        # Behind a host's proxy the real visitor address is in X-Forwarded-For.
        fwd = self.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() or self.client_address[0]

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/health":
            self._json(200, {"status": "ok"})
        elif path == "/api/enquiries":
            with LOCK:
                items = load()
            self._json(200, sorted(items, key=lambda i: i["created"], reverse=True))
        elif path == "/api/info":
            self._json(200, {"model": agent.MODEL_NAME, "models": MODELS, "demo": DEMO_MODE,
                             "remaining_today": LIMITS.remaining_today(), "daily_limit": DAILY_LIMIT})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        parts = self.path.strip("/").split("/")  # api / enquiries / <id> / <action>
        data = self._body()

        if parts == ["api", "enquiries"]:
            question = (data.get("question") or "").strip()
            if not question:
                return self._json(400, {"error": "Please type your question."})
            limited = LIMITS.take(self._visitor())
            if limited:
                return self._json(429, {"error": limited})
            item = {
                "id": uuid.uuid4().hex[:8],
                "name": (data.get("name") or "").strip()[:80],
                "question": question[:4000],
                "created": now(),
                "status": "drafting",
                "tools_used": [],
                "draft": "",
                "error": "",
            }
            with LOCK:
                items = load()
                items.append(item)
                save(items)
            start_draft(item["id"], item["question"])
            return self._json(201, item)

        if len(parts) == 4 and parts[:2] == ["api", "enquiries"]:
            eid, action = parts[2], parts[3]
            by = (data.get("by") or "Admissions staff").strip()[:80]
            if action == "save":
                item = update(eid, draft=data.get("draft", ""))
            elif action == "approve":
                item = update(eid, draft=data.get("draft", ""), status="approved",
                              decided_at=now(), decided_by=by, note="")
            elif action == "reject":
                item = update(eid, status="rejected", decided_at=now(), decided_by=by,
                              note=(data.get("note") or "").strip()[:500])
            elif action == "retry":
                limited = LIMITS.take(self._visitor())
                if limited:
                    return self._json(429, {"error": limited})
                item = update(eid, status="drafting", error="", draft="", tools_used=[], trace=[],
                              decided_at="", decided_by="", note="")
                if item:
                    start_draft(eid, item["question"])
            else:
                return self._json(404, {"error": "Unknown action"})
            return self._json(200, item) if item else self._json(404, {"error": "Not found"})

        self._json(404, {"error": "Not found"})


def seed_demo_data() -> None:
    """First start of a demo: load sample enquiries so the dashboard isn't empty.
    Times in the seed file are stored as 'minutes ago' so the sample always looks recent."""
    if DATA.exists() or not SEED.exists():
        return
    base = datetime.datetime.now()
    items = json.loads(SEED.read_text(encoding="utf-8"))
    for item in items:
        for key in ("created", "drafted_at", "decided_at"):
            ago = item.pop(f"{key}_minutes_ago", None)
            if ago is not None:
                item[key] = (base - datetime.timedelta(minutes=ago)).isoformat(timespec="seconds")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    save(items)
    print(f"Loaded {len(items)} sample enquiries from {SEED.name}")


def main() -> None:
    if DEMO_MODE:
        seed_demo_data()
    # Anything left "drafting" from a previous run was interrupted.
    with LOCK:
        items = load()
        for item in items:
            if item["status"] == "drafting":
                item.update(status="error", error="The server stopped before this finished. Press Try again.")
            if "category" not in item:  # enquiries saved before triage existed
                item.update(assess(item["question"], item.get("draft", ""), item.get("tools_used", [])))
            if item["status"] == "review":
                item["draft"] = plain(item.get("draft", ""))
                item["summary"] = plain(item.get("summary", ""))
            if item["status"] == "approved" and not item.get("decided_at"):
                item.update(decided_at=item["created"], decided_by="Admissions staff")
        save(items)
    print(f"Admissions Assistant. Models, in order of preference: {', '.join(MODELS)}")
    if DEMO_MODE:
        print(f"Demo mode: {DAILY_LIMIT} AI drafts a day, {VISITOR_LIMIT} per visitor per hour")
    shown = "127.0.0.1" if HOST == "0.0.0.0" else HOST
    print(f"Open http://{shown}:{PORT} in your browser. Press Ctrl+C to stop.", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
