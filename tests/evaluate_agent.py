"""Live evaluation of the full pipeline (agent + tools + triage) against real Gemini calls.

    python tests/evaluate_agent.py              # run all cases with the models in .env
    python tests/evaluate_agent.py --seed       # also save the results as demo_seed.json
    EVAL_MODELS=gemini-3.5-flash-lite python tests/evaluate_agent.py

Each enquiry costs roughly 3-5 model calls, so the full set uses ~40 calls of your quota.

Checks per enquiry:
  tools      the agent called at least the tools it needs (a subset check: an extra,
             reasonable tool call is not a failure)
  verdict    the fixed-rules eligibility verdict is the expected one (when relevant)
  triage     topic and priority match what staff should see
  facts      the drafted email contains the key fact from the college documents
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402

CASES = [
    {"name": "Aisha Khan",
     "question": "Hi, I'm 19 and I got two Ds in my A-levels (48 UCAS points) and GCSE English and Maths. Can I join the HND in Computing?",
     "tools": {"check_entry_requirements"}, "verdict": "does not meet requirements",
     "category": "entry_requirements", "priority": "high", "facts": ["64"]},
    {"name": "Li Wei",
     "question": "How much are the fees for the Level 3 Diploma in IT for international students?",
     "tools": {"search_policies"}, "category": "fees", "priority": "medium", "facts": ["9,000"]},
    {"name": "Daniel Okafor",
     "question": "I'm 25. I don't have any A-levels but I do have GCSE English and Maths. Could I still get onto the HND in Computing?",
     "tools": {"check_entry_requirements"}, "verdict": "possible, interview needed",
     "category": "entry_requirements", "priority": "high", "facts": ["interview"]},
    {"name": "Maria Santos",
     "question": "I'm 22, an international student with IELTS 5.5 and 70 UCAS points, plus GCSE English and Maths. Can I join the HND in Computing?",
     "tools": {"check_entry_requirements"}, "verdict": "does not meet requirements",
     "category": "entry_requirements", "priority": "high", "facts": ["6.0"]},
    {"name": "Sam Patel",
     "question": "If my student visa is refused, do I get my £2,000 deposit back?",
     "tools": {"search_policies"}, "category": "refunds", "priority": "high", "facts": ["refund"]},
    {"name": "Hannah Morris",
     "question": "I'm 15 with no qualifications yet. Can I join Python for Beginners, and how much does it cost?",
     "tools": {"check_entry_requirements"}, "verdict": "meets requirements",
     "category": "entry_requirements", "priority": None, "facts": ["180"]},
    {"name": "Tom Reed",
     "question": "I want to make a complaint. I applied over three weeks ago and nobody has replied to my emails.",
     "tools": {"search_policies"}, "category": "complaint", "priority": "high", "facts": ["complaint"]},
    {"name": "Yusuf Ali",
     "question": "How do I apply for the Foundation Year in Business, and how long until I hear back?",
     "tools": {"search_policies"}, "category": "application", "priority": None, "facts": ["10 working days"]},
]


def verdicts_from(trace) -> list:
    out = []
    for step in trace:
        if step["title"].startswith("check_entry_requirements returned"):
            r = app.parse_tool_result(step["detail"])
            if r:
                out.append(r.get("verdict"))
    return out


def run(seed: bool) -> None:
    if os.getenv("EVAL_MODELS"):
        app.MODELS = [m.strip() for m in os.environ["EVAL_MODELS"].split(",") if m.strip()]
    print(f"Models (in fallback order): {', '.join(app.MODELS)}\n")

    totals = {"tools": [0, 0], "verdict": [0, 0], "triage": [0, 0], "facts": [0, 0]}
    results = []
    for n, case in enumerate(CASES, 1):
        if n > 1:
            time.sleep(15)  # stay under the free tier's requests-per-minute limit
        fields, error = app.draft_once(case["question"])
        if error:
            print(f"[{n}] ERROR {case['question'][:60]}...\n     {error}\n")
            for k in totals:
                totals[k][1] += 1
            continue

        checks = {}
        checks["tools"] = case["tools"] <= set(fields["tools_used"])
        if case.get("verdict"):
            checks["verdict"] = case["verdict"] in verdicts_from(fields["trace"])
        checks["triage"] = (fields["category"] == case["category"]
                            and (case["priority"] is None or fields["priority"] == case["priority"]))
        checks["facts"] = all(f.lower() in fields["draft"].lower() for f in case["facts"])
        for k, ok in checks.items():
            totals[k][0] += ok
            totals[k][1] += 1

        mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
        print(f"[{n}] {case['question'][:70]}...")
        print(f"     model={fields['model']}  tools={fields['tools_used']}  "
              f"triage={fields['priority']}/{fields['category']}")
        print("     " + "  ".join(f"{k}:{mark(v)}" for k, v in checks.items()) + "\n")
        results.append({"case": case, "fields": fields, "checks": checks})

    print("=" * 60)
    for k, (ok, total) in totals.items():
        if total:
            print(f"{k:8} {ok}/{total}  ({ok / total:.0%})")

    if seed and results:
        save_seed(results)


def save_seed(results) -> None:
    """Turn real evaluation output into sample data for the public demo."""
    items = []
    minutes = [400, 330, 260, 190, 140, 95, 50, 20]
    approved = {"Li Wei": ("Priya Shah", 300), "Hannah Morris": ("Priya Shah", 120)}
    for r, ago in zip(results, minutes):
        case, f = r["case"], r["fields"]
        item = {"id": os.urandom(4).hex(), "name": case["name"], "question": case["question"],
                "created_minutes_ago": ago, "drafted_at_minutes_ago": ago - 1, "error": "", **f}
        item.pop("drafted_at", None)
        if case["name"] in approved:
            by, decided_ago = approved[case["name"]]
            item.update(status="approved", decided_by=by, decided_at_minutes_ago=decided_ago, note="")
        items.append(item)
    app.SEED.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(items)} sample enquiries to {app.SEED.name}")


if __name__ == "__main__":
    run(seed="--seed" in sys.argv)
