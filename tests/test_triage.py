"""Offline tests (no API key, no network) for the rules around the agent.

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

import app  # noqa: E402
import tools  # noqa: E402


def eligibility_messages(result: dict) -> list:
    import json
    return [
        AIMessage("", tool_calls=[{"name": "check_entry_requirements", "args": {}, "id": "1"}]),
        ToolMessage(json.dumps(result), name="check_entry_requirements", tool_call_id="1"),
        AIMessage("Summary of facts."),
    ]


class EntryRules(unittest.TestCase):
    def test_hnd_short_of_ucas(self):
        r = tools.check_entry_requirements("HND in Computing", age=19, ucas_points=48, has_gcse_english_maths=True)
        self.assertEqual(r["verdict"], "does not meet requirements")

    def test_mature_hnd_needs_interview(self):
        r = tools.check_entry_requirements("HND in Computing", age=25, has_gcse_english_maths=True)
        self.assertEqual(r["verdict"], "possible, interview needed")

    def test_international_ielts(self):
        r = tools.check_entry_requirements("HND in Computing", age=22, ucas_points=70, has_gcse_english_maths=True,
                                           ielts=5.5, is_international=True)
        self.assertIn("needs IELTS 6.0 (has 5.5)", r["reasons"])

    def test_unknown_course(self):
        self.assertEqual(tools.check_entry_requirements("Medicine", age=20)["verdict"], "unknown course")


class Triage(unittest.TestCase):
    def test_refund_is_high(self):
        t = app.assess("Can I get a refund?", "Dear student...", ["search_policies"])
        self.assertEqual((t["category"], t["priority"]), ("refunds", "high"))
        self.assertIn("Finance Office", t["recommendation"])

    def test_complaint_is_high_and_follows_procedure(self):
        t = app.assess("I want to complain about my application", "", ["search_policies"])
        self.assertEqual((t["category"], t["priority"]), ("complaint", "high"))
        self.assertIn("Student Services Manager", t["recommendation"])

    def test_not_eligible_is_high_with_reason(self):
        msgs = eligibility_messages({"course": "HND in Computing", "verdict": "does not meet requirements",
                                     "reasons": ["needs 64 UCAS points (has 48)"]})
        t = app.assess("Can I join the HND?", "Dear student", ["check_entry_requirements"], msgs)
        self.assertEqual(t["priority"], "high")
        self.assertTrue(any("needs 64 UCAS points" in r for r in t["reasons"]))
        self.assertEqual(t["summary"], "Summary of facts.")

    def test_simple_question_is_low(self):
        t = app.assess("What time do classes start?", "Classes start at 9am.", ["search_policies"])
        self.assertEqual((t["category"], t["priority"]), ("general", "low"))

    def test_no_tools_is_medium(self):
        self.assertEqual(app.assess("Hello", "Hi!", [])["priority"], "medium")

    def test_international_is_at_least_medium(self):
        t = app.assess("I'm an international student, what are the fees?", "x", ["search_policies"])
        self.assertIn(t["priority"], ("medium", "high"))


class ReplyCleanup(unittest.TestCase):
    def test_list_content_with_signature(self):
        parts = [{"type": "text", "text": "Dear **Student**,\n* Point one"}, {"type": "text", "text": "Thanks", "extras": {"signature": "abc"}}]
        self.assertEqual(app.reply_text(parts), "Dear Student,\n• Point one\nThanks")

    def test_plain_string(self):
        self.assertEqual(app.reply_text("  Hello  "), "Hello")


class ModelErrors(unittest.TestCase):
    def test_quota_and_busy_trigger_fallback(self):
        for msg in ("429 RESOURCE_EXHAUSTED", "503 UNAVAILABLE", "404 NOT_FOUND", "504 DEADLINE_EXCEEDED"):
            self.assertTrue(app.is_model_problem(Exception(msg)), msg)

    def test_bad_key_does_not_fallback(self):
        self.assertFalse(app.is_model_problem(Exception("400 API key not valid")))
        self.assertIn("API key", app.friendly(Exception("API key not valid")))


class DemoLimits(unittest.TestCase):
    def test_visitor_and_daily_limits(self):
        old = (app.DEMO_MODE, app.DAILY_LIMIT, app.VISITOR_LIMIT)
        app.DEMO_MODE, app.DAILY_LIMIT, app.VISITOR_LIMIT = True, 3, 2
        try:
            limits = app.Limits()
            self.assertIsNone(limits.take("a"))
            self.assertIsNone(limits.take("a"))
            self.assertIn("last hour", limits.take("a"))      # visitor limit
            self.assertIsNone(limits.take("b"))
            self.assertIn("limit of AI replies", limits.take("c"))  # daily limit
            self.assertEqual(limits.remaining_today(), 0)
        finally:
            app.DEMO_MODE, app.DAILY_LIMIT, app.VISITOR_LIMIT = old


if __name__ == "__main__":
    unittest.main()
