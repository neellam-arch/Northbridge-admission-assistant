"""
Tools the agent can use.

search_policies          -> finds relevant sections in the college documents
check_entry_requirements -> fixed Python rules, no AI, so the answer is always the same
"""
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are", "i", "my",
    "me", "do", "can", "what", "how", "if", "it", "on", "with", "be", "you", "your",
    "need", "want", "would", "like", "about", "please", "hi", "hello", "there", "this",
}


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9£]+", text.lower()) if w not in STOPWORDS]


def _load_chunks() -> list[dict]:
    """Split every document into chunks, one per '## ' section."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        title = ""
        for section in path.read_text(encoding="utf-8").split("\n## "):
            lines = section.strip().splitlines()
            if section.startswith("# "):          # document title, not a section
                title = lines[0].lstrip("# ").strip()
                continue
            chunks.append({
                "source": path.name,
                "heading": f"{title} > {lines[0].strip()}",
                "text": "\n".join(lines[1:]).strip(),
            })
    return chunks


CHUNKS = _load_chunks()


def search_policies(query: str, top_k: int = 3) -> list[dict]:
    """Return the top_k most relevant document sections for a query.

    Simple keyword scoring: words in the section heading count double,
    because headings are short and very specific.
    """
    q = set(_words(query))
    scored = []
    for c in CHUNKS:
        heading_hits = len(q & set(_words(c["heading"])))
        body_hits = len(q & set(_words(c["text"])))
        score = 2 * heading_hits + body_hits
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# Fixed entry rules, copied from entry_requirements.md
REQUIREMENTS = {
    "Level 3 Diploma in IT": {"min_age": 16, "ucas": 0, "gcse_eng_maths": True, "ielts": 5.0},
    "HND in Computing": {"min_age": 18, "ucas": 64, "gcse_eng_maths": True, "ielts": 6.0},
    "Foundation Year in Business": {"min_age": 18, "ucas": 48, "gcse_eng_maths": False, "ielts": 5.5},
    "Python for Beginners": {"min_age": 0, "ucas": 0, "gcse_eng_maths": False, "ielts": 0.0},
}


def check_entry_requirements(
    course: str,
    age: int,
    ucas_points: int = 0,
    has_gcse_english_maths: bool = False,
    ielts: float | None = None,
    is_international: bool = False,
    work_experience_years: int = 0,
) -> dict:
    """Check an applicant against the fixed rules. Returns a clear verdict and reasons."""
    if course not in REQUIREMENTS:
        return {"course": course, "verdict": "unknown course",
                "reasons": [f"Known courses: {', '.join(REQUIREMENTS)}"]}

    r = REQUIREMENTS[course]
    missing, notes = [], []

    if age < r["min_age"]:
        missing.append(f"minimum age is {r['min_age']}")

    if r["ucas"] and ucas_points < r["ucas"]:
        if course == "Foundation Year in Business" and work_experience_years >= 2:
            notes.append("below UCAS points but has 2+ years work experience: interview needed")
        elif course == "HND in Computing" and age >= 21:
            notes.append("mature applicant without UCAS points: may be considered after interview")
        else:
            missing.append(f"needs {r['ucas']} UCAS points (has {ucas_points})")

    if r["gcse_eng_maths"] and not has_gcse_english_maths:
        missing.append("needs GCSE English and Maths at grade 4 or above")

    if is_international and r["ielts"]:
        if ielts is None:
            missing.append(f"needs IELTS {r['ielts']} (no score given)")
        elif ielts < r["ielts"]:
            missing.append(f"needs IELTS {r['ielts']} (has {ielts})")

    if missing:
        verdict = "does not meet requirements"
    elif notes:
        verdict = "possible, interview needed"
    else:
        verdict = "meets requirements"
    return {"course": course, "verdict": verdict, "reasons": missing + notes}


if __name__ == "__main__":
    # Quick self-test: run `python tools.py`
    for r in search_policies("Can I get a refund if my visa is refused?"):
        print("FOUND:", r["heading"])
    print(check_entry_requirements("HND in Computing", age=19, ucas_points=48,
                                   has_gcse_english_maths=True))
    print(check_entry_requirements("HND in Computing", age=25, has_gcse_english_maths=True))
    print(check_entry_requirements("Foundation Year in Business", age=30, ucas_points=0,
                                   work_experience_years=3, is_international=True, ielts=6.0))
