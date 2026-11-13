"""
Step 3: the AI agent.

Flow:  enquiry -> agent (decides which tools to call, can loop) -> tools -> agent ... -> write_reply
The agent node is the "brain". It looks at the enquiry and the tool results so far,
and either calls another tool or says "I have enough information".
"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import tools as college_tools

load_dotenv()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------- 1. Tools the agent is allowed to use ----------
@tool
def search_policies(query: str) -> str:
    """Search the college documents (courses, entry requirements, fees, refunds,
    how to apply, offers, complaints). Use this for any question about college policy."""
    results = college_tools.search_policies(query)
    if not results:
        return "No matching policy found."
    return "\n\n".join(f"[{r['heading']}]\n{r['text']}" for r in results)


@tool
def check_entry_requirements(
    course: str,
    age: int,
    ucas_points: int = 0,
    has_gcse_english_maths: bool = False,
    ielts: float | None = None,
    is_international: bool = False,
    work_experience_years: int = 0,
) -> dict:
    """Check if an applicant meets the entry requirements for a course, using fixed rules.
    course must be one of: 'Level 3 Diploma in IT', 'HND in Computing',
    'Foundation Year in Business', 'Python for Beginners'.
    Only call this when the enquiry gives the applicant's age and qualifications."""
    return college_tools.check_entry_requirements(
        course, age, ucas_points, has_gcse_english_maths, ielts,
        is_international, work_experience_years,
    )


TOOLS = [search_policies, check_entry_requirements]


# ---------- 2. What the agent remembers while it works ----------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation + tool results
    enquiry: str
    draft_reply: str


AGENT_INSTRUCTIONS = """You are an admissions assistant for Northbridge College London (a fictional college).
Your job is to gather the facts needed to answer a student's enquiry.

Rules:
- Use search_policies to find the relevant college policy. Never answer policy questions from memory.
- If the student gives their age and qualifications and asks if they can join a course,
  call check_entry_requirements. Never decide eligibility yourself.
- You may call several tools. When you have enough information, reply with a short
  summary of the facts you found. Do not write the email yet."""

WRITER_INSTRUCTIONS = """Write a short, friendly, professional email reply to the student,
using ONLY the facts gathered below. Do not promise offers, refunds or exceptions.
If something needs a staff decision, say a member of the admissions team will follow up.
Sign off as 'Admissions Team, Northbridge College London'."""


def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    # max_retries: try again if Google is busy (503), but don't hang forever
    # timeout: give up on a single request after 60 seconds
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_retries=3, timeout=60)


# ---------- 3. The nodes (steps) of the graph ----------
def build_graph(llm=None):
    llm = llm or get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent(state: AgentState):
        """The brain: decide whether to call a tool or stop."""
        msgs = [SystemMessage(AGENT_INSTRUCTIONS)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(msgs)]}

    def write_reply(state: AgentState):
        """Turn the gathered facts into an email (the draft_reply step)."""
        facts = "\n".join(
            f"{m.type}: {m.content}" for m in state["messages"] if m.content
        )
        prompt = [SystemMessage(WRITER_INSTRUCTIONS),
                  HumanMessage(f"Student enquiry:\n{state['enquiry']}\n\nFacts gathered:\n{facts}")]
        return {"draft_reply": llm.invoke(prompt).content}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("write_reply", write_reply)

    g.add_edge(START, "agent")
    # If the agent asked for a tool -> run it; otherwise -> write the reply
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "write_reply"})
    g.add_edge("tools", "agent")          # after a tool runs, the agent thinks again
    g.add_edge("write_reply", END)
    return g.compile()


def run(enquiry: str, graph=None) -> dict:
    graph = graph or build_graph()
    result = graph.invoke({"enquiry": enquiry, "messages": [HumanMessage(enquiry)], "draft_reply": ""})
    tools_used = [tc["name"] for m in result["messages"] for tc in getattr(m, "tool_calls", []) or []]
    return {"tools_used": tools_used, "draft_reply": result["draft_reply"], "messages": result["messages"]}


if __name__ == "__main__":
    print(f"Using model: {MODEL_NAME}. Asking Gemini, this can take up to a minute...")
    out = run("Hi, I'm 19 and I got two Ds in my A-levels (48 UCAS points) and GCSE English "
              "and Maths. Can I join the HND in Computing?")
    print("TOOLS USED:", out["tools_used"])
    print("\nDRAFT REPLY:\n", out["draft_reply"])
