"""Offline test with a fake model (no API key needed) to check the graph wiring."""
from langchain_core.messages import AIMessage
import agent

class FakeLLM:
    def __init__(self): self.step = 0
    def bind_tools(self, tools): return self
    def invoke(self, msgs):
        self.step += 1
        if self.step == 1:
            return AIMessage("", tool_calls=[{"name": "search_policies", "args": {"query": "HND Computing entry requirements"}, "id": "1"}])
        if self.step == 2:
            return AIMessage("", tool_calls=[{"name": "check_entry_requirements", "args": {"course": "HND in Computing", "age": 19, "ucas_points": 48, "has_gcse_english_maths": True}, "id": "2"}])
        if self.step == 3:
            return AIMessage("Applicant does not meet HND requirements: needs 64 UCAS points.")
        return AIMessage("Dear student, thank you for your enquiry... (fake email)")

out = agent.run("test enquiry", graph=agent.build_graph(FakeLLM()))
print("TOOLS USED:", out["tools_used"])
for m in out["messages"]:
    if m.type == "tool": print("TOOL RESULT:", m.content[:120].replace("\n", " "))
print("DRAFT:", out["draft_reply"])
