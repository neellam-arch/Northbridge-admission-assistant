"""Try each Gemini model once and report which ones answer right now."""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
MODELS = [
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite",
    "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash",
]
working = []
for m in MODELS:
    try:
        llm = ChatGoogleGenerativeAI(model=m, max_retries=0, timeout=20)
        reply = llm.invoke("Reply with the word OK").content
        print(f"WORKS: {m}  ->  {str(reply)[:30]}")
        working.append(m)
    except Exception as e:
        print(f"busy/unavailable: {m}  ({str(e)[:60]})")

print("\nModels you can use right now:", working or "none, try again in 15 minutes")
