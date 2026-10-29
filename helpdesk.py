"""Run the Academic Helpdesk.

    ollama pull qwen3:8b ; ollama pull llama3.1:8b        (once; Ollama must be running)
    python helpdesk.py                       # interactive chat, multi-agent
    python helpdesk.py --mode single         # interactive chat, single-agent baseline
    python helpdesk.py --model ollama_chat/llama3.1:8b
    python helpdesk.py --demo                # run the assignment's sample benchmark questions
    python helpdesk.py --demo --trace        # also print which agent and tools were triggered
"""
import argparse
import asyncio
import inspect
import re
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config as C
import tools as T
import model_backend

APP, USER = "iitj_helpdesk", "student"
REFUSAL_RE = re.compile(
    r"(don't|do not|doesn't|cannot|can't|could not|couldn't|unable to)\s+(know|find|answer|provide)"
    r"|not (available )?in (the|my) (institute )?(documents|records|data)|no relevant",
    re.I,
)
THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


class Helpdesk:
    def __init__(self, mode: str = "multi"):
        from agents import build_agent            # imported late so --model can set config first
        self.mode = mode
        self.agent = build_agent(mode)
        self.svc = InMemorySessionService()
        self.runner = Runner(agent=self.agent, app_name=APP, session_service=self.svc)
        self.loop = asyncio.new_event_loop()      # ONE loop, reused (LiteLLM requires steady loops)
        self._sessions = set()

    # ---------------------------------------------------------------- core
    async def _ask(self, question: str, session_id: str) -> dict:
        if session_id not in self._sessions:
            res = self.svc.create_session(app_name=APP, user_id=USER, session_id=session_id)
            if inspect.isawaitable(res):           # async compatibility in ADK
                await res
            self._sessions.add(session_id)
        msg = types.Content(role="user", parts=[types.Part(text=question)])
        trace = {
            "question": question,
            "answer": "",
            "agents": [],
            "tool_calls": [],
            "tool_results": [],
            "error": "",
        }
        t0 = time.time()
        try:
            async for ev in self.runner.run_async(user_id=USER, session_id=session_id, new_message=msg):
                if ev.author and ev.author != "user" and ev.author not in trace["agents"]:
                    trace["agents"].append(ev.author)
                for fc in ev.get_function_calls() or []:
                    trace["tool_calls"].append({"name": fc.name, "args": dict(fc.args or {})})
                for fr in ev.get_function_responses() or []:
                    trace["tool_results"].append({"name": fr.name, "response": fr.response})
                if ev.is_final_response() and ev.content and ev.content.parts:
                    text = "".join(p.text for p in ev.content.parts if getattr(p, "text", None))
                    if text.strip():
                        trace["answer"] = THINK_RE.sub("", text).strip()
        except Exception as e:                     # connection refused, model down, bad tool call format
            err_msg = f"{type(e).__name__}: {e}"
            if "11434" in err_msg or "refused" in err_msg.lower():
                err_msg += (
                    " [HINT: Ollama is not running. Please start Ollama with 'ollama serve' "
                    "or verify that OLLAMA_API_BASE in config.py is correct.]"
                )
            trace["error"] = err_msg
        trace["seconds"] = round(time.time() - t0, 1)

        # Safety net: a refusal or error must always end up in the human-review log
        called = {c["name"] for c in trace["tool_calls"]}
        trace["refused"] = is_refusal(trace["answer"]) or not trace["answer"]
        trace["logged_unanswered"] = "log_unanswered_question" in called
        if trace["refused"] and not trace["logged_unanswered"]:
            T.log_unanswered(
                question,
                "auto-logged: agent could not answer"
                + (f" ({trace['error']})" if trace["error"] else ""),
            )
            trace["logged_unanswered"] = True
            trace["auto_logged"] = True
        if not trace["answer"]:
            trace["answer"] = "(no answer produced)" + (f" [{trace['error']}]" if trace["error"] else "")
        T.log_interaction(question, trace["answer"], sorted(called))
        return trace

    def ask(self, question: str, session_id: str | None = None) -> dict:
        return self.loop.run_until_complete(self._ask(question, session_id or str(uuid.uuid4())))


# ---------------------------------------------------------------- Sample Questions
DEMO = [
    "What is the attendance requirement to sit in the Major exam?",
    "What is the slot for the course Autonomous Systems?",
    "How many 400 level electives have been floated by the Humanities department in this semester?",
    "How many Open Elective credits do I need to earn in order to get a B.Tech. Degree?",
    "Write an email to the course instructor of CSL7570 that I need leave for 2 days",
    "What are the available dining options on the campus?",
    "What is the best way to reach the campus from Jodhpur city center?",
]


def show(r: dict, trace: bool):
    if trace:
        print(f"  [agents: {' > '.join(r['agents'])} | tools: {[c['name'] for c in r['tool_calls']]} "
              f"| {r['seconds']}s]")
    print(f"\n{r['answer']}\n")


def main():
    ap = argparse.ArgumentParser(description="IIT Jodhpur Academic Helpdesk Agent")
    ap.add_argument("--mode", choices=["single", "multi"], default="multi",
                    help="Agent mode: 'single' (base) or 'multi' (orchestrator + specialists)")
    ap.add_argument("--model", help="e.g. ollama_chat/qwen3:8b or ollama_chat/llama3.1:8b")
    ap.add_argument("--demo", action="store_true", help="Run the assignment benchmark questions")
    ap.add_argument("--trace", action="store_true", help="Show sub-agents and tool calls trace")
    a = ap.parse_args()
    if a.model:
        C.LLM_MODEL = a.model
    hd = Helpdesk(a.mode)
    print(f"=== IIT Jodhpur Academic Helpdesk [{a.mode}-agent | {C.LLM_MODEL}] ===")
    if a.demo:
        for q in DEMO:
            print("=" * 70, f"\nQ: {q}")
            show(hd.ask(q), a.trace)
        return
    print("IIT Jodhpur Academic Helpdesk (type 'exit' or 'quit' to quit)")
    sid = str(uuid.uuid4())                        # one session keeps conversation context
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if q:
            show(hd.ask(q, sid), a.trace)


if __name__ == "__main__":
    main()
