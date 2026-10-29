"""Test a diverse suite of new queries on the Helpdesk Multi-Agent and Single-Agent systems."""
import json
import time
from helpdesk import Helpdesk
from config import BASE_DIR

new_queries = [
    # 1. New Policy queries
    {
        "domain": "Policy",
        "question": "What is the minimum CGPA required for branch change at the end of the second semester?",
        "expected_domain": "policy"
    },
    {
        "domain": "Policy",
        "question": "Can a student register for 700 level courses in B.Tech, and what is the eligibility criteria?",
        "expected_domain": "policy"
    },
    {
        "domain": "Policy",
        "question": "How many credits are required for a B.Tech with Minor in Electrical Engineering?",
        "expected_domain": "policy"
    },
    # 2. New Schedule & Timetable queries
    {
        "domain": "Schedule",
        "question": "What is the course code, slot, and instructor for the course Optimization for ML?",
        "expected_domain": "schedule"
    },
    {
        "domain": "Schedule",
        "question": "In which room is the course Analog Circuits conducted and in which slot?",
        "expected_domain": "schedule"
    },
    {
        "domain": "Schedule",
        "question": "What is the L-T-P structure and credits for the course EEL3010?",
        "expected_domain": "schedule"
    },
    {
        "domain": "Schedule",
        "question": "How many courses are offered by the Chemical Engineering department this semester?",
        "expected_domain": "schedule"
    },
    # 3. Faculty contact & Email drafting
    {
        "domain": "Email",
        "question": "Draft an email to the instructor of EEL3010 requesting permission for a makeup exam due to medical hospitalization.",
        "expected_domain": "email"
    },
    {
        "domain": "Email",
        "question": "What is the email ID of the instructor teaching CSL7650?",
        "expected_domain": "schedule"
    },
    # 4. Out of Scope Refusals
    {
        "domain": "Out-of-Scope",
        "question": "What is the nearest railway station to IIT Jodhpur?",
        "expected_domain": "out_of_scope"
    },
    {
        "domain": "Out-of-Scope",
        "question": "Can you recommend good restaurants or cafes around Jodhpur?",
        "expected_domain": "out_of_scope"
    },
    {
        "domain": "Out-of-Scope",
        "question": "How can I book a sports badminton court in the indoor stadium?",
        "expected_domain": "out_of_scope"
    }
]

def run_tests():
    hd_multi = Helpdesk("multi")
    
    print(f"{'='*80}")
    print(f"RUNNING {len(new_queries)} BRAND NEW QUERIES THROUGH MULTI-AGENT HELPDESK")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, q in enumerate(new_queries, 1):
        question = q["question"]
        domain = q["domain"]
        print(f"[{idx}/{len(new_queries)}] Query ({domain}): {question}")
        
        t0 = time.time()
        res = hd_multi.ask(question)
        elapsed = time.time() - t0
        
        agents = res.get("agents", [])
        tools = [tc.get("name") for tc in res.get("tool_calls", [])]
        answer = res.get("answer", "")
        refused = res.get("refused", False)
        
        print(f"    Elapsed: {elapsed:.2f}s | Refused: {refused}")
        print(f"    Agents: {' -> '.join(agents)}")
        print(f"    Tools: {tools}")
        print(f"    Answer Preview: {answer[:180].replace(chr(10), ' ')}...")
        print(f"{'-'*80}")
        
        results.append({
            "idx": idx,
            "domain": domain,
            "question": question,
            "agents": agents,
            "tools": tools,
            "refused": refused,
            "answer": answer,
            "elapsed": round(elapsed, 2)
        })

    # Save results to scratch / results directory
    out_path = BASE_DIR / "results" / "new_queries_evaluation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll {len(new_queries)} new queries executed and saved to {out_path}!")

if __name__ == "__main__":
    run_tests()
