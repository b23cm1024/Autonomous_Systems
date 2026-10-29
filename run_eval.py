"""Run the evaluation question set and save raw answers + traces (no grading here).

    python run_eval.py --mode multi  --model ollama_chat/qwen3:8b    --runs 3
    python run_eval.py --mode multi  --model ollama_chat/llama3.1:8b --runs 3
    python run_eval.py --mode single --model ollama_chat/qwen3:8b    --runs 3   # baseline for comparison
    python run_eval.py --runs 1 --only O01,S01,E01                              # quick smoke test

Each run file is results/<mode>__<model>__runK.json. Existing run files are skipped, so you can
stop and resume. Then run:  python summarize.py
"""
import argparse
import json
import re
import time
from pathlib import Path

import config as C


def main():
    ap = argparse.ArgumentParser(description="Evaluate Academic Helpdesk System")
    ap.add_argument("--mode", choices=["single", "multi"], default="multi")
    ap.add_argument("--model", help="e.g. ollama_chat/llama3.1:8b (default: LLM_MODEL in config.py)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--only", help="comma-separated question ids or categories")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true", help="force overwrite existing run logs")
    a = ap.parse_args()
    if a.model:
        C.LLM_MODEL = a.model

    q_path = Path(a.questions)
    if not q_path.is_absolute():
        q_path = C.BASE_DIR / q_path
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    questions = json.load(open(q_path, encoding="utf-8"))
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        questions = [q for q in questions if q["id"] in keep or q["category"] in keep]

    out = Path(a.out)
    if not out.is_absolute():
        out = C.BASE_DIR / out
    out.mkdir(exist_ok=True, parents=True)

    tag = f"{a.mode}__{re.sub(r'[^A-Za-z0-9.]+', '-', C.LLM_MODEL)}"

    from helpdesk import Helpdesk              # after config is set
    hd = Helpdesk(a.mode)
    print(f"=== Running Evaluation: {len(questions)} questions x {a.runs} runs | {a.mode}-agent | {C.LLM_MODEL} ===")

    for run in range(1, a.runs + 1):
        path = out / f"{tag}__run{run}.json"
        if path.exists() and not a.force:
            print(f"run {run}: {path.name} exists, skipped")
            continue
        records, t0 = [], time.time()
        for i, q in enumerate(questions, 1):
            r = hd.ask(q["question"])          # fresh session per question: no cross-talk
            records.append({"id": q["id"], "category": q["category"], **r})
            status = "REFUSED" if r["refused"] else "answered"
            print(f"  run {run} [{i:2d}/{len(questions)}] {q['id']} {status:8s} {r['seconds']:5.1f}s "
                  f"agents={'>'.join(r['agents'])}" + (f"  ERROR {r['error'][:60]}" if r["error"] else ""))
        meta = {
            "mode": a.mode,
            "model": C.LLM_MODEL,
            "run": run,
            "temperature": C.LLM_TEMPERATURE,
            "num_ctx": C.LLM_NUM_CTX,
            "minutes": round((time.time() - t0) / 60, 1),
            "date": time.strftime("%Y-%m-%d %H:%M"),
        }
        path.write_text(json.dumps({"meta": meta, "records": records}, indent=1, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        print(f"run {run} saved -> {path} ({meta['minutes']} min)")


if __name__ == "__main__":
    main()
