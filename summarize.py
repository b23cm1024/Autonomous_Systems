"""Grade saved runs and write the report.

    python summarize.py                 # reads results/*.json  -> results/report.md, graded.csv, manual_review.csv

Metrics (per mode+model, averaged over runs)
  accuracy      answer contains the gold fact (computed from your CSVs) or the expected keywords;
                for out-of-scope questions = correctly refused; email-without-info questions = no draft produced
  groundedness  answered AND cites a document that the tools really returned (no invented sources)
                (+ the expected source file when the question defines one)
  refusal       out-of-scope questions answered with "I don't know"  /  in-scope questions wrongly refused
  logged        refused questions that reached the human-review log
  routing       multi-agent only: question went to the right expert (and out-of-scope to none)
Questions flagged manual=True have no auto-checkable answer: see results/manual_review.csv.
"""
import argparse
import csv
import difflib
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import config as C
import datafiles as D

TT = D.load_timetable(C.TIMETABLE_CSV)
FAC = D.load_faculty(C.FACULTY_CSV)
KNOWN_FILES = sorted({p.name for p in C.PDF_DIR.rglob("*.pdf")} | {C.TIMETABLE_CSV.name, C.FACULTY_CSV.name})
EXPERT = {"policy": "policy_agent", "schedule": "schedule_agent", "email": "email_agent"}


def norm(s):
    return " ".join(re.findall(r"[a-z0-9@.+-]+", str(s).lower()))


# ------------------------------------------------------------ gold answers from the CSVs
def gold_for(q):
    g = q.get("gold")
    if not g:
        return None
    if "csv" in g:
        s = g["csv"]
        df = (TT[TT.course_code.str.upper() == s["course_code"].upper()] if "course_code" in s
              else TT[TT.course_name.str.lower().str.contains(s["course_name"].lower())])
        return {"error": "course not in timetable"} if df.empty else {"kind": s["kind"], "value": df.iloc[0][s["field"]]}
    if "count" in g:
        s, df = g["count"], TT
        if "department_contains" in s:
            dept_pat = "humanit|liberal|sola" if s["department_contains"] in ("humanit", "hss", "liberal") else s["department_contains"]
            df = df[df.department.str.lower().str.contains(dept_pat)]
        if "level" in s:
            df = df[df.level == s["level"]]
        if "slot" in s:
            df = df[df.slot.str.upper() == s["slot"].upper()]
        return {"kind": "count", "value": str(len(df))}
    if "email_for_course" in g:
        row = TT[TT.course_code.str.upper() == g["email_for_course"].upper()]
        if row.empty:
            return {"error": "course not in timetable"}
        nm, ml = D.faculty_columns(FAC)
        sc = FAC[nm].map(lambda n: D.name_score(row.iloc[0].instructor, n))
        return {"kind": "email", "value": FAC[sc == sc.max()].iloc[0][ml]} if sc.max() >= 0.5 else {"error": "no email"}
    return None


def matches(kind, value, ans):
    if kind == "text":
        v = value.strip()
        if len(v) == 1:                                    # slot letters: case-sensitive, standalone
            return re.search(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![A-Za-z0-9])", ans) is not None
        return norm(v) in norm(ans)
    if kind == "num":
        return any(float(n) == float(value) for n in re.findall(r"\d+(?:\.\d+)?", ans))
    if kind == "count":
        return value in re.findall(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])", ans)
    if kind == "name":
        toks = D.name_tokens(value)
        return sum(t in D.name_tokens(ans) for t in toks) / max(len(toks), 1) >= 0.6
    if kind == "email":
        return value.lower() in ans.lower()
    return False


# ------------------------------------------------------------ tool trace helpers
def _unwrap(resp):
    if isinstance(resp, dict) and isinstance(resp.get("result"), dict):
        return resp["result"]
    return resp if isinstance(resp, dict) else {}


def retrieved_files(rec):
    files = set()
    for tr in rec["tool_results"]:
        r = _unwrap(tr["response"])
        if r.get("status") not in ("ok", None):
            continue
        for item in r.get("results", []) or []:
            if isinstance(item, dict) and item.get("source"):
                files.add(item["source"])
        if r.get("source"):
            files.add(r["source"])
        if tr["name"] == "draft_email" and r.get("status") == "ok":
            files |= {C.TIMETABLE_CSV.name, C.FACULTY_CSV.name}
    return files


def draft_ok(rec):
    return any(tr["name"] == "draft_email" and _unwrap(tr["response"]).get("status") == "ok"
               for tr in rec["tool_results"])


# ------------------------------------------------------------ grading
def grade(q, rec):
    ans, cat = rec["answer"], q["category"]
    out = {"accuracy": None, "grounded": None, "refused": rec["refused"], "logged": None,
           "routing": None, "manual": bool(q.get("manual")), "note": ""}
    if rec.get("error"):
        out["note"] = "ERROR " + rec["error"][:80]

    if cat == "out_of_scope":
        out["accuracy"] = rec["refused"]
        out["logged"] = rec["logged_unanswered"] if rec["refused"] else None
        out["routing"] = not (set(rec["agents"]) & set(EXPERT.values())) if len(rec["agents"]) and \
            rec["agents"][0] == "helpdesk_orchestrator" else None
        return out

    if q.get("expect_draft") is False:                     # must ASK or say not found, never draft
        out["accuracy"] = (not draft_ok(rec)) and "subject:" not in ans.lower()
    elif rec["refused"]:
        out["accuracy"] = False                            # over-refusal
        out["note"] = out["note"] or "wrongly refused"
    else:
        checks = []
        g = gold_for(q)
        if g:
            checks.append(False if "error" in g else matches(g["kind"], g["value"], ans))
            out["note"] = out["note"] or (f"gold={g.get('value', g.get('error'))}")
        if q.get("expected_all"):
            checks.append(all(norm(k) in norm(ans) for k in q["expected_all"]))
        if q.get("expected_any"):
            checks.append(any(norm(k) in norm(ans) for k in q["expected_any"]))
        if q.get("expect_draft"):
            checks.append(draft_ok(rec))
        out["accuracy"] = all(checks) if checks else None

    if not rec["refused"] and cat != "out_of_scope" and q.get("expect_draft") is not False:
        cited = {f for f in KNOWN_FILES if f.lower() in ans.lower()}
        exp = set(q.get("expected_source_any", []))
        out["grounded"] = bool(cited) and cited <= retrieved_files(rec) and (not exp or bool(cited & exp))

    if rec["agents"] and rec["agents"][0] == "helpdesk_orchestrator":
        out["routing"] = EXPERT[cat] in rec["agents"]
    return out


# ------------------------------------------------------------ report
def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--questions", default="eval/questions.json")
    a = ap.parse_args()
    q_path = Path(a.questions) if Path(a.questions).is_absolute() else C.BASE_DIR / a.questions
    res_path = Path(a.results) if Path(a.results).is_absolute() else C.BASE_DIR / a.results

    if not q_path.exists():
        print(f"Questions file not found: {q_path}")
        return

    Q = {q["id"]: q for q in json.load(open(q_path, encoding="utf-8"))}
    groups = defaultdict(list)                              # (mode, model) -> [(run, records)]
    for f in sorted(glob.glob(str(res_path / "*__run*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        groups[(d["meta"]["mode"], d["meta"]["model"])].append((d["meta"]["run"], d["records"], d["meta"]))
    if not groups:
        print(f"No run files found in {res_path}. Run run_eval.py first.")
        return

    rows, report = [], ["# Evaluation report\n"]
    summary_rows, per_q = [], defaultdict(dict)
    for (mode, model), runs in groups.items():
        g = [(run, rec, grade(Q[rec["id"]], rec)) for run, recs, _ in runs for rec in recs if rec["id"] in Q]
        for run, rec, gr in g:
            qid = rec["id"]
            q_obj = Q.get(qid, {})
            # extract expected criteria / gold fact
            exp_list = q_obj.get("expected_any", []) or q_obj.get("expected_all", [])
            gold_info = q_obj.get("gold")
            if exp_list:
                expected_text = "Keywords: " + ", ".join(exp_list)
            elif gold_info:
                g_eval = gold_for(q_obj)
                expected_text = f"Gold Fact: {g_eval.get('value', str(gold_info))}" if g_eval else str(gold_info)
            elif q_obj.get("category") == "out_of_scope":
                expected_text = "Out of scope: Must refuse with 'I don't know' and log"
            elif q_obj.get("expect_draft") is False:
                expected_text = "Missing Info: Must ask for course code and days"
            else:
                expected_text = "Verified official circular / timetable"

            # extract cited files
            cited_files = [f for f in KNOWN_FILES if f.lower() in rec["answer"].lower()]
            sources_str = ", ".join(cited_files) if cited_files else "None"

            rows.append({
                "mode": mode,
                "model": model,
                "run": run,
                "question_id": qid,
                "category": rec["category"],
                "question": q_obj.get("question", ""),
                "expected_answer_criteria": expected_text,
                "generated_answer": rec["answer"].replace("\n", " "),
                "accuracy": "PASS" if gr["accuracy"] is True else ("FAIL" if gr["accuracy"] is False else "N/A"),
                "grounded": "PASS" if gr["grounded"] is True else ("N/A" if rec["category"] == "out_of_scope" else "FAIL"),
                "refused": "YES" if rec["refused"] else "NO",
                "logged_for_human": "YES" if gr["logged"] else ("N/A" if rec["category"] != "out_of_scope" else "NO"),
                "routing_correct": "PASS" if gr["routing"] is True else ("N/A" if mode == "single" else "FAIL"),
                "agents_trace": ">".join(rec["agents"]),
                "tools_used": ",".join(c["name"] for c in rec["tool_calls"]),
                "sources_cited": sources_str,
                "latency_seconds": rec["seconds"],
                "eval_note": gr["note"]
            })
        cat = lambda c: [x[2] for x in g if x[1]["category"] == c]  # noqa: E731
        oos, ins = cat("out_of_scope"), [x[2] for x in g if x[1]["category"] != "out_of_scope"]
        per_run = [pct([x[2]["accuracy"] for x in g if x[0] == r and x[2]["accuracy"] is not None])
                   for r in sorted({x[0] for x in g})]
        summary_rows.append({
            "System": f"{mode}-agent / {model}", "Runs": len(runs),
            "Accuracy (all auto-graded)": pct([x[2]["accuracy"] for x in g]),
            "  policy": pct([x["accuracy"] for x in cat("policy")]),
            "  schedule": pct([x["accuracy"] for x in cat("schedule")]),
            "  email": pct([x["accuracy"] for x in cat("email")]),
            "Groundedness": pct([x[2]["grounded"] for x in g]),
            "Correct refusals (out-of-scope)": pct([x["accuracy"] for x in oos]),
            "Refused-question logged": pct([x["logged"] for x in oos]),
            "Wrong refusals (in-scope)": pct([x["refused"] for x in ins]),
            "Routing": pct([x[2]["routing"] for x in g]),
            "Avg s/question": f"{sum(x[1]['seconds'] for x in g) / len(g):.1f}",
            "Errors": sum(1 for x in g if x[1].get("error")),
            "Accuracy per run": ", ".join(per_run)})
        for run, rec, gr in g:
            per_q[rec["id"]].setdefault((mode, model), []).append((run, rec, gr))

    report.append("## Summary\n")
    keys = list(summary_rows[0])
    report.append("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys))
    report += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in summary_rows]

    report.append("\n## Per-question pass count (accuracy over runs; `-` = manual, `x` = nothing to grade)\n")
    gkeys = list(groups)
    report.append("| id | category | question | " + " | ".join(f"{m}/{mod.split('/')[-1]}" for m, mod in gkeys) + " |")
    report.append("|" + "---|" * (3 + len(gkeys)))
    
    matrix_rows = []
    for qid, q in Q.items():
        cells = []
        row_dict = {"question_id": qid, "category": q["category"], "question": q["question"]}
        for k in gkeys:
            items = per_q[qid].get(k, [])
            acc = [x[2]["accuracy"] for x in items]
            val = "-" if acc and all(v is None for v in acc) else (f"{sum(1 for v in acc if v)}/{len(acc)}" if acc else "x")
            cells.append(val)
            col_name = f"{k[0]}_{k[1].split('/')[-1]}"
            row_dict[col_name] = val
        matrix_rows.append(row_dict)
        report.append(f"| {qid} | {q['category']} | {q['question'][:70]} | " + " | ".join(cells) + " |")

    report.append("\n## Questions whose answers changed between runs\n")
    any_change = False
    variation_rows = []
    for k in gkeys:
        for qid, q in Q.items():
            items = sorted(per_q[qid].get(k, []), key=lambda x: x[0])
            if len(items) < 2:
                continue
            outcomes = {(x[2]["accuracy"], x[2]["grounded"], x[2]["refused"]) for x in items}
            sims = [difflib.SequenceMatcher(None, norm(a[1]["answer"]), norm(b[1]["answer"])).ratio()
                    for i, a in enumerate(items) for b in items[i + 1:]]
            kind = "OUTCOME CHANGED" if len(outcomes) > 1 else ("wording changed" if min(sims) < 0.8 else None)
            if kind:
                any_change = True
                report.append(f"### {qid} - {q['question']}\n*{k[0]}-agent / {k[1]} - {kind}*\n")
                var_dict = {
                    "question_id": qid,
                    "category": q["category"],
                    "question": q["question"],
                    "system": f"{k[0]}-agent / {k[1]}",
                    "variation_type": kind
                }
                for run, rec, gr in items:
                    report.append(f"- run {run} (acc={gr['accuracy']}, grounded={gr['grounded']}, "
                                  f"refused={gr['refused']}): {rec['answer'][:300].replace(chr(10), ' ')}")
                    var_dict[f"run_{run}_answer"] = rec["answer"].replace("\n", " ")
                variation_rows.append(var_dict)
                report.append("")
    if not any_change:
        report.append("No question changed between runs (or only 1 run per system).")

    out = res_path
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    
    # Save graded.csv (with fallback if user has it open in Excel)
    graded_target = out / "graded.csv"
    try:
        with open(graded_target, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        graded_target = out / "graded_latest.csv"
        with open(graded_target, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # Save metrics_summary.csv
    try:
        with open(out / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            w.writeheader()
            w.writerows(summary_rows)
    except PermissionError:
        pass

    # Save Multi-Tab Excel Workbook (.xlsx) with pandas and openpyxl
    try:
        import pandas as pd
        excel_path = out / "Academic_Helpdesk_Evaluation_Metrics.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary_Metrics", index=False)
            pd.DataFrame(rows).to_excel(writer, sheet_name="All_Graded_Records", index=False)
            pd.DataFrame(matrix_rows).to_excel(writer, sheet_name="Per_Question_Matrix", index=False)
            if variation_rows:
                pd.DataFrame(variation_rows).to_excel(writer, sheet_name="Answer_Variations", index=False)
            else:
                pd.DataFrame([{"Note": "No significant variations"}]).to_excel(writer, sheet_name="Answer_Variations", index=False)
        print(f"Generated multi-tab Excel spreadsheet: {excel_path}")
    except Exception as e:
        print(f"Excel export warning: {e}")

    # Save manual_review.csv (only if any manual questions exist)
    manual_rows = [r for r in rows if Q[r["question_id"]].get("manual")]
    if manual_rows:
        try:
            with open(out / "manual_review.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["mode", "model", "run", "id", "question", "answer", "auto_grounded", "manual_correct(Y/N)", "comment"])
                for r in manual_rows:
                    w.writerow([r["mode"], r["model"], r["run"], r["question_id"], r["question"], r["generated_answer"],
                                r["grounded"], "", Q[r["question_id"]].get("note", "")])
        except PermissionError:
            pass
    print("\n".join(report))
    print(f"\nSaved: {out/'report.md'}, {graded_target}, {out/'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
