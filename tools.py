"""Helpdesk tools for document search, timetable lookup, faculty directory, and email drafting."""
import datetime
import difflib
import json
import re
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CIRCULAR,
    COLLECTION,
    DB_DIR,
    EMBED_MODEL,
    FACULTY_CSV,
    LOG_FILE,
    MAX_DISTANCE,
    QUERY_PREFIX,
    TIMETABLE_CSV,
    UNANSWERED_FILE,
)
from datafiles import faculty_columns, load_faculty, load_timetable, name_score as _name_score

DEPT_ALIASES = {
    "cse": "computer science",
    "cs": "computer science",
    "ee": "electrical",
    "me": "mechanical",
    "ce": "civil",
    "civil": "civil",
    "chem": "chemical",
    "math": "mathematics",
    "maths": "mathematics",
    "phy": "physics",
    "humanities": "humanit",
    "hss": "humanit",
    "hum": "humanit",
    "sshs": "humanit",
    "bio": "bio",
    "mc": "mathematics",
    "aide": "artificial intelligence",
    "ai": "artificial intelligence",
}

STOP = {
    "what", "when", "where", "which", "who", "whom", "does", "have", "with", "from",
    "that", "this", "there", "about", "tell", "please", "should", "would", "could",
    "many", "much", "need", "the", "and", "for", "are", "can",
}

TT_NAME = TIMETABLE_CSV.name      # "Timetable.csv"
FAC_NAME = FACULTY_CSV.name       # "faculty.csv"


# ---------------- loaders ----------------
@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    return chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)


@lru_cache(maxsize=1)
def _timetable():
    return load_timetable(TIMETABLE_CSV)


@lru_cache(maxsize=1)
def _faculty():
    return load_faculty(FACULTY_CSV)


# ---------------- 1. document search ----------------
def search_docs(query: str, top: int = 6, k: int = 25,
                max_distance: float | None = None, keyword_guard: bool = True):
    """Semantic search with: superseded-rule filter, dedupe, distance cutoff, keyword guard."""
    max_distance = MAX_DISTANCE if max_distance is None else max_distance
    q = _model().encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=[q], n_results=k)

    kws = {w[:5] for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3 and w not in STOP}
    hits, seen, dropped_old = [], set(), 0
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if meta.get("status") != "current":
            dropped_old += 1
            continue
        key = re.sub(r"\W+", "", text.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        if dist > max_distance:
            continue
        if keyword_guard and kws and not any(kw in text.lower() for kw in kws):
            continue
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "distance": round(dist, 3),
        })

    if not hits:
        return {
            "status": "no_relevant_results",
            "message": "Nothing relevant found in the institute documents.",
        }

    # When querying attendance, prioritize the current circular Attendance_Policy.pdf
    if "attend" in query.lower():
        hits.sort(key=lambda h: (0 if h["source"] == CIRCULAR else 1, h["distance"]))

    out = {"status": "ok", "results": hits[:top]}
    if dropped_old:
        out["note"] = (
            "Older rules replaced by the 08 July 2024 attendance circular "
            "(Attendance_Policy.pdf) were excluded."
        )
    return out


# ---------------- 2. timetable ----------------
def _norm_code(c):
    return re.sub(r"[^A-Z0-9]", "", str(c).upper())


def _courses_ok(df, note=""):
    return {
        "status": "ok",
        "source": TT_NAME,
        "courses": df.to_dict("records"),
        "note": note,
    }


def get_course(course_code: str | None = None, course_name: str | None = None):
    df = _timetable()
    if course_code:
        m = df[df.course_code.map(_norm_code) == _norm_code(course_code)]
        if len(m):
            return _courses_ok(m)
        # partial match
        m2 = df[df.course_code.map(_norm_code).str.contains(_norm_code(course_code), na=False)]
        if len(m2):
            return _courses_ok(m2)
    if course_name:
        # 1. First check if course_name actually matches a course code directly
        norm_name = _norm_code(course_name)
        m_code = df[df.course_code.map(_norm_code) == norm_name]
        if len(m_code):
            return _courses_ok(m_code)
        # 2. Check if a course code is embedded in the string (e.g. "what is the course with code EEL3010")
        c_match = re.search(r"\b([A-Za-z]{2,4}\s*\d{3,4}[A-Za-z]?)\b", course_name)
        if c_match:
            norm_c = _norm_code(c_match.group(1))
            m_code2 = df[df.course_code.map(_norm_code) == norm_c]
            if len(m_code2):
                return _courses_ok(m_code2)

        # 3. Substring match against known course names (handles full questions like "In which room is the course Analog Circuits conducted?")
        q_exp = re.sub(r"\bml\b", "machine learning", course_name, flags=re.I)
        q_exp = re.sub(r"\bai\b", "artificial intelligence", q_exp, flags=re.I)
        q_lower = q_exp.lower()
        for cname in sorted(df.course_name.dropna().unique(), key=len, reverse=True):
            if len(cname) > 3 and cname.lower() in q_lower:
                m_sub = df[df.course_name.str.lower() == cname.lower()]
                if len(m_sub):
                    return _courses_ok(m_sub)

        # 4. Clean search query from sentence filler if needed
        clean_name = re.sub(r"(?i)^(what\s+is\s+(the\s+)?(course\s+)?(code|slot|room|classroom|instructor|name)?\s+(with\s+the\s+|for|of)?\s*)", "", course_name)
        clean_name = re.sub(r"(?i)(course\s+code\s+|the\s+course\s+)", "", clean_name).strip(" ?.")
        
        names = df.course_name.str.lower()
        n = (clean_name or course_name).lower().strip()
        m = df[names.str.contains(re.escape(n))]
        if len(m):
            return _courses_ok(m)
        close = difflib.get_close_matches(n, names.unique().tolist(), n=3, cutoff=0.6)
        if close:
            return _courses_ok(df[names.isin(close)], note="Closest name matches (no exact match).")
    pool = df.course_code.unique().tolist() if course_code else df.course_name.unique().tolist()
    sugg = difflib.get_close_matches(course_code or course_name or "", pool, n=3, cutoff=0.4)
    return {
        "status": "not_found",
        "message": f"No such course in {TT_NAME}.",
        "suggestions": sugg,
    }


def _resolve_dept(text):
    depts = _timetable().department.unique().tolist()
    tl = text.lower().strip()
    if tl in ("humanities", "hss", "hum", "sshs", "humanit", "liberal", "liberal arts", "sola", "school of liberal arts"):
        return [d for d in depts if any(k in d.lower() for k in ("liberal", "sola", "humanit"))]
    t = DEPT_ALIASES.get(tl, tl)
    m = [d for d in depts if t in d.lower()]
    return m or difflib.get_close_matches(text, depts, n=2, cutoff=0.5)


def count_courses(department: str | None = None, slot: str | None = None,
                  level: str | int | None = None, instructor: str | None = None,
                  credits: str | int | None = None):
    df, applied = _timetable(), []
    if department:
        depts = _resolve_dept(department)
        if not depts:
            return {
                "status": "ok",
                "source": TT_NAME,
                "count": 0,
                "courses": [],
                "note": f"Unknown department '{department}'. Known: {_timetable().department.unique().tolist()}",
            }
        df = df[df.department.isin(depts)]
        applied.append(f"department={depts}")
    if slot:
        df = df[df.slot.str.upper() == slot.strip().upper()]
        applied.append(f"slot={slot.upper()}")
    if level:
        df = df[df.level.astype(str) == str(level)]
        applied.append(f"level={level}")
    if credits:
        df = df[df.credits.astype(str) == str(credits)]
        applied.append(f"credits={credits}")
    if instructor:
        sc = df.instructor.map(lambda n: _name_score(instructor, n))
        co = df.co_instructor.map(lambda n: _name_score(instructor, n))
        df = df[(sc >= 0.5) | (co >= 0.5)]
        applied.append(f"instructor~{instructor}")
    note = "" if len(df) else f"No courses matched: {', '.join(applied) or 'no filters'}."
    return {
        "status": "ok",
        "source": TT_NAME,
        "count": len(df),
        "note": note,
        "courses": df.head(25).to_dict("records"),
    }


# ---------------- 3. faculty email ----------------
def get_instructor_email(instructor: str | None = None, course_code: str | None = None,
                         course_name: str | None = None):
    course = None
    if not instructor:
        c = get_course(course_code, course_name)
        if c["status"] != "ok":
            return c
        course = c["courses"][0]
        instructor = course["instructor"]
    fac = _faculty()
    name_col, mail_col = faculty_columns(fac)
    if not name_col or not mail_col:
        return {
            "status": "error",
            "message": f"{FAC_NAME} columns not recognised: {list(fac.columns)}",
        }
    scores = fac[name_col].map(lambda n: _name_score(instructor, n))
    best = scores.max()
    if best < 0.5:
        return {"status": "not_found", "message": f"No faculty match for '{instructor}'."}
    top = fac[scores == best]
    if top[name_col].nunique() > 1:
        return {"status": "ambiguous", "candidates": top[name_col].unique().tolist()}
    row = top.iloc[0]
    out = {
        "status": "ok",
        "source": FAC_NAME,
        "instructor": row[name_col],
        "email": row[mail_col],
    }
    if course:
        out["course_name"] = course["course_name"]
    return out


# ---------------- 4. email drafting ----------------
def draft_email(kind: str = "leave", course_code: str = "", student_name: str = "",
                days: int | str = "", reason: str = "", exam_name: str = "",
                instructor: str | None = None, instructor_email: str | None = None):
    """kind = 'leave' (sick/other leave) or 'makeup' (permission for a make-up exam)."""
    kind = (kind or "leave").lower()
    if not str(course_code).strip():
        return {"status": "need_info", "missing": ["course_code"]}
    if kind == "leave" and not str(days).strip():
        return {"status": "need_info", "missing": ["days"]}
    c = get_course(course_code=course_code)
    if c["status"] != "ok":
        return c
    course = c["courses"][0]
    instructor = instructor or course["instructor"]
    if not instructor_email:
        e = get_instructor_email(instructor=instructor)
        instructor_email = e.get("email", "") if e["status"] == "ok" else ""
    name = student_name.strip() or "[Your Name]"
    who = f"I am {student_name.strip()}, a student" if student_name.strip() else "I am a student"
    head = f"Dear Professor {instructor},\n\n{who} enrolled in {course['course_code']} ({course['course_name']}). "
    if kind == "makeup":
        exam = exam_name.strip() or "examination"
        why = f" due to {reason.strip()}" if reason.strip() else ""
        body = (
            head + f"I was unable to appear for the {exam}{why}. I would be grateful if you could "
            "kindly permit me to take a make-up examination. I can provide supporting documents "
            "(for example a medical certificate) if required."
        )
        subject = f"Request for make-up examination - {course['course_code']}"
    else:
        why = f" due to {reason.strip()}" if reason.strip() else ""
        body = (
            head + f"I am writing to request leave for {days} day(s){why}. I will catch up on the "
            "missed lectures and assignments, and I can provide supporting documents if required."
        )
        subject = f"Leave request - {course['course_code']} ({course['course_name']})"
    body += f"\n\nThank you for your understanding.\n\nSincerely,\n{name}"
    note = "Draft only - show it to the student; do not send it."
    if not student_name.strip():
        note += " Student name was not given: replace [Your Name]."
    if not instructor_email:
        note += f" Instructor email not found in {FAC_NAME}."
    return {
        "status": "ok",
        "to": instructor_email,
        "subject": subject,
        "body": body,
        "note": note,
    }


# ---------------- 5. logging ----------------
def log_unanswered(question: str, reason: str = ""):
    """Append a question the agent could not answer, for a human staff member to answer later."""
    rec = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "reason": reason,
        "status": "pending human answer",
    }
    with open(UNANSWERED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"status": "logged", "message": "Question recorded for staff follow-up."}


def log_interaction(question: str, answer: str = "", tools_used: list | None = None):
    """Append every question/answer pair (used for analysis and the audit trail)."""
    rec = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "tools": tools_used or [],
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"status": "logged"}
