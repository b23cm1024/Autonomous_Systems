"""Thin wrappers around tools.py for Google ADK.

ADK shows each function's name, type hints and DOCSTRING to the LLM, so these use only simple types
(str) with "" meaning "not given", and the docstrings are written for an 8B model.
"""
import tools as T


def search_documents(query: str) -> dict:
    """Search the institute's official documents: academic regulations (UG/PG), attendance policy,
    B.Tech credit requirements, academic calendar, office orders, program curricula and course syllabi.

    Args:
        query: A short, self-contained search query, e.g. "open elective credits required for B.Tech".

    Returns:
        status 'ok' with 'results' (each has text, source file and page), or 'no_relevant_results'.
    """
    return T.search_docs(query)


def lookup_course(course_code: str = "", course_name: str = "", query: str = "") -> dict:
    """Look up a course in this semester's timetable (Timetable.csv). Gives slot, room, instructor,
    co-instructor, credits, L-T-P, department and level.

    Args:
        course_code: Course code such as "CSL3050" (optional).
        course_name: Course name such as "Autonomous Systems" (optional).
        query: Generic search string matching code or name (optional).
    """
    if query and not course_code:
        import re
        if re.search(r"\b(how\s+many|count)\b", query, re.I) and "course" in query.lower():
            # LLM mistakenly called lookup_course for a count query
            dept = ""
            for d in ["chemical", "computer", "electrical", "mechanical", "civil", "physics", "chemistry", "mathematics", "humanities"]:
                if d in query.lower():
                    dept = d
                    break
            if dept:
                return count_courses(department=dept)
        code_match = re.search(r"\b([A-Za-z]{2,4}\s*\d{3,4}[A-Za-z]?)\b", query)
        if code_match:
            course_code = code_match.group(1).replace(" ", "")
        else:
            clean_q = re.sub(r"(?i)^(what\s+is\s+(the\s+)?(course\s+)?(code|slot|room|classroom|instructor|name)?\s+(with\s+the\s+|for|of)?\s*)", "", query)
            clean_q = re.sub(r"(?i)(course\s+code|the\s+course)", "", clean_q).strip(" ?.")
            course_name = clean_q or query.strip()
    return T.get_course(course_code or None, course_name or None)


def count_courses(department: str = "", slot: str = "", level: str = "",
                  instructor: str = "", credits: str = "") -> dict:
    """Count and list the courses in this semester's timetable that match ALL given filters.
    Use for questions like "how many 400 level courses does the Humanities department offer".

    Args:
        department: Department name or short form, e.g. "Humanities", "CSE", "EE" (optional).
        slot: Slot letter, e.g. "A" (optional).
        level: Course level 100-700, e.g. "400" (optional).
        instructor: Instructor name (optional).
        credits: Credits, e.g. "3" (optional).
    """
    return T.count_courses(department or None, slot or None, level or None,
                           instructor or None, credits or None)


def get_instructor_email(instructor: str = "", course_code: str = "", course_name: str = "") -> dict:
    """Get a faculty member's email ID (faculty.csv), either by their name or by a course they teach.

    Args:
        instructor: Faculty name (optional).
        course_code: Course code of a course they teach (optional).
        course_name: Course name of a course they teach (optional).
    """
    return T.get_instructor_email(instructor or None, course_code or None, course_name or None)


def draft_email(kind: str, course_code: str, student_name: str = "", days: str = "",
                reason: str = "", exam_name: str = "") -> dict:
    """Write an email DRAFT from the student to the course instructor. It is never sent.
    Pass only facts the student actually gave; never invent a name or a reason.

    Args:
        kind: "leave" for leave / sick leave, or "makeup" for permission to take a make-up exam.
        course_code: Course code, e.g. "CSL7570" (required). Find it with lookup_course if only the name is known.
        student_name: Student's name, if given.
        days: Number of days of leave, e.g. "2" (required for kind="leave").
        reason: Reason, e.g. "fever", only if the student gave one.
        exam_name: Which exam was missed, e.g. "Major exam" (for kind="makeup").

    Returns:
        status 'ok' with to / subject / body, or 'need_info' listing what to ask the student, or 'not_found'.
    """
    return T.draft_email(kind, course_code, student_name, days, reason, exam_name)


def log_unanswered_question(question: str, reason: str) -> dict:
    """Record a question that cannot be answered, so a human staff member can answer it later.
    Call this whenever you must say you don't know.

    Args:
        question: The student's question, copied exactly.
        reason: Why it cannot be answered, e.g. "out of scope" or "not found in documents".
    """
    return T.log_unanswered(question, reason)
