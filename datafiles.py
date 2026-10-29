"""Structured data (CSV) loading + validation. Used by both ingest.py and tools.py."""
import re

import pandas as pd

TITLES = {"dr", "prof", "professor", "mr", "mrs", "ms", "sir", "madam"}


# ---------------------------------------------------------------- reading / cleaning
def read_csv(path) -> pd.DataFrame:
    """Read a CSV saved by Excel (utf-8 / cp1252 / latin-1), all values as stripped strings."""
    last = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc).fillna("")
            break
        except UnicodeDecodeError as e:
            last = e
    else:
        raise last
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """'Course_Code'->course_code, 'Subject'->course_name, 'L-T-P'->ltp, 'Credit'->credits,
    'Co-Instructor'->co_instructor, 'Classroom'->room, 'Student Re...'->students."""
    rename = {"subject": "course_name", "course": "course_name", "l_t_p": "ltp",
              "credit": "credits", "classroom": "room"}
    cols = []
    for c in df.columns:
        k = re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        k = rename.get(k, k)
        if k.startswith("student"):
            k = "students"
        cols.append(k)
    df.columns = cols
    return df


def load_timetable(path) -> pd.DataFrame:
    df = norm_cols(read_csv(path))
    missing = [c for c in ("course_code", "course_name", "department", "instructor", "slot")
               if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing column(s) {missing}. Found: {list(df.columns)}")
    df = df[df.course_code != ""].drop_duplicates().reset_index(drop=True)
    if "co_instructor" not in df.columns:
        df["co_instructor"] = ""
    if "level" not in df.columns:               # CSL3050 -> 300, MSL73090 -> 700
        df["level"] = df.course_code.map(
            lambda c: (re.search(r"\d", c).group(0) + "00") if re.search(r"\d", c) else "")
    return df


def load_faculty(path) -> pd.DataFrame:
    df = norm_cols(read_csv(path))
    name_col, mail_col = faculty_columns(df)
    if not name_col or not mail_col:
        raise KeyError(f"{path}: need a name and an email column. Found: {list(df.columns)}")
    return df[df[name_col] != ""].reset_index(drop=True)


def faculty_columns(fac: pd.DataFrame):
    name_col = next((c for c in ("instructor", "name", "faculty_name", "faculty") if c in fac.columns),
                    next((c for c in fac.columns if "name" in c), None))
    mail_col = next((c for c in fac.columns if "mail" in c), None)
    return name_col, mail_col


# ---------------------------------------------------------------- name matching
def name_tokens(name) -> set:
    return {t for t in re.findall(r"[a-z]+", str(name).lower()) if t not in TITLES}


def name_score(query, candidate) -> float:
    """Fraction of the query's name tokens found in candidate (order/titles ignored)."""
    a, b = name_tokens(query), name_tokens(candidate)
    return len(a & b) / len(a) if a else 0.0


# ---------------------------------------------------------------- validation
def check_timetable(df: pd.DataFrame) -> list[str]:
    w = []
    if (n := (df.instructor == "").sum()):
        w.append(f"{n} course(s) have no instructor")
    if (n := (df.slot == "").sum()):
        w.append(f"{n} course(s) have no slot")
    if "credits" in df.columns:
        bad = df[~df.credits.str.match(r"^\d+(\.\d+)?$")]
        if len(bad):
            w.append(f"{len(bad)} course(s) with non-numeric credits, e.g. {bad.course_code.iloc[0]}")
    if (n := (df.level == "").sum()):
        w.append(f"{n} course code(s) contain no digit, so level is unknown")
    return w


def check_faculty(fac: pd.DataFrame, tt: pd.DataFrame) -> list[str]:
    w = []
    name_col, mail_col = faculty_columns(fac)
    bad = fac[~fac[mail_col].str.lower().str.endswith("@iitj.ac.in")]
    if len(bad):
        w.append(f"{len(bad)} faculty email(s) not ending in @iitj.ac.in, e.g. {bad[name_col].iloc[0]}")
    names = fac[name_col].tolist()
    no_email = sorted({i for i in tt.instructor.unique()
                       if i and max((name_score(i, n) for n in names), default=0) < 0.5})
    if no_email:
        w.append(f"{len(no_email)} timetable instructor(s) not found in faculty list "
                 f"(emails unavailable), e.g. {', '.join(no_email[:5])}")
    return w