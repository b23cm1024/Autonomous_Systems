"""Ingest ALL data files.

    python ingest.py              # CSV check + index new/changed PDFs (fast on re-runs)
    python ingest.py --rebuild    # wipe the index and re-index every PDF
    python ingest.py --csv-only   # just validate Timetable.csv / faculty.csv
    python ingest.py --pdf-only

PDFs  -> cleaned -> chunked -> embedded -> ChromaDB (searched by tools.search_docs)
CSVs  -> loaded, normalised, validated  (queried exactly with pandas by the timetable/email tools;
         rows are NOT embedded, because counts and slots must be exact, not "semantically similar")
"""
import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

import chromadb
import pymupdf as fitz
from sentence_transformers import SentenceTransformer

import config as C
import datafiles as D

try:
    import wordninja  # splits glued words: "courseunlessthestudent"
except ImportError:
    wordninja = None

MANIFEST = Path(C.DB_DIR) / "manifest.json"
OLD_RULE = re.compile(r"allowed to appear for the exams|required to score at least (a )?C grade", re.I)


# ============================================================ text cleaning & chunking
def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)                       # ﬀ -> ff
    text = text.replace("\u200b", " ")
    text = re.sub(r"[^\x00-\x7F\u2018\u2019\u201c\u201d\u2013\u2014]+", " ", text)   # junk glyphs
    return re.sub(r"\s+", " ", text).strip()


def split_glued_words(text: str) -> str:
    if wordninja is None:
        return text

    def fix(m):
        parts = wordninja.split(m.group(0))
        return " ".join(parts) if all(len(p) > 1 or p.lower() in "ai" for p in parts) else m.group(0)

    return re.sub(r"[A-Za-z]{15,}", fix, text)


def chunk_text(text: str, size=C.CHUNK_CHARS, overlap=C.OVERLAP_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", text)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur.strip())
            tail = cur[-overlap:]
            cur = f"{tail[tail.find(' ') + 1:]} {s}"                 # overlap, from a word boundary
        else:
            cur = f"{cur} {s}".strip()
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 60]


# ============================================================ PDFs
def get_pages(pdf: Path) -> list[str]:
    """Page texts. Uses OCR output 'X.pdf.txt' (from ocr_pdfs.py) when it exists next to the PDF."""
    ocr = pdf.with_name(pdf.name + ".txt")
    if ocr.exists() and ocr.stat().st_size > 50:
        txt = ocr.read_text(encoding="utf-8", errors="ignore")
        return txt.split("\f") if "\f" in txt else [txt]
    with fitz.open(pdf) as doc:
        return [p.get_text() for p in doc]


def status_for(source: str, text: str) -> str:
    """Old attendance wording in any file other than the 2024 circular is superseded."""
    return "superseded" if source != C.CIRCULAR and OLD_RULE.search(text) else "current"


def pdf_chunks(pdf: Path):
    """-> (list of (id, text, metadata), list of page numbers with no text)"""
    title, out, empty = pdf.stem.replace("_", " "), [], []
    for pno, raw in enumerate(get_pages(pdf)):
        if len(raw.strip()) < 50:
            empty.append(pno + 1)
            continue
        text = split_glued_words(clean(raw))
        if pdf.name == C.CIRCULAR:      # drop the obsolete "Existing policy" section
            text = re.sub(r"Existing policy:.*?(?=Revised Policy:)", "", text, flags=re.I | re.S)
        for i, ch in enumerate(chunk_text(text)):
            out.append((f"{pdf.stem}-{pno}-{i}", f"[Document: {title}] {ch}",
                        {"source": pdf.name, "page": pno + 1, "status": status_for(pdf.name, ch)}))
    return out, empty


def file_fingerprint(pdf: Path) -> str:
    h = hashlib.sha1(f"{C.INGEST_VERSION}|{C.EMBED_MODEL}|{C.CHUNK_CHARS}|{C.OVERLAP_CHARS}".encode())
    for p in (pdf, pdf.with_name(pdf.name + ".txt")):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def ingest_pdfs(rebuild: bool) -> bool:
    pdfs = sorted(C.PDF_DIR.rglob("*.pdf"))
    print(f"\n=== PDFs in {C.PDF_DIR} : {len(pdfs)} found ===")
    if not pdfs:
        print(f"No PDFs found. Put them in {C.PDF_DIR.resolve()} - index NOT changed.")
        return False

    if rebuild:
        shutil.rmtree(C.DB_DIR, ignore_errors=True)
    Path(C.DB_DIR).mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() and not rebuild else {}
    client = chromadb.PersistentClient(path=C.DB_DIR)
    col = client.get_or_create_collection(C.COLLECTION, metadata={"hnsw:space": "cosine"})
    model, ok = None, True

    for pdf in pdfs:
        fp = file_fingerprint(pdf)
        if manifest.get(pdf.name) == fp:
            print(f"  = {pdf.name}: unchanged, skipped")
            continue
        try:
            chunks, empty = pdf_chunks(pdf)
        except Exception as e:                  # e.g. OneDrive file not downloaded, corrupt PDF
            print(f"  ! {pdf.name}: FAILED ({type(e).__name__}: {e})")
            ok = False
            continue
        col.delete(where={"source": pdf.name})   # replace any old version of this file
        if chunks:
            model = model or SentenceTransformer(C.EMBED_MODEL)
            for i in range(0, len(chunks), 64):
                b = chunks[i:i + 64]
                col.add(ids=[x[0] for x in b], documents=[x[1] for x in b], metadatas=[x[2] for x in b],
                        embeddings=model.encode([x[1] for x in b], normalize_embeddings=True).tolist())
        if chunks:                              # zero-chunk files (pure scans) are retried next run
            manifest[pdf.name] = fp
        ocr = " [OCR text]" if pdf.with_name(pdf.name + ".txt").exists() else ""
        warn = f"  ({len(empty)} page(s) without text, e.g. p{empty[:3]} - scanned? run OCR)" if empty else ""
        print(f"  + {pdf.name}: {len(chunks)} chunks{ocr}{warn}")
        if not chunks:
            ok = False

    for gone in set(manifest) - {p.name for p in pdfs}:     # PDF deleted from disk
        col.delete(where={"source": gone})
        manifest.pop(gone)
        print(f"  - {gone}: removed from index")

    MANIFEST.write_text(json.dumps(manifest, indent=1))
    n_old = len(col.get(where={"status": "superseded"}, include=[])["ids"])
    print(f"Index now holds {col.count()} chunks ({n_old} marked superseded).")
    return ok


# ============================================================ CSVs
def ingest_csvs() -> bool:
    print("\n=== CSV files ===")
    ok = True
    try:
        tt = D.load_timetable(C.TIMETABLE_CSV)
        print(f"{C.TIMETABLE_CSV.name}: {len(tt)} courses, columns = {list(tt.columns)}")
        print("  departments:", {k: int(v) for k, v in tt.department.value_counts().items()})
        print("  levels     :", {k: int(v) for k, v in sorted(tt.level.value_counts().items())})
        print("  slots      :", {k: int(v) for k, v in sorted(tt.slot.value_counts().items())})
        for w in D.check_timetable(tt):
            print("  ! warning:", w)
    except Exception as e:
        print(f"  ! {C.TIMETABLE_CSV.name}: FAILED ({type(e).__name__}: {e})")
        return False
    try:
        fac = D.load_faculty(C.FACULTY_CSV)
        print(f"{C.FACULTY_CSV.name}: {len(fac)} faculty, columns = {list(fac.columns)}")
        for w in D.check_faculty(fac, tt):
            print("  ! warning:", w)
    except Exception as e:
        print(f"  ! {C.FACULTY_CSV.name}: FAILED ({type(e).__name__}: {e})")
        ok = False
    return ok


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="wipe index and re-embed every PDF")
    ap.add_argument("--pdf-only", action="store_true")
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()
    ok = True
    if not a.pdf_only:
        ok &= ingest_csvs()
    if not a.csv_only:
        ok &= ingest_pdfs(a.rebuild)
    print("\nDONE - all good." if ok else "\nDONE - with problems (see '!' lines above).")


if __name__ == "__main__":
    main()