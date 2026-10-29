"""All settings in one place. Edit paths and configuration here."""
import os
from pathlib import Path

# ---- base directory ----
BASE_DIR = Path(__file__).resolve().parent

# ---- data files ----
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "docs"                 # every *.pdf under here is ingested
TIMETABLE_CSV = DATA_DIR / "Timetable.csv"
FACULTY_CSV = DATA_DIR / "faculty.csv"

# ---- vector index ----
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "helpdesk"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_DISTANCE = 0.31                         # calibrated cutoff: in-scope <= 0.286, out-of-scope >= 0.342

# ---- chunking ----
CHUNK_CHARS = 700
OVERLAP_CHARS = 150
INGEST_VERSION = "3"                        # bump when ingest logic changes -> forces re-index

# ---- special documents ----
CIRCULAR = "Attendance_Policy.pdf"          # 08 Jul 2024 circular; supersedes older attendance rules

# ---- LLM (Google ADK -> LiteLLM -> Ollama) ----
LLM_MODEL = os.environ.get("HELP_DESK_MODEL", "ollama_chat/qwen3:8b")  # or "ollama_chat/llama3.1:8b"
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
LLM_TEMPERATURE = 0.3                       # >0 so repeated runs can differ; 0 = most deterministic
LLM_NUM_CTX = 8192                          # lower to 4096 if running tight on RAM on a 16GB laptop

# ---- logs ----
LOG_FILE = str(BASE_DIR / "interactions.jsonl")            # every question + answer + tools used
UNANSWERED_FILE = str(BASE_DIR / "unanswered_questions.jsonl")  # questions the agent could not answer (for human review)
