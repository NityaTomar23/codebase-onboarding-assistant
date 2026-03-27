"""
ingestor.py — Clone a GitHub repo, chunk source files by language, and embed into Chroma.
"""

import os
import shutil
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from git import Repo
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CLONE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cloned_repos")

SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "__pycache__",
    "venv", ".venv", "env", ".env", ".tox", ".eggs",
    "egg-info", ".mypy_cache", ".pytest_cache", ".next",
    "vendor", "target", "bin", "obj", "out",
}

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".class",
    ".lock", ".sum",
}

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".sol": Language.SOL,
}

MAX_FILE_SIZE_BYTES = 500_000  # skip files > 500 KB


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clone_repo(github_url: str) -> str:
    """Clone a GitHub repo into a local directory and return the path."""
    os.makedirs(CLONE_DIR, exist_ok=True)
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = os.path.join(CLONE_DIR, repo_name)

    if os.path.exists(dest):
        shutil.rmtree(dest)

    Repo.clone_from(github_url, dest, depth=1)
    return dest


def _clear_collection_if_exists(repo_name: str) -> None:
    """Delete an existing Chroma collection to prevent duplicate chunks on re-ingestion."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    if repo_name in existing:
        client.delete_collection(repo_name)


def _should_skip(path: Path) -> bool:
    """Return True if the file should be skipped."""
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        return True
    if path.stat().st_size == 0:
        return True
    return False


def _detect_language(path: Path) -> Language | None:
    """Map file extension to a LangChain Language enum value."""
    return EXTENSION_TO_LANGUAGE.get(path.suffix.lower())


def _load_and_chunk(file_path: Path, repo_name: str) -> list[Document]:
    """Load a single file and split it into chunks with metadata."""
    language = _detect_language(file_path)

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not content.strip():
        return []

    # Fix 6: Skip minified / auto-generated files (very few newlines, huge size)
    if content.count("\n") < 5 and len(content) > 10_000:
        return []

    # Build a language-aware splitter when possible, else fall back to generic
    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1500,
            chunk_overlap=200,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )

    chunks = splitter.create_documents(
        texts=[content],
        metadatas=[{
            "file_path": str(file_path),
            "language": language.value if language else "unknown",
            "repo_name": repo_name,
        }],
    )
    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def ingest_repo(github_url: str) -> str:
    """
    End-to-end ingestion: clone → walk → chunk → embed → store in Chroma.

    Returns the repo name.
    """
    repo_path = _clone_repo(github_url)
    repo_name = os.path.basename(repo_path)

    # Fix 3: Remove stale collection to prevent duplicate chunks on re-ingestion
    _clear_collection_if_exists(repo_name)

    all_chunks: list[Document] = []
    root = Path(repo_path)

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if _should_skip(file_path):
            continue
        chunks = _load_and_chunk(file_path, repo_name)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"No indexable source files found in {github_url}")

    # Persist into Chroma
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=repo_name,
    )

    return repo_name
