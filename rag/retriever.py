"""
retriever.py — Query the Chroma vector store and return relevant code chunks.
"""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


def get_vectorstore(repo_name: str) -> Chroma:
    """Load an existing Chroma collection for the given repo."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=repo_name,
        embedding_function=embeddings,
    )


def search(
    query: str,
    repo_name: str,
    k: int = 5,
    language_filter: str | None = None,
    path_prefix: str | None = None,
) -> list[dict]:
    """
    Semantic search over the codebase.

    Returns a list of dicts with keys: content, file_path, language, score.
    """
    vectorstore = get_vectorstore(repo_name)

    # Build optional metadata filter
    where_filter = None
    conditions = []
    if language_filter:
        conditions.append({"language": language_filter})
    if path_prefix:
        conditions.append({"file_path": {"$contains": path_prefix}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    results = vectorstore.similarity_search_with_relevance_scores(
        query, k=k, filter=where_filter
    )

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "file_path": doc.metadata.get("file_path", "unknown"),
            "language": doc.metadata.get("language", "unknown"),
            "score": round(score, 4),
        })

    return formatted


def get_file_chunks(file_path: str, repo_name: str) -> list[dict]:
    """Retrieve all indexed chunks for a specific file path.

    Uses Chroma's get() for a pure metadata fetch — no query vector needed.
    """
    vectorstore = get_vectorstore(repo_name)
    collection = vectorstore._collection  # access underlying chromadb collection

    results = collection.get(where={"file_path": file_path})

    if not results["documents"]:
        return []

    return [
        {
            "content": doc,
            "file_path": meta.get("file_path", "unknown"),
            "language": meta.get("language", "unknown"),
        }
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]
