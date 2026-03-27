"""Temporary script to verify all imports work after refactoring."""
import sys
print(f"Python: {sys.version}")
print(f"CWD: {__import__('os').getcwd()}")

try:
    from rag.chain import get_agent, CodebaseAgent
    print("[OK] rag.chain — get_agent, CodebaseAgent")
except Exception as e:
    print(f"[FAIL] rag.chain: {e}")

try:
    from rag.ingestor import ingest_repo, _clear_collection_if_exists
    print("[OK] rag.ingestor — ingest_repo, _clear_collection_if_exists")
except Exception as e:
    print(f"[FAIL] rag.ingestor: {e}")

try:
    from rag.retriever import search, get_file_chunks
    print("[OK] rag.retriever — search, get_file_chunks")
except Exception as e:
    print(f"[FAIL] rag.retriever: {e}")

try:
    from tools.search_code import make_search_tool
    t = make_search_tool("test-repo")
    assert t.name == "search_code", f"Expected 'search_code', got '{t.name}'"
    print("[OK] tools.search_code — make_search_tool factory works")
except Exception as e:
    print(f"[FAIL] tools.search_code: {e}")

try:
    from tools.explain_file import make_explain_tool
    t = make_explain_tool("test-repo")
    assert t.name == "explain_file", f"Expected 'explain_file', got '{t.name}'"
    print("[OK] tools.explain_file — make_explain_tool factory works")
except Exception as e:
    print(f"[FAIL] tools.explain_file: {e}")

try:
    from tools.generate_wiki import make_wiki_tool
    t = make_wiki_tool("test-repo")
    assert t.name == "generate_wiki", f"Expected 'generate_wiki', got '{t.name}'"
    print("[OK] tools.generate_wiki — make_wiki_tool factory works")
except Exception as e:
    print(f"[FAIL] tools.generate_wiki: {e}")

print("\n✅ All imports verified successfully!")
