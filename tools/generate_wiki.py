"""
generate_wiki.py — Agent tool: generate a markdown onboarding wiki for the codebase.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from rag import retriever

# ── Token-budget constants ───────────────────────────────────────────────────

TOKEN_BUDGET = 80_000
_enc = tiktoken.get_encoding("cl100k_base")


def make_wiki_tool(repo_name: str):
    """Factory: create a generate_wiki tool with repo_name baked in."""

    @tool
    def generate_wiki(topic: str = "general") -> str:
        """
        Generate a comprehensive onboarding wiki for the ingested codebase.

        Use this tool when the user wants an onboarding document, architecture
        overview, or a high-level guide to the repository. The topic parameter
        can be used to focus the wiki on a specific area (e.g. 'backend',
        'authentication', 'database') or left as 'general' for a full overview.

        Args:
            topic: The focus area for the wiki. Defaults to 'general'.
        """
        # Gather broad context: entry points, READMEs, configs, main modules
        queries = [
            "main entry point application setup configuration",
            "README documentation project overview",
            "architecture modules structure directory",
            f"{topic} implementation overview",
            "key classes functions API endpoints",
        ]

        # ── Fix 5: Parallel retriever searches ──────────────────────────────
        all_results: list[list[dict]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(retriever.search, q, repo_name, 4): q
                for q in queries
            }
            for future in as_completed(futures):
                all_results.append(future.result())

        # Deduplicate by file path
        all_snippets: list[str] = []
        seen_files: set[str] = set()

        for results in all_results:
            for r in results:
                fp = r["file_path"]
                if fp not in seen_files:
                    seen_files.add(fp)
                    all_snippets.append(
                        f"**File:** `{fp}` ({r['language']})\n```\n{r['content']}\n```"
                    )

        # ── Fix 6: Token-budget instead of hard slice ────────────────────────
        context_snippets: list[str] = []
        total_tokens = 0
        for snippet in all_snippets:
            t = len(_enc.encode(snippet))
            if total_tokens + t > TOKEN_BUDGET:
                break
            context_snippets.append(snippet)
            total_tokens += t

        context = "\n\n---\n\n".join(context_snippets)

        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        prompt = (
            "You are a senior developer writing an onboarding wiki for a new team member.\n"
            f"Repository: **{repo_name}**\n"
            f"Focus area: **{topic}**\n\n"
            "Using the code snippets below, produce a well-structured Markdown wiki with:\n"
            "1. **Project Overview** — what the project does\n"
            "2. **Architecture** — high-level module map\n"
            "3. **Key Modules & Files** — purpose of each important file\n"
            "4. **Setup & Running** — how to get started (inferred from code/config)\n"
            "5. **Common Workflows** — how typical tasks are performed in the codebase\n"
            "6. **Glossary** — project-specific terms\n\n"
            "Cite exact file paths for every claim. Use Markdown formatting.\n\n"
            f"---\n\n{context}"
        )

        response = llm.invoke(prompt)
        return response.content

    return generate_wiki
