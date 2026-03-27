"""
search_code.py — Agent tool: semantic search over the ingested codebase.
"""

from langchain_core.tools import tool

from rag import retriever


def make_search_tool(repo_name: str):
    """Factory: create a search_code tool with repo_name baked in."""

    @tool
    def search_code(query: str) -> str:
        """
        Search the ingested codebase for code relevant to the query.

        Use this tool when you need to find code snippets, functions, classes,
        or any source code related to a concept or feature. Returns the most
        relevant code chunks along with their file paths.

        Args:
            query: A natural-language description of what you are looking for.
        """
        results = retriever.search(query=query, repo_name=repo_name, k=5)

        if not results:
            return "No relevant code found for this query."

        output_parts = []
        for i, r in enumerate(results, 1):
            output_parts.append(
                f"### Result {i}  (relevance: {r['score']})\n"
                f"**File:** `{r['file_path']}`\n"
                f"**Language:** {r['language']}\n"
                f"```\n{r['content']}\n```"
            )

        return "\n\n".join(output_parts)

    return search_code
