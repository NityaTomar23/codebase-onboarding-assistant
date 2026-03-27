"""
explain_file.py — Agent tool: explain the purpose and structure of a specific file.
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from rag import retriever


def make_explain_tool(repo_name: str):
    """Factory: create an explain_file tool with repo_name baked in."""

    @tool
    def explain_file(file_path: str) -> str:
        """
        Explain the purpose and structure of a specific file in the codebase.

        Use this tool when the user asks about a specific file, wants to
        understand what a file does, or needs details about functions and
        classes inside a file.

        Args:
            file_path: The path of the file to explain (as shown in search results).
        """
        chunks = retriever.get_file_chunks(file_path=file_path, repo_name=repo_name)

        if not chunks:
            return f"No indexed content found for file: {file_path}"

        combined_code = "\n\n".join(chunk["content"] for chunk in chunks)
        language = chunks[0].get("language", "unknown")

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = (
            f"You are a senior software engineer. Explain the following {language} file "
            f"located at `{file_path}`.\n\n"
            "Provide:\n"
            "1. **Purpose** — what this file does in one or two sentences\n"
            "2. **Key functions / classes** — list each with a short description\n"
            "3. **Dependencies** — notable imports and what they are used for\n"
            "4. **How it fits** — how this file relates to the rest of the codebase\n\n"
            f"```{language}\n{combined_code}\n```"
        )

        response = llm.invoke(prompt)
        return response.content

    return explain_file
