"""
chain.py — Build a tool-calling agent with GPT-4o and the three codebase tools.

Uses llm.bind_tools() + a manual invocation loop, which is the stable
approach across all modern LangChain versions.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from tools.search_code import make_search_tool
from tools.explain_file import make_explain_tool
from tools.generate_wiki import make_wiki_tool

load_dotenv()

# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert codebase onboarding assistant. Your job is to help
developers understand a code repository quickly and thoroughly.

You have access to the following tools:
- **search_code**: Semantic search over the ingested codebase. Use it when
  you need to find functions, classes, patterns, or any code related to a
  concept.
- **explain_file**: Get a detailed explanation of a specific file. Use it
  when the user asks about a particular file.
- **generate_wiki**: Generate a comprehensive onboarding wiki. Use it when
  the user wants an architecture overview or onboarding document.

Rules:
1. ALWAYS cite the exact file path(s) in your answers.
2. When showing code, include the file path above the code block.
3. If you are unsure, search the codebase first before guessing.
4. Keep answers clear, structured, and actionable.
5. If the user asks a vague question, use search_code to gather context first.
"""

# ── Agent class ──────────────────────────────────────────────────────────────

class CodebaseAgent:
    """Thin wrapper around an LLM with bound tools and a conversation loop."""

    def __init__(self, repo_name: str):
        # Build tools via factories — repo_name is baked into each closure
        tools = [
            make_search_tool(repo_name),
            make_explain_tool(repo_name),
            make_wiki_tool(repo_name),
        ]
        self.tool_map = {t.name: t for t in tools}

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.llm = llm.bind_tools(tools)

    def invoke(self, input_dict: dict, config: dict | None = None) -> dict:
        """
        Run the agent.

        Args:
            input_dict: must contain 'input' (str) and 'chat_history' (list).
            config: optional LangChain config dict (e.g. {"callbacks": [...]}).

        Returns:
            dict with 'output' (str).
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add conversation history
        for msg in input_dict.get("chat_history", []):
            messages.append(msg)

        messages.append(HumanMessage(content=input_dict["input"]))

        # Agentic loop: keep calling tools until the LLM gives a final answer
        max_iterations = 10
        for _ in range(max_iterations):
            response = self.llm.invoke(messages, config=config)
            messages.append(response)

            # If no tool calls, we have a final answer
            if not response.tool_calls:
                return {"output": response.content}

            # Execute every tool call the LLM requested
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_fn = self.tool_map.get(tool_name)

                if tool_fn is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        result = f"Error running {tool_name}: {e}"

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        return {"output": response.content if response.content else "Max iterations reached."}


def get_agent(repo_name: str) -> CodebaseAgent:
    """Create and return a configured CodebaseAgent for the given repo."""
    return CodebaseAgent(repo_name)
