"""
app.py — Streamlit frontend for the Codebase Onboarding Assistant.
"""

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from rag.ingestor import ingest_repo
from rag.chain import get_agent


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Codebase Onboarding Assistant",
    page_icon="🧠",
    layout="wide",
)

# ── Custom Styles ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success {
        background: #0f5132;
        color: #a3cfbb !important;
    }
    .badge-error {
        background: #842029;
        color: #f5c2c7 !important;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State ────────────────────────────────────────────────────────────

if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# ── Sidebar: Repo Ingestion ─────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 Codebase Assistant")
    st.markdown("---")

    st.subheader("📦 Repository Ingestion")
    github_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo",
    )

    if st.button("🚀 Ingest Repo", use_container_width=True):
        if not github_url:
            st.error("Please enter a GitHub URL.")
        else:
            with st.spinner("Cloning and indexing repository…"):
                try:
                    repo_name = ingest_repo(github_url)
                    st.session_state.repo_name = repo_name
                    st.session_state.agent = get_agent(repo_name)
                    st.session_state.messages = []  # reset chat for new repo
                    st.success(f"✅ **{repo_name}** ingested successfully!")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    st.markdown("---")

    # Ingestion status
    if st.session_state.repo_name:
        st.markdown(
            f'<span class="status-badge badge-success">'
            f"● {st.session_state.repo_name}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No repo ingested yet. Paste a GitHub URL above.")

    st.markdown("---")

    # Wiki generation button
    st.subheader("📖 Onboarding Wiki")
    wiki_topic = st.text_input("Focus area (optional)", placeholder="e.g. backend, auth")
    if st.button("📝 Generate Wiki", use_container_width=True):
        if not st.session_state.repo_name:
            st.error("Ingest a repo first!")
        else:
            with st.spinner("Generating onboarding wiki…"):
                try:
                    if not st.session_state.agent:
                        st.session_state.agent = get_agent(st.session_state.repo_name)
                    topic = wiki_topic.strip() if wiki_topic else "general"
                    result = st.session_state.agent.invoke(
                        {
                            "input": f"Generate an onboarding wiki focused on: {topic}",
                            "chat_history": [],
                        }
                    )
                    st.session_state.wiki_content = result["output"]
                except Exception as e:
                    st.error(f"Wiki generation failed: {e}")

# ── Main Area ────────────────────────────────────────────────────────────────

st.title("🧠 Codebase Onboarding Assistant")

if not st.session_state.repo_name:
    st.markdown(
        """
        ### 👋 Welcome!

        Get started in three steps:
        1. **Paste a GitHub URL** in the sidebar
        2. **Click Ingest Repo** to clone and index the codebase
        3. **Ask questions** below — or generate an onboarding wiki!

        *Example questions:*
        - *"How does authentication work?"*
        - *"Where is rate limiting handled?"*
        - *"What does the PaymentService class do?"*
        """
    )
else:
    # ── Wiki display ─────────────────────────────────────────────────────
    if "wiki_content" in st.session_state and st.session_state.wiki_content:
        with st.expander("📖 Generated Onboarding Wiki", expanded=True):
            st.markdown(st.session_state.wiki_content)
            st.download_button(
                label="⬇️ Download Wiki (.md)",
                data=st.session_state.wiki_content,
                file_name=f"{st.session_state.repo_name}_onboarding_wiki.md",
                mime="text/markdown",
            )
            if st.button("Dismiss Wiki"):
                st.session_state.wiki_content = None
                st.rerun()

    # ── Chat history ─────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ───────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about the codebase…")

    if user_input:
        # Append & display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get agent response — with streaming callback for live feedback
        with st.chat_message("assistant"):
            try:
                if not st.session_state.agent:
                    st.session_state.agent = get_agent(st.session_state.repo_name)

                # Build chat history for context
                from langchain_core.messages import HumanMessage, AIMessage
                chat_history = []
                for m in st.session_state.messages[:-1]:  # exclude current
                    if m["role"] == "user":
                        chat_history.append(HumanMessage(content=m["content"]))
                    else:
                        chat_history.append(AIMessage(content=m["content"]))

                # Fix 4: StreamlitCallbackHandler gives live tool-call
                # visibility + streamed final answer
                callback = StreamlitCallbackHandler(st.container())
                result = st.session_state.agent.invoke(
                    {
                        "input": user_input,
                        "chat_history": chat_history,
                    },
                    config={"callbacks": [callback]},
                )
                answer = result["output"]
            except Exception as e:
                answer = f"❌ Error: {e}"

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
