import os
import sys
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ingestion.web_loader import load_page, chunk_documents
from ingestion.vectorstore import build_index
from retrieval.rag import answer, rewrite_question

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Web RAG Chatbot",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Web RAG Chatbot")
st.caption("Load any webpage and ask questions about it!")

# ─────────────────────────────────────────────
# Session state initialize
# ─────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None

if "llm" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.session_state.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            temperature=0,
        )
    else:
        st.session_state.llm = None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Load a Webpage")

    # ── HuggingFace Token (optional) ──────────
    st.subheader("🤗 HuggingFace Login")
    hf_token_default = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    hf_token_input = st.text_input(
        "HuggingFace Token (optional)",
        value=hf_token_default,
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Add your HF token if you face rate limit or gated model issues. Get it from huggingface.co/settings/tokens",
    )
    if hf_token_input.strip():
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token_input.strip()
        st.success("✅ HF Token set!")
    else:
        st.caption("Public models work without a token.")

    st.divider()

    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com/page",
        help="Enter the full URL starting with https://",
    )

    load_btn = st.button("🚀 Load & Index", use_container_width=True)

    if load_btn:
        if not url_input.strip():
            st.error("⚠️ URL is empty! Please enter a URL.")

        elif not (url_input.startswith("http://") or url_input.startswith("https://")):
            st.error("⚠️ Invalid URL. Must start with `https://`.")

        elif st.session_state.llm is None:
            st.error("⚠️ GROQ_API_KEY is missing from secrets.")

        else:
            try:
                with st.spinner("🌐 Loading webpage..."):
                    docs = load_page(url_input.strip())

                if not docs or not docs[0].page_content.strip():
                    st.error("⚠️ No readable text found on this page. Try another URL.")
                else:
                    with st.spinner("✂️ Splitting into chunks..."):
                        chunks = chunk_documents(docs)

                    with st.spinner(f"🧠 Embedding {len(chunks)} chunks..."):
                        st.session_state.faiss_index = build_index(chunks)

                    st.session_state.chat_history = []
                    st.session_state.loaded_url = url_input.strip()
                    st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                err_msg = str(e)
                if "Timeout" in err_msg or "timed out" in err_msg.lower():
                    st.error("⚠️ Timeout! The page did not respond.")
                elif "getaddrinfo" in err_msg:
                    st.error("⚠️ URL not reachable. Check your internet or URL.")
                else:
                    st.error(f"⚠️ Something went wrong: {err_msg}")

    if st.session_state.loaded_url:
        st.divider()
        st.markdown("**📌 Loaded:**")
        url = st.session_state.loaded_url
        display = f"{url[:50]}..." if len(url) > 50 else url
        st.markdown(f"[{display}]({url})")

    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────

if st.session_state.faiss_index is None:
    st.info(
        "👈 **Enter a URL in the sidebar and click 'Load & Index'.**\n\n"
        "Once loaded, you can ask questions about the page!"
    )
    st.stop()

# ─────────────────────────────────────────────
# Chat history render
# ─────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if (
            msg["role"] == "user"
            and msg.get("rewritten")
            and msg["rewritten"].strip().lower() != msg["content"].strip().lower()
        ):
            st.caption(f"🔄 Interpreted as: *{msg['rewritten']}*")

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(msg["sources"], 1):
                    source_url = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Source {i}:** [{source_url}]({source_url})")
                    preview = doc.page_content[:300].replace("\n", " ")
                    st.caption(f"> {preview}{'...' if len(doc.page_content) > 300 else ''}")
                    if i < len(msg["sources"]):
                        st.divider()

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────

user_question = st.chat_input("Ask a question about this page...")

if user_question:

    with st.chat_message("user"):

        history_for_rewrite = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]

        if history_for_rewrite:
            with st.spinner("🔄 Interpreting question..."):
                standalone_q = rewrite_question(
                    user_question,
                    history_for_rewrite,
                    st.session_state.llm,
                )
        else:
            standalone_q = user_question

        if standalone_q.strip().lower() != user_question.strip().lower():
            st.caption(f"🔄 Interpreted as: *{standalone_q}*")

        st.markdown(user_question)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question,
        "sources": [],
        "rewritten": standalone_q,
    })

    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            try:
                answer_text, source_docs = answer(
                    st.session_state.faiss_index,
                    standalone_q,
                    st.session_state.llm,
                )
            except Exception as e:
                answer_text = f"⚠️ Could not get an answer: {e}"
                source_docs = []

        st.markdown(answer_text)

        if source_docs:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(source_docs, 1):
                    source_url = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Source {i}:** [{source_url}]({source_url})")
                    preview = doc.page_content[:300].replace("\n", " ")
                    st.caption(f"> {preview}{'...' if len(doc.page_content) > 300 else ''}")
                    if i < len(source_docs):
                        st.divider()

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text,
        "sources": source_docs,
        "rewritten": "",
    })
