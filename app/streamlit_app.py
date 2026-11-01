# இந்த கோப்பு: RAG chatbot Streamlit UI — follow-up question support உடன்

import os
import sys
import warnings

# Warnings suppress பண்றோம்
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Project root-ஐ Python path-ல் சேர்க்கிறோம்
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
st.caption("எந்த webpage-ஐயும் load பண்ணி கேள்வி கேளுங்க!")

# ─────────────────────────────────────────────
# Session state initialize
# ─────────────────────────────────────────────

if "chat_history" not in st.session_state:
    # ஒவ்வொரு message-உம்: {"role", "content", "sources", "rewritten"}
    # rewritten: LLM rewrite பண்ணின question (UI-ல் caption-ஆக காட்ட)
    st.session_state.chat_history = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None

if "llm" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        # ஒரே LLM object - rewrite + answer இரண்டுக்கும் share பண்றோம்
        st.session_state.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            temperature=0,
        )
    else:
        st.session_state.llm = None

# ─────────────────────────────────────────────
# Sidebar: URL input
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Webpage Load பண்ணுங்க")

    # ── HuggingFace Token (optional) ──────────
    st.subheader("🤗 HuggingFace Login")
    hf_token_default = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    hf_token_input = st.text_input(
        "HuggingFace Token (optional)",
        value=hf_token_default,
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Gated models அல்லது rate limit பிரச்சனை இருந்தால் உங்கள் HF token போடுங்க. huggingface.co/settings/tokens-ல் பெறலாம்.",
    )
    if hf_token_input.strip():
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token_input.strip()
        st.success("✅ HF Token set!")
    else:
        st.caption("Token இல்லாமலும் public models வேலை செய்யும்.")

    st.divider()

    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com/page",
        help="Full URL போடுங்க (https:// தொடங்கணும்)",
    )

    load_btn = st.button("🚀 Load & Index", use_container_width=True)

    if load_btn:
        # ── Validation ──────────────────────────
        if not url_input.strip():
            st.error("⚠️ URL காலியாக இருக்கு! ஒரு URL போடுங்க.")

        elif not (url_input.startswith("http://") or url_input.startswith("https://")):
            st.error("⚠️ Valid URL இல்லை. `https://` தொடங்கணும்.")

        elif st.session_state.llm is None:
            st.error("⚠️ GROQ_API_KEY .env-ல் இல்லை.")

        else:
            # ── Load → Chunk → Index ─────────────
            try:
                with st.spinner("🌐 Webpage load ஆகிறது..."):
                    docs = load_page(url_input.strip())

                if not docs or not docs[0].page_content.strip():
                    st.error("⚠️ இந்த page-ல் readable text இல்லை. வேற URL try பண்ணுங்க.")
                else:
                    with st.spinner("✂️ Text chunk பண்றோம்..."):
                        chunks = chunk_documents(docs)

                    with st.spinner(f"🧠 {len(chunks)} chunks embed பண்றோம்..."):
                        st.session_state.faiss_index = build_index(chunks)

                    # புதிய URL load ஆனதும் chat history reset பண்றோம்
                    st.session_state.chat_history = []
                    st.session_state.loaded_url = url_input.strip()
                    st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                err_msg = str(e)
                if "Timeout" in err_msg or "timed out" in err_msg.lower():
                    st.error("⚠️ Timeout! Page respond பண்ணல.")
                elif "getaddrinfo" in err_msg:
                    st.error("⚠️ URL கிடைக்கவில்லை. Internet / URL சரி பார்க்கவும்.")
                else:
                    st.error(f"⚠️ ஏதோ பிரச்சனை: {err_msg}")

    # தற்போது load ஆன URL காட்டுகிறோம்
    if st.session_state.loaded_url:
        st.divider()
        st.markdown("**📌 Loaded:**")
        url = st.session_state.loaded_url
        display = f"{url[:50]}..." if len(url) > 50 else url
        st.markdown(f"[{display}]({url})")

    # Chat clear button
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Chat Clear பண்ணு", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────
# Index இல்லாதபோது welcome screen
# ─────────────────────────────────────────────

if st.session_state.faiss_index is None:
    st.info(
        "👈 **Sidebar-ல் ஒரு URL போட்டு 'Load & Index' click பண்ணுங்க.**\n\n"
        "Load ஆனதும் இங்கே கேள்வி கேட்கலாம்!"
    )
    st.stop()

# ─────────────────────────────────────────────
# Chat history render
# ─────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):

        # Follow-up question rewrite ஆகியிருந்தால் caption-ஆக காட்டுகிறோம்
        # User message-க்கு மட்டும் - rewritten != original question-ஆக இருந்தால்
        if (
            msg["role"] == "user"
            and msg.get("rewritten")
            and msg["rewritten"].strip().lower() != msg["content"].strip().lower()
        ):
            st.caption(f"🔄 Interpreted as: *{msg['rewritten']}*")

        st.markdown(msg["content"])

        # Assistant message-க்கு sources expander
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources பார்க்க"):
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

user_question = st.chat_input("இந்த page பத்தி கேள்வி கேளுங்க...")

if user_question:

    # ── Step 1: User message உடனே காட்டுகிறோம் ──
    with st.chat_message("user"):

        # Rewrite பண்ணுவதற்கு முன் — history-ல் user+assistant messages மட்டும் கொடுக்கிறோம்
        # sources, rewritten fields retrieval-க்கு தேவையில்லை
        history_for_rewrite = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]

        # ── Step 2: Question rewrite ──────────────
        # History இருந்தால் மட்டும் rewrite பண்றோம் - முதல் question-க்கு தேவையில்லை
        if history_for_rewrite:
            with st.spinner("🔄 Question interpret பண்றோம்..."):
                standalone_q = rewrite_question(
                    user_question,
                    history_for_rewrite,
                    st.session_state.llm,
                )
        else:
            # முதல் கேள்வி - rewrite தேவையில்லை
            standalone_q = user_question

        # Rewrite ஆகியிருந்தால் caption காட்டுகிறோம்
        if standalone_q.strip().lower() != user_question.strip().lower():
            st.caption(f"🔄 Interpreted as: *{standalone_q}*")

        st.markdown(user_question)

    # User message session state-ல் சேமிக்கிறோம் (rewritten question-உடன்)
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question,
        "sources": [],
        "rewritten": standalone_q,  # Transparency-க்காக store பண்றோம்
    })

    # ── Step 3: Retrieval + Answer ────────────
    with st.chat_message("assistant"):
        with st.spinner("💭 யோசிக்கிறேன்..."):
            try:
                # Rewritten standalone question-ஐ retrieval-க்கு பயன்படுத்துகிறோம்
                answer_text, source_docs = answer(
                    st.session_state.faiss_index,
                    standalone_q,
                    st.session_state.llm,
                )
            except Exception as e:
                answer_text = f"⚠️ பதில் கிடைக்கவில்லை: {e}"
                source_docs = []

        st.markdown(answer_text)

        # Sources expander
        if source_docs:
            with st.expander("📚 Sources பார்க்க"):
                for i, doc in enumerate(source_docs, 1):
                    source_url = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Source {i}:** [{source_url}]({source_url})")
                    preview = doc.page_content[:300].replace("\n", " ")
                    st.caption(f"> {preview}{'...' if len(doc.page_content) > 300 else ''}")
                    if i < len(source_docs):
                        st.divider()

    # Assistant message session state-ல் சேமிக்கிறோம்
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text,
        "sources": source_docs,
        "rewritten": "",
    })
