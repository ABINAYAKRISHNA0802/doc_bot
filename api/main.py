# இந்த கோப்பு: RAG pipeline-ஐ FastAPI REST API-ஆக expose பண்ணும்
#
# LIMITATION (முக்கியம்):
#   Index-ஐ Python process memory-ல் (in-memory dict) store பண்றோம்.
#   - Server restart ஆனால் எல்லா indexes போய்விடும்
#   - Multiple workers (gunicorn) use பண்ணினால் workers-இடையே share ஆகாது
#   - Production-ல் FAISS index-ஐ disk-ல் save பண்ணி (FAISS.save_local)
#     அல்லது Redis / persistent volume use பண்ணணும்

import os
import sys
import warnings
from contextlib import asynccontextmanager

# Warnings suppress பண்றோம்
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Project root import path-ல் சேர்க்கிறோம்
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from ingestion.web_loader import load_page, chunk_documents
from ingestion.vectorstore import build_index
from retrieval.rag import answer, rewrite_question

load_dotenv()

# ─────────────────────────────────────────────
# Global state
# LIMITATION: in-memory store — single process மட்டுமே valid
# ─────────────────────────────────────────────

# { url_string: FAISS index } — ஒவ்வொரு indexed URL-க்கும் ஒரு entry
index_store: dict[str, FAISS] = {}

# LLM singleton — ஒரே ஒரு முறை initialize பண்றோம்
_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """
    LLM object-ஐ lazy initialize பண்ணி return பண்றோம்.
    GROQ_API_KEY இல்லாட்டா startup-லேயே error தெரியும்.
    """
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable இல்லை.")
        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            temperature=0,
        )
    return _llm


# ─────────────────────────────────────────────
# Lifespan: startup-ல் LLM warm-up பண்றோம்
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server start ஆகும்போது LLM initialize பண்றோம் — first request slow ஆகாமல் இருக்க."""
    try:
        get_llm()
        print("[INFO] LLM initialized successfully.")
    except RuntimeError as e:
        print(f"[WARNING] {e}")
    yield
    # Shutdown: cleanup தேவையில்லை — in-memory store auto-clear ஆகும்


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Web RAG API",
    description="URL-ஐ index பண்ணி questions-க்கு answers தரும் RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Pydantic models — Request / Response schemas
# ─────────────────────────────────────────────

class IndexRequest(BaseModel):
    """POST /index request body"""
    url: HttpUrl  # Valid URL மட்டுமே accept பண்ணும் — Pydantic validate பண்ணும்


class IndexResponse(BaseModel):
    """POST /index response"""
    message: str
    url: str
    chunks_indexed: int


class AskRequest(BaseModel):
    """POST /ask request body"""
    url: HttpUrl                        # எந்த index-ஐ use பண்ணணும்
    question: str                       # பயனர் கேட்கும் கேள்வி
    chat_history: list[dict] = []       # Follow-up support-க்கு optional history
                                        # Format: [{"role": "user"/"assistant", "content": "..."}]


class SourceDoc(BaseModel):
    """ஒரு source chunk-இன் details"""
    source_url: str
    preview: str    # முதல் 300 characters


class AskResponse(BaseModel):
    """POST /ask response"""
    question: str           # Original question
    rewritten_question: str # Rewrite ஆனது (same-ஆ இருந்தால் original)
    answer: str
    sources: list[SourceDoc]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", summary="Health check")
async def root():
    """API running-ஆ என்று confirm பண்ண."""
    return {"status": "ok", "message": "Web RAG API is running."}


@app.get("/indexes", summary="Indexed URL list")
async def list_indexes():
    """
    தற்போது memory-ல் இருக்கும் indexed URLs-ஐ list பண்ணும்.
    LIMITATION: server restart-ல் இந்த list காலியாகும்.
    """
    return {
        "indexed_urls": list(index_store.keys()),
        "count": len(index_store),
    }


@app.post("/index", response_model=IndexResponse, summary="URL index பண்ணு")
async def index_url(req: IndexRequest):
    """
    கொடுக்கப்பட்ட URL-ஐ load செய்து FAISS index உருவாக்கும்.
    Already indexed URL-ஐ மறுபடியும் POST பண்ணினால் re-index ஆகும்.

    LIMITATION: Index process sync-ஆக நடக்கும் — large pages-க்கு timeout வரலாம்.
    Production-ல் background task (Celery / FastAPI BackgroundTasks) use பண்ணணும்.
    """
    url_str = str(req.url)

    try:
        # Web page load பண்றோம்
        docs = load_page(url_str)

        if not docs or not docs[0].page_content.strip():
            raise HTTPException(
                status_code=422,
                detail=f"'{url_str}' page-ல் readable text இல்லை."
            )

        # Chunk + embed + index
        chunks = chunk_documents(docs)
        faiss_index = build_index(chunks)

        # Memory-ல் store பண்றோம்
        index_store[url_str] = faiss_index

        return IndexResponse(
            message="Index successfully created.",
            url=url_str,
            chunks_indexed=len(chunks),
        )

    except HTTPException:
        raise  # Already formatted error-ஐ re-raise பண்றோம்

    except ValueError as e:
        # HTTP 404 / 403 errors
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        err = str(e)
        if "timed out" in err.lower() or "Timeout" in err:
            raise HTTPException(status_code=504, detail=f"Timeout: '{url_str}' respond பண்ணல.")
        raise HTTPException(status_code=500, detail=f"Index build பிரச்சனை: {err}")


@app.post("/ask", response_model=AskResponse, summary="கேள்வி கேளு")
async def ask_question(req: AskRequest):
    """
    Indexed URL-க்கு question கேட்டு answer + sources return பண்ணும்.

    Follow-up support:
      chat_history-ல் முந்தைய messages கொடுத்தால் pronoun rewriting நடக்கும்.
      Example: [{"role":"user","content":"What is a list?"},
                {"role":"assistant","content":"A list is..."}]

    LIMITATION: Chat history server-ல் store ஆவதில்லை — client-இடமிருந்து
    ஒவ்வொரு request-லும் history அனுப்பணும் (stateless design).
    """
    url_str = str(req.url)

    # Index இருக்கா என்று check பண்றோம்
    if url_str not in index_store:
        raise HTTPException(
            status_code=404,
            detail=f"'{url_str}' index ஆகவில்லை. முதலில் POST /index call பண்ணுங்க."
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question காலியாக இருக்கு.")

    try:
        llm = get_llm()
        faiss_index = index_store[url_str]

        # Follow-up question rewrite — history இருந்தால் மட்டும்
        if req.chat_history:
            standalone_q = rewrite_question(req.question, req.chat_history, llm)
        else:
            standalone_q = req.question

        # Retrieval + Answer generation
        answer_text, source_docs = answer(faiss_index, standalone_q, llm)

        # Source documents-ஐ response format-ல் மாற்றுகிறோம்
        sources = [
            SourceDoc(
                source_url=doc.metadata.get("source", "unknown"),
                preview=doc.page_content[:300].replace("\n", " "),
            )
            for doc in source_docs
        ]

        return AskResponse(
            question=req.question,
            rewritten_question=standalone_q,
            answer=answer_text,
            sources=sources,
        )

    except HTTPException:
        raise

    except RuntimeError as e:
        # GROQ_API_KEY missing போன்ற config errors
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation பிரச்சனை: {str(e)}")


# ─────────────────────────────────────────────
# Local dev run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # reload=True: code மாறும்போது auto-restart — dev மட்டும்
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
