# இந்த கோப்பு: FAISS retrieval + question rewriting + answer generation

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────
# Prompt 1: Question Rewriter
# Chat history-ஐ பார்த்து follow-up question-ஐ standalone-ஆக மாற்றும்
# ─────────────────────────────────────────────

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a question rewriter. Given a conversation history and a follow-up question,
rewrite the follow-up question to be fully self-contained and standalone.
Replace all pronouns (it, that, this, adhu, idhu, avai, etc.) with the actual subject from history.
If the question is already standalone, return it as-is without any changes.
Return ONLY the rewritten question. No explanation, no prefix, no quotes.""",
    ),
    (
        "human",
        """Conversation history:
{history}

Follow-up question: {question}

Standalone question:""",
    ),
])

# ─────────────────────────────────────────────
# Prompt 2: RAG Answer
# Context-ஐ மட்டுமே பயன்படுத்தி பதில் சொல்லும்
# ─────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, respond with exactly:
"இந்த website-ல் அந்த தகவல் இல்லை. (This information is not available on the website.)"
Do not use any outside knowledge.

Context:
{context}""",
    ),
    ("human", "{question}"),
])

# Similarity search-ல் எத்தனை chunks எடுக்கணும்
TOP_K = 4

# Chat history-இல் எத்தனை messages rewrite-க்கு பயன்படுத்துவது
HISTORY_WINDOW = 6


def rewrite_question(
    question: str,
    chat_history: list[dict],
    llm: BaseLanguageModel,
) -> str:
    """
    Follow-up question-ஐ chat history-ஐ பார்த்து standalone question-ஆக rewrite பண்ணும்.

    Args:
        question: பயனர் தற்போது கேட்கும் கேள்வி (pronoun இருக்கலாம்)
        chat_history: [{"role": "user"/"assistant", "content": "..."}] format
        llm: ChatGroq LLM object

    Returns:
        Rewritten standalone question string
    """

    # History இல்லாட்டா rewrite தேவையில்லை - அப்படியே return பண்றோம்
    if not chat_history:
        return question

    # Last HISTORY_WINDOW messages மட்டும் எடுக்கிறோம் - context window சிக்கனம்
    recent = chat_history[-HISTORY_WINDOW:]

    # History-ஐ readable format-ல் string-ஆக மாற்றுகிறோம்
    history_str = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in recent
    )

    # LLM-கிட்ட rewrite கேட்கிறோம்
    prompt = REWRITE_PROMPT.invoke({
        "history": history_str,
        "question": question,
    })
    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    # Empty response வந்தால் original question-ஐ திரும்ப கொடுக்கிறோம்
    return rewritten if rewritten else question


def answer(
    index: FAISS,
    question: str,
    llm: BaseLanguageModel,
) -> tuple[str, list[Document]]:
    """
    Standalone question-ஐ FAISS index-ல் தேடி LLM மூலம் பதில் தரும்.
    (rewrite_question() call பண்ணிய பிறகு இதை call பண்ணணும்)

    Args:
        index: build_index() திரும்ப கொடுத்த FAISS vectorstore
        question: rewrite_question() திரும்ப கொடுத்த standalone question
        llm: ChatGroq LLM object

    Returns:
        (answer_text, source_documents) tuple
    """

    # FAISS-ல் semantic similarity search
    relevant_docs = index.similarity_search(question, k=TOP_K)

    # எல்லா chunks-ஐயும் ஒரே context string-ஆக இணைக்கிறோம்
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # RAG prompt உருவாக்கி LLM-க்கு அனுப்புகிறோம்
    prompt = RAG_PROMPT.invoke({"context": context, "question": question})
    response = llm.invoke(prompt)

    return response.content, relevant_docs
