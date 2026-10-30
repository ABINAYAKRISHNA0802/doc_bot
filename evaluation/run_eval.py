# இந்த கோப்பு: RAG pipeline-ஐ evaluate செய்து CSV report save பண்ணும்

import os
import sys
import csv
import warnings
from datetime import datetime

# Warnings suppress பண்றோம்
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Project root path-ல் சேர்க்கிறோம்
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ingestion.web_loader import load_page, chunk_documents
from ingestion.vectorstore import build_index
from retrieval.rag import answer, rewrite_question

load_dotenv()

# ─────────────────────────────────────────────
# Evaluate பண்ண வேண்டிய URL
# ─────────────────────────────────────────────

EVAL_URL = "https://docs.python.org/3/tutorial/introduction.html"

# ─────────────────────────────────────────────
# Test cases — (question, expected_keywords, category, prev_question)
#
# expected_keywords: இந்த words answer-ல் இருந்தால் "correct" என்று mark பண்ணும்
# category:
#   direct      — page-ல் clearly இருக்கும்
#   detailed    — specific fact தேட வேண்டும்
#   not_in_page — page-ல் இல்லாத info, "இல்லை" வரணும்
#   followup    — முந்தைய question-ஐ depend பண்ணும்
#   followup_base — follow-up-க்கு base question
#
# prev_question: follow-up question-க்கு முந்தைய question (history simulate பண்ண)
# ─────────────────────────────────────────────

TEST_CASES = [
    # ── Category 1: Direct questions (5) ────────────────────────────
    {
        "question": "What operator is used to calculate powers in Python?",
        "expected_keywords": ["**"],
        "category": "direct",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "What is the type of a number with a fractional part in Python?",
        "expected_keywords": ["float"],
        "category": "direct",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "How do you create a multi-line string in Python?",
        "expected_keywords": ['"""', "'''", "triple"],
        "category": "direct",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "What does the // operator do in Python?",
        "expected_keywords": ["floor", "integer", "fractional"],
        "category": "direct",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "What suffix is used to indicate the imaginary part of a complex number?",
        "expected_keywords": ["j", "J"],
        "category": "direct",
        "prev_question": None,
        "prev_answer": None,
    },

    # ── Category 2: Detailed questions (3) ──────────────────────────
    {
        "question": "What is the result of 17 divided by 3 using classic division in Python?",
        "expected_keywords": ["5.666", "5.67"],
        "category": "detailed",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "What happens when you use a variable that has not been defined in Python?",
        "expected_keywords": ["error", "NameError", "not defined", "undefined"],
        "category": "detailed",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "What types of numbers does Python support besides int and float?",
        "expected_keywords": ["Decimal", "Fraction", "complex"],
        "category": "detailed",
        "prev_question": None,
        "prev_answer": None,
    },

    # ── Category 3: "Illatha" questions (3) ─────────────────────────
    # இந்த questions-க்கு "இல்லை" அல்லது "not available" வரணும்
    {
        "question": "Who created the Python programming language?",
        "expected_keywords": ["இல்லை", "not available", "not found", "not in", "cannot", "don't"],
        "category": "not_in_page",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "What is the latest version of Python available for download?",
        "expected_keywords": ["இல்லை", "not available", "not found", "not in", "cannot", "don't"],
        "category": "not_in_page",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "How do you connect Python to a MySQL database?",
        "expected_keywords": ["இல்லை", "not available", "not found", "not in", "cannot", "don't"],
        "category": "not_in_page",
        "prev_question": None,
        "prev_answer": None,
    },

    # ── Category 4: Follow-up questions (2 pairs) ───────────────────
    # followup_base: தனியாக கேட்கும் - answer சேமிக்கப்படும்
    {
        "question": "What is a string in Python?",
        "expected_keywords": ["str", "text", "character", "sequence"],
        "category": "followup_base",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "How do you concatenate them?",  # "them" = strings
        "expected_keywords": ["+", "concatenat", "join", "combine"],
        "category": "followup",
        "prev_question": "What is a string in Python?",
        "prev_answer": None,  # run_eval-ல் dynamically fill ஆகும்
    },
    {
        "question": "What is a list in Python?",
        "expected_keywords": ["list", "square bracket", "[", "ordered", "collection", "sequence"],
        "category": "followup_base",
        "prev_question": None,
        "prev_answer": None,
    },
    {
        "question": "How do you access its elements?",  # "its" = list
        "expected_keywords": ["index", "[0]", "slice", "bracket", "position"],
        "category": "followup",
        "prev_question": "What is a list in Python?",
        "prev_answer": None,  # run_eval-ல் dynamically fill ஆகும்
    },
]


def check_answer_correct(actual: str, expected_keywords: list[str], category: str) -> bool:
    """
    Answer சரியா என்று check பண்றோம்.

    not_in_page category: "இல்லை" அல்லது "not available" type response வந்தால் correct.
    மற்றவை: expected_keywords-ல் ஏதாவது ஒன்று answer-ல் இருந்தால் correct.
    """
    actual_lower = actual.lower()

    if category == "not_in_page":
        # எந்த keyword-உம் match ஆனால் correct — LLM "not available" சொல்லியிருக்கும்
        return any(kw.lower() in actual_lower for kw in expected_keywords)
    else:
        # குறைந்தபட்சம் ஒரு keyword match ஆனால் correct
        return any(kw.lower() in actual_lower for kw in expected_keywords)


def check_source_correct(sources: list, expected_url: str) -> bool:
    """
    Retrieval source சரியான URL-இலிருந்து வந்ததா என்று check பண்றோம்.
    Trailing slash differences handle பண்ண rstrip use பண்றோம்.
    """
    if not sources:
        return False
    expected_clean = expected_url.rstrip("/")
    return any(
        doc.metadata.get("source", "").rstrip("/") == expected_clean
        for doc in sources
    )


def run_evaluation():
    """
    எல்லா test cases-ஐயும் run பண்ணி CSV-ல் save பண்ணும்.
    """

    print("=" * 60)
    print("RAG Evaluation Starting...")
    print("=" * 60)

    # ── Step 1: Pipeline setup ───────────────────────────────────────
    print(f"\n[1/3] URL load + index build: {EVAL_URL}")
    docs = load_page(EVAL_URL)
    chunks = chunk_documents(docs)
    index = build_index(chunks)
    print(f"      {len(chunks)} chunks indexed.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY இல்லை. .env சரி பார்க்கவும்.")
        sys.exit(1)

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
    )
    print("      LLM ready.")

    # ── Step 2: Test cases run ───────────────────────────────────────
    print(f"\n[2/3] {len(TEST_CASES)} test cases run பண்றோம்...\n")

    results = []

    # followup_base answers-ஐ இங்கே store பண்றோம் — follow-up history-க்கு பயன்படுத்துவோம்
    base_answers: dict[str, str] = {}

    for i, tc in enumerate(TEST_CASES, 1):
        question    = tc["question"]
        expected_kw = tc["expected_keywords"]
        category    = tc["category"]
        prev_q      = tc["prev_question"]

        print(f"  [{i:02d}/{len(TEST_CASES)}] {category.upper()}: {question[:60]}...")

        # ── Chat history simulate பண்றோம் (follow-up cases-க்கு) ──
        history = []
        if prev_q and prev_q in base_answers:
            # முந்தைய question + answer-ஐ history-ஆக கொடுக்கிறோம்
            history = [
                {"role": "user",      "content": prev_q},
                {"role": "assistant", "content": base_answers[prev_q]},
            ]

        # ── Question rewrite (history இருந்தால்) ──
        if history:
            standalone_q = rewrite_question(question, history, llm)
        else:
            standalone_q = question

        # ── Answer generate ──────────────────────
        try:
            actual_answer, sources = answer(index, standalone_q, llm)
        except Exception as e:
            actual_answer = f"ERROR: {e}"
            sources = []

        # followup_base answer சேமிக்கிறோம் — அடுத்த follow-up-க்கு history ஆகும்
        if category == "followup_base":
            base_answers[question] = actual_answer

        # ── Correctness check ─────────────────────
        ans_correct = check_answer_correct(actual_answer, expected_kw, category)
        src_correct = check_source_correct(sources, EVAL_URL)

        # Source URL list (display-க்கு)
        source_urls = "; ".join(
            doc.metadata.get("source", "?") for doc in sources
        ) if sources else "—"

        results.append({
            "question":        question,
            "category":        category,
            "rewritten_q":     standalone_q if standalone_q != question else "—",
            "expected":        ", ".join(expected_kw),
            "actual":          actual_answer.replace("\n", " ").strip(),
            "answer_correct":  "Y" if ans_correct else "N",
            "source_correct":  "Y" if src_correct else "N",
            "sources":         source_urls,
        })

        status = "✅" if ans_correct else "❌"
        print(f"         Answer: {status}  |  Source: {'✅' if src_correct else '❌'}")

    # ── Step 3: CSV save ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, f"eval_results_{timestamp}.csv")

    fieldnames = [
        "question", "category", "rewritten_q",
        "expected", "actual",
        "answer_correct", "source_correct", "sources",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        # utf-8-sig: Excel-ல் Tamil/Unicode சரியாக திறக்கும்
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[3/3] CSV saved: {csv_path}")

    # ── Accuracy summary print ───────────────────────────────────────
    total         = len(results)
    ans_correct   = sum(1 for r in results if r["answer_correct"] == "Y")
    src_correct   = sum(1 for r in results if r["source_correct"] == "Y")

    # Category-wise breakdown
    categories = ["direct", "detailed", "not_in_page", "followup_base", "followup"]
    cat_stats: dict[str, dict] = {c: {"total": 0, "correct": 0} for c in categories}
    for r in results:
        c = r["category"]
        if c in cat_stats:
            cat_stats[c]["total"] += 1
            if r["answer_correct"] == "Y":
                cat_stats[c]["correct"] += 1

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total questions   : {total}")
    print(f"  Answer accuracy   : {ans_correct}/{total} = {ans_correct/total*100:.1f}%")
    print(f"  Source accuracy   : {src_correct}/{total} = {src_correct/total*100:.1f}%")
    print()
    print("  Category breakdown:")
    for cat, stat in cat_stats.items():
        if stat["total"] > 0:
            pct = stat["correct"] / stat["total"] * 100
            bar = "█" * stat["correct"] + "░" * (stat["total"] - stat["correct"])
            print(f"    {cat:<15} [{bar}] {stat['correct']}/{stat['total']} ({pct:.0f}%)")
    print("=" * 60)
    print(f"\nDetailed results: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
