import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ingestion.web_loader import load_page, chunk_documents
from ingestion.vectorstore import build_index
from retrieval.rag import answer

load_dotenv()

index = build_index(chunk_documents(load_page("https://docs.python.org/3/tutorial/introduction.html")))

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

print("\n" + "="*50, flush=True)
text, sources = answer(index, "What is a list in Python?", llm)
print("\n[ANSWER 1]:", flush=True)
print(text, flush=True)
print("\n[SOURCES]:", flush=True)
print([d.metadata["source"] for d in sources], flush=True)

print("\n" + "="*50, flush=True)
text, _ = answer(index, "Who is the Prime Minister of India?", llm)
print("\n[ANSWER 2]:", flush=True)
print(text, flush=True)
sys.stdout.flush()
