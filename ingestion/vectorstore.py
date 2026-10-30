# இந்த கோப்பு: Document chunks-ஐ embeddings ஆக மாற்றி FAISS index-ல் சேமிக்கும்

import os
import warnings

# HuggingFace symlinks warning-ஐ suppress பண்றோம் - Windows Developer Mode இல்லாதபோது வரும்
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# LangChain deprecation warnings suppress பண்றோம்
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# langchain-huggingface install ஆகியிருந்தால் அதை use பண்ணும், இல்லாட்டா community version
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


# Embedding model - sentence-transformers இலிருந்து lightweight model பயன்படுத்துகிறோம்
# முதல் முறை download ஆகும், பிறகு cache ஆகும்
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_index(chunks: list[Document]) -> FAISS:
    """
    Document chunks-ஐ எடுத்து FAISS vector index உருவாக்கும்.

    Args:
        chunks: chunk_documents() திரும்ப கொடுத்த Document list

    Returns:
        தேடலுக்கு தயாரான FAISS vectorstore object
    """

    if not chunks:
        raise ValueError("[ERROR] Chunks empty - index உருவாக்க முடியாது.")

    print(f"[INFO] Embedding model load ஆகிறது: {EMBEDDING_MODEL}")

    # HuggingFace sentence-transformers model - local-ஆக run ஆகும், API தேவையில்லை
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"[INFO] {len(chunks)} chunks-ஐ FAISS index-ல் சேர்க்கிறோம்...")

    # Chunks-ஐ embed செய்து FAISS index உருவாக்குகிறோம்
    index = FAISS.from_documents(chunks, embeddings)

    print(f"[INFO] FAISS index தயார். {len(chunks)} vectors சேமிக்கப்பட்டன.")
    return index
