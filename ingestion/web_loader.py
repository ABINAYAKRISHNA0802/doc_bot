# இந்த கோப்பு: Web URL-இலிருந்து text extract செய்து LangChain Documents ஆக மாற்றும்

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# தேவையில்லாத HTML tags - இவற்றை நீக்கிவிடுவோம்
REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "noscript"]


def load_page(url: str) -> list[Document]:
    """load_web_page-க்கு alias - main.py compatibility-க்காக"""
    return load_web_page(url)


def load_web_page(url: str) -> list[Document]:
    """
    கொடுக்கப்பட்ட URL-ஐ load செய்து, clean text-ஐ
    LangChain Document list-ஆக திரும்ப கொடுக்கும்.

    Args:
        url: படிக்க வேண்டிய web page-இன் முழு URL

    Returns:
        LangChain Document objects-இன் list (metadata-ல் source URL இருக்கும்)

    Raises:
        ValueError: 404 அல்லது வேறு HTTP error வந்தால்
        requests.exceptions.Timeout: connection timeout ஆனால்
        requests.exceptions.RequestException: பிற network பிரச்சனைகள்
    """

    print(f"[INFO] URL load ஆகிறது: {url}")

    # Browser போல் தெரிய User-Agent header சேர்க்கிறோம் - bot-blocking தவிர்க்க
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        # 10 seconds timeout வைக்கிறோம் - மிகவும் slow sites-க்கு உதவும்
        response = requests.get(url, headers=headers, timeout=10)

        # 404, 403, 500 போன்ற HTTP error codes-ஐ இங்கே பிடிக்கிறோம்
        response.raise_for_status()

    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(
            f"[ERROR] Timeout: '{url}' 10 seconds-க்குள் respond செய்யவில்லை."
        )
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "?"
        raise ValueError(
            f"[ERROR] HTTP {status_code} error: '{url}' தவறான response கொடுத்தது. ({e})"
        )
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"[ERROR] Network பிரச்சனை: '{url}' load ஆகவில்லை. ({e})"
        )

    # HTML-ஐ parse செய்கிறோம்
    soup = BeautifulSoup(response.text, "html.parser")

    # தேவையில்லாத tags-ஐ DOM-இலிருந்து முழுவதும் நீக்குகிறோம்
    for tag in REMOVE_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    # மீதமுள்ள clean text-ஐ எடுக்கிறோம், extra whitespace நீக்குகிறோம்
    raw_text = soup.get_text(separator="\n")
    clean_text = "\n".join(
        line.strip() for line in raw_text.splitlines() if line.strip()
    )

    if not clean_text:
        print(f"[WARNING] '{url}' page-இல் எந்த text-உம் கிடைக்கவில்லை.")
        return []

    # LangChain Document object உருவாக்குகிறோம்
    # metadata-ல் "source" key-ல் URL சேமிக்கிறோம்
    document = Document(
        page_content=clean_text,
        metadata={"source": url},
    )

    print(f"[INFO] Text extract வெற்றிகரமாக முடிந்தது. Characters: {len(clean_text)}")
    return [document]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Document list-ஐ சிறிய chunks ஆக பிரிக்கும்.
    RAG-க்கு ஏற்ற size-ல் split செய்கிறோம்.

    Args:
        documents: load_web_page() திரும்ப கொடுத்த Document list

    Returns:
        Split ஆன சிறிய Document chunks-இன் list
    """

    # chunk_size=500: ஒவ்வொரு chunk-லும் அதிகபட்சம் 500 characters
    # chunk_overlap=50: அடுத்த chunk-உடன் 50 characters overlap - context தொடர்ச்சி கிடைக்கும்
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)
    print(f"[INFO] {len(documents)} document(s) → {len(chunks)} chunks ஆக பிரிக்கப்பட்டது.")
    return chunks


# நேரடியாக இந்த script-ஐ run செய்தால் சோதனை நடக்கும்
if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

    # Step 1: URL load செய்து documents எடுக்கிறோம்
    docs = load_web_page(test_url)

    if docs:
        print(f"\n--- Document Preview (முதல் 300 characters) ---")
        print(docs[0].page_content[:300])
        print(f"\nMetadata: {docs[0].metadata}")

        # Step 2: Documents-ஐ chunks ஆக பிரிக்கிறோம்
        chunks = chunk_documents(docs)
        print(f"\n--- முதல் Chunk ---")
        print(chunks[0].page_content)
        print(f"Chunk Metadata: {chunks[0].metadata}")
