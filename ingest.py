import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROFILE_DIR = Path("data/profile")
CHROMA_DIR = Path("data/profile_chroma")
COLLECTION_NAME = "profile_chunks"

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in resp.data]

def main() -> None:
    if not PROFILE_DIR.exists():
        raise SystemExit(f"Missing folder: {PROFILE_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    chroma = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    try:
        collection.delete(where={})
    except Exception:
        pass

    documents = []
    metadatas = []
    ids = []

    txt_files = list(PROFILE_DIR.glob("*.txt"))
    for file in txt_files:
        raw = file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(raw)

        for idx, chunk in enumerate(chunks):
            ids.append(f"{file.stem}::chunk{idx}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source": file.stem,
                    "chunk_index": idx,
                    "path": str(file),
                }
            )

    if not documents:
        raise SystemExit("No .txt content found in data/profile.")

    BATCH = 64
    for start in range(0, len(documents), BATCH):
        batch_docs = documents[start:start + BATCH]
        batch_ids = ids[start:start + BATCH]
        batch_metas = metadatas[start:start + BATCH]
        embeddings = embed(batch_docs)

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )

    print(f"Ingested {len(documents)} chunks from {len(txt_files)} files.")

if __name__ == "__main__":
    main()
