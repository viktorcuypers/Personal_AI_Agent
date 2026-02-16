import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from prompts import system_prompt
from logger import log_unanswered

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

CHROMA_DIR = Path("data/profile_chroma")
COLLECTION_NAME = "profile_chunks"

chroma = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

class ChatRequest(BaseModel):
    message: str

def retrieve_profile_context(query: str, k: int = 5) -> tuple[str, list[str], float | None]:
    if not CHROMA_DIR.exists():
        return "", [], None

    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    ).data[0].embedding

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]  # lower = better

    if not docs:
        return "", [], None

    best_dist = dists[0]

    parts = []
    sources = []
    for doc, meta, dist in zip(docs, metas, dists):
        src = meta.get("source", "unknown")
        sources.append(src)
        parts.append(f"#source: {src} (chunk={meta.get('chunk_index')}, dist={dist:.3f})\n{doc}")

    seen = set()
    sources = [s for s in sources if not (s in seen or seen.add(s))]

    return "\n\n---\n\n".join(parts), sources, best_dist



RELEVANCE_THRESHOLD = 0.35  

@app.post("/chat")
def chat(req: ChatRequest):
    context, sources, best_dist = retrieve_profile_context(req.message)

    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": req.message})

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=messages,
    )

    answer_text = response.output_text or ""


    if best_dist is None or best_dist > RELEVANCE_THRESHOLD:
        log_unanswered(
            question=req.message,
            answer=answer_text,
            sources=sources,
            reason=f"No relevant context (best_dist={best_dist}, threshold={RELEVANCE_THRESHOLD})",
        )

    return {
        "answer": answer_text,
        "used_context": bool(context) and (best_dist is not None and best_dist <= RELEVANCE_THRESHOLD),
        "sources": sources,
    }



@app.get("/health")
def health_check():
    return {"ok": True}
