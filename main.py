import os
from fastapi import FastAPI
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path
from prompts import system_prompt
from logger import log_unanswered



load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

ME_DIR = Path("data/me")

class ChatRequest(BaseModel):
    message: str

def retrieve_me_context(query: str, k: int = 3) ->tuple[str, list[str]]:


    # Placeholder for actual retrieval logic
    # In a real implementation, this would involve embedding the query and retrieving relevant documents
    if not ME_DIR.exists():
        return "", []

    q_words = [w for w in query.lower().split() if len(w) > 2]

    scored: list[tuple[int, str, str]] = []
    for file in ME_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        score = sum(lower.count(w) for w in q_words)
        if score > 0:
            scored.append((score, file.stem, text))
    
    scored.sort(reverse=True)
    top = scored[:k]

    sources = [name for _, name, _ in top]
    context = "\n\n".join([f"#source: {name}\n{txt[:2000]}" for _, name, txt in top])
    return context, sources

@app.post("/chat")
def chat(req: ChatRequest):
    context, sources = retrieve_me_context(req.message)

    if not sources:
        log_unanswered(
            question= req.message,
            sources= sources,
            reason= "No relevant context found"
        )

    messages = [{"role": "system", "content": system_prompt}]

    if context: 
        messages.append({"role": "system", "content": f"Context:\n{context}"})

    messages.append({"role": "user", "content": req.message})


    response = client.responses.create(
        model="gpt-4.1-nano",
        input=messages,
    )

    return {
        "answer": response.output_text,
        "used_context": bool(context),
        "sources": sources,
        }





@app.get("/health")
def health_check():
    return {"ok": True}
