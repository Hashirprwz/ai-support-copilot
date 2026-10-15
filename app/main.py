from fastapi import FastAPI, Query
from app.search import search_documents, load_documents

app = FastAPI(title="AI Support Copilot")

@app.on_event("startup")
def startup_event():
    # Load documents when the server starts
    load_documents()

@app.get("/")
def read_root():
    return {"status": "AI Support Copilot is alive"}

@app.get("/search")
def search(q: str = Query(..., description="The search query")):
    """Semantic search endpoint."""
    results = search_documents(query=q, top_k=1)
    return {"query": q, "results": results}
