import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a tiny, free, local embedding model
# This model runs on your CPU and requires no API key or credits
print("Loading local embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
print("Model loaded.")

# This list will hold our documents and their embeddings
DOCUMENTS = []

def get_embedding(text: str) -> list[float]:
    """Turns text into a vector using a local model."""
    # model.encode() returns a numpy array, we convert it to a list
    embedding = model.encode(text)
    return embedding.tolist()

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    # Dot product divided by the product of their magnitudes
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_documents():
    """Reads text files, chunks them, and creates embeddings for them."""
    docs_folder = "data/docs"
    
    if not os.path.exists(docs_folder):
        print(f"Error: {docs_folder} folder not found.")
        return

    for filename in os.listdir(docs_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            print(f"Embedding {filename}...")
            embedding = get_embedding(text)
            
            DOCUMENTS.append({
                "filename": filename,
                "text": text,
                "embedding": embedding
            })
    
    print(f"Successfully loaded {len(DOCUMENTS)} documents.")

def search_documents(query: str, top_k: int = 1) -> list[dict]:
    """Searches documents by calculating cosine similarity."""
    if not DOCUMENTS:
        load_documents()
        
    # 1. Turn the user's query into an embedding
    query_embedding = get_embedding(query)
    
    # 2. Calculate similarity against every document
    results = []
    for doc in DOCUMENTS:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        results.append({
            "filename": doc["filename"],
            "text": doc["text"],
            "score": similarity
        })
    
    # 3. Sort by highest similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # 4. Return the top_k results
    return results[:top_k]
