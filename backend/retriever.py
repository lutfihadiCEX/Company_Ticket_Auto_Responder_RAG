import os
from chromadb import Client
from chromadb.config import Settings
from ollama_client import ollama_client

kb_folder = r"C:\MLCourse\ticket-auto-responder\kb"

if not os.path.exists(kb_folder):
    raise FileNotFoundError(f"KB folder not found at: {kb_folder}")
                            
documents = []

for filename in os.listdir(kb_folder):
    if filename.endswith(".txt"):
        path = os.path.join(kb_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            documents.append({"id": filename, "text": text})

def embed_text(text: str):
    """
    Input: raw text
    Output: embedding vector (floats)
    """
    embedding_resp = ollama_client.embeddings("mxbai-embed-large", text)
    
    
    print("Ollama response object:", embedding_resp)
    
    
    try:
        vector = embedding_resp.embedding  # first try this
    except AttributeError:
        
        vector = embedding_resp.data[0].embedding
    
    if not isinstance(vector, list):
        raise ValueError("Embedding not in expected list format.")
    
    return vector


chroma_client = Client(Settings(
    persist_directory="../data/chroma",
    anonymized_telemetry=False
))

collection = chroma_client.get_or_create_collection("kb_docs")

all_docs = collection.get(include=["documents", "metadatas"])
existing_ids = set(meta.get("id") for meta in all_docs["metadatas"])

for doc in documents:
    if doc["id"] in existing_ids:
        continue
        
    embedding_vector = embed_text(doc["text"])
    collection.add(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[embedding_vector]
    )

def retrieve_documents(query: str, top_k: int = 3) -> list:
    """
    Input: text from email
    Output: list of top k relevant KB documents
    """
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]  # list of text

