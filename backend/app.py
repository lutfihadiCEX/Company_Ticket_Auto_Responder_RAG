from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from classifier import classify_ticket
from retriever import retrieve_documents
from reply_generator import generate_reply

app = FastAPI()

class EmailRequest(BaseModel):

    subject: str
    body: str
    sender: str

class EmailResponse(BaseModel):
    reply: str
    category: str
    retrieved_docs: List[str] = []

@app.get("/health")
def health_check():
    """For health check endpoint."""

    return {"status": "ok"}

@app.post("/process_email", response_model=EmailResponse)
def process_email(req: EmailRequest):
    """
    Full processing pipeline:
    1. Classify ticket
    2. Retrieve relevant KB docs
    3. Generate professional reply
    """

    
    category = classify_ticket(req.subject, req.body)

    
    kb_docs = retrieve_documents(req.body, top_k=3)

    
    reply = generate_reply(category, req.body, kb_docs)

    
    # log_ticket(req.sender, req.subject, category, kb_docs, reply)

    return EmailResponse(reply=reply, category=category, retrieved_docs=kb_docs)