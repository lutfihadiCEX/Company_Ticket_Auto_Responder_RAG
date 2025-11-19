from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from classifier import classify_ticket
from retriever import retrieve_documents
from reply_generator import generate_reply
from logging_config import logger
from logging_utils import log_ticket


app = FastAPI()

class EmailRequest(BaseModel):

    subject: str
    body: str
    sender: str

class EmailResponse(BaseModel):
    reply: str
    category: str
    confidence: float = None
    retrieved_docs: List[dict] = []

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

    try:
        category, classifier_conf = classify_ticket(req.subject, req.body)
        logger.info(f"Category: {category}, Classifier Confidence: {classifier_conf:.2f}")
        
        kb_docs = retrieve_documents(req.body, top_k=3)
        retrieval_conf = max([doc.get("similarity", 0) for doc in kb_docs], default=0)
        logger.info(f"Retrieved {len(kb_docs)} KB docs, Max Retrieval Confidence: {retrieval_conf:.2f}")

        overall_confidence = classifier_conf * retrieval_conf
        logger.info(f"Overall Confidence: {overall_confidence:.2f}")

        reply = generate_reply(category, req.body, kb_docs)
        logger.info(f"Generated reply: {reply[:200]}")

        log_ticket(
            sender=req.sender,
            subject=req.subject,
            body=req.body,
            category=category,
            confidence=overall_confidence,
            kb_docs=kb_docs,
            reply=reply
        )


        return EmailResponse(reply=reply, category=category, confidence=overall_confidence, retrieved_docs=kb_docs)

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        raise e
