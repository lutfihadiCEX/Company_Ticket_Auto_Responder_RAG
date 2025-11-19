import json
import os
from datetime import datetime

TICKETS_LOG_DIR = "logs/tickets"

if not os.path.exists(TICKETS_LOG_DIR):
    os.makedirs(TICKETS_LOG_DIR)

def log_ticket(sender: str, subject: str, body: str, category: str, confidence: float, kb_docs: list, reply: str):
    """
    Logs ticket info in structured JSON format.
    Each ticket is appended to a daily .jsonl file (one JSON per line).
    """

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "sender": sender,
        "subject": subject,
        "body": body[:500],  # optional truncation
        "category": category,
        "confidence": confidence,
        "retrieved_docs": [{"id": doc["id"], "content": doc["content"][:300]} for doc in kb_docs],
        "reply": reply[:500]  # optional truncation
    }

    log_filename = os.path.join(TICKETS_LOG_DIR, f"{datetime.utcnow().date()}.jsonl")

    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")