from ollama_client import ollama_client
import json
import math



ALLOWED_CATEGORIES = [
    "email_verification_issue",
    "login_issue",
    "password_reset",
    "subscription_billing",
    "payment_failure",
    "account_update",
    "feature_request",
    "bug_report",
    "general_question",
    "technical_error"
]



def classify_ticket(subject: str, body: str) -> tuple[str, float]:
    """
    Returns:
        category: predicted category string
        confidence: float (0.0 - 1.0) based on model scores
    """
    prompt_text = f"""
    
You are a strict customer support ticket classifier.
Classify the email into ONE category from: {ALLOWED_CATEGORIES}
Return JSON ONLY in this format: {{"category": "<category>", "scores": {{category_name: score}}}}

Email Subject: {subject}
Email Body: {body}
"""

    response = ollama_client.generate(model="gemma2:9b", prompt=prompt_text)

    response_text = getattr(response, "content", getattr(response, "response", "{}"))

    try:
        first_brace = response_text.find("{")
        last_brace = response_text.rfind("}") + 1
        json_part = response_text[first_brace:last_brace]
        data = json.loads(json_part)

        category = data.get("category", "general_question")
        scores = data.get("scores", {})

        def softmax(scores_dict):
            exps = {k: math.exp(v) for k, v in scores_dict.items()}
            total = sum(exps.values())
            return {k: v/total for k, v in exps.items()}

        
        scores_norm = softmax(scores)

        confidence = scores_norm.get(category, 0.75)

    except Exception:
        category = "general_question"
        confidence = 0.75

    if category not in ALLOWED_CATEGORIES:
        category = "general_question"
    
    return category, confidence