from ollama_client import ollama_client
import json
import math

CATEGORY_DEFINITIONS = {
    "email_verification_issue": "Issues related to receiving or using the account activation/verification email.",
    "login_issue": "The user cannot log in, excluding password resets and 2FA failures.",
    "password_reset": "The user needs to reset their password or is having trouble with the reset link/process.",
    "subscription_billing": "Questions about invoices, pricing, refunds, plan changes (upgrade/downgrade), or overage charges.",
    "payment_failure": "The transaction failed due to a bank, credit card, or checkout system error (e.g., card declined, payment unauthorized).",
    "account_update": "The user wants to change their profile details, email, team roles, or privacy settings.",
    "feature_request": "The user is suggesting a new feature, improvement, or integration.",
    "bug_report": "The user is reporting a specific, reproducible malfunction in the software's functionality or UI.",
    "technical_error": "Systemic or environment-related issues affecting availability, performance, or a general system error (e.g., 404, 500, slow loading, API issues).",
    "general_question": "A simple informational inquiry, policy question, or request for documentation/tutorials. Do NOT use this for bugs, failures, or specific account changes."
}

ALLOWED_CATEGORIES = list(CATEGORY_DEFINITIONS.keys())



def classify_ticket(subject: str, body: str) -> tuple[str, float]:
    """
    Classifies a customer support ticket using Gemma 2:9b with strict semantic rules.
    Returns:
        category: predicted category string
        confidence: float (0.0 - 1.0) based on model scores
    """

    definitions_list = [f"- {cat}: {desc}" for cat, desc in CATEGORY_DEFINITIONS.items()]
    definitions_str = "\n".join(definitions_list)

    prompt_text = f"""
    
You are a strict customer support ticket classifier. Your goal is high precision.
Classify the email into ONE category based on the following strict definitions:

{definitions_str}

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