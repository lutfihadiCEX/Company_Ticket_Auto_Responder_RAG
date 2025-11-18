from ollama_client import ollama_client
import json



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
    "unknown"
]



def classify_ticket(subject: str, body: str) -> str:
    """
    Process ticket category based on email subject and body.

    """

    prompt_text = f"""

You are a strict customer support ticket classifier. 
Your job is to categorize the user's issue into ONE of the following categories:

- email_verification_issue   → expired link, missing verification email, can't verify account
- login_issue                → can't log in, invalid credentials, 2FA issues
- password_reset             → forgot password, reset not working
- subscription_billing       → subscription, cancel, refund, invoice, upgrade, downgrade
- payment_failure            → card declined, payment error, transaction failed
- account_update             → change email, update profile, delete account
- feature_request            → asking for new features
- bug_report                 → errors, crashes, something not working
- general_question           → anything that doesn't fit categories above
- unknown                    → unclear or insufficient info

Return ONLY JSON: {{"category": "<category>"}}

### Examples ###
Email: "My email verification link expired"
Category: email_verification_issue

Email: "I want to cancel my subscription"
Category: subscription_billing

Email: "Your app crashes when uploading files"
Category: bug_report

Email: "How do I change my email address?"
Category: account_update

Email: "My credit card was declined"
Category: payment_failure

Now classify this email:

Subject: {subject}
Body: {body}


    """

    response = ollama_client.generate(model="gemma2:9b", prompt=prompt_text)

    
    response_text = ""
    if hasattr(response, "content"):
        response_text = response.content
    elif hasattr(response, "response"):
        response_text = response.response
    elif isinstance(response, str):
        response_text = response
    else:
        response_text = "{}"

    
    try:
        first_brace = response_text.find("{")
        last_brace = response_text.rfind("}") + 1
        json_part = response_text[first_brace:last_brace]
        data = json.loads(json_part)
        category = data.get("category", "general_question")
    except Exception:
        category = "general_question"

    
    if category not in ALLOWED_CATEGORIES:
        category = "general_question"

    return category
