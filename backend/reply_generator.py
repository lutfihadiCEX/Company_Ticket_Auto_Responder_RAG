from ollama_client import ollama_client


def generate_reply(category: str, email_body: str, kb_docs: list) -> str:
    """
    Inputs:
        category: ticket category string
        email_body: original customer email
        kb_docs: list of relevant KB document texts
    Returns:
        reply: professional email reply string
    """

    email_body = email_body.strip()
    #kb_text = "\n".join([doc.strip() for doc in kb_docs]) if kb_docs else "No additional knowledge base info."

    if not kb_docs:
        
        return (
            "Hello,\n\nThank you for contacting us. "
            "We are unable to process your request as it falls outside the scope of our support knowledge base.\n\n"
            "Please reach out with queries related to our products or services.\n\n"
            "Best regards,\nYour AI Customer Support Assistant"
        )

    kb_text = "\n".join([doc.strip() for doc in kb_docs])

    prompt_text = f"""
    You are an AI customer support assistant.

    The ticket category is: {category}
    The customer email is: "{email_body}"

    Use ONLY the relevant information from the knowledge base documents below:

    --- KB DOCUMENTS START ---
    {kb_text}
    --- KB DOCUMENTS END ---

    Write a professional email reply that includes:
    - Greeting
    - Understanding of user's issue
    - Steps/solution based on the KB documents
    - Offer extra assistance if needed
    - Closing & signature

    If the KB does NOT contain the answer, reply:
    "Hello, thank you for reaching out. We are looking into your request and will respond shortly."

"""

    response = ollama_client.generate(model="gemma2:9b", prompt=prompt_text)

    #print("DEBUG RAW RESPONSE:", response)

    if hasattr(response, "response") and response.response.strip():
        reply = response.response.strip()
    elif hasattr(response, "text") and response.text.strip():
        reply = response.text.strip()
    else:
        reply = "Hello, thank you for reaching out. We are looking into your request and will respond shortly."

    reply = reply.replace("\n", " ")
    
    return reply

    