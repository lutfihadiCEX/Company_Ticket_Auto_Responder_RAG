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

    kb_text = "\n".join([doc["content"].strip() for doc in kb_docs])

    prompt_text = f"""
    You are a highly professional Customer Support AI Assistant.
    Your task is to write a response email based strictly on the retrieved knowledge base (KB) content.

    ## RULES
    - Use only the information provided in the KB chunks below.
    - Do NOT add any steps or details that are not explicitly stated in the KB.
    - If the KB lacks relevant information, politely state that the information is unavailable.
    - The reply must be helpful, polite, and clear.
    - Do NOT mention that KB chunks were used.
    - Write the reply in a natural, customer-friendly tone.
    - Include a short subject line.

    ## CUSTOMER EMAIL
    {email_body}

    ## CATEGORY
    {category}

    ## RETRIEVED KB CHUNKS
    {kb_text}

    ## TASK
    Write a complete reply email addressing the customer's request using only the KB content.

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

    