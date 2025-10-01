# agent_chain.py
import os
import requests
from ollama import generate
from prompts import retriever_agent_prompt


MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "pYXacKkJmYtAwnQtDu8JTyAFVphLeDLB"


# ---------------- Classifier Agent ----------------
def classifier_agent(user_input: str) -> str:
    """
    Calls Mistral API to classify text as 'chat' or 'knowledge'.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""
    You are a classifier. Your task:
    - If the user text is a casual conversation (like greetings, jokes, small talk),
      classify as: chat
    - If the text is about problem-solving, knowledge, or technical issues,
      classify as: knowledge

    Output only one word: "chat" or "knowledge".

    User text: {user_input}
    """

    body = {
        "model": "mistral-tiny",   # small & fast model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }

    resp = requests.post(MISTRAL_API_URL, headers=headers, json=body)
    resp.raise_for_status()

    classification = resp.json()["choices"][0]["message"]["content"].strip().lower()

    return "chat" if "chat" in classification else "knowledge"


# ---------------- Chat Agent ----------------
def chat_agent(user_input: str) -> str:
    """
    Conversational agent for casual chat using Mistral API.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "mistral-small",  # better for conversation
        "messages": [
            {"role": "system", "content": "You are a friendly assistant that chats casually with the user."},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7
    }

    resp = requests.post(MISTRAL_API_URL, headers=headers, json=body)
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"].strip()

def retriever_agent_chain(retriever):
    """
    Returns a callable agent function that takes a question
    and returns structured reasoning based on retriever context.
    """

    def run_agent(question: str):
        # Step 1: Retrieve documents
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([d.page_content for d in docs])

        # Step 2: Format prompt
        formatted_prompt = retriever_agent_prompt.format(context=context, question=question)

        # Step 3: Call Ollama LLM
        response = generate(model="mistral:latest", prompt=formatted_prompt)

        return response["response"]

    return run_agent
