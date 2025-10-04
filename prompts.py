from langchain.prompts import ChatPromptTemplate
# Define the agent's reasoning behavior
agent_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the provided context to answer the question as accurately as possible.

Context:
{context}

Question:
{question}

Instructions:
- Base your answer strictly on the given context.
- If the context contains the answer, extract and summarize the most relevant information clearly and concisely.
- Do not add information that is not present in the context.
- If the answer cannot be found in the context, respond with: 
  "I could not find relevant information in the provided documents."

Final Answer:
""")