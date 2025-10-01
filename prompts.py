from langchain.prompts import ChatPromptTemplate
# Define the agent's reasoning behavior
retriever_agent_prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant helping identify issues and missing key points.

Steps:
1. Analyze the user's input and retrieved context.
2. Identify the most relevant issue.
3. List what key details are missing to fully resolve the issue.
4. If details are missing, ask a clear follow-up question to obtain them.
5. Finally, summarize your output in a structured way.

Context:
{context}

User Question:
{question}

Your Output (follow these steps carefully):
- Relevant Issue: ...
- Missing Key Points: ...
- Follow-up Question (if needed): ...
- Final Answer: ...
""")