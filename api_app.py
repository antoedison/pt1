from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import time

# Import your RAG functions
from rag_app import (
    load_document,
    chunk_document,
    create_faiss_index,
    load_faiss_index,
    get_qa_chain,
)

from agents import (
    classifier_agent,
    chat_agent,
    retriever_agent_chain
)


app = FastAPI(title="RAG API with LangChain + Ollama")

# ---------------- Templating setup ----------------
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------- Utility function ----------------
def respond(request: Request, data: dict, template_name: str = "index.html"):
    """
    Returns HTML if the client accepts it, otherwise JSON.
    """
    if "text/html" in request.headers.get("accept", ""):
        return templates.TemplateResponse(template_name, {"request": request, **data})
    return JSONResponse(data)


# ---------------- Home Page ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return respond(request, {})


# ---------------- Upload Document Endpoint ----------------
@app.post("/upload")
async def upload_document(request: Request, file: UploadFile):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docs = load_document(file_path)
        chunks = chunk_document(docs)

        index_path = f"faiss_index_{os.path.splitext(file.filename)[0]}"
        create_faiss_index(chunks, index_path=index_path)

        return respond(request, {
            "message": "Index created successfully",
            "index_path": index_path
        })

    except Exception as e:
        return respond(request, {"error": str(e)})


# ---------------- Query Document Endpoint ----------------
@app.post("/query")
async def query_document(
    request: Request,
    index_name: str = Form(...),
    question: str = Form(...)
):
    try:
        db = load_faiss_index(index_path=index_name)
        retriever = db.as_retriever()
        qa_chain = get_qa_chain(retriever=retriever)

        answer = qa_chain.run(question)

        return respond(request, {
            "query": question,
            "answer": answer,
            "index_used": index_name
        })

    except Exception as e:
        return respond(request, {"error": str(e)})
    
# ---------------- Agent Endpoint ----------------
@app.post("/agent")
async def agent_endpoint(
    request: Request,
    index_name: str = Form(...),
    question: str = Form(...),
):
    try:
        # Step 1: Classify the question
        classification = classifier_agent(question)

        if classification == "chat":
            # Step 2a: Chat Agent
            # Delay before calling chat agent
            time.sleep(1.1)
            response = chat_agent(question)
            return respond(
                request,
                {
                    "query": question,
                    "agent_type": "chat",
                    "agent_response": response,
                },
            )

        else:
            # Step 2b: Knowledge Agent (Retriever Agent)
            db = load_faiss_index(index_path=index_name)
            retriever = db.as_retriever()
            agent = retriever_agent_chain(retriever)
            response = agent(question)

            return respond(
                request,
                {
                    "query": question,
                    "agent_type": "knowledge",
                    "agent_response": response,
                    "index_used": index_name,
                },
            )

    except Exception as e:
        return respond(request, {"error": str(e)})