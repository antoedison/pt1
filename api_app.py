from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import time
from dotenv import load_dotenv
from supabase import create_client, Client
load_dotenv()  # Loads .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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
def respond(request: Request, data: dict, template_name: str = None):
    """
    Returns HTML if the client accepts it, otherwise JSON.
    - If template_name is passed, renders that template.
    - Otherwise defaults to JSON.
    """
    if "text/html" in request.headers.get("accept", "") and template_name:
        return templates.TemplateResponse(template_name, {"request": request, **data})
    return JSONResponse(data)


# Home/Login page
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "message": ""})

# Login form submission
@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, register_number: str = Form(...), password: str = Form(...)):
    # Fetch user from Supabase
    response = supabase.table("users").select("*").eq("register_number", register_number).execute()
    user_data = response.data
    
    if not user_data:
        return templates.TemplateResponse("login.html", {"request": request, "message": "User not found"})
    
    user = user_data[0]
    if user["password"] == password:
        # Redirect based on role
        if user["role"].strip().lower() == "student":
            return RedirectResponse(url="/chatbot", status_code=302)
        elif user["role"].strip().lower() == "admin":
            return RedirectResponse(url="/admin", status_code=302)
        else:
            return templates.TemplateResponse("login.html", {"request": request, "message": "Unknown role"})
    else:
        return templates.TemplateResponse("login.html", {"request": request, "message": "Incorrect password"})


# ---------------- Admin Page ----------------
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


# ---------------- Chatbot Page ----------------
@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})


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
        }, template_name="admin.html")

    except Exception as e:
        return respond(request, {"error": str(e)}, template_name="admin.html")


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
        }, template_name="admin.html")

    except Exception as e:
        return respond(request, {"error": str(e)}, template_name="admin.html")
    

# ---------------- Agent Endpoint ----------------
@app.post("/agent")
async def agent_endpoint(
    request: Request,
    index_name: str = Form("faiss_index"),  # default set here
    question: str = Form(...),
):
    try:
        # Step 1: Classify the question
        classification = classifier_agent(question)

        if classification == "chat":
            # Step 2a: Chat Agent
            time.sleep(1.1)  # slight delay to mimic thinking
            response = chat_agent(question)
            return respond(
                request,
                {
                    "query": question,
                    "agent_type": "chat",
                    "agent_response": response,
                },
                template_name="chatbot.html"
            )

        else:
            # Step 2b: Knowledge Agent
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
                template_name="chatbot.html"
            )

    except Exception as e:
        return respond(request, {"error": str(e)}, template_name="chatbot.html")

