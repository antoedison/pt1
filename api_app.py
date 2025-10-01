from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import shutil

# Import your RAG functions
from rag_app import (
    load_document,
    chunk_document,
    create_faiss_index,
    load_faiss_index,
    get_qa_chain,
)

app = FastAPI(title="RAG API with LangChain + Ollama")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------- Upload Document Endpoint -----------
@app.post("/upload")
async def upload_document(file: UploadFile):
    """
    Upload a document, process it, and create FAISS index.
    The index will be saved as 'faiss_index_<filename>'.
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process document
        docs = load_document(file_path)
        chunks = chunk_document(docs)

        # Create FAISS index with filename-based ID
        index_path = f"faiss_index_{os.path.splitext(file.filename)[0]}"
        db = create_faiss_index(chunks, index_path=index_path)

        return JSONResponse({
            "message": f"Index created successfully",
            "index_path": index_path
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ----------- Query Endpoint -----------
@app.post("/query")
async def query_document(index_name: str = Form(...), question: str = Form(...)):
    """
    Ask a question against a given FAISS index.
    """
    try:
        # Load FAISS index
        db = load_faiss_index(index_path=index_name)
        retriever = db.as_retriever()

        # Create QA chain
        qa_chain = get_qa_chain(retriever=retriever)

        # Run query
        answer = qa_chain.run(question)

        return JSONResponse({
            "query": question,
            "answer": answer,
            "index_used": index_name
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
