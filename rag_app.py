"""
Standalone RAG script using LangChain + Ollama

Supports:
- PDF (via PyPDFLoader)
- Other file types can be added with UnstructuredFileLoader
- Ollama embeddings & generation
- FAISS vector store
"""

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ---------------- CONFIG ----------------
EMBED_MODEL = "mxbai-embed-large:latest"
GEN_MODEL = "mistral:latest"
INDEX_PATH = "faiss_index"


# ---------- File Loading ----------
def load_document(file_path):
    """
    Loads a document based on file type.
    Currently supports PDF. Add more loaders for DOCX, TXT, etc.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        # fallback loader (requires additional deps)
        loader = UnstructuredFileLoader(file_path)
    return loader.load()


# ---------- Text Chunking ----------
def chunk_document(docs):
    """
    Split documents into smaller chunks for embedding & retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20
    )
    return text_splitter.split_documents(docs)


# ---------- Vector Store ----------
def create_faiss_index(chunks, embedding_model_name=EMBED_MODEL, index_path=INDEX_PATH):
    """
    Create FAISS vector store and save locally.
    """
    print("Creating FAISS vector store...")
    embedding_model = OllamaEmbeddings(model=embedding_model_name)
    faiss_db = FAISS.from_documents(chunks, embedding_model)
    faiss_db.save_local(index_path)
    print(f"FAISS index saved to '{index_path}'.")
    return faiss_db


def load_faiss_index(embedding_model_name=EMBED_MODEL, index_path=INDEX_PATH):
    """
    Load existing FAISS vector store.
    """
    print(f"Loading FAISS index from '{index_path}'...")
    embedding_model = OllamaEmbeddings(model=embedding_model_name)
    faiss_db = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully.")
    return faiss_db


# ---------- RAG Chain ----------
def get_qa_chain(llm_model_name=GEN_MODEL, retriever=None):
    """
    Create RetrievalQA chain using Ollama + FAISS retriever.
    """
    llm = Ollama(model=llm_model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )


# ---------- Usage Example ----------
if __name__ == "__main__":
    if not INDEX_PATH:
        # Load and process document
        docs = load_document("it_helpdesk_solution.pdf")
        chunks = chunk_document(docs)

        # Create / Load FAISS index
        db = create_faiss_index(chunks)
        retriever = db.as_retriever()
    else:
        db = load_faiss_index()
        retriever = db.as_retriever()

    # Create QA chain
    qa_chain = get_qa_chain(retriever=retriever)

    # Run query
    query = "My outlook is not working"
    result = qa_chain.run(query)
    print("Q:", query)
    print("A:", result)
