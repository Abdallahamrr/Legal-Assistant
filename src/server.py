"""
Legal AI Assistant — FastAPI Backend Server
Capstone Project | Gen AI Course 2026
"""

import os
import sys
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

# Ensure parent directory is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag_pipeline import (
    DATA_DIR,
    VECTOR_STORE_DIR,
    load_vector_store,
    load_documents,
    split_documents,
    build_vector_store,
    build_legal_agent,
    AgentState,
    run_evaluation,
    save_eval_report,
    TEST_QUERIES
)

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP CONFIG
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal AI Assistant API",
    description="Production-ready FastAPI backend for the Agentic RAG Legal AI Assistant.",
    version="1.0.0"
)

# Add CORS Middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared memory agent instance
shared_vector_store = None
shared_agent = None

def init_resources():
    global shared_vector_store, shared_agent
    if VECTOR_STORE_DIR.exists():
        try:
            shared_vector_store = load_vector_store()
            print("[Backend API] Vector store loaded successfully.")
        except Exception as e:
            print(f"[Backend API] Failed to load vector store: {e}. Rebuilding...")
            docs = load_documents()
            chunks = split_documents(docs)
            shared_vector_store = build_vector_store(chunks)
    else:
        docs = load_documents()
        chunks = split_documents(docs)
        shared_vector_store = build_vector_store(chunks)
    
    shared_agent = build_legal_agent(shared_vector_store)

# Initialize resources on startup
@app.on_event("startup")
def startup_event():
    init_resources()


# ─────────────────────────────────────────────────────────────────────────────
# API SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or legal scenario")
    history: List[ChatMessage] = Field(default=[], description="Conversation history")
    model_provider: Optional[str] = Field("gemini", description="LLM provider: 'gemini' or 'deepseek'")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Grounded, cited analysis")
    sources: List[str] = Field(..., description="Source citations")
    latency_ms: float = Field(..., description="Response latency in ms")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "Legal AI Assistant Backend API",
        "vector_store_initialized": shared_vector_store is not None,
        "agent_initialized": shared_agent is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Invoke the Agentic RAG pipeline with query reformulation, sufficiency check, and security.
    """
    global shared_agent
    if not shared_agent:
        raise HTTPException(status_code=503, detail="Agent is not initialized yet.")
        
    t0 = time.time()
    try:
        # Re-build langchain message history
        messages = []
        for h in request.history:
            if h.role == "user":
                messages.append(HumanMessage(content=h.content))
            else:
                messages.append(AIMessage(content=h.content))
        
        # Append latest message
        messages.append(HumanMessage(content=request.message))
        
        # Invoke agent state
        state = AgentState(messages=messages)
        # Explicit model swap inside build_legal_agent if needed, or dynamically re-compile
        if request.model_provider == "deepseek":
            dynamic_agent = build_legal_agent(shared_vector_store, model_provider="deepseek")
            result = dynamic_agent.invoke(state)
        else:
            result = shared_agent.invoke(state)
            
        latency = (time.time() - t0) * 1000
        
        if isinstance(result, dict):
            answer = result.get("answer", str(result))
            sources = result.get("sources", [])
        else:
            answer = result.answer if hasattr(result, "answer") else str(result)
            sources = result.sources if hasattr(result, "sources") else []
            
        return ChatResponse(
            answer=answer,
            sources=sources,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {str(e)}")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document (PDF or TXT) and rebuild index dynamically on the backend.
    """
    global shared_vector_store, shared_agent
    
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    try:
        # Save file to disk
        DATA_DIR.mkdir(exist_ok=True)
        file_path = DATA_DIR / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Rebuild RAG index
        print(f"[Backend API] Dynamic ingestion of: {file.filename}")
        docs = load_documents()
        chunks = split_documents(docs)
        shared_vector_store = build_vector_store(chunks)
        shared_agent = build_legal_agent(shared_vector_store)
        
        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {file.filename}",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


@app.post("/evaluate")
def run_evaluations_endpoint(background_tasks: BackgroundTasks):
    """
    Trigger the capstone evaluation benchmark suite.
    """
    global shared_agent
    if not shared_agent:
        raise HTTPException(status_code=503, detail="Agent is not initialized.")
        
    def run_suite():
        results = run_evaluation(TEST_QUERIES, shared_agent)
        save_eval_report(results)
        
    background_tasks.add_task(run_suite)
    return {
        "status": "triggered",
        "message": "Evaluation suite is running in the background. Check outputs/eval_report.json once complete."
    }

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
