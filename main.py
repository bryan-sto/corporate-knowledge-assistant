import logging
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Corporate Knowledge Assistant API",
    description="RAG-powered API for internal corporate policy Q&A built with FastAPI, FAISS, and LLM Guardrails.",
    version="1.0.0"
)

# Enable browser-compliant CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class IngestResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "success"})
    filename: str = Field(..., json_schema_extra={"example": "sample_policy.pdf"})
    chunks_ingested: int = Field(..., json_schema_extra={"example": 11})


class SourceItem(BaseModel):
    bab: str = Field(..., json_schema_extra={"example": "BAB III: TUNJANGAN & KLAIM KESEHATAN"})
    section_number: str = Field(..., json_schema_extra={"example": "3.1"})
    section_title: str = Field(..., json_schema_extra={"example": "Klaim Kacamata (Optical)"})
    similarity_score: float = Field(..., json_schema_extra={"example": 0.7045})


class ChatRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "Berapa plafon klaim kacamata untuk level Manager?"})


class ChatResponse(BaseModel):
    answer: str = Field(..., json_schema_extra={"example": "Plafon klaim kacamata untuk level Manager adalah maksimal Rp 2.500.000,- setiap 2 tahun sekali."})
    sources: List[SourceItem] = Field(default_factory=list)
    top_similarity_score: Optional[float] = Field(None, json_schema_extra={"example": 0.7045})
    guardrail_triggered: Optional[str] = Field(None, json_schema_extra={"example": None})


@app.get("/health", tags=["Health Check"])
@app.get("/", tags=["Health Check"])
async def root():
    return {
        "status": "online",
        "service": "Intelligent Corporate Knowledge Assistant API",
        "docs_url": "/docs"
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Accepts a PDF document upload, performs heading-aware chunking,
    generates normalized embeddings, and persists the FAISS vector index.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
        result = rag.ingest_pdf_bytes(contents, file.filename)
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the document. Check server logs for details."
        )


@app.post("/chat", response_model=ChatResponse, tags=["RAG Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Receives employee query, performs Cosine similarity search in FAISS,
    evaluates 2-layer guardrail, and returns a grounded policy answer.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty."
        )

    try:
        result = await rag.answer_question(request.question)
        if "error" in result:
            raise HTTPException(
                status_code=result.get("status_code", status.HTTP_400_BAD_REQUEST),
                detail=result["error"]
            )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating the answer. Check server logs for details."
        )
