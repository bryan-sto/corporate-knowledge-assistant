import io
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
import httpx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global singleton model & index state
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model


def clean_section_title(raw_title: str) -> str:
    """
    Isolates clean section title from raw line text where pypdf flattens heading and body together.
    """
    raw_title = raw_title.strip()
    
    # Split if sub-clause marker like ' a. ', ' b. ' exists
    sub_clause_match = re.search(r'\s+[a-z]\.\s+', raw_title)
    if sub_clause_match:
        raw_title = raw_title[:sub_clause_match.start()].strip()

    # Parentheses match: e.g. "Klaim Kacamata (Optical) Perusahaan..." -> "Klaim Kacamata (Optical)"
    paren_match = re.search(r'^(.*?\([^)]+\))(?:\s+[A-Za-z]|$)', raw_title)
    if paren_match:
        result = paren_match.group(1).strip()
    else:
        # Transition to body sentence: if word i+1 is lowercase, word i is the start of body
        words = raw_title.split()
        if len(words) > 1:
            title_words = []
            for i, w in enumerate(words):
                if i > 0 and i + 1 < len(words) and words[i+1].islower():
                    break
                title_words.append(w)
            if title_words:
                raw_title = " ".join(title_words)
        result = " ".join(raw_title.split()[:6]).strip()

    # Remove internal quotes to avoid JSON escaped quotes
    return result.replace('"', '').strip()


class HeadingAwareChunker:
    """
    Splits policy documents along structural headers (BAB and numbered sub-sections)
    to keep policy clauses and numeric conditions attached to their titles.
    """
    def __init__(self, target_chunk_size: int = 800, chunk_overlap: int = 100):
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            full_text.append(page_text)
        return "\n".join(full_text)

    def parse_and_chunk(self, raw_text: str, filename: str) -> List[Dict[str, Any]]:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        current_bab = "UMUM"
        current_section_num = "0.0"
        current_section_title = "Ketentuan Umum"
        
        sections: List[Dict[str, Any]] = []
        current_buffer: List[str] = []

        bab_pattern = re.compile(r'^(BAB\s+[IVXLCDM\d]+[:\s]?.*)', re.IGNORECASE)
        section_pattern = re.compile(r'^(\d+\.\d+)\s+(.*)')

        def flush_buffer():
            nonlocal current_buffer, sections, current_bab, current_section_num, current_section_title
            if not current_buffer:
                return
            content = " ".join(current_buffer)
            if len(content) > 1000:
                sub_chunks = self.fallback_splitter.split_text(content)
                for idx, sub_text in enumerate(sub_chunks):
                    sections.append({
                        "text": sub_text,
                        "bab": current_bab,
                        "section_number": current_section_num,
                        "section_title": current_section_title,
                        "sub_index": idx,
                        "filename": filename
                    })
            else:
                sections.append({
                    "text": content,
                    "bab": current_bab,
                    "section_number": current_section_num,
                    "section_title": current_section_title,
                    "sub_index": 0,
                    "filename": filename
                })
            current_buffer = []

        for line in lines:
            bab_match = bab_pattern.match(line)
            sec_match = section_pattern.match(line)

            if bab_match:
                flush_buffer()
                current_bab = bab_match.group(1).strip()
            elif sec_match:
                flush_buffer()
                current_section_num = sec_match.group(1).strip()
                raw_title_rest = sec_match.group(2).strip()
                current_section_title = clean_section_title(raw_title_rest)
                current_buffer.append(line)
            else:
                current_buffer.append(line)

        flush_buffer()
        return sections


class FAISSVectorStore:
    """
    Manages FAISS IndexFlatIP (Inner Product / Cosine Similarity with normalized vectors)
    and sidecar metadata persistence.
    """
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self.load_if_exists()

    def load_if_exists(self) -> bool:
        if config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists():
            try:
                self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
                with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self.index = None
                self.metadata = []
        return False

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
            with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {config.FAISS_INDEX_PATH}")

    def build_and_save(self, chunks: List[Dict[str, Any]], model: SentenceTransformer):
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = chunks
        self.save()

    def search(self, query: str, model: SentenceTransformer, top_k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata) and idx >= 0:
                results.append((float(score), self.metadata[idx]))
        return results


async def call_llm(prompt: str, system_prompt: str) -> str:
    """
    Asynchronously calls Cloudflare Workers AI with fallback to Groq API.
    """
    if not config.NO_CLOUDFLARE and config.CF_ACCOUNT_ID and config.CF_API_TOKEN:
        url = f"https://api.cloudflare.com/client/v4/accounts/{config.CF_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        headers = {
            "Authorization": f"Bearer {config.CF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": config.LLM_TEMPERATURE
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success") and "result" in data:
                        return data["result"]["response"].strip()
        except Exception as e:
            logger.warning(f"Cloudflare Workers AI failed, attempting Groq fallback: {e}")

    if config.GROQ_API_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": config.LLM_TEMPERATURE
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")

    raise RuntimeError("No working LLM provider available. Check API keys.")


vector_store = FAISSVectorStore()


def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    chunker = HeadingAwareChunker()
    raw_text = chunker.extract_text_from_pdf(pdf_bytes)
    chunks = chunker.parse_and_chunk(raw_text, filename=filename)

    model = get_embedding_model()
    vector_store.build_and_save(chunks, model)

    return {
        "status": "success",
        "filename": filename,
        "chunks_ingested": len(chunks)
    }


async def answer_question(question: str) -> Dict[str, Any]:
    if vector_store.index is None or vector_store.index.ntotal == 0:
        return {
            "error": "Vector index is empty. Please ingest a document via /ingest first.",
            "status_code": 400
        }

    model = get_embedding_model()
    results = vector_store.search(question, model, top_k=config.TOP_K_CHUNKS)

    if not results:
        return {
            "answer": "Maaf, saya hanya bisa menjawab terkait kebijakan internal.",
            "sources": [],
            "guardrail_triggered": "Layer 1 (No Results)"
        }

    top_score, top_chunk = results[0]
    logger.info(f"Top retrieval similarity score: {top_score:.4f} (Threshold: {config.RETRIEVAL_SIMILARITY_THRESHOLD})")

    if top_score < config.RETRIEVAL_SIMILARITY_THRESHOLD:
        return {
            "answer": "Maaf, saya hanya bisa menjawab terkait kebijakan internal.",
            "sources": [],
            "guardrail_triggered": f"Layer 1 (Similarity score {top_score:.4f} < {config.RETRIEVAL_SIMILARITY_THRESHOLD})",
            "top_similarity_score": round(top_score, 4)
        }

    context_blocks = []
    sources = []
    seen_sections = set()

    for score, chunk in results:
        context_blocks.append(f"[{chunk['bab']} - Section {chunk['section_number']}: {chunk['section_title']}]\n{chunk['text']}")
        sec_key = (chunk['bab'], chunk['section_number'])
        if sec_key not in seen_sections:
            seen_sections.add(sec_key)
            sources.append({
                "bab": chunk["bab"],
                "section_number": chunk["section_number"],
                "section_title": chunk["section_title"],
                "similarity_score": round(score, 4)
            })

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an internal corporate policy assistant for PT Teknologi Masa Depan (TMD).\n"
        "Answer employee questions strictly based on the provided context below.\n\n"
        "Rules:\n"
        "1. If the question is outside the scope of internal policy or cannot be answered using the provided context "
        "(e.g. general trivia, politics, external facts), reply politely with:\n"
        '   "Maaf, saya hanya bisa menjawab terkait kebijakan internal."\n'
        "2. Do not attempt to guess or use external knowledge.\n"
        "3. Keep the answer concise, accurate, clear, and professional in Bahasa Indonesia.\n"
        "4. For multi-part questions (e.g. covering multiple policies), organize the explanation with clear structured points.\n"
        "5. Do not enclose words or policy titles in unnecessary quotation marks."
    )

    user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION: {question}"

    answer = await call_llm(user_prompt, system_prompt)

    # Post-process answer to ensure clean formatting
    if answer.startswith('"') and answer.endswith('"') and len(answer) > 2:
        answer = answer[1:-1].strip()

    return {
        "answer": answer,
        "sources": sources,
        "top_similarity_score": round(top_score, 4),
        "guardrail_triggered": None
    }
