# app.py
import os
import json
import time
import hashlib
import logging
import random
import threading
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import requests
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag-fastapi")

# ---------------------------
# Config (env-driven)
# ---------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
JSON_FILE_PATH = os.getenv("JSON_FILE_PATH", "json_output/all_records_combined.json")

# Rate limiting (RPM)
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "12"))

# Embedding progress controls
EMB_LOG_EVERY = int(os.getenv("EMB_LOG_EVERY", "25"))
EMB_CKPT_EVERY = int(os.getenv("EMB_CKPT_EVERY", "500"))
EMB_STOP_AFTER = int(os.getenv("EMB_STOP_AFTER", "0"))  # 0 = disabled

# Chroma collection
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "incident_hazard_records")

# Paths
Path("./vector_db").mkdir(exist_ok=True, parents=True)
EMB_CACHE_JSON = "vector_db/embedding_cache.json"
EMB_CACHE_SQLITE = "vector_db/emb_cache.sqlite"
EMB_PROGRESS = "vector_db/embeddings.progress"

# ---------------------------
# Rate limiter (token bucket)
# ---------------------------
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> float:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= amount:
                self.tokens -= amount
                return 0.0
            deficit = amount - self.tokens
            wait_s = deficit / self.refill_rate if self.refill_rate > 0 else 5.0
            self.tokens = 0.0
            return max(wait_s, 0.0)

def sleep_with_jitter(seconds: float):
    if seconds <= 0:
        return
    time.sleep(seconds + random.uniform(0, 0.25))

bucket = TokenBucket(capacity=max(GEMINI_RPM, 1), refill_rate=max(GEMINI_RPM, 1) / 60.0)

def gemini_guardrail():
    wait = bucket.consume(1.0)
    if wait > 0:
        sleep_with_jitter(wait)

# ---------------------------
# Embedding cache (JSON + SQLite)
# ---------------------------
class EmbeddingCache:
    def __init__(self, json_path: str, sqlite_path: str):
        self.json_path = json_path
        self.sqlite_path = sqlite_path
        self.mem: Dict[str, List[float]] = {}
        self._load_json()
        self._ensure_sqlite()

    def _load_json(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    self.mem = json.load(f)
                logger.info("Loaded embedding cache (JSON)")
            except Exception as e:
                logger.warning(f"Could not load JSON cache: {e}")

    def _ensure_sqlite(self):
        try:
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        hash TEXT PRIMARY KEY,
                        dim  INTEGER NOT NULL,
                        vec  BLOB NOT NULL
                    )
                """)
        except Exception as e:
            logger.warning(f"Could not init sqlite cache: {e}")

    def _vec_to_blob(self, vec: List[float]) -> bytes:
        arr = np.asarray(vec, dtype=np.float32)
        return arr.tobytes()

    def _blob_to_vec(self, blob: bytes, dim: int) -> List[float]:
        return np.frombuffer(blob, dtype=np.float32, count=dim).tolist()

    def get(self, key: str) -> Optional[List[float]]:
        v = self.mem.get(key)
        if v is not None:
            return v
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                row = conn.execute("SELECT dim, vec FROM embeddings WHERE hash=?", (key,)).fetchone()
                if row:
                    dim, blob = row
                    vec = self._blob_to_vec(blob, dim)
                    self.mem[key] = vec
                    return vec
        except Exception:
            pass
        return None

    def set(self, key: str, vec: List[float]):
        self.mem[key] = vec
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    "REPLACE INTO embeddings(hash, dim, vec) VALUES(?, ?, ?)",
                    (key, len(vec), self._vec_to_blob(vec))
                )
        except Exception:
            pass

    def flush_json(self):
        try:
            with open(self.json_path, "w") as f:
                json.dump(self.mem, f)
            logger.info("Saved embedding cache (JSON)")
        except Exception as e:
            logger.warning(f"Could not save JSON cache: {e}")

# ---------------------------
# Core RAG engine
# ---------------------------
class RAGEngine:
    def __init__(self, gemini_api_key: str, openrouter_api_key: str, collection_name: str):
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        if not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")

        genai.configure(api_key=gemini_api_key)

        self.chroma = chromadb.PersistentClient(
            path="./vector_db",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        self.cache = EmbeddingCache(EMB_CACHE_JSON, EMB_CACHE_SQLITE)
        self.openrouter_api_key = openrouter_api_key

    def _get_or_create_collection(self):
        try:
            col = self.chroma.get_collection(self.collection_name)
            logger.info("Using existing Chroma collection")
            return col
        except Exception:
            col = self.chroma.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new Chroma collection")
            return col

    def _get_all_existing_ids(self, page_size: int = 1000) -> set:
        existing = set()
        try:
            offset = 0
            while True:
                data = self.collection.get(include=['ids'], limit=page_size, offset=offset)
                ids = data.get('ids', [])
                if not ids:
                    break
                existing.update(ids)
                offset += len(ids)
                if len(ids) < page_size:
                    break
        except Exception as e:
            logger.warning(f"Enumerate IDs failed: {e}")
        return existing

    def _text_hash(self, text: str) -> str:
        norm = " ".join(str(text).split())
        return hashlib.md5(norm.encode("utf-8")).hexdigest()

    def _embed_with_retries(self, text: str, max_tries: int = 6) -> List[float]:
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                gemini_guardrail()
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text[:8192],
                    task_type="retrieval_document"
                )
                emb = result["embedding"]
                if not emb or not isinstance(emb, list):
                    raise RuntimeError("Empty embedding")
                return emb
            except Exception as e:
                last_err = e
                base = 1.5
                wait_s = min(60.0, (base ** attempt))
                logger.warning(f"[Gemini] attempt {attempt}/{max_tries} failed: {e}. Backoff {wait_s:.1f}s")
                sleep_with_jitter(wait_s)
        logger.error(f"Failed to embed text after {max_tries} attempts: {last_err}")
        return [0.0] * 768

    def get_embedding(self, text: str) -> List[float]:
        key = self._text_hash(text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        emb = self._embed_with_retries(text)
        self.cache.set(key, emb)
        return emb

    def _fallback_text(self, rec: Dict) -> str:
        parts = []
        if 'sheet_name' in rec:
            parts.append(f"Sheet: {rec['sheet_name']}")
        if 'data' in rec and rec['data']:
            for k, v in rec['data'].items():
                if v is not None and str(v).strip() and str(v) != 'nan':
                    parts.append(f"{k}: {v}")
        return " | ".join(parts) if parts else "Empty record"

    def ingest_json(self, json_path: str, limit_records: Optional[int] = None) -> Dict:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        logger.info(f"Loaded {len(records)} total records")
        existing_ids = self._get_all_existing_ids()
        new_records = [r for r in records if str(r.get("id")) not in existing_ids]
        logger.info(f"New records to index: {len(new_records)}")

        if limit_records is not None and limit_records > 0:
            new_records = new_records[:limit_records]
            logger.info(f"Capped to first {len(new_records)} new records (limit_records={limit_records})")

        if not new_records:
            return {"added": 0, "skipped_existing": len(existing_ids), "embedded": 0}

        # Prepare texts + hashes
        def rec_to_text(r: Dict) -> str:
            txt = r.get("text_content", "") or self._fallback_text(r)
            return (txt.strip() or f"Record from {r.get('sheet_name', 'Unknown')} sheet")

        texts = [rec_to_text(r) for r in new_records]
        hashes = [self._text_hash(t) for t in texts]

        # Deduplicate text hashes across new records
        unique_map: Dict[str, str] = {}
        for h, t in zip(hashes, texts):
            if h not in unique_map:
                unique_map[h] = t

        # Only embed unseen hashes
        unknown_hashes = [h for h in unique_map if self.cache.get(h) is None]
        total = len(unknown_hashes)
        logger.info(f"Unique new texts: {len(unique_map)} | Need embeddings: {total}")

        # Progress
        start_ts, done, last_log = time.time(), 0, 0
        micro = max(1, min(10, GEMINI_RPM // 2))

        for i in range(0, total, micro):
            chunk = unknown_hashes[i:i+micro]
            for h in chunk:
                if EMB_STOP_AFTER and done >= EMB_STOP_AFTER:
                    logger.info(f"EMB_STOP_AFTER={EMB_STOP_AFTER} reached; stopping early.")
                    break
                emb = self._embed_with_retries(unique_map[h])
                self.cache.set(h, emb)
                done += 1

                if done - last_log >= EMB_LOG_EVERY or done == total:
                    elapsed = time.time() - start_ts
                    rate = done / max(elapsed, 1e-6)
                    remaining = total - done
                    eta = int(remaining / max(rate, 1e-6))
                    logger.info(f"[Embeddings] {done}/{total} ({done/total:.1%}) | {rate:.2f}/s | ETA {eta}s")
                    last_log = done

                if done % EMB_CKPT_EVERY == 0:
                    self.cache.flush_json()
                    try:
                        with open(EMB_PROGRESS, "a") as fp:
                            fp.write(f"{done}/{total} at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                    except Exception:
                        pass

            if EMB_STOP_AFTER and done >= EMB_STOP_AFTER:
                break
            sleep_with_jitter(5.0)

        self.cache.flush_json()
        try:
            with open(EMB_PROGRESS, "a") as fp:
                fp.write(f"COMPLETED {done}/{total} at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        except Exception:
            pass

        # Add NEW records to Chroma
        added = 0
        BATCH = 25
        for i in range(0, len(new_records), BATCH):
            batch = new_records[i:i+BATCH]
            ids, embs, docs, metas = [], [], [], []
            for rec, txt, h in zip(batch, texts[i:i+BATCH], hashes[i:i+BATCH]):
                emb = self.cache.get(h) or self._embed_with_retries(txt)
                ids.append(str(rec["id"]))
                docs.append(txt)
                metas.append(self._metadata_from_record(rec))
                embs.append(emb)
            if ids:
                try:
                    self.collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
                    added += len(ids)
                except Exception as e:
                    logger.error(f"Chroma add failed: {e}")

        return {"added": added, "skipped_existing": len(existing_ids), "embedded": done}

    def _metadata_from_record(self, rec: Dict) -> Dict:
        md = {
            "sheet_name": str(rec.get("sheet_name", "")),
            "row_number": str(rec.get("row_number", "")),
            "record_id": str(rec.get("id", "")),
            "non_null_fields": str(rec.get("metadata", {}).get("non_null_fields", 0)),
            "source_sheet": str(rec.get("metadata", {}).get("source_sheet", "")),
        }
        if rec.get("data"):
            for field in ["Incident Type(s)", "Status", "Category", "Department",
                          "Date of Occurrence", "Location", "Group Company"]:
                if field in rec["data"] and rec["data"][field]:
                    clean = field.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    md[clean] = str(rec["data"][field])[:100]
        return md

    def query(self, text: str, n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        q_emb = self.get_embedding(text)
        try:
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances'],
                where=where
            )
            return res
        except Exception as e:
            logger.error(f"Chroma query error: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _prepare_context(self, results: Dict) -> str:
        if not results.get("documents") or not results["documents"][0]:
            return "No relevant data found in the database."
        parts = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            parts.append(f"\n--- Record {i+1} (Similarity: {1 - dist:.3f}) ---")
            parts.append(f"Sheet: {meta.get('sheet_name', 'Unknown')}")
            parts.append(f"Record ID: {meta.get('record_id', 'Unknown')}")
            for k, v in meta.items():
                if k not in ["sheet_name", "record_id", "row_number"] and v:
                    parts.append(f"{k.replace('_', ' ').title()}: {v}")
            parts.append(f"Content: {doc[:800]}...")
        return "\n".join(parts)

    def chat_with_data(self, question: str, model: str = "openai/gpt-4o-mini") -> str:
        results = self.query(question, n_results=5)
        context = self._prepare_context(results)

        system_prompt = (
            "You are an AI assistant specialized in analyzing incident and hazard data from industrial facilities. "
            "Use the provided context to answer precisely, cite incident numbers/dates when possible, and focus on actionable safety insights."
        )
        user_prompt = f"User Question: {question}\n\nRelevant Data from Database:\n{context}\n\nAnswer comprehensively."

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "RAG Industrial Safety System",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1500,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=500, detail=f"OpenRouter error {resp.status_code}: {detail}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat error: {e}")

    def chat_with_data_stream(self, question: str, model: str = "openai/gpt-4o-mini"):
        results = self.query(question, n_results=5)
        context = self._prepare_context(results)

        system_prompt = (
            "You are an AI assistant specialized in analyzing incident and hazard data from industrial facilities. "
            "Use the provided context to answer precisely, cite incident numbers/dates when possible, and focus on actionable safety insights."
        )
        user_prompt = f"User Question: {question}\n\nRelevant Data from Database:\n{context}\n\nAnswer comprehensively."

        def event_generator():
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "HTTP-Referer": "https://example.com",
                        "X-Title": "RAG Industrial Safety System",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1500,
                        "stream": True,
                    },
                    timeout=300,
                    stream=True,
                )
                if resp.status_code != 200:
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = resp.text
                    yield f"event: error\ndata: {json.dumps({'status': resp.status_code, 'detail': str(detail)})}\n\n"
                    return

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith(":"):
                        # pass through comments/heartbeats silently
                        continue
                    if line.startswith("data: "):
                        payload = line[len("data: "):].strip()
                        if payload == "[DONE]":
                            yield "event: done\ndata: [DONE]\n\n"
                            break
                        try:
                            obj = json.loads(payload)
                            # OpenAI-compatible delta format
                            choices = obj.get("choices", [])
                            if choices:
                                first = choices[0] or {}
                                delta = first.get("delta", {}) or {}
                                content = (
                                    delta.get("content")
                                    or (first.get("message") or {}).get("content")
                                    or first.get("text")
                                )
                                if content:
                                    # Wrap token in SSE data for client consumption
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        except Exception:
                            # If cannot parse JSON, forward raw text
                            yield f"data: {json.dumps({'content': payload})}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

        return event_generator()

# ---------------------------
# FastAPI app & schemas
# ---------------------------
app = FastAPI(title="RAG FastAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine(GEMINI_API_KEY, OPENROUTER_API_KEY, COLLECTION_NAME)

class IngestRequest(BaseModel):
    json_path: Optional[str] = None
    limit_records: Optional[int] = None  # applies to NEW records only

class IngestResponse(BaseModel):
    added: int
    skipped_existing: int
    embedded: int

class ChatRequest(BaseModel):
    question: str
    model: Optional[str] = "openai/gpt-4o-mini"

class ChatResponse(BaseModel):
    answer: str

class QueryRequest(BaseModel):
    text: str
    n_results: Optional[int] = 5
    where: Optional[dict] = None

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    json_path = req.json_path or JSON_FILE_PATH
    try:
        result = engine.ingest_json(json_path=json_path, limit_records=req.limit_records)
        return IngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        answer = engine.chat_with_data(req.question, model=req.model or "openai/gpt-4o-mini")
        return ChatResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    try:
        generator = engine.chat_with_data_stream(req.question, model=req.model or "openai/gpt-4o-mini")
        return StreamingResponse(generator, media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat stream failed")
        raise HTTPException(status_code=500, detail=f"Chat stream failed: {e}")

@app.get("/chat/stream")
def chat_stream_get(question: str, model: Optional[str] = "openai/gpt-4o-mini"):
    try:
        generator = engine.chat_with_data_stream(question, model=model or "openai/gpt-4o-mini")
        return StreamingResponse(generator, media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat stream (GET) failed")
        raise HTTPException(status_code=500, detail=f"Chat stream (GET) failed: {e}")

@app.post("/query")
def query(req: QueryRequest):
    try:
        res = engine.query(req.text, n_results=req.n_results or 5, where=req.where)
        return res
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
