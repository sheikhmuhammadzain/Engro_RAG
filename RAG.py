import json
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import sqlite3
from pathlib import Path
import hashlib
import random
import threading

# ===========================
# Logging
# ===========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# For super-verbose output, switch to DEBUG:
# logger.setLevel(logging.DEBUG)

plt.style.use('default')
try:
    sns.set_palette("husl")
except Exception:
    pass

# ===========================
# ### RATE LIMITING & RESILIENCE
# ===========================
class TokenBucket:
    """
    Simple token-bucket limiter for RPM-limited APIs.
    capacity = max tokens (allowed calls in window)
    refill_rate = tokens per second (capacity / 60)
    """
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, amount: float = 1.0):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= amount:
                self.tokens -= amount
                return 0.0  # no wait
            deficit = amount - self.tokens
            wait_s = deficit / self.refill_rate if self.refill_rate > 0 else 5.0
            self.tokens = 0.0
            return max(wait_s, 0.0)

def sleep_with_jitter(seconds: float):
    if seconds <= 0:
        return
    jitter = random.uniform(0, 0.25)
    time.sleep(seconds + jitter)

def parse_retry_after(resp) -> Optional[float]:
    try:
        ra = resp.headers.get("Retry-After")
        if not ra:
            return None
        return float(ra)  # assume seconds
    except Exception:
        return None

# conservative default; set GEMINI_RPM in .env if you know your actual cap
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "12"))
_gemini_bucket = TokenBucket(capacity=max(GEMINI_RPM, 1), refill_rate=max(GEMINI_RPM, 1) / 60.0)

def gemini_guardrail():
    wait = _gemini_bucket.consume(1.0)
    if wait > 0:
        logger.debug(f"Rate limit: sleeping {wait:.2f}s")
        sleep_with_jitter(wait)

# ===========================
# Persistent embedding cache (JSON + optional SQLite)
# ===========================
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
                with open(self.json_path, 'r') as f:
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
        except Exception as e:
            logger.debug(f"sqlite get failed: {e}")
        return None

    def set(self, key: str, vec: List[float]):
        self.mem[key] = vec
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    "REPLACE INTO embeddings(hash, dim, vec) VALUES(?, ?, ?)",
                    (key, len(vec), self._vec_to_blob(vec))
                )
        except Exception as e:
            logger.debug(f"sqlite set failed: {e}")

    def flush_json(self):
        try:
            with open(self.json_path, 'w') as f:
                json.dump(self.mem, f)
            logger.info("Saved embedding cache (JSON)")
        except Exception as e:
            logger.warning(f"Could not save JSON cache: {e}")

# ===========================
# RAG System
# ===========================
class RAGVectorSystem:
    """Complete RAG system with Gemini embeddings, local ChromaDB, and OpenRouter GPT communication"""

    def __init__(self, gemini_api_key: str, openrouter_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.openrouter_api_key = openrouter_api_key

        genai.configure(api_key=gemini_api_key)

        Path("./vector_db").mkdir(exist_ok=True)
        Path("./plots").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path="./vector_db",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "incident_hazard_records"
        self.collection = None

        # unified caches
        self.embedding_cache = EmbeddingCache(
            json_path="vector_db/embedding_cache.json",
            sqlite_path="vector_db/emb_cache.sqlite"
        )

    def _get_text_hash(self, text: str) -> str:
        norm = " ".join(str(text).split())  # normalize whitespace for better cache hits
        return hashlib.md5(norm.encode("utf-8")).hexdigest()

    # ---------- Helpers for incremental mode ----------
    def _get_or_create_collection(self, recreate: bool = False):
        """
        Get existing collection if present. If recreate=True, delete and create new.
        """
        try:
            if recreate:
                try:
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info("Deleted existing collection (recreate=True)")
                except Exception:
                    pass
            # try to get existing
            self.collection = self.chroma_client.get_collection(self.collection_name)
            if recreate:
                # If we wanted a fresh one after deletion, create again
                try:
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info("Created new collection (fresh)")
                except Exception:
                    pass
            else:
                logger.info("Using existing collection")
        except Exception:
            # doesn't exist -> create
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new collection")

    def _get_all_existing_ids(self, page_size: int = 1000) -> set:
        """
        Pull all existing IDs from Chroma collection (paged).
        """
        existing = set()
        try:
            # chroma get supports limit/offset; some versions use 'where' filters only
            # We'll page until fewer than page_size returned.
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
            logger.warning(f"Could not enumerate existing IDs: {e}")
        return existing

    def _embed_with_retries(self, text: str, max_tries: int = 6) -> List[float]:
        """
        Embed with:
        - global token-bucket (RPM friendly)
        - exponential backoff + jitter
        - Retry-After header support
        """
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                gemini_guardrail()
                result = genai.embed_content(
                    model="models/text-embedding-004",  # newer embedding model
                    content=text[:8192],
                    task_type="retrieval_document"
                )
                emb = result["embedding"]
                if not emb or not isinstance(emb, list):
                    raise RuntimeError("Empty embedding")
                return emb
            except Exception as e:
                last_err = e
                wait_s = None
                try:
                    resp = getattr(e, "response", None)
                    if resp is not None and getattr(resp, "status_code", None) in (429, 503):
                        wait_s = parse_retry_after(resp)
                except Exception:
                    pass

                if wait_s is None:
                    base = 1.5
                    wait_s = min(60.0, (base ** attempt))
                logger.warning(f"[Gemini] attempt {attempt}/{max_tries} failed: {e}. Backing off {wait_s:.1f}s")
                sleep_with_jitter(wait_s)

        logger.error(f"Failed to embed after {max_tries} attempts: {last_err}")
        return [0.0] * 768  # safe fallback for text-embedding-004

    def get_gemini_embedding(self, text: str) -> List[float]:
        key = self._get_text_hash(text)
        cached = self.embedding_cache.get(key)
        if cached is not None:
            return cached
        emb = self._embed_with_retries(text)
        self.embedding_cache.set(key, emb)
        return emb

    def create_vector_database(self, json_file_path: str, batch_size: int = 25,
                               limit_records: Optional[int] = None, recreate: bool = False):
        logger.info("Loading JSON records...")

        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")

        with open(json_file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        logger.info(f"Loaded {len(records)} total records from JSON")

        # Prepare collection (incremental by default)
        self._get_or_create_collection(recreate=recreate)

        # Determine which IDs are already present to skip them
        existing_ids = self._get_all_existing_ids()
        if existing_ids:
            logger.info(f"Existing records in DB: {len(existing_ids)}")
        else:
            logger.info("No existing records found in DB (fresh build or empty)")

        # Keep only new records (not already in collection by id)
        new_records = [r for r in records if str(r.get('id')) not in existing_ids]
        logger.info(f"New records to process: {len(new_records)}")

        # Apply optional hard cap for testing to the NEW records only
        if limit_records is not None and limit_records > 0:
            new_records = new_records[:limit_records]
            logger.info(f"Processing capped new records: {len(new_records)} (limit_records={limit_records})")

        if not new_records:
            logger.info("No new records to embed/index. Exiting create_vector_database early.")
            return

        # Build normalized text & deduplicate among NEW records to minimize API calls
        def record_to_text(rec: Dict) -> str:
            txt = rec.get("text_content", "") or self._create_fallback_text(rec)
            txt = txt.strip() or f"Record from {rec.get('sheet_name', 'Unknown')} sheet"
            return txt

        texts = [record_to_text(r) for r in new_records]
        hashes = [self._get_text_hash(t) for t in texts]
        unique_map: Dict[str, str] = {}
        for h, t in zip(hashes, texts):
            if h not in unique_map:
                unique_map[h] = t

        # Only embed those hashes that are not already cached
        unknown_hashes = [h for h in unique_map.keys() if self.embedding_cache.get(h) is None]
        logger.info(f"Unique texts (NEW): {len(unique_map)} | Need embeddings (NEW): {len(unknown_hashes)}")

        # --- Progress, ETA, and resumable checkpoints (for NEW only) ---
        micro = max(1, min(10, GEMINI_RPM // 2))  # conservative micro-batch
        total = len(unknown_hashes)
        start_ts = time.time()
        LOG_EVERY = int(os.getenv("EMB_LOG_EVERY", "25"))       # log every N embeddings
        CKPT_EVERY = int(os.getenv("EMB_CKPT_EVERY", "500"))    # flush cache every N
        STOP_AFTER = int(os.getenv("EMB_STOP_AFTER", "0"))      # 0 = no cap; otherwise stop after N for testing

        done = 0
        last_log = 0
        for i in range(0, total, micro):
            chunk = unknown_hashes[i:i+micro]

            for h in chunk:
                if STOP_AFTER and done >= STOP_AFTER:
                    logger.info(f"EMB_STOP_AFTER={STOP_AFTER} reached; stopping early (resumable).")
                    break
                emb = self._embed_with_retries(unique_map[h])
                self.embedding_cache.set(h, emb)
                done += 1

                if done - last_log >= LOG_EVERY or done == total:
                    elapsed = time.time() - start_ts
                    rate = done / max(elapsed, 1e-6)
                    remaining = total - done
                    eta_s = remaining / max(rate, 1e-6)
                    logger.info(f"[Embeddings NEW] {done}/{total} ({done/total:.1%}) "
                                f"| {rate:.2f}/s | ETA ~ {int(eta_s//60)}m {int(eta_s%60)}s")
                    last_log = done

                if done % CKPT_EVERY == 0:
                    self.embedding_cache.flush_json()
                    try:
                        with open("vector_db/embeddings.progress", "a") as fp:
                            fp.write(f"NEW {done}/{total} at {datetime.now().isoformat()}\n")
                    except Exception:
                        pass

            if STOP_AFTER and done >= STOP_AFTER:
                break

            # gentle pause between micro-batches to avoid bursts
            sleep_with_jitter(5.0)

        # final flush + marker
        self.embedding_cache.flush_json()
        try:
            with open("vector_db/embeddings.progress", "a") as fp:
                fp.write(f"NEW COMPLETED {done}/{total} at {datetime.now().isoformat()}\n")
        except Exception:
            pass

        # Add only NEW records to Chroma in batches (reads embeddings from cache)
        total_batches = (len(new_records) + batch_size - 1) // batch_size
        for i in range(0, len(new_records), batch_size):
            batch = new_records[i:i + batch_size]
            batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

            current_batch = i // batch_size + 1
            logger.info(f"Processing NEW batch {current_batch}/{total_batches}")

            for rec, txt, h in zip(batch, texts[i:i+batch_size], hashes[i:i+batch_size]):
                try:
                    emb = self.embedding_cache.get(h) or self._embed_with_retries(txt)
                    metadata = {
                        'sheet_name': str(rec.get('sheet_name', '')),
                        'row_number': str(rec.get('row_number', '')),
                        'record_id': str(rec.get('id', '')),
                        'non_null_fields': str(rec.get('metadata', {}).get('non_null_fields', 0)),
                        'source_sheet': str(rec.get('metadata', {}).get('source_sheet', ''))
                    }
                    if 'data' in rec and rec['data']:
                        data = rec['data']
                        key_fields = ['Incident Type(s)', 'Status', 'Category', 'Department',
                                      'Date of Occurrence', 'Location', 'Group Company']
                        for field in key_fields:
                            if field in data and data[field]:
                                clean_key = field.lower().replace(' ', '_').replace('(', '').replace(')', '')
                                metadata[clean_key] = str(data[field])[:100]

                    batch_ids.append(str(rec['id']))
                    batch_embeddings.append(emb)
                    batch_documents.append(txt)
                    batch_metadatas.append(metadata)
                except Exception as e:
                    logger.error(f"Error processing record {rec.get('id', 'unknown')}: {e}")
                    continue

            if batch_ids:
                try:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    logger.info(f"Added {len(batch_ids)} NEW records to database")
                except Exception as e:
                    logger.error(f"Error adding NEW batch to collection: {str(e)}")

            sleep_with_jitter(2.0)

        logger.info("Incremental vector database update completed!")

    def _create_fallback_text(self, record: Dict) -> str:
        parts = []
        if 'sheet_name' in record:
            parts.append(f"Sheet: {record['sheet_name']}")
        if 'data' in record and record['data']:
            for k, v in record['data'].items():
                if v is not None and str(v).strip() and str(v) != 'nan':
                    parts.append(f"{k}: {v}")
        return " | ".join(parts) if parts else "Empty record"

    def load_existing_database(self):
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info("Loaded existing vector database")
            return True
        except Exception as e:
            logger.error(f"Could not load existing database: {e}")
            return False

    def query_vector_database(self, query_text: str, n_results: int = 5,
                              filter_metadata: Optional[Dict] = None) -> Dict:
        if not self.collection:
            if not self.load_existing_database():
                raise Exception("No vector database found. Please create one first.")
        query_embedding = self.get_gemini_embedding(query_text)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances'],
                where=filter_metadata
            )
            return results
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    def chat_with_data(self, user_question: str, model: str = "openai/gpt-4o-mini") -> str:
        try:
            search_results = self.query_vector_database(user_question, n_results=5)
            context = self._prepare_context(search_results)
            system_prompt = """You are an AI assistant specialized in analyzing incident and hazard data from industrial facilities.
Use the provided context to answer precisely, cite incident numbers/dates when possible, and focus on actionable safety insights."""
            user_prompt = f"User Question: {user_question}\n\nRelevant Data from Database:\n{context}\n\nAnswer comprehensively based on the data above."

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "RAG Industrial Safety System"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1500
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                error_msg = f"API call failed with status {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail}"
                except Exception:
                    error_msg += f": {response.text}"
                return f"Error: {error_msg}"
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"

    def _prepare_context(self, search_results: Dict) -> str:
        if not search_results['documents'] or not search_results['documents'][0]:
            return "No relevant data found in the database."
        parts = []
        for i, (doc, metadata, distance) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0],
            search_results['distances'][0]
        )):
            parts.append(f"\n--- Record {i+1} (Similarity: {1-distance:.3f}) ---")
            parts.append(f"Sheet: {metadata.get('sheet_name', 'Unknown')}")
            parts.append(f"Record ID: {metadata.get('record_id', 'Unknown')}")
            for k, v in metadata.items():
                if k not in ['sheet_name', 'record_id', 'row_number'] and v:
                    parts.append(f"{k.replace('_', ' ').title()}: {v}")
            parts.append(f"Content: {doc[:800]}...")
            parts.append("")
        return "\n".join(parts)

    def generate_analytics_plots(self, analysis_request: str) -> str:
        try:
            search_results = self.query_vector_database(analysis_request, n_results=20)
            context = self._prepare_context(search_results)
            system_prompt = """You are a data visualization expert specializing in industrial safety analytics.
Generate complete, executable Python code to create insightful plots and analytics."""
            user_prompt = f"Analysis Request: {analysis_request}\n\nAvailable Data Context:\n{context}\n\nGenerate Python code for the visualization."
            return self.chat_with_data(user_prompt, model="openai/gpt-4o")
        except Exception as e:
            return f"Error generating plot code: {str(e)}"

# ===========================
# Analyzer / App
# ===========================
class DataAnalyzer:
    def __init__(self, rag_system: RAGVectorSystem):
        self.rag_system = rag_system

    def get_data_summary(self) -> Dict:
        try:
            if not self.rag_system.collection:
                if not self.rag_system.load_existing_database():
                    return {"error": "No database found"}
            all_records = self.rag_system.collection.get(include=['documents', 'metadatas'])
            if not all_records['metadatas']:
                return {"error": "No data in database"}

            sheets, departments, incident_types = {}, {}, {}
            for metadata in all_records['metadatas']:
                sheet_name = metadata.get('sheet_name', 'Unknown')
                sheets[sheet_name] = sheets.get(sheet_name, 0) + 1

                dept = metadata.get('department', 'Unknown')
                if dept and dept != 'Unknown':
                    departments[dept] = departments.get(dept, 0) + 1
                inc_type = metadata.get('incident_types', 'Unknown')
                if inc_type and inc_type != 'Unknown':
                    incident_types[inc_type] = incident_types.get(inc_type, 0) + 1

            return {
                'total_records': len(all_records['documents']),
                'sheets_breakdown': sheets,
                'departments_breakdown': departments,
                'incident_types_breakdown': incident_types,
                'collection_name': self.rag_system.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {"error": str(e)}

    def create_basic_plots(self):
        try:
            summary = self.get_data_summary()
            if "error" in summary:
                print(f"Error creating plots: {summary['error']}")
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Database Overview Dashboard', fontsize=16, fontweight='bold')

            if summary['sheets_breakdown']:
                sheets = list(summary['sheets_breakdown'].keys())
                counts = list(summary['sheets_breakdown'].values())
                axes[0,0].pie(counts, labels=sheets, autopct='%1.1f%%', startangle=90)
                axes[0,0].set_title('Distribution of Records by Sheet')
                colors = plt.cm.Set3(np.linspace(0, 1, len(sheets)))
                axes[0,1].bar(sheets, counts, color=colors)
                axes[0,1].set_title('Record Count by Sheet')
                axes[0,1].set_xlabel('Sheet Name')
                axes[0,1].set_ylabel('Number of Records')
                axes[0,1].tick_params(axis='x', rotation=45)

            if summary['departments_breakdown']:
                depts = list(summary['departments_breakdown'].keys())[:10]
                dept_counts = [summary['departments_breakdown'][d] for d in depts]
                colors = plt.cm.viridis(np.linspace(0, 1, len(depts)))
                axes[1,0].barh(depts, dept_counts, color=colors)
                axes[1,0].set_title('Top 10 Departments by Record Count')
                axes[1,0].set_xlabel('Number of Records')

            stats_text = f"""Database Statistics:

Total Records: {summary['total_records']:,}

Sheets: {len(summary['sheets_breakdown'])}

Departments: {len(summary['departments_breakdown'])}

Incident Types: {len(summary['incident_types_breakdown'])}"""

            axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            axes[1,1].set_title('Summary Statistics')
            axes[1,1].axis('off')

            plt.tight_layout()
            plt.savefig('./plots/database_overview.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Basic analytics plots created and saved to './plots/database_overview.png'")
        except Exception as e:
            logger.error(f"Error creating basic plots: {e}")
            print(f"Error creating plots: {e}")

    def export_data_summary(self, filename: str = "data_summary.json"):
        summary = self.get_data_summary()
        summary['export_timestamp'] = datetime.now().isoformat()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📄 Data summary exported to {filename}")

class RAGApplication:
    def __init__(self, gemini_api_key: str, openrouter_api_key: str):
        self.rag_system = RAGVectorSystem(gemini_api_key, openrouter_api_key)
        self.analyzer = DataAnalyzer(self.rag_system)
        self.chat_history = []

    def setup_database(self, json_file_path: str, record_limit: Optional[int] = None, recreate: bool = False):
        try:
            logger.info("Setting up vector database...")
            print("🔄 Creating vector database...")
            self.rag_system.create_vector_database(json_file_path,
                                                   limit_records=record_limit,
                                                   recreate=recreate)
            summary = self.analyzer.get_data_summary()
            if "error" not in summary:
                print("\n" + "="*60)
                print("✅ DATABASE SETUP COMPLETE")
                print("="*60)
                print(f"📊 Total records: {summary['total_records']:,}")
                print("\n📋 Records by sheet:")
                for sheet, count in summary['sheets_breakdown'].items():
                    print(f"   • {sheet}: {count:,}")
                if summary['departments_breakdown']:
                    print(f"\n🏢 Departments found: {len(summary['departments_breakdown'])}")
                print("="*60)
                print("🚀 Database is ready for queries!")
            else:
                print(f"❌ Error in database setup: {summary['error']}")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            print(f"❌ Database setup failed: {e}")

    def interactive_chat(self):
        print("\n" + "="*60)
        print("🤖 RAG INDUSTRIAL SAFETY CHAT SYSTEM")
        print("="*60)
        print("💡 Example queries:")
        print("   • 'What are the most common incident types?'")
        print("   • 'Show me equipment failures in 2022'")
        print("   • 'What safety violations occurred?'")
        print("   • 'plot: Create incident timeline by month'")
        print("\n📝 Commands:")
        print("   • Type 'quit' or 'exit' to end session")
        print("   • Type 'plot:' before requests for visualizations")
        print("   • Type 'help' for more options")
        print("   • Type 'summary' for database overview")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using the RAG system! Stay safe!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help(); continue
                elif user_input.lower() == 'summary':
                    self._show_summary(); continue
                elif user_input.lower() == 'history':
                    self._show_history(); continue

                self.chat_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': user_input,
                    'type': 'plot' if user_input.lower().startswith('plot:') else 'chat'
                })

                print("🤔 Analyzing data...")
                if user_input.lower().startswith('plot:'):
                    plot_request = user_input[5:].strip()
                    print("🎨 Generating visualization code...")
                    response = self.rag_system.generate_analytics_plots(plot_request)
                else:
                    print("🔍 Searching database...")
                    response = self.rag_system.chat_with_data(user_input)

                self.chat_history[-1]['response'] = response
                print(f"\n🤖 AI Assistant:")
                print("-" * 40)
                print(response)
                print("-" * 40)
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print(f"❌ Error: {str(e)}")

    def _show_help(self):
        print("""
        🆘 HELP - Available Commands:

        💬 Chat Commands:
           • Ask questions about incidents, safety, equipment, etc.
           • Example: "What incidents happened in PVC department?"

        📊 Plot Commands:
           • plot: [description] - Generate visualizations
           • Example: "plot: Show incident trends by month"

        🛠️ System Commands:
           • help - Show this help message
           • summary - Show database statistics
           • history - Show recent queries
           • quit/exit - End the session

        💡 Tips:
           • Be specific in your queries for better results
           • Ask about trends, patterns, departments, dates
           • Request specific incident types or safety categories
        """)

    def _show_summary(self):
        summary = self.analyzer.get_data_summary()
        if "error" in summary:
            print(f"❌ Error getting summary: {summary['error']}")
            return
        print(f"\n📊 DATABASE SUMMARY:")
        print(f"   Total Records: {summary['total_records']:,}")
        print(f"   Sheets: {len(summary['sheets_breakdown'])}")
        if summary['departments_breakdown']:
            print(f"   Departments: {len(summary['departments_breakdown'])}")
        print("\n📋 Sheet Breakdown:")
        for sheet, count in summary['sheets_breakdown'].items():
            print(f"   • {sheet}: {count:,}")

    def _show_history(self):
        if not self.chat_history:
            print("📝 No chat history available."); return
        print("\n📝 RECENT QUERIES:")
        for i, entry in enumerate(self.chat_history[-5:], 1):
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            query_type = entry['type'].upper()
            query = entry['query'][:50] + "..." if len(entry['query']) > 50 else entry['query']
            print(f"   {i}. [{timestamp}] ({query_type}) {query}")

    def run_sample_queries(self):
        sample_queries = [
            "What are the most common types of incidents in the database?",
            "Show me incidents related to equipment failure",
            "What safety violations occurred in 2022 and 2023?",
            "Which departments have the highest number of incidents?",
            "plot: Create a bar chart showing incidents by department"
        ]
        print("\n" + "="*60)
        print("🔍 RUNNING SAMPLE QUERIES")
        print("="*60)
        for i, query in enumerate(sample_queries, 1):
            print(f"\n📋 Query {i}: {query}")
            print("-" * 50)
            try:
                if query.startswith('plot:'):
                    response = self.rag_system.generate_analytics_plots(query[5:])
                    print("🎨 Generated visualization code:")
                else:
                    response = self.rag_system.chat_with_data(query)
                    print("🤖 Response:")
                print(response[:500] + "..." if len(response) > 500 else response)
                print("-" * 50)
            except Exception as e:
                logger.error(f"Error in sample query: {e}")
                print(f"❌ Error: {str(e)}")
            time.sleep(1)

# ===========================
# Env / Main
# ===========================
def load_environment_variables():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        gemini_key = os.getenv('GEMINI_API_KEY')
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if not gemini_key:
            logger.error("GEMINI_API_KEY not found in .env file")
            return None, None
        if not openrouter_key:
            logger.error("OPENROUTER_API_KEY not found in .env file")
            return None, None
        return gemini_key, openrouter_key
    except ImportError:
        logger.error("python-dotenv not installed. Please install it: pip install python-dotenv")
        return None, None
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return None, None

def create_sample_env_file():
    env_file_path = ".env"
    if not os.path.exists(env_file_path):
        sample_content = """# API Keys for RAG System
# Get your Gemini API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Get your OpenRouter API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: JSON file path
JSON_FILE_PATH=json_output/all_records_combined.json

# Optional: Gemini RPM cap (default 12)
GEMINI_RPM=12

# Optional: embedding progress controls
EMB_LOG_EVERY=25
EMB_CKPT_EVERY=500
# EMB_STOP_AFTER=0

# Optional: limit records for testing (e.g., 100). Omit or 0 = no limit
RECORD_LIMIT=0
"""
        with open(env_file_path, 'w') as f:
            f.write(sample_content)
        print(f"📝 Created sample .env file at {env_file_path}")
        print("Please edit the .env file and add your actual API keys.")
        return False
    return True

def main():
    print("🚀 RAG Industrial Safety Analysis System")
    print("="*60)
    if not create_sample_env_file():
        return
    print("🔑 Loading API keys from .env file...")
    GEMINI_API_KEY, OPENROUTER_API_KEY = load_environment_variables()
    if not GEMINI_API_KEY or not OPENROUTER_API_KEY:
        print("❌ Error: Could not load API keys from .env file")
        print("\n📋 Setup Instructions:")
        print("1. Edit the .env file in this directory")
        print("2. Add your Gemini API key from: https://makersuite.google.com/app/apikey")
        print("3. Add your OpenRouter API key from: https://openrouter.ai/keys")
        print("4. Save the file and run this script again")
        return

    JSON_FILE_PATH = os.getenv('JSON_FILE_PATH', "json_output/all_records_combined.json")

    # Optional: cap the number of NEW records for testing (e.g., 100)
    record_limit_env = os.getenv('RECORD_LIMIT', '').strip()
    RECORD_LIMIT = int(record_limit_env) if record_limit_env.isdigit() and int(record_limit_env) > 0 else None

    if not os.path.exists(JSON_FILE_PATH):
        print(f"⚠️  JSON file not found: {JSON_FILE_PATH}")
        print("Please run the Excel to JSON conversion script first")
        print("Or update the JSON_FILE_PATH in your .env file")
        return

    print("✅ Configuration loaded successfully!")
    try:
        print("🔧 Initializing RAG system...")
        app = RAGApplication(GEMINI_API_KEY, OPENROUTER_API_KEY)
        db_exists = app.rag_system.load_existing_database()
        if not db_exists:
            print("📊 No existing database found.")
            setup_db = input("Create new vector database? (y/n): ").lower().strip()
            if setup_db in ['y', 'yes']:
                app.setup_database(JSON_FILE_PATH, record_limit=RECORD_LIMIT, recreate=False)
            else:
                print("⚠️  Cannot proceed without a vector database.")
                return
        else:
            print("✅ Loaded existing vector database")
            recreate_ans = input("Recreate database from scratch (delete & rebuild)? (y/n): ").lower().strip()
            recreate_flag = recreate_ans in ['y', 'yes']
            app.setup_database(JSON_FILE_PATH, record_limit=RECORD_LIMIT, recreate=recreate_flag)

        while True:
            print("\n" + "="*60)
            print("🎯 MAIN MENU")
            print("="*60)
            print("1. 💬 Interactive chat with data")
            print("2. 🔍 Run sample queries")
            print("3. 📊 Create basic analytics plots")
            print("4. 📋 Show database summary")
            print("5. 📤 Export data summary")
            print("6. ❌ Exit")

            choice = input("\nEnter your choice (1-6): ").strip()
            if choice == '1':
                app.interactive_chat()
            elif choice == '2':
                app.run_sample_queries()
            elif choice == '3':
                app.analyzer.create_basic_plots()
            elif choice == '4':
                app._show_summary()
            elif choice == '5':
                filename = input("Enter filename for export (default: data_summary.json): ").strip() or "data_summary.json"
                app.analyzer.export_data_summary(filename)
            elif choice == '6':
                print("👋 Thank you for using the RAG System! Stay safe!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Critical error: {str(e)}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    required_packages = [
        "chromadb",
        "google-generativeai",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "requests",
        "python-dotenv"
    ]
    print("📦 Required packages:")
    for package in required_packages:
        print(f"   pip install {package}")

    print(f"\n🔧 Setup Instructions:")
    print("1. Install required packages (see above)")
    print("2. Create a .env file in this directory")
    print("3. Add your API keys to the .env file:")
    print("   GEMINI_API_KEY=your_actual_key")
    print("   OPENROUTER_API_KEY=your_actual_key")
    print("4. (Optional) Set GEMINI_RPM=12 (or your known limit)")
    print("5. (Optional) Tune EMB_LOG_EVERY / EMB_CKPT_EVERY")
    print("6. (Optional) Set RECORD_LIMIT=100 for a small NEW-records test run")
    print("\n🚀 Starting application...\n")

    try:
        import dotenv  # noqa
    except ImportError:
        print("❌ python-dotenv not found. Installing...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "python-dotenv"])
        except Exception as e:
            print(f"❌ Could not install python-dotenv: {e}")
            print("Please install manually: pip install python-dotenv")
            exit(1)

    main()
