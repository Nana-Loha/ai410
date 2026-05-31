"""
Lab 7.1 — Reproducible LlamaIndex Ingestion over Clinical Guideline Corpus
===========================================================================
Course: AI410 Spring 2026 | Bellevue College
Student: Pitchanan Lohavanichbutr

Run:
    uv run python ingest.py
    uv run python ingest.py --rebuild
"""

from __future__ import annotations

import json
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from xml.parsers.expat import model

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import MockEmbedding

BASE_DIR       = Path(__file__).resolve().parent
GUIDELINES_DIR = BASE_DIR / "data" / "guidelines"
STORAGE_DIR    = BASE_DIR / "storage" / "index"
MANIFEST_PATH  = STORAGE_DIR / "ingestion_manifest.json"

CHUNK_SIZE    = 256
CHUNK_OVERLAP = 40
EMBED_DIM     = 384

TOPIC_MAP = {
    "chest_pain":              "cardiology",
    "warfarin":                "pharmacology",
    "aspirin":                 "pharmacology",
    "hypertension":            "cardiology",
    "diabetes":                "endocrinology",
    "a1c":                     "endocrinology",
    "asthma":                  "pulmonology",
}

SOURCE_MAP = {
    "chest_pain":              "ACC/AHA 2021 Chest Pain Guideline",
    "warfarin_aspirin":        "FDA Drug Label + ISMP + ACC/AHA 2019",
    "hypertension_followup":   "ACC/AHA 2017 + JNC 8 + ADA 2023",
    "diabetes_a1c_management": "ADA Standards of Medical Care 2023",
    "asthma_controller_stepup":"GINA 2023 + NAEPP EPR-3",
}


def configure_embeddings() -> str:
    model = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding(model_name=model)
        print(f"[Embed] HuggingFace Local: {model}")
        return "huggingface_local"
    except ImportError:
        raise ImportError("Run: uv add llama-index-embeddings-huggingface")


def enrich_metadata(doc, file_name: str) -> None:
    stem   = Path(file_name).stem.lower()
    topic  = next((t for k, t in TOPIC_MAP.items() if k in stem), "general")
    source = SOURCE_MAP.get(stem, "Unknown Guideline")
    doc_id = hashlib.md5(file_name.encode()).hexdigest()[:8]
    doc.metadata.update({"topic": topic, "guideline_source": source, "doc_id": doc_id, "file_name": file_name})


def ingest(force_rebuild: bool = False) -> dict:
    if not GUIDELINES_DIR.exists():
        raise FileNotFoundError(f"Missing: {GUIDELINES_DIR}")
    doc_files = sorted(GUIDELINES_DIR.glob("*.txt"))
    if not doc_files:
        raise ValueError("No .txt files in data/guidelines/")

    if not force_rebuild and (STORAGE_DIR / "docstore.json").exists():
        print("[Ingest] Loading existing index (use --rebuild to force)")
        return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    print(f"[Ingest] Building index from {len(doc_files)} files...")
    backend = configure_embeddings()

    docs = SimpleDirectoryReader(input_dir=str(GUIDELINES_DIR), required_exts=[".txt"]).load_data()
    for doc in docs:
        enrich_metadata(doc, doc.metadata.get("file_name", "unknown"))
        print(f"  → {doc.metadata['file_name']} | topic={doc.metadata['topic']}")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes    = splitter.get_nodes_from_documents(docs)
    print(f"[Ingest] {len(nodes)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    manifest = {
        "timestamp_utc":     datetime.now(UTC).isoformat(),
        "embedding_backend": backend,
        "chunk_size":        CHUNK_SIZE,
        "chunk_overlap":     CHUNK_OVERLAP,
        "doc_count":         len(doc_files),
        "node_count":        len(nodes),
        "documents":         [p.name for p in doc_files],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Ingest] ✅ Done — manifest: {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    import sys
    manifest = ingest(force_rebuild="--rebuild" in sys.argv)
    print(json.dumps(manifest, indent=2))