"""LlamaIndex RAG pipeline with reproducible ingestion for Week 7 labs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter


DISCLAIMER = "Not a substitute for professional medical advice"


@dataclass
class PipelineConfig:
    chunk_size: int = 256
    chunk_overlap: int = 40
    similarity_top_k: int = 3
    enable_metadata_filter: bool = False
    enable_lexical_rerank: bool = False
    embedding_backend: str = "huggingface_local"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    source_id: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    confidence: float


class ClinicalRAGPipeline:
    def __init__(self, base_dir: Path, config: PipelineConfig | None = None) -> None:
        self.base_dir = base_dir
        self.guidelines_dir = base_dir / "data" / "guidelines"
        self.storage_dir = base_dir / "storage" / "index"
        self.config = config or PipelineConfig()

    def _configure_embeddings(self) -> None:
        backend = self.config.embedding_backend.lower().strip()

        if backend == "mock":
            Settings.embed_model = MockEmbedding(embed_dim=384)
            return

        if backend in ("huggingface_local", "huggingface_api"):
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=self.config.embedding_model_name
                )
                return
            except ImportError as exc:
                raise ImportError(
                    "Run: uv add llama-index-embeddings-huggingface"
                ) from exc

        raise ValueError(
            "Unsupported embedding_backend. Use 'mock' or 'huggingface_local'."
        )

    @staticmethod
    def _topic_from_filename(file_name: str) -> str:
        stem = Path(file_name).stem.lower()
        if "chest_pain" in stem:
            return "cardiology"
        if "warfarin" in stem or "interaction" in stem:
            return "pharmacology"
        if "hypertension" in stem:
            return "cardiology"
        if "diabetes" in stem or "a1c" in stem:
            return "endocrinology"
        if "asthma" in stem:
            return "pulmonology"
        return "general"

    @staticmethod
    def _query_topic(query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["chest pain", "hypertension", "cardiac", "heart", "stemi", "troponin", "ecg", "bp", "blood pressure"]):
            return "cardiology"
        if any(k in q for k in ["warfarin", "aspirin", "drug", "medication", "interaction", "inr", "anticoagul"]):
            return "pharmacology"
        if any(k in q for k in ["diabetes", "a1c", "glucose", "insulin", "metformin", "sglt", "glp"]):
            return "endocrinology"
        if any(k in q for k in ["asthma", "inhaler", "wheeze", "fev1", "gina", "laba", "ics"]):
            return "pulmonology"
        return "general"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.strip(".,:;!?()[]{}\"'").lower() for t in text.split() if t.strip()}

    def ingest(self) -> dict:
        if not self.guidelines_dir.exists():
            raise FileNotFoundError(f"Missing guideline directory: {self.guidelines_dir}")

        doc_files = sorted(self.guidelines_dir.glob("*.txt"))
        if not doc_files:
            raise ValueError("No guideline .txt files found for ingestion")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._configure_embeddings()

        splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        docs = SimpleDirectoryReader(
            input_dir=str(self.guidelines_dir), required_exts=[".txt"]
        ).load_data()
        for doc in docs:
            file_name = str(doc.metadata.get("file_name", "unknown"))
            doc.metadata["topic"] = self._topic_from_filename(file_name)
        nodes = splitter.get_nodes_from_documents(docs)

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=str(self.storage_dir))

        manifest = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "guideline_dir": str(self.guidelines_dir),
            "storage_dir": str(self.storage_dir),
            "doc_count": len(doc_files),
            "node_count": len(nodes),
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "similarity_top_k": self.config.similarity_top_k,
                "enable_metadata_filter": self.config.enable_metadata_filter,
                "enable_lexical_rerank": self.config.enable_lexical_rerank,
                "embedding_backend": self.config.embedding_backend,
                "embedding_model_name": self.config.embedding_model_name,
            },
            "documents": [p.name for p in doc_files],
        }
        (self.storage_dir / "ingestion_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def _load_or_build_index(self) -> VectorStoreIndex:
        if Settings.embed_model is None or isinstance(Settings.embed_model, MockEmbedding):
            self._configure_embeddings()
        if (self.storage_dir / "docstore.json").exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            return load_index_from_storage(storage_context)
        self.ingest()
        storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
        return load_index_from_storage(storage_context)
    
    def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult(chunks=[], confidence=0.0)

        index = self._load_or_build_index()
        retriever = index.as_retriever(similarity_top_k=self.config.similarity_top_k)
        nodes = retriever.retrieve(query)

        if self.config.enable_metadata_filter:
            target_topic = self._query_topic(query)
            if target_topic != "general":
                nodes = [n for n in nodes if str(n.metadata.get("topic", "")) == target_topic]

        chunks: list[RetrievedChunk] = []
        for node in nodes:
            source_id = str(node.metadata.get("file_name", "unknown")) if node.metadata else "unknown"
            chunks.append(RetrievedChunk(
                source_id=source_id,
                text=node.text,
                score=float(node.score or 0.0),
            ))

        if self.config.enable_lexical_rerank and chunks:
            q_tokens = self._tokenize(query)
            reranked = []
            for chunk in chunks:
                c_tokens = self._tokenize(chunk.text)
                overlap = len(q_tokens.intersection(c_tokens))
                lexical = overlap / max(len(q_tokens), 1)
                combined = 0.7 * chunk.score + 0.3 * lexical
                reranked.append(RetrievedChunk(
                    source_id=chunk.source_id, text=chunk.text, score=combined
                ))
            reranked.sort(key=lambda c: c.score, reverse=True)
            chunks = reranked[: self.config.similarity_top_k]

        confidence = max((c.score for c in chunks), default=0.0)
        return RetrievalResult(chunks=chunks, confidence=confidence)


def build_response_with_citations(chunks: list[RetrievedChunk]) -> tuple[str, list[str], float]:
    if not chunks:
        return (
            f"I could not find enough guideline evidence. Please consult a licensed clinician.\n\nDisclaimer: {DISCLAIMER}.",
            [], 0.0
        )
    citations = [c.source_id for c in chunks]
    confidence = max(c.score for c in chunks)
    lines = [f"- [{c.source_id}] {c.text[:220].strip()}" for c in chunks]
    response = (
        "Based on retrieved guideline evidence:\n\n"
        + "\n".join(lines)
        + f"\n\nDisclaimer: {DISCLAIMER}."
    )
    return response, citations, confidence


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    pipeline = ClinicalRAGPipeline(base_dir=base_dir)
    manifest = pipeline.ingest()
    print(json.dumps(manifest, indent=2))