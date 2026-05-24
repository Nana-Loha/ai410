"""LlamaIndex RAG pipeline with reproducible ingestion for Week 7 labs."""

from __future__ import annotations

import json
import os
import importlib
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
    """Configurable retrieval parameters used for experiments."""

    chunk_size: int = 256
    chunk_overlap: int = 40
    similarity_top_k: int = 3
    enable_metadata_filter: bool = False
    enable_lexical_rerank: bool = False
    embedding_backend: str = "mock"  # mock | huggingface_api
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    """Normalized retrieval hit used by node and evaluation code."""

    source_id: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    """Full retrieval payload."""

    chunks: list[RetrievedChunk]
    confidence: float


class ClinicalRAGPipeline:
    """Build and query a local RAG index over guideline files."""

    def __init__(self, base_dir: Path, config: PipelineConfig | None = None) -> None:
        self.base_dir = base_dir
        self.guidelines_dir = base_dir / "data" / "guidelines"
        self.storage_dir = base_dir / "storage" / "index"
        self.config = config or PipelineConfig()

    def _configure_embeddings(self) -> None:
        """Set embedding model backend for index build and retrieval."""
        backend = self.config.embedding_backend.lower().strip()

        if backend == "mock":
            Settings.embed_model = MockEmbedding(embed_dim=384)
            return

        if backend == "huggingface_api":
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            if not token:
                raise ValueError(
                    "Hugging Face token not found. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
                )

            try:
                module = importlib.import_module("llama_index.embeddings.huggingface_api")
                huggingface_embedding_cls = getattr(module, "HuggingFaceInferenceAPIEmbedding")
            except Exception as exc:
                raise ImportError(
                    "Missing Hugging Face API embedding package. Install with: "
                    "uv add llama-index-embeddings-huggingface-api"
                ) from exc

            Settings.embed_model = huggingface_embedding_cls(
                model_name=self.config.embedding_model_name,
                token=token,
            )
            return

        raise ValueError(
            "Unsupported embedding_backend. Use 'mock' or 'huggingface_api'."
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
        if any(k in q for k in ["chest pain", "hypertension", "cardiac", "heart"]):
            return "cardiology"
        if any(k in q for k in ["warfarin", "aspirin", "drug", "medication", "interaction"]):
            return "pharmacology"
        if any(k in q for k in ["diabetes", "a1c", "glucose"]):
            return "endocrinology"
        if any(k in q for k in ["asthma", "inhaler", "wheeze"]):
            return "pulmonology"
        return "general"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.strip(".,:;!?()[]{}\"'").lower() for t in text.split() if t.strip()}

    def ingest(self) -> dict:
        """Rebuild index from the guideline corpus and write manifest."""
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

        docs = SimpleDirectoryReader(input_dir=str(self.guidelines_dir), required_exts=[".txt"]).load_data()
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
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return manifest

    def _load_or_build_index(self) -> VectorStoreIndex:
        """Load persisted index or rebuild if missing."""
        self._configure_embeddings()
        if (self.storage_dir / "docstore.json").exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            return load_index_from_storage(storage_context)

        self.ingest()
        storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
        return load_index_from_storage(storage_context)

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve top-k chunks for a query."""
        if not query.strip():
            return RetrievalResult(chunks=[], confidence=0.0)

        index = self._load_or_build_index()
        retriever = index.as_retriever(similarity_top_k=self.config.similarity_top_k)
        nodes = retriever.retrieve(query)

        if self.config.enable_metadata_filter:
            target_topic = self._query_topic(query)
            if target_topic != "general":
                nodes = [
                    n for n in nodes if str(n.metadata.get("topic", "")) == target_topic
                ]

        chunks: list[RetrievedChunk] = []
        for node in nodes:
            source_id = "unknown"
            if node.metadata and "file_name" in node.metadata:
                source_id = str(node.metadata["file_name"])
            score = float(node.score or 0.0)
            chunks.append(
                RetrievedChunk(
                    source_id=source_id,
                    text=node.text,
                    score=score,
                )
            )

        if self.config.enable_lexical_rerank and chunks:
            q_tokens = self._tokenize(query)
            reranked: list[RetrievedChunk] = []
            for chunk in chunks:
                c_tokens = self._tokenize(chunk.text)
                overlap = len(q_tokens.intersection(c_tokens))
                lexical = overlap / max(len(q_tokens), 1)
                combined = 0.7 * chunk.score + 0.3 * lexical
                reranked.append(
                    RetrievedChunk(
                        source_id=chunk.source_id,
                        text=chunk.text,
                        score=combined,
                    )
                )
            reranked.sort(key=lambda c: c.score, reverse=True)
            chunks = reranked[: self.config.similarity_top_k]

        confidence = max((c.score for c in chunks), default=0.0)
        return RetrievalResult(chunks=chunks, confidence=confidence)


def build_response_with_citations(chunks: list[RetrievedChunk]) -> tuple[str, list[str], float]:
    """Create grounded response text from retrieved evidence."""
    if not chunks:
        response = (
            "I could not find enough guideline evidence for this question. "
            "Please consult a licensed clinician.\n\n"
            f"Disclaimer: {DISCLAIMER}."
        )
        return response, [], 0.0

    citations = [c.source_id for c in chunks]
    confidence = max(c.score for c in chunks)
    lines = [f"- [{c.source_id}] {c.text[:220].strip()}" for c in chunks]

    response = (
        "Based on retrieved guideline evidence, here are relevant findings:\n\n"
        + "\n".join(lines)
        + "\n\n"
        + "Use this as decision support only. "
        + f"Disclaimer: {DISCLAIMER}."
    )
    return response, citations, confidence


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pipeline = ClinicalRAGPipeline(base_dir=base_dir)
    manifest = pipeline.ingest()
    print("Ingestion complete")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
