"""Stage 7 (optional): Post-processing — chunking, embeddings, enrichment.

Provides:
  - chunk_text: character-based text chunking with overlap
  - Chunker / ChunkConfig: document chunking from predictions or files
  - Embedder / EmbedConfig: semantic embeddings via sentence-transformers
  - Enricher / EnrichConfig / EnrichmentResult: LLM-based document enrichment
  - run_chunking / run_embedding / run_enrichment: high-level entry points
  - ImageKnowledgeGraph / EntityInfo / KGQueryResult: cross-document knowledge graph

Lazy imports are used for sentence-transformers and httpx to avoid heavy
dependencies at package load time.
"""

from __future__ import annotations

from vit_curator.post.chunk import ChunkConfig, Chunker, chunk_text, run_chunking
from vit_curator.post.knowledge_graph import EntityInfo, ImageKnowledgeGraph, KGQueryResult

__all__ = [
    "ChunkConfig",
    "ChunkTextResult",
    "Chunker",
    "EmbedConfig",
    "Embedder",
    "EnrichConfig",
    "Enricher",
    "EnrichmentResult",
    "EntityInfo",
    "ImageKnowledgeGraph",
    "KGQueryResult",
    "chunk_text",
    "run_chunking",
    "run_embedding",
    "run_enrichment",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy imports for modules requiring optional dependencies."""
    if name in ("EmbedConfig", "Embedder", "run_embedding"):
        from vit_curator.post import embed as _mod  # noqa: PLC0415

        return getattr(_mod, name)
    if name in ("Enricher", "EnrichmentResult", "run_enrichment"):
        from vit_curator.post import enrich as _mod  # noqa: PLC0415

        return getattr(_mod, name)
    if name == "EnrichConfig":
        from vit_curator.config import EnrichConfig  # noqa: PLC0415

        return EnrichConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
