"""Vector-memory backends: Chroma (local) and pgvector (prod).

Both share the same embedding function, which defaults to OpenAI's
``text-embedding-3-small``. Override by passing ``embed_fn`` to the backend
constructor — e.g. to use a local sentence-transformers model.

Example:
    >>> import asyncio, tempfile
    >>> from agent_kit.memory.vector import ChromaVectorMemory
    >>> async def demo() -> None:
    ...     with tempfile.TemporaryDirectory() as d:
    ...         mem = ChromaVectorMemory(
    ...             path=d,
    ...             embed_fn=lambda t: [float(len(t))] * 4,
    ...         )
    ...         await mem.upsert(id="a", text="hello", metadata={})
    ...         hits = await mem.search("hello", k=1)
    ...         assert hits[0].id == "a"
    >>> asyncio.run(demo())
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_kit.memory.base import VectorMemory, VectorRecord

EmbedFn = Callable[[str], list[float] | Awaitable[list[float]]]


async def _call_embed(fn: EmbedFn, text: str) -> list[float]:
    """Call ``fn`` and await if it returns a coroutine."""
    result = fn(text)
    if asyncio.iscoroutine(result):
        return list(await result)
    return list(result)  # type: ignore[arg-type]


async def _openai_embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Default embedding function using OpenAI's ``text-embedding-3-small``."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.embeddings.create(input=text, model=model)
    return [float(x) for x in response.data[0].embedding]


class ChromaVectorMemory(VectorMemory):
    """Local vector store backed by a persistent Chroma collection."""

    def __init__(
        self,
        *,
        path: str | Path = "./.agent_kit/chroma",
        collection: str = "agent_kit",
        embed_fn: EmbedFn | None = None,
    ) -> None:
        """Open (or create) a Chroma persistent client and collection."""
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_fn: EmbedFn = embed_fn or _openai_embed

    async def embed(self, text: str) -> list[float]:
        """Compute the embedding vector for ``text``."""
        return await _call_embed(self._embed_fn, text)

    async def upsert(
        self,
        *,
        id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a record keyed by ``id``."""
        embedding = await self.embed(text)
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[id],
            documents=[text],
            embeddings=[embedding],  # type: ignore[arg-type]
            metadatas=[metadata or {"_": True}],
        )

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """Return the ``k`` nearest records to ``query``."""
        embedding = await self.embed(query)
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=k,
            where=where,
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        records: list[VectorRecord] = []
        for i, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            # Chroma returns cosine *distance*; convert to similarity in [0, 1].
            score = max(0.0, 1.0 - float(dist))
            records.append(
                VectorRecord(
                    id=i,
                    text=doc,
                    embedding=[],
                    metadata=dict(meta) if meta else {},
                    score=score,
                )
            )
        return records

    async def delete(self, ids: list[str]) -> None:
        """Delete records by id."""
        if not ids:
            return
        await asyncio.to_thread(self._collection.delete, ids=ids)


_PG_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS agent_kit_vectors (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(%(dim)s) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_kit_vectors_embedding_idx
    ON agent_kit_vectors USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class PgVectorMemory(VectorMemory):
    """pgvector-backed vector store for production."""

    def __init__(
        self,
        *,
        dsn: str,
        embed_fn: EmbedFn | None = None,
        dimension: int = 1536,
    ) -> None:
        """Open a pool and ensure the schema exists."""
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(dsn, min_size=1, max_size=4, open=True)
        self._embed_fn: EmbedFn = embed_fn or _openai_embed
        self._dim = dimension
        with self._pool.connection() as conn:
            conn.execute(_PG_DDL % {"dim": dimension})
            conn.commit()

    async def embed(self, text: str) -> list[float]:
        """Compute the embedding vector for ``text``."""
        return await _call_embed(self._embed_fn, text)

    async def upsert(
        self,
        *,
        id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a record keyed by ``id``."""
        embedding = await self.embed(text)
        vector = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
        await asyncio.to_thread(self._upsert_sync, id, text, vector, metadata or {})

    def _upsert_sync(self, id_: str, text: str, vector: str, metadata: dict[str, Any]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_kit_vectors (id, text, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                    SET text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """,
                (id_, text, vector, json.dumps(metadata, default=str)),
            )
            conn.commit()

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorRecord]:
        """Return the ``k`` nearest records to ``query``."""
        embedding = await self.embed(query)
        vector = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
        rows = await asyncio.to_thread(self._search_sync, vector, k, where)
        return [
            VectorRecord(
                id=row[0],
                text=row[1],
                embedding=[],
                metadata=row[2] or {},
                score=max(0.0, 1.0 - float(row[3])),
            )
            for row in rows
        ]

    def _search_sync(
        self, vector: str, k: int, where: dict[str, Any] | None
    ) -> list[tuple[Any, ...]]:
        where_clause = ""
        params: list[Any] = [vector, k]
        if where:
            where_clause = "WHERE metadata @> %s::jsonb"
            params.insert(0, json.dumps(where))
        query = f"""
            SELECT id, text, metadata,
                   embedding <=> %s::vector AS distance
            FROM agent_kit_vectors
            {where_clause}
            ORDER BY distance ASC
            LIMIT %s
        """
        with self._pool.connection() as conn:
            cur = conn.execute(query, params)
            return list(cur.fetchall())

    async def delete(self, ids: list[str]) -> None:
        """Delete records by id."""
        if not ids:
            return
        await asyncio.to_thread(self._delete_sync, ids)

    def _delete_sync(self, ids: list[str]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM agent_kit_vectors WHERE id = ANY(%s)",
                (ids,),
            )
            conn.commit()


__all__ = ["ChromaVectorMemory", "PgVectorMemory"]
