import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .embeddings import E5EmbeddingFunction

logger = logging.getLogger(__name__)


@dataclass
class ToSSearchResult:
    section_title: str
    section_content: str
    document_title: str
    category: str
    parent_content: str
    effective_date: str
    source_url: str
    score: float
    id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_title": self.section_title,
            "section_content": self.section_content,
            "document_title": self.document_title,
            "category": self.category,
            "parent_content": self.parent_content,
            "effective_date": self.effective_date,
            "source_url": self.source_url,
            "score": self.score,
            "id": self.id,
        }


class ToSChunker:
    SECTION_PATTERN = re.compile(r"^제\s*(\d+)\s*조\s*[\(（]([^)）]+)[\)）]")
    SUBSECTION_PATTERN = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]|^\d+\.|^[가나다라마바사][\.\)]")

    def __init__(self, include_parent_context: bool = True, max_chunk_length: int = 2000):
        self.include_parent_context = include_parent_context
        self.max_chunk_length = max_chunk_length

    def chunk_document(self, tos_item: dict[str, Any]) -> list[dict[str, Any]]:
        chunks = []
        document_title = tos_item.get("title", "")
        category = tos_item.get("category", "")
        effective_date = tos_item.get("effective_date", "")
        source_url = tos_item.get("source_url", "")
        crawled_at = tos_item.get("crawled_at", datetime.now().isoformat())

        sections = tos_item.get("sections", [])
        if sections:
            for section in sections:
                chunk = self._create_section_chunk(
                    section=section,
                    document_title=document_title,
                    category=category,
                    effective_date=effective_date,
                    source_url=source_url,
                    crawled_at=crawled_at,
                    full_content=tos_item.get("content", ""),
                )
                if chunk:
                    chunks.append(chunk)

        if not chunks:
            content = tos_item.get("content", "")
            if content:
                parsed_sections = self._parse_sections_from_content(content)
                for section in parsed_sections:
                    chunk = self._create_section_chunk(
                        section=section,
                        document_title=document_title,
                        category=category,
                        effective_date=effective_date,
                        source_url=source_url,
                        crawled_at=crawled_at,
                        full_content=content,
                    )
                    if chunk:
                        chunks.append(chunk)

        if not chunks and tos_item.get("content"):
            chunks.append(
                {
                    "section_title": document_title,
                    "section_content": tos_item["content"][: self.max_chunk_length],
                    "document_title": document_title,
                    "category": category,
                    "parent_content": "",
                    "effective_date": effective_date,
                    "source_url": source_url,
                    "crawled_at": crawled_at,
                }
            )

        return chunks

    def _create_section_chunk(
        self,
        section: dict[str, str],
        document_title: str,
        category: str,
        effective_date: str,
        source_url: str,
        crawled_at: str,
        full_content: str,
    ) -> dict[str, Any] | None:
        title = section.get("title", "").strip()
        content = section.get("content", "").strip()

        if not title and not content:
            return None

        combined = f"{title}\n{content}" if content else title
        if len(combined) < 10:
            return None

        parent_content = ""
        if self.include_parent_context and full_content:
            parent_content = self._get_parent_context(title, full_content)

        return {
            "section_title": title,
            "section_content": content[: self.max_chunk_length],
            "document_title": document_title,
            "category": category,
            "parent_content": parent_content[:500],
            "effective_date": effective_date,
            "source_url": source_url,
            "crawled_at": crawled_at,
        }

    def _parse_sections_from_content(self, content: str) -> list[dict[str, str]]:
        sections = []
        lines = content.split("\n")

        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = self.SECTION_PATTERN.match(line)
            if match:
                if current_section:
                    sections.append(
                        {
                            "title": current_section,
                            "content": "\n".join(current_content),
                        }
                    )

                current_section = line
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections.append(
                {
                    "title": current_section,
                    "content": "\n".join(current_content),
                }
            )

        return sections

    def _get_parent_context(self, section_title: str, full_content: str) -> str:
        match = self.SECTION_PATTERN.match(section_title)
        if not match:
            return ""

        section_num = int(match.group(1))
        if section_num <= 1:
            return ""

        prev_section_pattern = rf"제\s*{section_num - 1}\s*조\s*[\(（][^)）]+[\)）]"
        prev_match = re.search(prev_section_pattern, full_content)
        if prev_match:
            start = prev_match.start()
            end = min(start + 500, len(full_content))
            return full_content[start:end]

        return ""


class ToSVectorStore:
    COLLECTION_NAME = "tos"

    def __init__(
        self,
        persist_directory: str | Path = "data/vectordb/tos",
        embedding_model: str | None = None,
        device: str | None = None,
        enable_hybrid_search: bool = False,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_fn = E5EmbeddingFunction(
            model_name=embedding_model,
            device=device,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

        self.chunker = ToSChunker()

        # Hybrid search (lazy initialization)
        self._enable_hybrid_search = enable_hybrid_search
        self._hybrid_search = None

        logger.info(
            f"ToS Vector Store initialized. Collection '{self.COLLECTION_NAME}' "
            f"has {self.collection.count()} documents. "
            f"Hybrid search: {'enabled' if enable_hybrid_search else 'disabled'}"
        )

    @property
    def hybrid_search(self):
        """Get or create hybrid search instance."""
        if self._hybrid_search is None and self._enable_hybrid_search:
            from src.tos_search import ToSHybridSearch, ToSRuleMatcher, TripletStore

            triplet_path = self.persist_directory / "triplets.json"
            self._hybrid_search = ToSHybridSearch(
                vector_store=self,
                rule_matcher=ToSRuleMatcher(),
                triplet_store=TripletStore(persist_path=triplet_path),
            )
        return self._hybrid_search

    def add_tos_document(self, tos_item: dict[str, Any]) -> list[str]:
        chunks = self.chunker.chunk_document(tos_item)
        return self._add_chunks(chunks)

    def add_tos_batch(
        self,
        tos_items: list[dict[str, Any]],
        batch_size: int = 50,
    ) -> list[str]:
        all_chunks = []
        for item in tos_items:
            chunks = self.chunker.chunk_document(item)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks to add: {len(all_chunks)}")
        return self._add_chunks(all_chunks, batch_size)

    def _add_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 50,
    ) -> list[str]:
        all_ids = []
        seen_ids: set[str] = set()
        skipped_duplicates = 0
        skipped_existing = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = []
            documents = []
            metadatas = []

            for chunk in batch:
                section_title = chunk.get("section_title", "")
                section_content = chunk.get("section_content", "")

                embed_text = f"{section_title}\n{section_content}".strip()
                if not embed_text:
                    continue

                chunk_id = hashlib.sha256(
                    f"{chunk['document_title']}_{section_title}_{section_content[:100]}".encode()
                ).hexdigest()[:16]

                if chunk_id in seen_ids:
                    skipped_duplicates += 1
                    continue

                seen_ids.add(chunk_id)

                ids.append(chunk_id)
                documents.append(embed_text)
                metadatas.append(
                    {
                        "section_title": section_title,
                        "section_content": section_content[:1000],
                        "document_title": chunk.get("document_title", ""),
                        "category": chunk.get("category", ""),
                        "parent_content": chunk.get("parent_content", ""),
                        "effective_date": chunk.get("effective_date", ""),
                        "source_url": chunk.get("source_url", ""),
                        "crawled_at": chunk.get("crawled_at", ""),
                    }
                )

            if ids:
                existing = self.collection.get(ids=ids)
                existing_ids = set(existing["ids"])
                skipped_existing += len(existing_ids)
                filtered = [
                    (cid, doc, meta)
                    for cid, doc, meta in zip(ids, documents, metadatas, strict=False)
                    if cid not in existing_ids
                ]
                if filtered:
                    filtered_ids, filtered_docs, filtered_metas = zip(*filtered, strict=False)
                    self.collection.add(
                        ids=list(filtered_ids),
                        documents=list(filtered_docs),
                        metadatas=list(filtered_metas),
                    )
                    all_ids.extend(filtered_ids)

            logger.info(f"Added batch {i // batch_size + 1}: {len(ids)} chunks")

        logger.info(f"Total added: {len(all_ids)} chunks")
        if skipped_duplicates:
            logger.warning(f"Skipped duplicate chunk IDs: {skipped_duplicates}")
        if skipped_existing:
            logger.warning(f"Skipped existing chunk IDs: {skipped_existing}")
        return all_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[ToSSearchResult]:
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        query_embedding = self.embedding_fn.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,  # type: ignore[arg-type]
            include=["documents", "metadatas", "distances"],  # type: ignore[arg-type]
        )

        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance

                if score_threshold is not None and score < score_threshold:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                search_results.append(
                    ToSSearchResult(
                        section_title=str(metadata.get("section_title", "")),
                        section_content=str(metadata.get("section_content", "")),
                        document_title=str(metadata.get("document_title", "")),
                        category=str(metadata.get("category", "")),
                        parent_content=str(metadata.get("parent_content", "")),
                        effective_date=str(metadata.get("effective_date", "")),
                        source_url=str(metadata.get("source_url", "")),
                        score=score,
                        id=chunk_id,
                    )
                )

        return search_results

    def get_by_id(self, chunk_id: str) -> ToSSearchResult | None:
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],  # type: ignore[arg-type]
        )

        if result["ids"]:
            metadata = result["metadatas"][0] if result["metadatas"] else {}

            return ToSSearchResult(
                section_title=str(metadata.get("section_title", "")),
                section_content=str(metadata.get("section_content", "")),
                document_title=str(metadata.get("document_title", "")),
                category=str(metadata.get("category", "")),
                parent_content=str(metadata.get("parent_content", "")),
                effective_date=str(metadata.get("effective_date", "")),
                source_url=str(metadata.get("source_url", "")),
                score=1.0,
                id=chunk_id,
            )

        return None

    def delete(self, chunk_id: str) -> None:
        self.collection.delete(ids=[chunk_id])
        logger.debug(f"Deleted ToS chunk: {chunk_id}")

    def count(self) -> int:
        return self.collection.count()

    def load_from_json(self, json_path: str | Path) -> list[str]:
        json_path = Path(json_path)

        data = json.loads(json_path.read_text(encoding="utf-8"))

        logger.info(f"Loading {len(data)} ToS documents from {json_path}")
        return self.add_tos_batch(data)

    def clear(self) -> None:
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        # Clear hybrid search state
        if self._hybrid_search:
            self._hybrid_search.triplet_store.clear()
        logger.info("Cleared all ToS chunks")

    def search_hybrid(
        self,
        query: str,
        n_results: int = 5,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search using hybrid method (vector + rule + triplet).

        Args:
            query: Search query
            n_results: Number of results
            category_filter: Optional category filter

        Returns:
            List of search results with combined scores
        """
        if not self._enable_hybrid_search or not self.hybrid_search:
            # Fallback to regular search
            results = self.search(query, n_results, category_filter)
            return [r.to_dict() for r in results]

        hybrid_results = self.hybrid_search.search(
            query=query,
            n_results=n_results,
            category_filter=category_filter,
        )
        return [r.to_dict() for r in hybrid_results]

    def build_triplet_index(self) -> int:
        """Build triplet index from stored documents.

        Returns:
            Number of triplets extracted
        """
        if not self._enable_hybrid_search:
            logger.warning("Hybrid search is not enabled")
            return 0

        if self.hybrid_search:
            return self.hybrid_search.build_triplet_index()
        return 0
