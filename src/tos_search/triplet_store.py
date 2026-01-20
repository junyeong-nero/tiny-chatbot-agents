"""Triplet (Subject-Predicate-Object) extraction and storage for ToS search.

Provides:
- Triplet extraction from ToS text using pattern matching
- Simple JSON-based triplet storage
- Search by subject, object, or relation
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """A Subject-Predicate-Object triplet."""

    subject: str
    predicate: str
    obj: str  # 'object' is reserved
    source_chunk_id: str = ""
    confidence: float = 1.0

    @property
    def id(self) -> str:
        """Generate unique ID from triplet content."""
        content = f"{self.subject}:{self.predicate}:{self.obj}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Triplet":
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            obj=data["object"],
            source_chunk_id=data.get("source_chunk_id", ""),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class TripletSearchResult:
    """Result from triplet search."""

    triplet: Triplet
    match_type: str  # 'subject', 'predicate', 'object'
    score: float
    source_chunk_id: str


class TripletExtractor:
    """Extract triplets from ToS text using pattern matching.

    Focuses on common ToS patterns like:
    - "회사는 X를 할 수 있다" → (회사, 가능, X)
    - "고객은 X를 해야 한다" → (고객, 의무, X)
    - "X는 Y로 한다" → (X, 정의, Y)
    """

    # Pattern definitions for Korean ToS
    PATTERNS = [
        # 주체 + 조동사 패턴
        (
            r"(?P<subj>회사|당사|고객|이용자|납부자|가입자|사용자|투자자)(?:는|은|가|이)\s+"
            r"(?P<obj>[^을를]+)[을를]\s+"
            r"(?P<pred>할 수 있|하여야 하|해야 하|거부할 수 있|거절할 수 있|제한할 수 있)",
            lambda m: (m.group("subj"), _normalize_predicate(m.group("pred")), m.group("obj").strip()),
        ),
        # 정의 패턴: "X란 Y를 말한다" or "X라 함은 Y를 의미한다"
        (
            r'"?(?P<term>[^"]+)"?(?:이)?라\s*(?:함은|란)\s+(?P<def>[^을를]+)[을를]?\s*(?:말한다|의미한다|뜻한다)',
            lambda m: (m.group("term").strip(), "정의", m.group("def").strip()),
        ),
        # 조건 패턴: "X하는 경우 Y한다"
        (
            r"(?P<cond>[가-힣\s]+)(?:하는|한)\s*경우(?:에)?[는는,]?\s*(?P<subj>회사|고객|이용자)?(?:는|은)?\s*"
            r"(?P<action>[가-힣\s]{2,20})(?:합니다|한다|됩니다|된다)",
            lambda m: (
                m.group("subj") or "회사",
                m.group("cond").strip() + "_시",
                m.group("action").strip(),
            ),
        ),
        # 금지 패턴: "X하여서는 아니 된다"
        (
            r"(?P<subj>회사|고객|이용자)?(?:는|은)?\s*(?P<action>[가-힣\s]+)[을를]?\s*"
            r"(?:하여서는|해서는)\s*(?:아니|안)\s*(?:된다|됩니다)",
            lambda m: (m.group("subj") or "행위자", "금지", m.group("action").strip()),
        ),
        # 책임 패턴: "X에 대하여 책임을 지지 아니한다"
        (
            r"(?P<subj>회사|당사)(?:는|은)\s+(?P<obj>[가-힣\s]+)에\s*(?:대하여|대해)\s*"
            r"(?:책임을?\s*)?(?:지지\s*(?:아니|않|안)|면책)",
            lambda m: (m.group("subj"), "면책", m.group("obj").strip()),
        ),
    ]

    def __init__(self, min_confidence: float = 0.5):
        """Initialize extractor.

        Args:
            min_confidence: Minimum confidence threshold for including triplets
        """
        self.min_confidence = min_confidence
        self._compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), extractor)
            for pattern, extractor in self.PATTERNS
        ]

    def extract(self, text: str, source_chunk_id: str = "") -> list[Triplet]:
        """Extract triplets from text.

        Args:
            text: ToS text content
            source_chunk_id: ID of the source chunk

        Returns:
            List of extracted triplets
        """
        triplets = []
        seen = set()

        for pattern, extractor in self._compiled_patterns:
            for match in pattern.finditer(text):
                try:
                    subj, pred, obj = extractor(match)

                    # Clean up
                    subj = _clean_text(subj)
                    pred = _clean_text(pred)
                    obj = _clean_text(obj)

                    if not subj or not pred or not obj:
                        continue

                    # Skip duplicates
                    key = (subj, pred, obj)
                    if key in seen:
                        continue
                    seen.add(key)

                    triplet = Triplet(
                        subject=subj,
                        predicate=pred,
                        obj=obj,
                        source_chunk_id=source_chunk_id,
                        confidence=0.8,  # Pattern-based extraction
                    )
                    triplets.append(triplet)

                except (IndexError, AttributeError) as e:
                    logger.debug(f"Failed to extract triplet: {e}")
                    continue

        return triplets

    def extract_from_chunks(
        self, chunks: list[dict[str, Any]]
    ) -> list[Triplet]:
        """Extract triplets from multiple chunks.

        Args:
            chunks: List of ToS chunks with 'id' and 'section_content'

        Returns:
            All extracted triplets
        """
        all_triplets = []

        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            content = chunk.get("section_content", "")
            title = chunk.get("section_title", "")

            full_text = f"{title}\n{content}"
            triplets = self.extract(full_text, chunk_id)
            all_triplets.extend(triplets)

        logger.info(f"Extracted {len(all_triplets)} triplets from {len(chunks)} chunks")
        return all_triplets


class TripletStore:
    """Simple JSON-based triplet storage.

    Provides search by subject, object, or predicate.
    """

    def __init__(self, persist_path: str | Path | None = None):
        """Initialize triplet store.

        Args:
            persist_path: Path to JSON file for persistence (optional)
        """
        self.persist_path = Path(persist_path) if persist_path else None
        self.triplets: dict[str, Triplet] = {}

        # Indexes for fast lookup
        self._subject_index: dict[str, set[str]] = {}
        self._predicate_index: dict[str, set[str]] = {}
        self._object_index: dict[str, set[str]] = {}
        self._chunk_index: dict[str, set[str]] = {}

        if self.persist_path and self.persist_path.exists():
            self.load()

    def add(self, triplet: Triplet) -> str:
        """Add a triplet to the store.

        Args:
            triplet: Triplet to add

        Returns:
            Triplet ID
        """
        triplet_id = triplet.id

        # Skip if already exists
        if triplet_id in self.triplets:
            return triplet_id

        self.triplets[triplet_id] = triplet

        # Update indexes
        self._add_to_index(self._subject_index, triplet.subject.lower(), triplet_id)
        self._add_to_index(self._predicate_index, triplet.predicate.lower(), triplet_id)
        self._add_to_index(self._object_index, triplet.obj.lower(), triplet_id)

        if triplet.source_chunk_id:
            self._add_to_index(self._chunk_index, triplet.source_chunk_id, triplet_id)

        return triplet_id

    def add_batch(self, triplets: list[Triplet]) -> list[str]:
        """Add multiple triplets.

        Args:
            triplets: List of triplets to add

        Returns:
            List of triplet IDs
        """
        ids = [self.add(t) for t in triplets]
        if self.persist_path:
            self.save()
        return ids

    def _add_to_index(
        self, index: dict[str, set[str]], key: str, triplet_id: str
    ) -> None:
        """Add triplet ID to an index."""
        if key not in index:
            index[key] = set()
        index[key].add(triplet_id)

    def search_by_subject(
        self, subject: str, fuzzy: bool = True
    ) -> list[TripletSearchResult]:
        """Search triplets by subject.

        Args:
            subject: Subject to search for
            fuzzy: If True, do partial matching

        Returns:
            List of matching triplet results
        """
        return self._search_index(
            self._subject_index, subject.lower(), "subject", fuzzy
        )

    def search_by_predicate(
        self, predicate: str, fuzzy: bool = True
    ) -> list[TripletSearchResult]:
        """Search triplets by predicate (relation).

        Args:
            predicate: Predicate to search for
            fuzzy: If True, do partial matching

        Returns:
            List of matching triplet results
        """
        return self._search_index(
            self._predicate_index, predicate.lower(), "predicate", fuzzy
        )

    def search_by_object(
        self, obj: str, fuzzy: bool = True
    ) -> list[TripletSearchResult]:
        """Search triplets by object.

        Args:
            obj: Object to search for
            fuzzy: If True, do partial matching

        Returns:
            List of matching triplet results
        """
        return self._search_index(
            self._object_index, obj.lower(), "object", fuzzy
        )

    def search_by_chunk(self, chunk_id: str) -> list[Triplet]:
        """Get all triplets from a specific chunk.

        Args:
            chunk_id: Source chunk ID

        Returns:
            List of triplets from that chunk
        """
        triplet_ids = self._chunk_index.get(chunk_id, set())
        return [self.triplets[tid] for tid in triplet_ids if tid in self.triplets]

    def _search_index(
        self,
        index: dict[str, set[str]],
        query: str,
        match_type: str,
        fuzzy: bool,
    ) -> list[TripletSearchResult]:
        """Search an index."""
        results = []

        if fuzzy:
            # Partial matching
            for key, triplet_ids in index.items():
                if query in key or key in query:
                    score = len(query) / max(len(key), len(query))
                    for tid in triplet_ids:
                        if tid in self.triplets:
                            results.append(
                                TripletSearchResult(
                                    triplet=self.triplets[tid],
                                    match_type=match_type,
                                    score=score,
                                    source_chunk_id=self.triplets[tid].source_chunk_id,
                                )
                            )
        else:
            # Exact matching
            triplet_ids = index.get(query, set())
            for tid in triplet_ids:
                if tid in self.triplets:
                    results.append(
                        TripletSearchResult(
                            triplet=self.triplets[tid],
                            match_type=match_type,
                            score=1.0,
                            source_chunk_id=self.triplets[tid].source_chunk_id,
                        )
                    )

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def search(self, query: str) -> list[TripletSearchResult]:
        """Search triplets by any field.

        Args:
            query: Search query

        Returns:
            Combined results from subject, predicate, and object search
        """
        results = []
        seen = set()

        for search_fn in [
            self.search_by_subject,
            self.search_by_predicate,
            self.search_by_object,
        ]:
            for result in search_fn(query, fuzzy=True):
                if result.triplet.id not in seen:
                    results.append(result)
                    seen.add(result.triplet.id)

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def count(self) -> int:
        """Return number of triplets."""
        return len(self.triplets)

    def save(self) -> None:
        """Save triplets to JSON file."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = [t.to_dict() for t in self.triplets.values()]
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(data)} triplets to {self.persist_path}")

    def load(self) -> None:
        """Load triplets from JSON file."""
        if not self.persist_path or not self.persist_path.exists():
            return

        with open(self.persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            triplet = Triplet.from_dict(item)
            self.add(triplet)

        logger.info(f"Loaded {len(data)} triplets from {self.persist_path}")

    def clear(self) -> None:
        """Clear all triplets."""
        self.triplets.clear()
        self._subject_index.clear()
        self._predicate_index.clear()
        self._object_index.clear()
        self._chunk_index.clear()

        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()


def _normalize_predicate(pred: str) -> str:
    """Normalize predicate to standard form."""
    pred = pred.strip()

    # Map common patterns to normalized predicates
    mappings = {
        "할 수 있": "가능",
        "하여야 하": "의무",
        "해야 하": "의무",
        "거부할 수 있": "거부_가능",
        "거절할 수 있": "거절_가능",
        "제한할 수 있": "제한_가능",
    }

    return mappings.get(pred, pred)


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Remove trailing particles
    text = re.sub(r"[을를이가은는의에서로]$", "", text)

    return text.strip()
