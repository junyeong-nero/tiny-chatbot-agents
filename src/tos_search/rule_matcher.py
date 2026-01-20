"""Rule-based keyword and pattern matching for ToS search.

Provides:
- Section reference extraction (제1조, 제2조 등)
- Keyword boosting for legal/ToS terms
- Category-based filtering
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Legal/ToS-specific keywords with weights
KEYWORD_WEIGHTS: dict[str, float] = {
    # 계약 관련
    "해지": 0.8,
    "해제": 0.8,
    "취소": 0.7,
    "철회": 0.7,
    "종료": 0.6,
    # 금전 관련
    "환불": 0.9,
    "위약금": 0.9,
    "수수료": 0.8,
    "손해배상": 0.8,
    "배상": 0.7,
    "보상": 0.7,
    "청구": 0.6,
    # 책임 관련
    "면책": 0.9,
    "책임": 0.7,
    "의무": 0.6,
    "권리": 0.6,
    # 개인정보 관련
    "개인정보": 0.8,
    "정보보호": 0.7,
    "동의": 0.6,
    # 기타
    "약관": 0.5,
    "규정": 0.5,
    "조항": 0.5,
}


@dataclass
class SectionRef:
    """Extracted section reference from query."""

    article_num: int  # 조 번호 (제N조)
    clause_num: int | None = None  # 항 번호 (N항)
    subclause_num: int | None = None  # 호 번호 (N호)
    title: str | None = None  # 조 제목 (용어의 정의)

    def to_pattern(self) -> str:
        """Convert to regex pattern for matching."""
        pattern = rf"제\s*{self.article_num}\s*조"
        if self.title:
            pattern += rf"\s*[\(（]{self.title}[\)）]"
        return pattern

    def __str__(self) -> str:
        result = f"제{self.article_num}조"
        if self.title:
            result += f"({self.title})"
        if self.clause_num:
            result += f" {self.clause_num}항"
        if self.subclause_num:
            result += f" {self.subclause_num}호"
        return result


@dataclass
class RuleMatchResult:
    """Result from rule-based matching."""

    chunk_id: str
    section_title: str
    section_content: str
    document_title: str
    category: str
    rule_score: float
    matched_keywords: list[str] = field(default_factory=list)
    section_ref_match: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_title": self.section_title,
            "section_content": self.section_content,
            "document_title": self.document_title,
            "category": self.category,
            "rule_score": self.rule_score,
            "matched_keywords": self.matched_keywords,
            "section_ref_match": self.section_ref_match,
            "metadata": self.metadata,
        }


class ToSRuleMatcher:
    """Rule-based matcher for ToS documents.

    Performs:
    1. Section reference matching (제1조, 제2조 등)
    2. Keyword boosting for legal terms
    3. Category filtering
    """

    # Pattern for section references in queries
    SECTION_QUERY_PATTERN = re.compile(
        r"제\s*(\d+)\s*조(?:\s*[\(（]([^)）]+)[\)）])?(?:\s*(\d+)\s*항)?(?:\s*(\d+)\s*호)?"
    )

    # Pattern for section headers in content
    SECTION_HEADER_PATTERN = re.compile(r"^제\s*(\d+)\s*조\s*[\(（]([^)）]+)[\)）]")

    def __init__(
        self,
        keyword_weights: dict[str, float] | None = None,
        section_match_boost: float = 1.0,
        keyword_match_weight: float = 0.5,
    ):
        """Initialize rule matcher.

        Args:
            keyword_weights: Custom keyword weights (merged with defaults)
            section_match_boost: Score boost for exact section matches
            keyword_match_weight: Weight for keyword score in final score
        """
        self.keyword_weights = {**KEYWORD_WEIGHTS}
        if keyword_weights:
            self.keyword_weights.update(keyword_weights)

        self.section_match_boost = section_match_boost
        self.keyword_match_weight = keyword_match_weight

    def extract_section_reference(self, query: str) -> SectionRef | None:
        """Extract section reference from query.

        Args:
            query: User query (e.g., "제1조는 뭐야?", "제3조 2항 알려줘")

        Returns:
            SectionRef if found, None otherwise
        """
        match = self.SECTION_QUERY_PATTERN.search(query)
        if not match:
            return None

        article_num = int(match.group(1))
        title = match.group(2)
        clause_num = int(match.group(3)) if match.group(3) else None
        subclause_num = int(match.group(4)) if match.group(4) else None

        return SectionRef(
            article_num=article_num,
            title=title,
            clause_num=clause_num,
            subclause_num=subclause_num,
        )

    def calculate_keyword_score(self, query: str, content: str) -> tuple[float, list[str]]:
        """Calculate keyword match score.

        Args:
            query: User query
            content: Document content

        Returns:
            Tuple of (score, list of matched keywords)
        """
        query_lower = query.lower()
        content_lower = content.lower()

        matched_keywords = []
        total_score = 0.0

        for keyword, weight in self.keyword_weights.items():
            if keyword in query_lower and keyword in content_lower:
                matched_keywords.append(keyword)
                total_score += weight

        # Normalize by number of keywords checked
        if matched_keywords:
            total_score = min(1.0, total_score / len(matched_keywords))

        return total_score, matched_keywords

    def match_section(self, section_ref: SectionRef, section_title: str) -> bool:
        """Check if section title matches section reference.

        Args:
            section_ref: Extracted section reference from query
            section_title: Section title from document

        Returns:
            True if matches
        """
        header_match = self.SECTION_HEADER_PATTERN.match(section_title)
        if not header_match:
            return False

        doc_article_num = int(header_match.group(1))
        if doc_article_num != section_ref.article_num:
            return False

        # If title specified in query, check it too
        if section_ref.title:
            doc_title = header_match.group(2)
            if section_ref.title not in doc_title:
                return False

        return True

    def match(
        self,
        query: str,
        documents: list[dict[str, Any]],
        category_filter: str | None = None,
    ) -> list[RuleMatchResult]:
        """Perform rule-based matching on documents.

        Args:
            query: User query
            documents: List of ToS chunks with metadata
            category_filter: Optional category filter

        Returns:
            List of RuleMatchResult sorted by score
        """
        results = []
        section_ref = self.extract_section_reference(query)

        for doc in documents:
            # Apply category filter
            doc_category = doc.get("category", "")
            if category_filter and doc_category != category_filter:
                continue

            section_title = doc.get("section_title", "")
            section_content = doc.get("section_content", "")
            combined_content = f"{section_title}\n{section_content}"

            # Check section reference match
            section_ref_match = False
            if section_ref:
                section_ref_match = self.match_section(section_ref, section_title)

            # Calculate keyword score
            keyword_score, matched_keywords = self.calculate_keyword_score(
                query, combined_content
            )

            # Calculate final score
            rule_score = 0.0

            if section_ref_match:
                rule_score = self.section_match_boost


            if keyword_score > 0:
                rule_score = max(
                    rule_score,
                    keyword_score * self.keyword_match_weight,
                )

            # Include if any rule matched
            if rule_score > 0 or section_ref_match or matched_keywords:
                results.append(
                    RuleMatchResult(
                        chunk_id=doc.get("id", ""),
                        section_title=section_title,
                        section_content=section_content,
                        document_title=doc.get("document_title", ""),
                        category=doc_category,
                        rule_score=rule_score,
                        matched_keywords=matched_keywords,
                        section_ref_match=section_ref_match,
                        metadata={
                            "effective_date": doc.get("effective_date", ""),
                            "source_url": doc.get("source_url", ""),
                        },
                    )
                )

        # Sort by score descending
        results.sort(key=lambda x: x.rule_score, reverse=True)
        return results

    def boost_vector_results(
        self,
        query: str,
        vector_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Boost vector search results with rule-based scores.

        Args:
            query: User query
            vector_results: Results from vector search (with 'score' field)

        Returns:
            Results with added 'rule_score' and updated 'combined_score'
        """
        section_ref = self.extract_section_reference(query)
        boosted_results = []

        for result in vector_results:
            section_title = result.get("section_title", "")
            section_content = result.get("section_content", "")
            combined_content = f"{section_title}\n{section_content}"

            # Section reference match
            section_ref_match = False
            if section_ref:
                section_ref_match = self.match_section(section_ref, section_title)

            # Keyword score
            keyword_score, matched_keywords = self.calculate_keyword_score(
                query, combined_content
            )

            # Calculate rule score
            rule_score = 0.0
            if section_ref_match:
                rule_score = self.section_match_boost
            elif keyword_score > 0:
                rule_score = keyword_score * self.keyword_match_weight

            # Add rule info to result
            boosted_result = dict(result)
            boosted_result["rule_score"] = rule_score
            boosted_result["matched_keywords"] = matched_keywords
            boosted_result["section_ref_match"] = section_ref_match

            boosted_results.append(boosted_result)

        return boosted_results
