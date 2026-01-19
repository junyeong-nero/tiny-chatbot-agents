"""Triplet extractor for building knowledge graph from ToS documents.

Extracts relationships between sections:
- Cross-references (제N조 참조)
- Exception/condition relationships (단, ~의 경우)
- Prerequisite relationships (전항에 따라)
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """A knowledge graph triplet (subject, predicate, object)."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
        }


class TripletExtractor:
    """Extract triplets from ToS document sections.

    Supports two extraction modes:
    1. Rule-based: Fast, deterministic extraction using regex patterns
    2. LLM-based: More comprehensive but requires LLM API (optional)

    Patterns detected:
    - REFERENCES: "제N조에 따라", "제N조를 준용"
    - EXCEPTION_OF: "다만,", "단,"
    - PREREQUISITE: "전항에 따라", "제N항의 규정에 의하여"
    - DEFINES: Term definitions in parentheses
    """

    # Pattern for section references
    SECTION_REF_PATTERN = re.compile(
        r"제\s*(\d+)\s*조(?:\s*제?\s*(\d+)\s*항)?(?:에\s*(?:따라|의해|의거|준용|규정))"
    )

    # Pattern for exception clauses
    EXCEPTION_PATTERN = re.compile(r"(?:다만|단),?\s+(.+?)(?:경우|때)",)

    # Pattern for previous section reference
    PREV_SECTION_PATTERN = re.compile(r"(전항|전조|전호|제\s*\d+\s*항)에?\s*(?:따라|의하여|불구하고)")

    # Pattern for term definitions
    DEFINITION_PATTERN = re.compile(r'"([^"]+)"(?:이?라\s*함은|이란|이?라\s*한다)')

    def __init__(self, use_llm: bool = False, llm_client: Any = None) -> None:
        """Initialize the triplet extractor.

        Args:
            use_llm: Whether to use LLM for extraction
            llm_client: LLM client instance (required if use_llm is True)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

    def extract_from_section(
        self,
        section: dict[str, str],
        document_id: str = "",
    ) -> list[Triplet]:
        """Extract triplets from a single section.

        Args:
            section: Section dict with 'title' and 'content'
            document_id: Parent document ID for context

        Returns:
            List of extracted Triplet objects
        """
        triplets = []
        title = section.get("title", "")
        content = section.get("content", "")
        full_text = f"{title}\n{content}"

        # Extract section references
        triplets.extend(self._extract_references(title, content))

        # Extract exception relationships
        triplets.extend(self._extract_exceptions(title, content))

        # Extract definitions
        triplets.extend(self._extract_definitions(title, content))

        # Optional: LLM-based extraction for more complex relationships
        if self.use_llm and self.llm_client:
            triplets.extend(self._extract_with_llm(full_text))

        return triplets

    def extract_from_document(
        self,
        tos_item: dict[str, Any],
    ) -> list[Triplet]:
        """Extract all triplets from a ToS document.

        Args:
            tos_item: Complete ToS document dict

        Returns:
            List of all extracted triplets
        """
        all_triplets = []
        document_title = tos_item.get("title", "")
        sections = tos_item.get("sections", [])

        # If sections not parsed, parse from content
        if not sections and tos_item.get("content"):
            sections = self._parse_sections(tos_item["content"])

        for section in sections:
            triplets = self.extract_from_section(section, document_title)
            all_triplets.extend(triplets)

        logger.info(f"Extracted {len(all_triplets)} triplets from '{document_title}'")
        return all_triplets

    def _extract_references(self, title: str, content: str) -> list[Triplet]:
        """Extract REFERENCES relationships from content."""
        triplets = []

        # Get current section number
        current_section = self._get_section_number(title)
        if not current_section:
            return triplets

        # Find all referenced sections
        for match in self.SECTION_REF_PATTERN.finditer(content):
            ref_section = match.group(1)
            ref_subsection = match.group(2)

            ref_target = f"제{ref_section}조"
            if ref_subsection:
                ref_target += f" 제{ref_subsection}항"

            triplets.append(
                Triplet(
                    subject=f"제{current_section}조",
                    predicate="REFERENCES",
                    object=ref_target,
                    confidence=0.95,
                )
            )

        return triplets

    def _extract_exceptions(self, title: str, content: str) -> list[Triplet]:
        """Extract EXCEPTION_OF relationships from exception clauses."""
        triplets = []

        current_section = self._get_section_number(title)
        if not current_section:
            return triplets

        # Check for exception patterns
        for match in self.EXCEPTION_PATTERN.finditer(content):
            exception_content = match.group(1)[:50]  # Limit length

            triplets.append(
                Triplet(
                    subject=f"제{current_section}조 예외",
                    predicate="EXCEPTION_OF",
                    object=f"제{current_section}조",
                    confidence=0.8,
                )
            )

        return triplets

    def _extract_definitions(self, title: str, content: str) -> list[Triplet]:
        """Extract DEFINES relationships from term definitions."""
        triplets = []

        current_section = self._get_section_number(title)
        if not current_section:
            return triplets

        for match in self.DEFINITION_PATTERN.finditer(content):
            term = match.group(1)

            triplets.append(
                Triplet(
                    subject=f"제{current_section}조",
                    predicate="DEFINES",
                    object=term,
                    confidence=0.9,
                )
            )

        return triplets

    def _get_section_number(self, title: str) -> str | None:
        """Extract section number from title."""
        match = re.search(r"제\s*(\d+)\s*조", title)
        return match.group(1) if match else None

    def _parse_sections(self, content: str) -> list[dict[str, str]]:
        """Parse sections from raw content."""
        sections = []
        section_pattern = re.compile(r"^제\s*(\d+)\s*조\s*[\(（]([^)）]+)[\)）]")

        lines = content.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = section_pattern.match(line)
            if match:
                if current_section:
                    sections.append({
                        "title": current_section,
                        "content": "\n".join(current_content),
                    })
                current_section = line
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections.append({
                "title": current_section,
                "content": "\n".join(current_content),
            })

        return sections

    def _extract_with_llm(self, text: str) -> list[Triplet]:
        """Extract triplets using LLM (optional, requires LLM client)."""
        if not self.llm_client:
            return []

        # This would call the LLM to extract relationships
        # Implementation depends on the specific LLM API being used
        logger.debug("LLM-based extraction not yet implemented")
        return []
