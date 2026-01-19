"""ToS Graph Store using Neo4j for relationship-based search.

This module provides a graph database interface for storing and querying
Terms of Service documents with relationship-based search capabilities.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    """Search result from Graph Store."""

    section_id: str
    section_title: str
    section_content: str
    document_title: str
    related_sections: list[dict[str, Any]]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "section_content": self.section_content,
            "document_title": self.document_title,
            "related_sections": self.related_sections,
            "score": self.score,
        }


class ToSGraphStore:
    """Graph store for Terms of Service using Neo4j.

    Stores ToS sections as nodes with relationships:
    - (:Section)-[:REFERENCES]->(:Section) - cross-references between sections
    - (:Section)-[:PART_OF]->(:Document) - section belongs to document
    - (:Section)-[:NEXT]->(:Section) - sequential ordering

    Attributes:
        driver: Neo4j driver instance
        database: Database name
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        """Initialize ToS Graph Store.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Neo4j database."""
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            # Verify connection
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            logger.warning("neo4j package not installed. Graph features disabled.")
            self.driver = None
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}. Graph features disabled.")
            self.driver = None

    def is_available(self) -> bool:
        """Check if Neo4j connection is available."""
        return self.driver is not None

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def _init_schema(self) -> None:
        """Initialize graph schema with constraints and indexes."""
        if not self.driver:
            return

        with self.driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            session.run("""
                CREATE CONSTRAINT section_id IF NOT EXISTS
                FOR (s:Section) REQUIRE s.section_id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT document_id IF NOT EXISTS
                FOR (d:Document) REQUIRE d.document_id IS UNIQUE
            """)
            # Create index for title search
            session.run("""
                CREATE INDEX section_title IF NOT EXISTS
                FOR (s:Section) ON (s.title)
            """)
            logger.info("Graph schema initialized")

    def add_document(self, tos_item: dict[str, Any]) -> str:
        """Add a ToS document and its sections to the graph.

        Args:
            tos_item: ToS document with sections

        Returns:
            Document ID
        """
        if not self.driver:
            logger.warning("Neo4j not available. Skipping add_document.")
            return ""

        document_title = tos_item.get("title", "")
        document_id = self._generate_id(document_title)
        sections = tos_item.get("sections", [])
        content = tos_item.get("content", "")

        # Parse sections if not provided
        if not sections and content:
            sections = self._parse_sections(content)

        with self.driver.session(database=self.database) as session:
            # Create document node
            session.run(
                """
                MERGE (d:Document {document_id: $doc_id})
                SET d.title = $title,
                    d.effective_date = $effective_date,
                    d.category = $category
                """,
                doc_id=document_id,
                title=document_title,
                effective_date=tos_item.get("effective_date", ""),
                category=tos_item.get("category", ""),
            )

            # Create section nodes and relationships
            prev_section_id = None
            for section in sections:
                section_id = self._add_section(
                    session, section, document_id, document_title
                )
                if prev_section_id:
                    # Create NEXT relationship
                    session.run(
                        """
                        MATCH (s1:Section {section_id: $prev_id})
                        MATCH (s2:Section {section_id: $curr_id})
                        MERGE (s1)-[:NEXT]->(s2)
                        """,
                        prev_id=prev_section_id,
                        curr_id=section_id,
                    )
                prev_section_id = section_id

            # Extract and create REFERENCES relationships
            self._extract_references(session, sections, document_id)

        logger.info(f"Added document '{document_title}' with {len(sections)} sections")
        return document_id

    def _add_section(
        self,
        session: Any,
        section: dict[str, str],
        document_id: str,
        document_title: str,
    ) -> str:
        """Add a section node to the graph."""
        title = section.get("title", "")
        content = section.get("content", "")
        section_id = self._generate_id(f"{document_id}_{title}")

        session.run(
            """
            MERGE (s:Section {section_id: $section_id})
            SET s.title = $title,
                s.content = $content,
                s.document_title = $document_title
            WITH s
            MATCH (d:Document {document_id: $doc_id})
            MERGE (s)-[:PART_OF]->(d)
            """,
            section_id=section_id,
            title=title,
            content=content[:2000],  # Limit content size
            document_title=document_title,
            doc_id=document_id,
        )

        return section_id

    def _extract_references(
        self,
        session: Any,
        sections: list[dict[str, str]],
        document_id: str,
    ) -> None:
        """Extract cross-references between sections and create relationships."""
        # Pattern to match "제N조" references
        ref_pattern = re.compile(r"제\s*(\d+)\s*조")

        section_map = {}
        for section in sections:
            title = section.get("title", "")
            match = re.search(r"제\s*(\d+)\s*조", title)
            if match:
                section_num = match.group(1)
                section_id = self._generate_id(f"{document_id}_{title}")
                section_map[section_num] = section_id

        # Find references in content and create relationships
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            source_id = self._generate_id(f"{document_id}_{title}")

            # Find all referenced section numbers
            references = ref_pattern.findall(content)
            for ref_num in set(references):
                if ref_num in section_map and section_map[ref_num] != source_id:
                    target_id = section_map[ref_num]
                    session.run(
                        """
                        MATCH (s1:Section {section_id: $source_id})
                        MATCH (s2:Section {section_id: $target_id})
                        MERGE (s1)-[:REFERENCES]->(s2)
                        """,
                        source_id=source_id,
                        target_id=target_id,
                    )

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

    def search(
        self,
        query: str,
        n_results: int = 5,
        include_related: bool = True,
    ) -> list[GraphSearchResult]:
        """Search for sections matching the query.

        Uses text matching on titles and content.
        For semantic search, use HybridSearch with vector store.

        Args:
            query: Search query
            n_results: Maximum number of results
            include_related: Whether to include related sections

        Returns:
            List of GraphSearchResult
        """
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            # Full-text search on title and content
            result = session.run(
                """
                MATCH (s:Section)
                WHERE s.title CONTAINS $query OR s.content CONTAINS $query
                RETURN s.section_id AS section_id,
                       s.title AS title,
                       s.content AS content,
                       s.document_title AS document_title
                LIMIT $limit
                """,
                query=query,
                limit=n_results,
            )

            search_results = []
            for record in result:
                related = []
                if include_related:
                    related = self._get_related_sections(
                        session, record["section_id"]
                    )

                search_results.append(
                    GraphSearchResult(
                        section_id=record["section_id"],
                        section_title=record["title"],
                        section_content=record["content"],
                        document_title=record["document_title"],
                        related_sections=related,
                        score=1.0,  # Text match score
                    )
                )

            return search_results

    def get_related_sections(
        self,
        section_id: str,
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get sections related to a given section.

        Args:
            section_id: Source section ID
            relationship_types: Optional list of relationship types to filter

        Returns:
            List of related section info
        """
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            return self._get_related_sections(session, section_id, relationship_types)

    def _get_related_sections(
        self,
        session: Any,
        section_id: str,
        relationship_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get related sections from a session."""
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            query = f"""
                MATCH (s:Section {{section_id: $section_id}})-[r:{rel_filter}]-(related:Section)
                RETURN related.section_id AS section_id,
                       related.title AS title,
                       type(r) AS relationship
            """
        else:
            query = """
                MATCH (s:Section {section_id: $section_id})-[r]-(related:Section)
                RETURN related.section_id AS section_id,
                       related.title AS title,
                       type(r) AS relationship
            """

        result = session.run(query, section_id=section_id)

        return [
            {
                "section_id": record["section_id"],
                "title": record["title"],
                "relationship": record["relationship"],
            }
            for record in result
        ]

    def clear(self) -> None:
        """Clear all nodes and relationships from the graph."""
        if not self.driver:
            return

        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared all graph data")

    def count(self) -> dict[str, int]:
        """Count nodes and relationships in the graph."""
        if not self.driver:
            return {"sections": 0, "documents": 0, "relationships": 0}

        with self.driver.session(database=self.database) as session:
            sections = session.run(
                "MATCH (s:Section) RETURN count(s) AS count"
            ).single()["count"]
            documents = session.run(
                "MATCH (d:Document) RETURN count(d) AS count"
            ).single()["count"]
            relationships = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()["count"]

            return {
                "sections": sections,
                "documents": documents,
                "relationships": relationships,
            }

    def _generate_id(self, text: str) -> str:
        """Generate a unique ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:12]
