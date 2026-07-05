"""Cross-document knowledge graph for entity linking and semantic relationships.

Builds a multi-modal knowledge graph from extracted entities across
multiple documents, enabling:
- Cross-document entity linking ("this person appears in 47 documents")
- Semantic similarity search between images
- Concept hierarchy inference
- Co-occurrence analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class EntityInfo:
    """An entity extracted from a document."""

    text: str
    label: str = "entity"
    confidence: float = 1.0
    image_id: str = ""


@dataclass
class KGQueryResult:
    """Result of a knowledge graph query."""

    similar_images: list[tuple[str, float]] = field(default_factory=list)
    related_entities: list[tuple[str, str, float]] = field(default_factory=list)
    entity_connections: list[dict[str, Any]] = field(default_factory=list)
    concept_hierarchy: list[dict[str, Any]] = field(default_factory=list)


class ImageKnowledgeGraph:
    """Build and query a cross-document knowledge graph.

    Nodes represent images, entities (extracted text/concepts), and
    hierarchical concepts. Edges represent containment, co-occurrence,
    and semantic relationships.

    Use cases:
    - Find all images containing a specific entity
    - Find images similar to a given image (shared entities)
    - Discover entity co-occurrence patterns
    - Build concept hierarchies from extracted entities
    """

    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self._image_nodes: set[str] = set()
        self._entity_nodes: set[str] = set()
        self._concept_nodes: set[str] = set()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_image(self, image_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an image as a node in the knowledge graph.

        Args:
            image_id: Unique identifier for the image (e.g., file_pk or path).
            metadata: Optional dict of image metadata (path, dimensions, etc.).
        """
        self.G.add_node(image_id, type="image", **(metadata or {}))
        self._image_nodes.add(image_id)

    def add_entity(
        self,
        entity_text: str,
        entity_type: str = "entity",
        confidence: float = 1.0,
    ) -> str:
        """Add an entity node (deduplicated by text).

        Args:
            entity_text: The entity text (e.g., "Acme Corp", "invoice", "2024-01-15").
            entity_type: Type of entity - "person", "organization", "date", "amount", etc.
            confidence: Extraction confidence (0.0-1.0).

        Returns:
            The entity node ID.
        """
        entity_id = f"entity:{entity_text}"

        if entity_id not in self.G:
            self.G.add_node(
                entity_id,
                type="entity",
                text=entity_text,
                entity_type=entity_type,
                confidence=confidence,
            )
            self._entity_nodes.add(entity_id)
        else:
            # Update confidence if new is higher
            existing = self.G.nodes[entity_id].get("confidence", 0.0)
            if confidence > existing:
                self.G.nodes[entity_id]["confidence"] = confidence

        return entity_id

    def add_concept(self, concept_name: str, parent: str | None = None) -> str:
        """Add a hierarchical concept node.

        Args:
            concept_name: The concept name (e.g., "invoice", "financial_document").
            parent: Optional parent concept for hierarchy.

        Returns:
            The concept node ID.
        """
        concept_id = f"concept:{concept_name}"

        if concept_id not in self.G:
            self.G.add_node(concept_id, type="concept", name=concept_name)
            self._concept_nodes.add(concept_id)

        if parent:
            parent_id = f"concept:{parent}"
            if parent_id not in self.G:
                self.G.add_node(parent_id, type="concept", name=parent)
                self._concept_nodes.add(parent_id)
            self.G.add_edge(parent_id, concept_id, relation="is_a")

        return concept_id

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def link_image_to_entity(
        self,
        image_id: str,
        entity_text: str,
        entity_type: str = "entity",
        confidence: float = 1.0,
    ) -> None:
        """Link an image to an entity it contains.

        Args:
            image_id: The image node ID.
            entity_text: The entity text found in the image.
            entity_type: Type of entity.
            confidence: Extraction confidence.
        """
        if image_id not in self.G:
            self.add_image(image_id)

        entity_id = self.add_entity(entity_text, entity_type, confidence)
        self.G.add_edge(image_id, entity_id, relation="contains", confidence=confidence)

    def link_entity_to_concept(self, entity_text: str, concept_name: str) -> None:
        """Link an entity to a concept (e.g., "Acme Corp" → "organization").

        Args:
            entity_text: The entity text.
            concept_name: The concept name.
        """
        entity_id = f"entity:{entity_text}"
        concept_id = f"concept:{concept_name}"

        if entity_id not in self.G:
            self.add_entity(entity_text)
        if concept_id not in self.G:
            self.add_concept(concept_name)

        self.G.add_edge(entity_id, concept_id, relation="instance_of")

    def add_co_occurrence(
        self,
        entity_a: str,
        entity_b: str,
        image_id: str,
    ) -> None:
        """Record that two entities co-occur in the same image.

        Args:
            entity_a: First entity text.
            entity_b: Second entity text.
            image_id: The image where they co-occur.
        """
        id_a = f"entity:{entity_a}"
        id_b = f"entity:{entity_b}"

        if id_a not in self.G:
            self.add_entity(entity_a)
        if id_b not in self.G:
            self.add_entity(entity_b)

        if self.G.has_edge(id_a, id_b):
            self.G[id_a][id_b]["weight"] = self.G[id_a][id_b].get("weight", 0) + 1
        else:
            self.G.add_edge(id_a, id_b, relation="co_occurs", weight=1, image=image_id)

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------

    def load_from_entities(
        self,
        image_id: str,
        entities: list[EntityInfo],
        image_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Load an image and its extracted entities into the graph.

        Args:
            image_id: Unique image identifier.
            entities: List of EntityInfo objects extracted from the image.
            image_metadata: Optional image metadata.
        """
        self.add_image(image_id, image_metadata)

        for entity in entities:
            self.link_image_to_entity(
                image_id,
                entity.text,
                entity.label,
                entity.confidence,
            )

        # Add co-occurrence edges between all entity pairs in this image
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                self.add_co_occurrence(e1.text, e2.text, image_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_similar_images(
        self,
        image_id: str,
        min_shared: int = 1,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find images similar to a given image based on shared entities.

        Uses Jaccard similarity on entity sets.

        Args:
            image_id: The reference image ID.
            min_shared: Minimum number of shared entities required.
            top_k: Maximum number of results.

        Returns:
            List of (image_id, similarity_score) tuples, sorted by score descending.
        """
        if image_id not in self.G:
            return []

        # Get entities in the reference image
        ref_entities: set[str] = set()
        for _, target, data in self.G.out_edges(image_id, data=True):
            if data.get("relation") == "contains":
                ref_entities.add(target)

        if not ref_entities:
            return []

        # Find images sharing those entities
        scores: dict[str, float] = {}
        shared_counts: dict[str, int] = {}
        for entity_id in ref_entities:
            for source, _, data in self.G.in_edges(entity_id, data=True):
                if data.get("relation") == "contains" and source != image_id:
                    if source not in scores:
                        scores[source] = 0.0
                        shared_counts[source] = 0
                    shared_counts[source] += 1

        # Compute Jaccard similarity
        results: list[tuple[str, float]] = []
        for other_id, shared_count in shared_counts.items():
            if shared_count < min_shared:
                continue

            other_entities: set[str] = set()
            for _, target, data in self.G.out_edges(other_id, data=True):
                if data.get("relation") == "contains":
                    other_entities.add(target)

            union = len(ref_entities | other_entities)
            jaccard = shared_count / union if union > 0 else 0.0
            results.append((other_id, round(jaccard, 4)))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def find_entity_connections(
        self,
        entity_text: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Find all connections for an entity in the graph.

        Args:
            entity_text: The entity text to search for.
            max_depth: Maximum traversal depth (reserved for future use).

        Returns:
            List of connection dicts with source, target, relation, and weight.
        """
        entity_id = f"entity:{entity_text}"
        if entity_id not in self.G:
            return []

        connections: list[dict[str, Any]] = []

        # Outgoing edges (what this entity relates to)
        for _, target, data in self.G.out_edges(entity_id, data=True):
            target_info = self.G.nodes[target]
            connections.append(
                {
                    "source": entity_text,
                    "target": target_info.get("text", target_info.get("name", target)),
                    "relation": data.get("relation", "unknown"),
                    "weight": data.get("weight", 1.0),
                    "direction": "outgoing",
                }
            )

        # Incoming edges (what relates to this entity)
        for source, _, data in self.G.in_edges(entity_id, data=True):
            source_info = self.G.nodes[source]
            connections.append(
                {
                    "source": source_info.get("text", source_info.get("name", source)),
                    "target": entity_text,
                    "relation": data.get("relation", "unknown"),
                    "weight": data.get("weight", 1.0),
                    "direction": "incoming",
                }
            )

        return connections

    def find_co_occurring_entities(
        self,
        entity_text: str,
        min_weight: int = 1,
        top_k: int = 20,
    ) -> list[tuple[str, int]]:
        """Find entities that frequently co-occur with a given entity.

        Args:
            entity_text: The entity text.
            min_weight: Minimum co-occurrence count.
            top_k: Maximum number of results.

        Returns:
            List of (entity_text, co_occurrence_count) tuples.
        """
        entity_id = f"entity:{entity_text}"
        if entity_id not in self.G:
            return []

        co_occurrences: list[tuple[str, int]] = []
        for _, target, data in self.G.out_edges(entity_id, data=True):
            if data.get("relation") == "co_occurs":
                weight = data.get("weight", 0)
                if weight >= min_weight:
                    target_text = self.G.nodes[target].get("text", target)
                    co_occurrences.append((target_text, weight))

        co_occurrences.sort(key=lambda x: -x[1])
        return co_occurrences[:top_k]

    def get_concept_hierarchy(
        self,
        concept_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get the concept hierarchy, optionally starting from a specific concept.

        Args:
            concept_name: Optional root concept. If None, returns all hierarchies.

        Returns:
            List of hierarchy dicts with name, parent, and children.
        """
        hierarchies: list[dict[str, Any]] = []

        if concept_name:
            concept_id = f"concept:{concept_name}"
            if concept_id not in self.G:
                return []

            # Get children
            children = []
            for _, child, data in self.G.out_edges(concept_id, data=True):
                if data.get("relation") == "is_a":
                    child_name = self.G.nodes[child].get("name", child)
                    children.append(child_name)

            # Get parent
            parent = None
            for source, _, data in self.G.in_edges(concept_id, data=True):
                if data.get("relation") == "is_a":
                    parent = self.G.nodes[source].get("name", source)

            hierarchies.append(
                {
                    "name": concept_name,
                    "parent": parent,
                    "children": children,
                }
            )
        else:
            # Return all concept hierarchies
            for node in self._concept_nodes:
                name = self.G.nodes[node].get("name", node)
                children = []
                for _, child, data in self.G.out_edges(node, data=True):
                    if data.get("relation") == "is_a":
                        children.append(self.G.nodes[child].get("name", child))
                parent = None
                for source, _, data in self.G.in_edges(node, data=True):
                    if data.get("relation") == "is_a":
                        parent = self.G.nodes[source].get("name", source)
                hierarchies.append(
                    {
                        "name": name,
                        "parent": parent,
                        "children": children,
                    }
                )

        return hierarchies

    def get_images_for_entity(
        self,
        entity_text: str,
    ) -> list[str]:
        """Find all images containing a specific entity.

        Args:
            entity_text: The entity text.

        Returns:
            List of image IDs containing this entity.
        """
        entity_id = f"entity:{entity_text}"
        if entity_id not in self.G:
            return []

        images: list[str] = []
        for source, _, data in self.G.in_edges(entity_id, data=True):
            if data.get("relation") == "contains" and source in self._image_nodes:
                images.append(source)

        return images

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Get knowledge graph statistics."""
        return {
            "total_nodes": len(self.G),
            "total_edges": len(self.G.edges()),
            "images": len(self._image_nodes),
            "entities": len(self._entity_nodes),
            "concepts": len(self._concept_nodes),
        }

    def get_top_entities(self, top_k: int = 20) -> list[tuple[str, int]]:
        """Get the most frequently occurring entities.

        Args:
            top_k: Maximum number of results.

        Returns:
            List of (entity_text, occurrence_count) tuples.
        """
        entity_counts: dict[str, int] = {}
        for entity_id in self._entity_nodes:
            count = 0
            for _, _, data in self.G.in_edges(entity_id, data=True):
                if data.get("relation") == "contains":
                    count += 1
            if count > 0:
                text = self.G.nodes[entity_id].get("text", entity_id)
                entity_counts[text] = count

        sorted_entities = sorted(entity_counts.items(), key=lambda x: -x[1])
        return sorted_entities[:top_k]

    def to_graphml(self) -> str:
        """Serialize the knowledge graph to GraphML format."""
        result = ""
        for line in nx.generate_graphml(self.G):
            result += line
        return result
