"""Document layout graph analysis using NetworkX.

Builds spatial relationship graphs from OCR/label output for
reading order inference, table detection, and region grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class LayoutBlock:
    """A text block or region detected in a document."""

    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    label: str = "text"
    confidence: float = 1.0


@dataclass
class LayoutGraphResult:
    """Result of document layout graph analysis."""

    num_blocks: int
    num_edges: int
    reading_order: list[int]  # node indices in reading order
    tables: list[list[int]]  # groups of nodes forming tables
    regions: list[list[int]]  # groups of nodes forming regions
    graph_ml: str = ""  # GraphML serialization


class DocumentLayoutGraph:
    """Build and analyze spatial relationship graphs from document layout blocks.

    Nodes represent text blocks/regions. Edges represent spatial relationships
    (left_of, above, same_row, etc.). Graph algorithms enable:
    - Reading order inference via topological-like traversal
    - Table detection via community detection on aligned blocks
    - Region grouping via connected components
    """

    def __init__(self, row_tolerance: float = 20.0, col_tolerance: float = 20.0):
        """Initialize the layout graph builder.

        Args:
            row_tolerance: Vertical pixel tolerance for same-row detection.
            col_tolerance: Horizontal pixel tolerance for same-column detection.
        """
        self.G = nx.Graph()
        self.row_tolerance = row_tolerance
        self.col_tolerance = col_tolerance

    def add_blocks(self, blocks: list[LayoutBlock]) -> None:
        """Add text blocks as nodes and compute spatial edges.

        Args:
            blocks: List of LayoutBlock objects with text and bounding boxes.
        """
        # Add nodes
        for i, block in enumerate(blocks):
            x0, y0, x1, y1 = block.bbox
            self.G.add_node(
                i,
                text=block.text,
                bbox=block.bbox,
                label=block.label,
                confidence=block.confidence,
                x_center=(x0 + x1) / 2.0,
                y_center=(y0 + y1) / 2.0,
                width=x1 - x0,
                height=y1 - y0,
            )

        # Add spatial edges
        nodes = list(self.G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self._add_spatial_edges(nodes[i], nodes[j])

    def _add_spatial_edges(self, i: int, j: int) -> None:
        """Add spatial relationship edges between two nodes."""
        bi = self.G.nodes[i]
        bj = self.G.nodes[j]

        yi_center = bi["y_center"]
        yj_center = bj["y_center"]
        xi_center = bi["x_center"]
        xj_center = bj["x_center"]

        # Same row detection
        if abs(yi_center - yj_center) < self.row_tolerance:
            self.G.add_edge(i, j, relation="same_row", weight=1.0)
            if xi_center < xj_center:
                self.G.add_edge(i, j, relation="left_of", weight=1.0)
            else:
                self.G.add_edge(i, j, relation="right_of", weight=1.0)

        # Same column detection
        if abs(xi_center - xj_center) < self.col_tolerance:
            self.G.add_edge(i, j, relation="same_column", weight=1.0)
            if yi_center < yj_center:
                self.G.add_edge(i, j, relation="above", weight=1.0)
            else:
                self.G.add_edge(i, j, relation="below", weight=1.0)

        # Proximity edge (always added if close enough)
        dist = ((xi_center - xj_center) ** 2 + (yi_center - yj_center) ** 2) ** 0.5
        proximity_threshold = max(bi["width"], bi["height"]) * 2.0
        if dist < proximity_threshold:
            self.G.add_edge(i, j, relation="near", weight=1.0 / max(dist, 1.0))

    def get_reading_order(self) -> list[int]:
        """Infer reading order using spatial graph traversal.

        Returns node indices in approximate reading order (top-to-bottom,
        left-to-right within rows).
        """
        if len(self.G) == 0:
            return []

        # Group nodes by row (using y_center with tolerance)
        nodes_by_row: dict[int, list[int]] = {}
        for node in self.G.nodes():
            y = self.G.nodes[node]["y_center"]
            row_key = int(y / self.row_tolerance)
            if row_key not in nodes_by_row:
                nodes_by_row[row_key] = []
            nodes_by_row[row_key].append(node)

        # Sort rows top-to-bottom, then left-to-right within each row
        reading_order: list[int] = []
        for row_key in sorted(nodes_by_row.keys()):
            row_nodes = sorted(
                nodes_by_row[row_key],
                key=lambda n: self.G.nodes[n]["x_center"],
            )
            reading_order.extend(row_nodes)

        return reading_order

    def detect_tables(self, min_columns: int = 2, min_rows: int = 2) -> list[list[int]]:
        """Detect table structures using community detection on aligned blocks.

        A table is a group of blocks that share row/column alignment.

        Args:
            min_columns: Minimum number of columns to consider a table.
            min_rows: Minimum number of rows to consider a table.

        Returns:
            List of tables, each a list of node indices.
        """
        if len(self.G) < min_columns * min_rows:
            return []

        # Find connected components of same_row edges (these form table rows)
        row_graph = nx.Graph()
        for u, v, data in self.G.edges(data=True):
            if data.get("relation") == "same_row":
                row_graph.add_edge(u, v)

        tables: list[list[int]] = []
        for component in nx.connected_components(row_graph):
            if len(component) >= min_columns:
                # Check if there are multiple rows in this component
                subgraph = self.G.subgraph(component)
                rows: set[int] = set()
                for node in subgraph.nodes():
                    rows.add(int(subgraph.nodes[node]["y_center"] / self.row_tolerance))
                if len(rows) >= min_rows:
                    tables.append(sorted(component))

        return tables

    def get_regions(self) -> list[list[int]]:
        """Group blocks into regions using connected components.

        Returns:
            List of regions, each a list of node indices.
        """
        if len(self.G) == 0:
            return []

        # Use proximity edges for region detection
        proximity_graph = nx.Graph()
        for u, v, data in self.G.edges(data=True):
            if data.get("relation") == "near":
                proximity_graph.add_edge(u, v, weight=data.get("weight", 1.0))

        # Also add same_row/same_column edges
        for u, v, data in self.G.edges(data=True):
            if data.get("relation") in ("same_row", "same_column"):
                proximity_graph.add_edge(u, v)

        components = list(nx.connected_components(proximity_graph))
        return [sorted(comp) for comp in components]

    def to_graphml(self) -> str:
        """Serialize the graph to GraphML format.

        Note: bbox tuples are converted to strings since GraphML
        does not support tuple/list attribute types.
        """
        # GraphML doesn't support tuple/list attributes — convert bbox to string
        export_graph = nx.Graph()
        for node, data in self.G.nodes(data=True):
            export_data = dict(data)
            if "bbox" in export_data and isinstance(export_data["bbox"], tuple):
                export_data["bbox"] = ",".join(str(v) for v in export_data["bbox"])
            export_graph.add_node(node, **export_data)
        for u, v, data in self.G.edges(data=True):
            export_graph.add_edge(u, v, **data)
        result = ""
        for line in nx.generate_graphml(export_graph):
            result += line
        return result

    def analyze(self) -> LayoutGraphResult:
        """Run full analysis and return structured results."""
        reading_order = self.get_reading_order()
        tables = self.detect_tables()
        regions = self.get_regions()

        return LayoutGraphResult(
            num_blocks=len(self.G),
            num_edges=len(self.G.edges()),
            reading_order=reading_order,
            tables=tables,
            regions=regions,
            graph_ml=self.to_graphml(),
        )


def build_layout_graph_from_labels(
    labels_data: list[dict[str, Any]],
    row_tolerance: float = 20.0,
    col_tolerance: float = 20.0,
) -> DocumentLayoutGraph:
    """Build a layout graph from label/OCR output data.

    Args:
        labels_data: List of dicts with 'text', 'bbox' (or 'quad_boxes'),
                     and optional 'label', 'confidence' fields.
        row_tolerance: Vertical pixel tolerance for same-row detection.
        col_tolerance: Horizontal pixel tolerance for same-column detection.

    Returns:
        Configured DocumentLayoutGraph with blocks added.
    """
    graph = DocumentLayoutGraph(row_tolerance=row_tolerance, col_tolerance=col_tolerance)

    blocks: list[LayoutBlock] = []
    for item in labels_data:
        text = item.get("text", item.get("label", ""))
        bbox = item.get("bbox", item.get("quad_boxes", [0, 0, 0, 0]))

        # Handle quad_boxes format (8 values → take bounding rect)
        if isinstance(bbox, list) and len(bbox) == 8:
            xs = [bbox[0], bbox[2], bbox[4], bbox[6]]
            ys = [bbox[1], bbox[3], bbox[5], bbox[7]]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        elif isinstance(bbox, list) and len(bbox) == 4:
            bbox = tuple(bbox)
        else:
            bbox = (0.0, 0.0, 0.0, 0.0)

        blocks.append(
            LayoutBlock(
                text=str(text),
                bbox=bbox,
                label=str(item.get("label", "text")),
                confidence=float(item.get("confidence", 1.0)),
            )
        )

    graph.add_blocks(blocks)
    return graph
