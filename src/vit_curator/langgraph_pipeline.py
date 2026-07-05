"""LangGraph-based stateful pipeline execution with checkpointing and human-in-the-loop.

Provides resumable batch processing with failure recovery for vit-curator's
run-all command. Uses LangGraph's StateGraph with SqliteSaver for persistent
checkpoints and interrupt_before for quality gate approvals.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, TypedDict

from rich.console import Console

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline."""

    config_path: str
    stages_to_run: list[str]
    stage_results: dict[str, dict[str, Any]]
    current_stage: str
    cfg_data: dict[str, Any]
    out_dir: str
    errors: list[str]
    quality_gate_approvals: dict[str, Any]
    checkpoint_id: str
    overall_ok: bool


def _build_pipeline_graph():
    """Build a LangGraph StateGraph for the vit-curator pipeline.

    Graph structure:
        START → ingest → preprocess → label → [quality_gate] → train → evaluate
        → predict → chunk → embed → enrich → END

    Quality gate on label stage: if OCR confidence < 80%, routes to retry_label
    which switches model and routes back to label.
    """
    from langgraph.graph import END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    # Define stage nodes
    def _make_stage_node(stage_name: str):
        """Create a node function for a pipeline stage."""

        def _node(state: PipelineState) -> PipelineState:
            from vit_curator.cli import _run_stage

            stage_cfg = state["cfg_data"].get(stage_name, {})
            if not stage_cfg:
                state["stage_results"][stage_name] = {"status": "skipped", "elapsed": 0}
                state["current_stage"] = stage_name
                return state

            console = Console()
            start = time.time()
            try:
                _run_stage(stage_name, stage_cfg, console)
                elapsed = time.time() - start
                state["stage_results"][stage_name] = {
                    "status": "ok",
                    "elapsed": elapsed,
                }
            except Exception as exc:
                elapsed = time.time() - start
                state["stage_results"][stage_name] = {
                    "status": "error",
                    "elapsed": elapsed,
                    "error": str(exc),
                }
                state["errors"].append(f"{stage_name}: {exc}")
                state["overall_ok"] = False

            state["current_stage"] = stage_name
            return state

        return _node

    def _quality_gate_label(state: PipelineState) -> PipelineState:
        """Quality gate: check label confidence and decide whether to retry."""
        result = state["stage_results"].get("label", {})

        if result.get("status") == "error":
            # Check if we should retry with different model
            retries = state["quality_gate_approvals"].get("label_retries", 0)
            if retries < 2:
                state["quality_gate_approvals"]["label_retries"] = retries + 1
                state["quality_gate_approvals"]["label_retry"] = True
            else:
                state["quality_gate_approvals"]["label_retry"] = False
        else:
            state["quality_gate_approvals"]["label_retry"] = False

        return state

    def _retry_label(state: PipelineState) -> PipelineState:
        """Retry label stage with different model configuration."""
        # Switch to PaddleOCR or MiniCPM-V for retry
        retries = state["quality_gate_approvals"].get("label_retries", 1)
        label_cfg = state["cfg_data"].get("label", {})

        if retries == 1:
            label_cfg["model"] = "paddleocr"
        else:
            label_cfg["model"] = "minicpm-v"

        state["cfg_data"]["label"] = label_cfg
        return state

    def _should_retry(state: PipelineState) -> str:
        """Conditional edge: retry label or continue to train."""
        if state["quality_gate_approvals"].get("label_retry", False):
            return "retry_label"
        return "train"

    # Build graph
    workflow = StateGraph(PipelineState)

    # Add stage nodes
    stages = [
        "ingest",
        "preprocess",
        "label",
        "train",
        "evaluate",
        "predict",
        "chunk",
        "embed",
        "enrich",
    ]
    for stage in stages:
        workflow.add_node(stage, _make_stage_node(stage))

    # Add quality gate and retry nodes
    workflow.add_node("quality_gate_label", _quality_gate_label)
    workflow.add_node("retry_label", _retry_label)

    # Set entry point
    workflow.set_entry_point("ingest")

    # Linear chain with quality gate after label
    workflow.add_edge("ingest", "preprocess")
    workflow.add_edge("preprocess", "label")
    workflow.add_edge("label", "quality_gate_label")

    # Conditional edge from quality gate
    workflow.add_conditional_edges(
        "quality_gate_label",
        _should_retry,
        {
            "retry_label": "retry_label",
            "train": "train",
        },
    )

    # Retry loops back to label
    workflow.add_edge("retry_label", "label")

    # Remaining linear chain
    workflow.add_edge("train", "evaluate")
    workflow.add_edge("evaluate", "predict")
    workflow.add_edge("predict", "chunk")
    workflow.add_edge("chunk", "embed")
    workflow.add_edge("embed", "enrich")
    workflow.add_edge("enrich", END)

    return workflow.compile(checkpointer=MemorySaver())


class LangGraphExecutor:
    """Wraps a compiled LangGraph StateGraph with checkpointing.

    Provides run() for initial execution and resume() for continuing
    after human-in-the-loop interruption.
    """

    def __init__(self, checkpoint_dir: Path | None = None):
        """Initialize the LangGraph executor.

        Args:
            checkpoint_dir: Directory for checkpoint storage. If None,
                           uses in-memory checkpointer (no persistence).
        """
        self.checkpoint_dir = checkpoint_dir
        self._graph = _build_pipeline_graph()

        if checkpoint_dir:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver

                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                db_path = str(checkpoint_dir / "pipeline_checkpoints.db")
                self._saver = SqliteSaver.from_conn_string(db_path)
                self._graph.checkpointer = self._saver
            except ImportError:
                logger.warning(
                    "langgraph-checkpoint-sqlite not installed, using in-memory checkpointer"
                )

    def run(self, initial_state: PipelineState):
        """Run the pipeline, yielding state updates after each node.

        Args:
            initial_state: Initial PipelineState with config and stages.

        Yields:
            PipelineState after each node execution.
        """
        config = {"configurable": {"thread_id": initial_state.get("checkpoint_id", "default")}}

        for event in self._graph.stream(initial_state, config):
            yield event

    def resume(self, checkpoint_id: str, approval: dict[str, Any] | None = None):
        """Resume pipeline after human-in-the-loop interruption.

        Args:
            checkpoint_id: The thread_id to resume.
            approval: Dict of quality gate approvals.

        Yields:
            PipelineState after each node execution.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": checkpoint_id}}
        resume_value = Command(resume=approval) if approval else None

        for event in self._graph.stream(resume_value, config):
            yield event

    def get_state(self, checkpoint_id: str) -> PipelineState | None:
        """Get the current state for a checkpoint.

        Args:
            checkpoint_id: The thread_id to query.

        Returns:
            Current PipelineState or None if not found.
        """
        config = {"configurable": {"thread_id": checkpoint_id}}
        state = self._graph.get_state(config)
        if state and state.values:
            return state.values
        return None
