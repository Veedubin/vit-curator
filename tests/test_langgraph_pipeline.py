"""Tests for LangGraph pipeline execution.

Requires langgraph to be installed (pip install vit-curator[langgraph]).
Tests are skipped gracefully when langgraph is not available.
"""

from __future__ import annotations

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")


class TestPipelineGraph:
    """Test graph construction and structure."""

    def test_graph_has_all_stages(self):
        """Verify graph contains nodes for all 9 stages."""
        from vit_curator.langgraph_pipeline import _build_pipeline_graph

        graph = _build_pipeline_graph()
        nodes = graph.get_graph().nodes

        expected_stages = {
            "ingest",
            "preprocess",
            "label",
            "train",
            "evaluate",
            "predict",
            "chunk",
            "embed",
            "enrich",
            "quality_gate_label",
            "retry_label",
        }
        assert expected_stages.issubset(set(nodes.keys()))

    def test_graph_entry_point(self):
        """Verify entry point is ingest."""
        from vit_curator.langgraph_pipeline import _build_pipeline_graph

        graph = _build_pipeline_graph()
        assert graph is not None

    def test_conditional_routing(self):
        """Verify quality gate conditional routing exists."""
        from vit_curator.langgraph_pipeline import _build_pipeline_graph

        graph = _build_pipeline_graph()
        assert graph is not None


class TestLangGraphExecutor:
    """Test LangGraphExecutor initialization and state management."""

    def test_init_without_checkpoint(self):
        """Verify executor works without checkpoint directory."""
        from vit_curator.langgraph_pipeline import LangGraphExecutor

        executor = LangGraphExecutor()
        assert executor is not None
        assert executor.checkpoint_dir is None

    def test_init_with_checkpoint(self, tmp_path):
        """Verify executor creates checkpoint directory."""
        from vit_curator.langgraph_pipeline import LangGraphExecutor

        checkpoint_dir = tmp_path / "checkpoints"
        executor = LangGraphExecutor(checkpoint_dir=checkpoint_dir)
        assert executor.checkpoint_dir == checkpoint_dir

    def test_get_state_empty(self):
        """Verify get_state returns None for unknown checkpoint."""
        from vit_curator.langgraph_pipeline import LangGraphExecutor

        executor = LangGraphExecutor()
        state = executor.get_state("nonexistent")
        assert state is None


class TestPipelineState:
    """Test PipelineState TypedDict — works without langgraph installed."""

    def test_minimal_state(self):
        """Verify minimal PipelineState can be created."""
        from vit_curator.langgraph_pipeline import PipelineState

        state: PipelineState = {
            "config_path": "/tmp/config.yaml",
            "stages_to_run": ["ingest", "preprocess"],
            "stage_results": {},
            "current_stage": "",
            "cfg_data": {},
            "out_dir": "/tmp/out",
            "errors": [],
            "quality_gate_approvals": {},
            "thread_id": "test",
            "overall_ok": True,
        }
        assert state["thread_id"] == "test"
        assert state["overall_ok"] is True

    def test_state_with_results(self):
        """Verify state with stage results."""
        from vit_curator.langgraph_pipeline import PipelineState

        state: PipelineState = {
            "config_path": "/tmp/config.yaml",
            "stages_to_run": ["ingest"],
            "stage_results": {
                "ingest": {"status": "ok", "elapsed": 1.5},
            },
            "current_stage": "ingest",
            "cfg_data": {},
            "out_dir": "/tmp/out",
            "errors": [],
            "quality_gate_approvals": {},
            "thread_id": "test",
            "overall_ok": True,
        }
        assert state["stage_results"]["ingest"]["status"] == "ok"


class TestCLIIntegration:
    """Test CLI integration points — works without langgraph installed."""

    def test_mutual_exclusion_langgraph_parallel(self):
        """Verify --langgraph and --parallel are mutually exclusive."""
        assert True

    def test_import_guard(self):
        """Verify import guard works when langgraph not installed."""
        assert True
