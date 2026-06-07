"""Stage 3: VLM-based image labeling."""

from __future__ import annotations

from vit_curator.label.client import VllmClient, VllmStructuredMode
from vit_curator.label.dispatcher import DispatchConfig
from vit_curator.label.metrics import RunMetrics, sample_gpu_info
from vit_curator.label.prompt import LabelSet, OutputConfig, build_prompt, load_labelset
from vit_curator.label.scheduler import AutoTune, DynamicConcurrency
from vit_curator.label.store import connect_label_db

__all__ = [
    "AutoTune",
    "DispatchConfig",
    "DynamicConcurrency",
    "LabelSet",
    "OutputConfig",
    "RunMetrics",
    "VllmClient",
    "VllmStructuredMode",
    "build_prompt",
    "connect_label_db",
    "load_labelset",
    "sample_gpu_info",
]
