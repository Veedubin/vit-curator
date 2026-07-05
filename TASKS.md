# vit-curator — Active Task Backlog

**Last updated:** 2026-07-04 (Session: vit-curator/videre-mcp improvement sprint)
**Branch:** `main`

---

## Open Tasks

- [ ] Test libvips decode backend with real image batches
- [ ] Test NetworkX pipeline DAG with real multi-stage configs
- [ ] Test knowledge graph with real multi-document datasets
- [ ] Test LangGraph pipeline with real multi-stage configs

---

## Closed (2026-07-04 — improvement sprint, round 3)

- [x] P3: LangGraph batch pipelines — new `langgraph_pipeline.py` with `PipelineState` TypedDict, `_build_pipeline_graph()` (StateGraph with 9 stages + quality gate + conditional retry), `LangGraphExecutor` class (run/resume/get_state with SqliteSaver). `--langgraph` flag on `run-all`. Mutual exclusion with `--parallel`. 10 tests. `langgraph>=0.2.0` as optional dep.

## Closed (2026-07-04 — improvement sprint, round 2)

- [x] P2: NetworkX knowledge graph — new `post/knowledge_graph.py` with `ImageKnowledgeGraph` class. Cross-document entity linking, Jaccard similarity search, co-occurrence analysis, concept hierarchy. New `knowledge-graph` CLI command.

## Closed (2026-07-04 — improvement sprint, round 1)

- [x] P0-2: libvips (pyvips) optional decode backend — `decode_rgb_u8_chw_vips()` with PIL fallback. `backend="auto"|"vips"|"pil"` parameter. 3-10x faster batch decode. `pyvips>=2.2.0` as optional dep.
- [x] P1-4: NetworkX pipeline DAG — `--parallel` flag on `run-all`. `_build_pipeline_dag()`, `_run_stages_parallel()` with `ThreadPoolExecutor`, `_run_stages_sequential()`. `networkx>=3.2` added to deps.
- [x] P1-5: NetworkX document layout graphs — new `post/layout_graph.py` with `DocumentLayoutGraph` class. New `layout-graph` CLI command. Reading order inference, table detection, region grouping, GraphML export.
