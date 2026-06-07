"""Tests for vit_curator.post.enrich — LLM document enrichment.

All network calls are mocked; no real HTTP requests are made.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import UTC, datetime

import duckdb
import pytest

from vit_curator.config import EnrichConfig
from vit_curator.post.enrich import (
    Enricher,
    EnrichmentResult,
    _count_target_files,
    _extract_json_from_text,
    _fetch_target_files,
    _insert_enrichment,
    build_system_prompt,
    build_user_prompt,
    call_llm,
    parse_enrichment_json,
    run_enrichment,
)
from vit_curator.shared.db import ensure_schema

SKIP_HTTPX = not importlib.util.find_spec("httpx")


def _insert_file(con: duckdb.DuckDBPyConnection, file_pk: int, rel_path: str) -> None:
    """Insert a minimal file row."""

    rel_blob = os.fsencode(rel_path)
    con.execute(
        "INSERT INTO files "
        "(file_pk, rel_path_blob, rel_path_hash, ext_blob, size_bytes, mtime_ns, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [file_pk, rel_blob, rel_blob + b"_hash", os.fsencode("jpg"), 1234, 1000000, 1],
    )


def _insert_prediction(con: duckdb.DuckDBPyConnection, file_pk: int, text: str) -> None:
    """Insert a prediction row with text."""
    con.execute(
        "INSERT INTO predictions "
        "(file_pk, run_id, labels, text, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [file_pk, "00000000-0000-0000-0000-000000000001", [1], text, datetime.now(UTC).isoformat()],
    )


@pytest.fixture
def fresh_db(tmp_path) -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB with schema initialized."""
    con = duckdb.connect(str(tmp_path / "test.duckdb"))
    ensure_schema(con)
    return con


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_build_system_prompt_returns_string() -> None:
    """System prompt should be non-empty and mention JSON."""
    prompt = build_system_prompt()
    assert isinstance(prompt, str)
    assert "JSON" in prompt
    assert "subject" in prompt
    assert "summary" in prompt


def test_build_user_prompt_returns_string() -> None:
    """User prompt should include file path and text."""
    doc_text = "Hello world"
    file_path = "/tmp/test.jpg"
    prompt = build_user_prompt(doc_text, file_path)
    assert isinstance(prompt, str)
    assert file_path in prompt
    assert doc_text in prompt


# ---------------------------------------------------------------------------
# JSON extraction / parsing
# ---------------------------------------------------------------------------


def test_extract_json_from_text_plain() -> None:
    """Plain JSON should be returned unchanged."""
    raw = '{"subject": "S"}'
    assert _extract_json_from_text(raw) == raw


def test_extract_json_from_text_fenced() -> None:
    """JSON inside markdown code fences should be stripped."""
    raw = '```json\n{"subject": "S"}\n```'
    assert _extract_json_from_text(raw) == '{"subject": "S"}'


def test_extract_json_from_text_backtick_fenced() -> None:
    """JSON inside triple backticks without json tag."""
    raw = '```\n{"subject": "S"}\n```'
    assert _extract_json_from_text(raw) == '{"subject": "S"}'


def test_parse_enrichment_json_valid() -> None:
    """Valid JSON should produce a fully populated EnrichmentResult."""
    payload = json.dumps(
        {
            "subject": "Meeting Notes",
            "summary": "Discussed Q3 goals",
            "entities": {"persons": ["Alice"], "organizations": ["Acme"]},
            "tags": ["memo", "internal"],
        }
    )
    result = parse_enrichment_json(payload)
    assert result.subject == "Meeting Notes"
    assert result.summary == "Discussed Q3 goals"
    assert result.doc_type == "memo"
    assert result.entities_json is not None
    assert result.tags_json is not None


def test_parse_enrichment_json_malformed() -> None:
    """Malformed JSON should return an empty EnrichmentResult."""
    result = parse_enrichment_json("not json")
    assert result.subject is None
    assert result.summary is None
    assert result.doc_type is None
    assert result.entities_json is None
    assert result.tags_json is None


def test_parse_enrichment_json_empty() -> None:
    """Empty string should return an empty EnrichmentResult."""
    result = parse_enrichment_json("")
    assert result.subject is None
    assert result.summary is None


def test_parse_enrichment_json_partial() -> None:
    """Partial JSON (only subject) should still work."""
    result = parse_enrichment_json('{"subject": "OnlySubject"}')
    assert result.subject == "OnlySubject"
    assert result.summary is None
    assert result.doc_type is None
    assert result.entities_json is None
    assert result.tags_json is None


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


def test_enrichment_result_defaults() -> None:
    """EnrichmentResult defaults should all be None."""
    result = EnrichmentResult()
    assert result.subject is None
    assert result.summary is None
    assert result.doc_type is None
    assert result.entities_json is None
    assert result.tags_json is None


def test_enrichment_result_fields() -> None:
    """EnrichmentResult should store provided fields."""
    result = EnrichmentResult(
        subject="S", summary="Sum", doc_type="D", entities_json="{}", tags_json="[]"
    )
    assert result.subject == "S"
    assert result.summary == "Sum"
    assert result.doc_type == "D"
    assert result.entities_json == "{}"
    assert result.tags_json == "[]"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def test_count_target_files_zero(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """When there are no predictions, count should be 0."""
    n = _count_target_files(fresh_db, "test-model", reprocess_existing=False)
    assert n == 0


def test_count_target_files_with_predictions(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Count should reflect predictions with text."""
    _insert_file(fresh_db, 1, "/a.jpg")
    _insert_prediction(fresh_db, 1, "some text")
    n = _count_target_files(fresh_db, "test-model", reprocess_existing=True)
    assert n == 1


def test_fetch_target_files(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Fetch should return (file_pk, text, rel_path) tuples."""
    _insert_file(fresh_db, 1, "/a.jpg")
    _insert_prediction(fresh_db, 1, "hello")
    rows = _fetch_target_files(fresh_db, "test-model", max_docs=None, reprocess_existing=True)
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] == "hello"
    assert rows[0][2] == "/a.jpg"


def test_insert_enrichment(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """Insert should write a row to doc_enrichments."""
    _insert_file(fresh_db, 1, "/a.jpg")
    result = EnrichmentResult(subject="S", summary="Sum")
    _insert_enrichment(fresh_db, 1, "m", result, "stop", False, 10, 2, "{}")
    row = fresh_db.execute(
        "SELECT subject, summary, finish_reason, truncated FROM doc_enrichments WHERE file_pk = 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "S"
    assert row[1] == "Sum"
    assert row[2] == "stop"
    assert row[3] is False


# ---------------------------------------------------------------------------
# Enricher class
# ---------------------------------------------------------------------------


def test_enricher_init_default() -> None:
    """Enricher should create a default EnrichConfig when none is given."""
    enricher = Enricher()
    assert isinstance(enricher.config, EnrichConfig)


def test_enricher_init_custom() -> None:
    """Enricher should accept a custom EnrichConfig."""
    cfg = EnrichConfig(db_path=EnrichConfig.__dataclass_fields__["db_path"].default, model="custom")
    enricher = Enricher(cfg)
    assert enricher.config.model == "custom"


def test_enricher_enrich_no_targets(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """When no targets exist, enrich should return 0 without error."""
    enricher = Enricher()
    n = enricher.enrich(fresh_db)
    assert n == 0


def test_enricher_enrich_invalid_config() -> None:
    """Invalid token/word ratios should raise ValueError."""

    cfg = EnrichConfig()
    object.__setattr__(cfg, "tokens_per_word", 0)
    enricher = Enricher(cfg)
    with pytest.raises(ValueError, match="tokens_per_word and chars_per_word must be > 0"):
        enricher.enrich(duckdb.connect(":memory:"))


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def test_run_enrichment_no_targets(fresh_db: duckdb.DuckDBPyConnection) -> None:
    """run_enrichment should return 0 when there is nothing to enrich."""
    n = run_enrichment(fresh_db)
    assert n == 0


# ---------------------------------------------------------------------------
# httpx-dependent tests (skipped if httpx is not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(SKIP_HTTPX, reason="httpx not installed")
def test_call_llm_mock(monkeypatch) -> None:
    """call_llm should parse the response from an OpenAI-compatible endpoint."""
    import httpx

    def _mock_post(*args, **kwargs):
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"subject": "Test"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
            request=httpx.Request("POST", "http://test"),
        )
        return response

    monkeypatch.setattr(httpx.Client, "post", _mock_post)
    content, finish = call_llm(
        server_url="http://localhost:9001",
        api_key="",
        model="test-model",
        max_output_tokens=64,
        doc_text="hello",
        file_path="/tmp/test.jpg",
    )
    assert content == '{"subject": "Test"}'
    assert finish == "stop"


@pytest.mark.skipif(SKIP_HTTPX, reason="httpx not installed")
def test_call_llm_list_content(monkeypatch) -> None:
    """call_llm should handle list-shaped content (some models return arrays)."""
    import httpx

    def _mock_post(*args, **kwargs):
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": [{"text": "part1"}, {"text": "part2"}]},
                        "finish_reason": "length",
                    }
                ]
            },
            request=httpx.Request("POST", "http://test"),
        )
        return response

    monkeypatch.setattr(httpx.Client, "post", _mock_post)
    content, finish = call_llm(
        server_url="http://localhost:9001",
        api_key="",
        model="m",
        max_output_tokens=64,
        doc_text="t",
        file_path="p",
    )
    assert content == "part1\npart2"
    assert finish == "length"


@pytest.mark.skipif(SKIP_HTTPX, reason="httpx not installed")
def test_call_llm_auth_header(monkeypatch) -> None:
    """call_llm should include Authorization header when api_key is provided."""
    import httpx

    captured = {}

    def _mock_post(self, url, *, headers=None, **kwargs):
        captured["headers"] = headers
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "{}"},
                        "finish_reason": "stop",
                    }
                ]
            },
            request=httpx.Request("POST", "http://test"),
        )

    monkeypatch.setattr(httpx.Client, "post", _mock_post)
    call_llm(
        server_url="http://localhost:9001",
        api_key="secret-key",
        model="m",
        max_output_tokens=64,
        doc_text="t",
        file_path="p",
    )
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


@pytest.mark.skipif(SKIP_HTTPX, reason="httpx not installed")
def test_enricher_enrich_with_mock_llm(fresh_db: duckdb.DuckDBPyConnection, monkeypatch) -> None:
    """Enricher should enrich predictions when LLM call is mocked."""
    import httpx

    _insert_file(fresh_db, 1, "/a.jpg")
    _insert_prediction(fresh_db, 1, "Document about machine learning.")

    def _mock_post(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "subject": "ML Doc",
                                    "summary": "Overview of ML.",
                                    "entities": {"persons": [], "organizations": ["OpenAI"]},
                                    "tags": ["tech"],
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            request=httpx.Request("POST", "http://test"),
        )

    monkeypatch.setattr(httpx.Client, "post", _mock_post)

    cfg = EnrichConfig(
        db_path=EnrichConfig.__dataclass_fields__["db_path"].default,
        model="test-model",
        server_url="http://localhost:9001",
        max_tokens=100,
        max_output_tokens=64,
    )
    enricher = Enricher(cfg)
    n = enricher.enrich(fresh_db)
    assert n == 1

    row = fresh_db.execute(
        "SELECT subject, summary FROM doc_enrichments WHERE file_pk = 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "ML Doc"
    assert row[1] == "Overview of ML."


@pytest.mark.skipif(SKIP_HTTPX, reason="httpx not installed")
def test_enricher_enrich_skip_too_long(fresh_db: duckdb.DuckDBPyConnection, monkeypatch) -> None:
    """Enricher should skip too-long docs when skip_too_long=True."""
    import httpx

    long_text = "word " * 1000
    _insert_file(fresh_db, 1, "/a.jpg")
    _insert_prediction(fresh_db, 1, long_text)

    def _mock_post(*args, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", "http://test"),
        )

    monkeypatch.setattr(httpx.Client, "post", _mock_post)

    cfg = EnrichConfig(
        db_path=EnrichConfig.__dataclass_fields__["db_path"].default,
        model="m",
        server_url="http://localhost:9001",
        max_tokens=10,
        skip_too_long=True,
    )
    enricher = Enricher(cfg)
    n = enricher.enrich(fresh_db)
    assert n == 0


@pytest.mark.skipif(SKIP_HTTPX, reason="httpx not installed")
def test_enricher_enrich_truncates(fresh_db: duckdb.DuckDBPyConnection, monkeypatch) -> None:
    """Enricher should truncate (not skip) when skip_too_long=False."""
    import httpx

    long_text = "word " * 1000
    _insert_file(fresh_db, 1, "/a.jpg")
    _insert_prediction(fresh_db, 1, long_text)

    def _mock_post(*args, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", "http://test"),
        )

    monkeypatch.setattr(httpx.Client, "post", _mock_post)

    cfg = EnrichConfig(
        db_path=EnrichConfig.__dataclass_fields__["db_path"].default,
        model="m",
        server_url="http://localhost:9001",
        max_tokens=10,
        skip_too_long=False,
    )
    enricher = Enricher(cfg)
    n = enricher.enrich(fresh_db)
    assert n == 1

    row = fresh_db.execute("SELECT truncated FROM doc_enrichments WHERE file_pk = 1").fetchone()
    assert row is not None
    assert row[0] is True
