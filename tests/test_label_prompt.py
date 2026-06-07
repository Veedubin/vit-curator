"""Tests for vit_curator.label.prompt — LabelSet, PromptBuilder, build_prompt."""

from __future__ import annotations

import hashlib

import orjson
import pytest

from vit_curator.label.prompt import LabelSet, OutputConfig, build_prompt, load_labelset


def test_labelset_rejects_duplicate_ids() -> None:
    """Test that LabelSet rejects labels with duplicate IDs."""
    with pytest.raises(Exception):  # noqa: B017
        LabelSet.model_validate(
            {
                "version": "v",
                "labels": [
                    {"id": 1, "name": "a", "desc": "aa"},
                    {"id": 1, "name": "b", "desc": "bb"},
                ],
            }
        )


def test_build_prompt_deterministic() -> None:
    """Test that build_prompt produces deterministic output regardless of label order."""
    ls1 = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 2, "name": "b", "desc": "bb"},
                {"id": 1, "name": "a", "desc": "aa"},
            ],
        }
    )
    ls2 = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "a", "desc": "aa"},
                {"id": 2, "name": "b", "desc": "bb"},
            ],
        }
    )

    b1 = build_prompt(ls1)
    b2 = build_prompt(ls2)

    assert b1.prompt == b2.prompt
    assert b1.schema == b2.schema
    assert b1.prompt_version == b2.prompt_version
    assert b1.allowed_label_ids == (1, 2)


def test_prompt_minimizes_whitespace() -> None:
    """Test that prompts normalize whitespace in label descriptions."""
    ls = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "  A  ", "desc": "line1\n   line2"},
            ],
        }
    )
    b = build_prompt(ls)
    assert "1=A (line1 line2)" in b.prompt


def test_prompt_schema_has_labels_and_restricts_ids() -> None:
    """Test that the generated JSON schema is properly structured."""
    ls = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "a", "desc": "aa"},
                {"id": 2, "name": "b", "desc": "bb"},
            ],
        }
    )
    b = build_prompt(ls)

    assert "Labels:" in b.prompt
    assert b.schema is not None
    assert b.schema["type"] == "object"
    assert b.schema["additionalProperties"] is False
    assert b.schema["required"] == ["labels", "text"]

    labels_schema = b.schema["properties"]["labels"]
    assert labels_schema["type"] == "array"
    assert labels_schema["uniqueItems"] is True
    assert labels_schema.get("minItems", 0) == 0

    items = labels_schema["items"]
    assert items["type"] == "integer"
    assert items["enum"] == [1, 2]


def test_prompt_schema_includes_optional_fields() -> None:
    """Test that optional fields are included in schema when configured."""
    ls = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "a", "desc": "aa"},
            ],
        }
    )
    b = build_prompt(
        ls,
        output=OutputConfig(include_subject=True, include_entities=True, include_summary=True),
    )
    assert b.schema is not None
    assert b.schema["properties"]["text"]["type"] == "string"
    assert b.schema["properties"]["subject"]["type"] == "string"
    assert b.schema["properties"]["entities"]["type"] == "array"
    assert b.schema["properties"]["summary"]["type"] == "string"
    assert set(b.schema["required"]) == {"labels", "text", "subject", "entities", "summary"}


def test_prompt_schema_labels_only() -> None:
    """Test labels-only mode omits text field from required and schema."""
    ls = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "a", "desc": "aa"},
            ],
        }
    )
    b = build_prompt(ls, output=OutputConfig(labels_only=True))
    assert b.schema is not None
    assert b.schema["required"] == ["labels"]
    assert "text" not in b.schema["properties"]


def test_prompt_version_changes_when_labels_change() -> None:
    """Test that changing label descriptions changes the prompt version."""
    ls = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "a", "desc": "aa"},
            ],
        }
    )
    b1 = build_prompt(ls)

    ls_changed = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "a", "desc": "aa changed"},
            ],
        }
    )
    b2 = build_prompt(ls_changed)

    assert b1.prompt_version != b2.prompt_version


def test_prompt_version_is_sha256() -> None:
    """Test that prompt_version is a deterministic SHA-256 hash."""
    ls = LabelSet.model_validate(
        {
            "version": "v",
            "labels": [
                {"id": 1, "name": "a", "desc": "aa"},
                {"id": 2, "name": "b", "desc": "bb"},
            ],
        }
    )
    b = build_prompt(ls)
    assert b.schema is not None

    h = hashlib.sha256()
    h.update(ls.version.encode("utf-8"))
    h.update(b.prompt.encode("utf-8"))
    h.update(orjson.dumps(b.schema, option=orjson.OPT_SORT_KEYS))
    assert b.prompt_version == h.hexdigest()


def test_load_labelset_round_trip(tmp_path) -> None:
    """Test loading a LabelSet from a JSON file."""
    p = tmp_path / "labels.json"
    p.write_bytes(
        orjson.dumps(
            {
                "version": "v",
                "labels": [
                    {"id": 2, "name": "b", "desc": "bb"},
                    {"id": 1, "name": "a", "desc": "aa"},
                ],
            }
        )
    )
    ls = load_labelset(str(p))
    # Validator should normalize ordering
    assert [label.id for label in ls.labels] == [1, 2]
