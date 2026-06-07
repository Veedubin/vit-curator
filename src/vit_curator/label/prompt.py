"""Pydantic label definitions, prompt + JSON schema builder."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import orjson
from pydantic import BaseModel, Field, field_validator


class LabelDef(BaseModel):
    id: int = Field(..., ge=1)
    name: str = Field(..., min_length=1, max_length=64)
    desc: str = Field(..., min_length=1, max_length=120)

    @field_validator("name", "desc")
    @classmethod
    def _normalize_whitespace(cls, v: str) -> str:
        v2 = " ".join(v.split())
        if not v2:
            raise ValueError("must not be blank")
        return v2


class LabelSet(BaseModel):
    version: str
    labels: list[LabelDef]

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: list[LabelDef]) -> list[LabelDef]:
        ids = [x.id for x in v]
        if len(ids) != len(set(ids)):
            raise ValueError("labels contain duplicate ids")
        return sorted(v, key=lambda x: x.id)


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    schema: dict | None
    prompt_version: str  # sha256 hex of prompt + schema + label version
    allowed_label_ids: tuple[int, ...]


@dataclass(frozen=True)
class OutputConfig:
    include_text: bool = True
    include_subject: bool = False
    include_entities: bool = False
    include_summary: bool = False
    labels_only: bool = False

    def normalized(self) -> OutputConfig:
        if not self.labels_only:
            return self
        return OutputConfig(
            include_text=False,
            include_subject=False,
            include_entities=False,
            include_summary=False,
            labels_only=True,
        )


def load_labelset(path: str) -> LabelSet:
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    return LabelSet.model_validate(data)


def _select_labels(labelset: LabelSet, enabled_ids: Sequence[int] | None) -> list[LabelDef]:
    labels = sorted(labelset.labels, key=lambda x: x.id)
    if enabled_ids is None:
        return labels
    enabled = set(enabled_ids)
    return [label for label in labels if label.id in enabled]


def _output_fields(cfg: OutputConfig) -> list[str]:
    fields = ["labels"]
    if cfg.include_text:
        fields.append("text")
    if cfg.include_subject:
        fields.append("subject")
    if cfg.include_entities:
        fields.append("entities")
    if cfg.include_summary:
        fields.append("summary")
    return fields


def _build_prompt_lines(labels: Sequence[LabelDef], cfg: OutputConfig) -> str:
    lines: list[str] = ["Choose zero or more label IDs that apply to the image."]
    lines.append(f"Output MUST be JSON with keys: {', '.join(_output_fields(cfg))}.")
    lines.append('Use ascending unique integers for "labels". If none apply, output {"labels":[]}.')
    if cfg.include_text:
        lines.append('Include "text" with OCR text content (string).')
    if cfg.include_subject:
        lines.append('Include "subject" as a short noun phrase (string).')
    if cfg.include_entities:
        lines.append('Include "entities" as an array of strings (named entities).')
    if cfg.include_summary:
        lines.append('Include "summary" as 2-3 sentences, <=150 words.')
    lines.append("Labels:")
    for label in labels:
        lines.append(f"{label.id}={label.name} ({label.desc})")
    return "\n".join(lines)


def _build_schema(allowed_ids: Sequence[int], cfg: OutputConfig) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "integer", "enum": list(allowed_ids)},
                "uniqueItems": True,
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }
    optional_fields: list[tuple[str, dict[str, Any], bool]] = [
        ("text", {"type": "string"}, cfg.include_text),
        ("subject", {"type": "string"}, cfg.include_subject),
        ("entities", {"type": "array", "items": {"type": "string"}}, cfg.include_entities),
        ("summary", {"type": "string"}, cfg.include_summary),
    ]
    for name, field_schema, enabled in optional_fields:
        if enabled:
            schema["properties"][name] = field_schema
            schema["required"].append(name)
    return schema


def build_prompt(
    labelset: LabelSet,
    *,
    enabled_ids: Sequence[int] | None = None,
    output: OutputConfig | None = None,
) -> PromptBundle:
    labels = _select_labels(labelset, enabled_ids)
    allowed_ids = tuple(label.id for label in labels)
    cfg = (output or OutputConfig()).normalized()

    prompt = _build_prompt_lines(labels, cfg)
    schema = _build_schema(allowed_ids, cfg)

    h = hashlib.sha256()
    h.update(labelset.version.encode("utf-8"))
    h.update(prompt.encode("utf-8"))
    h.update(orjson.dumps(schema, option=orjson.OPT_SORT_KEYS))
    prompt_version = h.hexdigest()

    return PromptBundle(
        prompt=prompt,
        schema=schema,
        prompt_version=prompt_version,
        allowed_label_ids=allowed_ids,
    )
