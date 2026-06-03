from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_HIGH_PRECISION_RELATIONS = (
    "import_relation",
    "test_source_relation",
    "hyperlink_relation",
)

DEFAULT_OPTIONAL_AFTER_AUDIT_RELATIONS = ("config_script_relation",)

DEFAULT_NOISY_RELATIONS = (
    "readme_code_relation",
    "docs_code_relation",
    "example_code_relation",
)

DEFAULT_RELATION_RELIABILITY = {
    "import_relation": 1.0,
    "test_source_relation": 0.95,
    "hyperlink_relation": 0.95,
    "config_script_relation": 0.75,
    "readme_code_relation": 0.20,
    "docs_code_relation": 0.25,
    "example_code_relation": 0.15,
}


@dataclass(frozen=True)
class RelationConfig:
    name: str = "main_high_precision"
    allowed_relations: tuple[str, ...] = DEFAULT_HIGH_PRECISION_RELATIONS
    optional_relations: tuple[str, ...] = DEFAULT_OPTIONAL_AFTER_AUDIT_RELATIONS
    noisy_relations: tuple[str, ...] = DEFAULT_NOISY_RELATIONS
    relation_reliability: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RELATION_RELIABILITY)
    )


def load_relation_config(path: str | Path | None = None) -> RelationConfig:
    if path is None:
        return RelationConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    reliability = dict(DEFAULT_RELATION_RELIABILITY)
    reliability.update(
        {
            str(relation): float(value)
            for relation, value in dict(raw.get("relation_reliability", {})).items()
        }
    )

    return RelationConfig(
        name=str(raw.get("name") or config_path.stem),
        allowed_relations=_tuple(raw.get("allowed_relations"), DEFAULT_HIGH_PRECISION_RELATIONS),
        optional_relations=_tuple(
            raw.get("optional_relations") or raw.get("optional_after_audit"),
            DEFAULT_OPTIONAL_AFTER_AUDIT_RELATIONS,
        ),
        noisy_relations=_tuple(raw.get("noisy_relations"), DEFAULT_NOISY_RELATIONS),
        relation_reliability=reliability,
    )


def _tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
