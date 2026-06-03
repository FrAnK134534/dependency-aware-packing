from dapacking.relation_config import (
    DEFAULT_HIGH_PRECISION_RELATIONS,
    load_relation_config,
)


def test_load_relation_config_defaults_to_high_precision() -> None:
    config = load_relation_config()

    assert config.allowed_relations == DEFAULT_HIGH_PRECISION_RELATIONS
    assert config.relation_reliability["import_relation"] == 1.0
    assert "readme_code_relation" in config.noisy_relations


def test_load_relation_config_overrides_allowed_and_reliability(tmp_path) -> None:
    config_path = tmp_path / "relations.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: custom",
                "allowed_relations:",
                "  - import_relation",
                "relation_reliability:",
                "  import_relation: 0.8",
            ]
        ),
        encoding="utf-8",
    )

    config = load_relation_config(config_path)

    assert config.name == "custom"
    assert config.allowed_relations == ("import_relation",)
    assert config.relation_reliability["import_relation"] == 0.8
