"""Tests for PPR preset system in PrunedHippoRAGNeo4jRetrievalConfig.

Verifies:
- Backward compatibility: configs without preset still work
- Preset application: preset fills in advanced defaults
- Priority: explicit params override preset values
- All presets are valid and produce valid configs
- Helper methods (describe_essential, get_preset, list_presets)
- Cross-field validation warnings
"""
import warnings

import pytest

from config.core.retrieval.pruned_hipporag_neo4j_config import (
    PPR_PRESET_NAMES,
    PrunedHippoRAGNeo4jRetrievalConfig,
    _PPR_PRESETS,
)


# ---------------------------------------------------------------------------
# Minimal graph_config dict (needed for all PrunedHippoRAGNeo4jRetrievalConfig instances).
# We never actually build a Neo4j connection in these unit tests.
# ---------------------------------------------------------------------------
_MINIMAL_GRAPH_CONFIG = {
    "type": "pruned_hipporag_neo4j",
    "embedding": {"type": "qwen_embedding"},
}


def _make_cfg(**overrides):
    """Helper: create a PrunedHippoRAGNeo4jRetrievalConfig with minimal required fields."""
    data = {
        "type": "pruned_hipporag_neo4j_retrieval",
        "graph_config": _MINIMAL_GRAPH_CONFIG,
    }
    data.update(overrides)
    return PrunedHippoRAGNeo4jRetrievalConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Backward compatibility: no preset => uses raw Pydantic defaults
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_preset_uses_field_defaults(self):
        """Configs without a preset field should use the original Pydantic defaults."""
        cfg = _make_cfg()
        assert cfg.preset is None
        # Check some field defaults match the original values
        assert cfg.damping_factor == 0.3
        assert cfg.expansion_hops == 2
        assert cfg.max_neighbors == 30
        assert cfg.fact_retrieval_top_k == 20  # Original default was 20
        assert cfg.ppr_push_epsilon == 1e-6
        assert cfg.dense_file_closure_enabled is True
        assert cfg.dense_file_prior_enabled is True

    def test_explicit_params_without_preset(self):
        """Explicit params without preset should set exactly those values."""
        cfg = _make_cfg(damping_factor=0.5, expansion_hops=3)
        assert cfg.preset is None
        assert cfg.damping_factor == 0.5
        assert cfg.expansion_hops == 3
        # Other fields remain at default
        assert cfg.max_neighbors == 30


# ---------------------------------------------------------------------------
# Preset application
# ---------------------------------------------------------------------------

class TestPresetApplication:
    @pytest.mark.parametrize("preset_name", PPR_PRESET_NAMES)
    def test_preset_creates_valid_config(self, preset_name):
        """Every named preset should produce a valid config."""
        cfg = _make_cfg(preset=preset_name)
        assert cfg.preset == preset_name
        # Verify all fields are populated (no validation error)
        assert cfg.damping_factor > 0
        assert cfg.expansion_hops >= 1
        assert cfg.max_neighbors >= 1

    def test_balanced_preset_matches_production_defaults(self):
        """The 'balanced' preset should match current production defaults."""
        cfg = _make_cfg(preset="balanced")
        assert cfg.damping_factor == 0.3
        assert cfg.expansion_hops == 2
        assert cfg.max_neighbors == 30
        assert cfg.ppr_backend == "push"
        assert cfg.dense_file_closure_enabled is True
        assert cfg.dense_file_prior_enabled is True

    def test_fast_preset_reduces_scope(self):
        """The 'fast' preset should have fewer hops and smaller budgets."""
        cfg = _make_cfg(preset="fast")
        assert cfg.expansion_hops == 1
        assert cfg.max_neighbors < 30  # Less than balanced
        assert cfg.dense_file_closure_enabled is False
        assert cfg.dense_file_prior_enabled is False

    def test_thorough_preset_increases_scope(self):
        """The 'thorough' preset should have more hops and larger budgets."""
        cfg = _make_cfg(preset="thorough")
        assert cfg.expansion_hops == 3
        assert cfg.max_neighbors > 30  # More than balanced
        assert cfg.fact_retrieval_top_k > 10

    def test_preset_values_are_applied(self):
        """Preset values should fill in parameters not explicitly provided."""
        preset_values = _PPR_PRESETS["fast"]
        cfg = _make_cfg(preset="fast")
        for key, expected in preset_values.items():
            actual = getattr(cfg, key)
            assert actual == expected, f"{key}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Priority: explicit > preset > field default
# ---------------------------------------------------------------------------

class TestPriority:
    def test_explicit_overrides_preset(self):
        """Explicitly set params should override preset values."""
        cfg = _make_cfg(preset="fast", expansion_hops=5, damping_factor=0.9)
        assert cfg.expansion_hops == 5  # Explicit overrides fast preset's 1
        assert cfg.damping_factor == 0.9  # Explicit overrides fast preset's 0.3
        # Other preset values should still apply
        assert cfg.max_neighbors == _PPR_PRESETS["fast"]["max_neighbors"]

    def test_preset_overrides_field_default(self):
        """Preset values should override Pydantic field defaults."""
        # The 'fast' preset sets ppr_push_epsilon to 1e-5, while field default is 1e-6
        cfg = _make_cfg(preset="fast")
        assert cfg.ppr_push_epsilon == _PPR_PRESETS["fast"]["ppr_push_epsilon"]

    def test_non_preset_fields_use_field_defaults(self):
        """Fields not covered by the preset should use Pydantic field defaults."""
        cfg = _make_cfg(preset="balanced")
        # These fields are NOT in the preset, so they use Pydantic defaults
        assert cfg.fact_groundability_enabled is True  # Pydantic default
        assert cfg.chunk_selection_strategy == "top_ppr_chunks"  # Pydantic default
        assert cfg.similarity_edge_relation == "SIMILAR_TO"  # Pydantic default


# ---------------------------------------------------------------------------
# Unknown preset handling
# ---------------------------------------------------------------------------

class TestUnknownPreset:
    def test_unknown_preset_raises_validation_error(self):
        """Unknown preset name should be rejected by Pydantic Literal validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="literal_error"):
            _make_cfg(preset="nonexistent_preset")


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_describe_essential(self):
        """describe_essential() should return a compact dict with key params."""
        cfg = _make_cfg(preset="balanced")
        desc = cfg.describe_essential()
        assert isinstance(desc, dict)
        assert desc["preset"] == "balanced"
        assert "damping_factor" in desc
        assert "expansion_hops" in desc
        assert "max_neighbors" in desc

    def test_get_preset(self):
        """get_preset() should return a copy of the named preset."""
        preset = PrunedHippoRAGNeo4jRetrievalConfig.get_preset("fast")
        assert isinstance(preset, dict)
        assert preset == _PPR_PRESETS["fast"]
        # Verify it's a copy, not a reference
        preset["damping_factor"] = 999.0
        assert _PPR_PRESETS["fast"]["damping_factor"] != 999.0

    def test_get_preset_raises_on_unknown(self):
        """get_preset() should raise ValueError for unknown presets."""
        with pytest.raises(ValueError, match="Unknown PPR preset"):
            PrunedHippoRAGNeo4jRetrievalConfig.get_preset("does_not_exist")

    def test_list_presets(self):
        """list_presets() should return all available presets."""
        presets = PrunedHippoRAGNeo4jRetrievalConfig.list_presets()
        assert isinstance(presets, dict)
        assert set(presets.keys()) == set(PPR_PRESET_NAMES)
        for name, values in presets.items():
            assert isinstance(values, dict)
            assert len(values) > 0

    def test_ppr_preset_names_constant(self):
        """PPR_PRESET_NAMES should contain all preset names."""
        assert set(PPR_PRESET_NAMES) == set(_PPR_PRESETS.keys())
        assert "balanced" in PPR_PRESET_NAMES
        assert "fast" in PPR_PRESET_NAMES
        assert "thorough" in PPR_PRESET_NAMES


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------

class TestCrossFieldValidation:
    def test_no_errors_with_valid_config(self):
        """Valid configs should not raise errors."""
        cfg = _make_cfg(preset="balanced")
        assert cfg is not None

    def test_high_hops_without_pruning_logs_debug(self):
        """expansion_hops >= 3 with enable_pruning=False should not crash."""
        cfg = _make_cfg(expansion_hops=3, enable_pruning=False)
        assert cfg.expansion_hops == 3
        assert cfg.enable_pruning is False


# ---------------------------------------------------------------------------
# JSON serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    @pytest.mark.parametrize("preset_name", PPR_PRESET_NAMES)
    def test_model_dump_round_trip(self, preset_name):
        """Config with preset should survive model_dump -> model_validate round-trip."""
        cfg1 = _make_cfg(preset=preset_name)
        data = cfg1.model_dump()
        cfg2 = PrunedHippoRAGNeo4jRetrievalConfig.model_validate(data)
        # All essential fields should match
        assert cfg1.preset == cfg2.preset
        assert cfg1.damping_factor == cfg2.damping_factor
        assert cfg1.expansion_hops == cfg2.expansion_hops
        assert cfg1.max_neighbors == cfg2.max_neighbors
        assert cfg1.ppr_push_epsilon == cfg2.ppr_push_epsilon

    def test_json_config_with_preset(self):
        """Simulate a JSON config that uses the simplified preset form."""
        json_data = {
            "type": "pruned_hipporag_neo4j_retrieval",
            "preset": "fast",
            "graph_config": _MINIMAL_GRAPH_CONFIG,
            # User only sets the few params they care about
            "damping_factor": 0.4,
        }
        cfg = PrunedHippoRAGNeo4jRetrievalConfig.model_validate(json_data)
        assert cfg.preset == "fast"
        assert cfg.damping_factor == 0.4  # Explicit override
        assert cfg.expansion_hops == 1  # From fast preset
        assert cfg.max_neighbors == 15  # From fast preset
