"""Invented-only independent checks for prospective memory-gated routing."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("gated_independent_tests",
    Path(__file__).resolve().parents[1] / "scripts/verify_gated_learning.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_gate_starts_isolated_with_empty_observed_memory():
    result = audit.independent_gate(np.zeros(4, np.int8), "gated_balanced")
    assert result == {"full_known_before": False, "mode": "isolated", "balanced_branch": False}


def test_fourth_discovery_cannot_retroactively_change_its_own_update_mode():
    before = np.array([1, 1, -1, 0], np.int8)
    after = np.array([1, 1, -1, -1], np.int8)
    assert audit.independent_gate(before, "gated_balanced")["mode"] == "isolated"
    assert audit.independent_gate(after, "gated_balanced")["mode"] == "balanced"


@pytest.mark.parametrize("method", audit.METHODS[:4])
@pytest.mark.parametrize("known", [np.zeros(4, np.int8), np.array([1, 1, -1, -1], np.int8)])
def test_other_registered_methods_do_not_acquire_an_unrequested_gate(method, known):
    assert audit.independent_gate(known, method)["mode"] == method


@pytest.mark.parametrize("known", [np.zeros(3, int), np.zeros(4, float), np.array([1, 2, 0, 0])])
def test_gate_does_not_silently_complete_or_coerce_observation_table(known):
    with pytest.raises(ValueError): audit.independent_gate(known, "gated_balanced")


def test_unregistered_gate_method_rejected():
    with pytest.raises(ValueError): audit.independent_gate(np.zeros(4, int), "oracle_gated")


def fixture_config():
    from drosolearn.gated_learning import SCIENCE
    cfg = copy.deepcopy(SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[90085, 90086], training_trials=16,
        checkpoints=[0, 4, 8, 16], population_pass_count=2, positive_seed_count=2, protection_nonworse_seed_count=2)
    return cfg


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


@pytest.fixture(scope="module")
def producer_fixture(tmp_path_factory):
    from drosolearn import gated_learning as producer
    from drosolearn.graph import Projection
    cfg, raw = fixture_config(), tiny_graph()
    root = tmp_path_factory.mktemp("gated_invented_replay")
    n = raw["n_kc"]
    graph = Projection(raw["pre"], raw["post"], raw["counts"], n, n,
        [str(i) for i in range(n)], [str(i) for i in range(n)], {})
    catalog = root / audit.INPUT_CATALOG
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"study_id": "representation_update_v1", "representations": list(audit.REPRESENTATIONS),
        "catalog": {key: value.tolist() for key, value in audit.prior.independent_catalog_arrays().items()}}), encoding="utf8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(producer, "load_projection", lambda _: graph)
        arrays, rows = producer.produce(cfg, root)
    expected, wanted_rows = audit.expected_artifacts(cfg, raw, arrays)
    return cfg, raw, arrays, rows, expected, wanted_rows


def test_current_five_method_producer_matches_independent_fresh_replay(producer_fixture):
    cfg, _, arrays, rows, expected, wanted_rows = producer_fixture
    audit.validate_science(cfg, allow_fixture=True)
    audit.check_artifacts(arrays, expected, rows, wanted_rows)
    assert len(arrays) == 195 and len(rows) == 960 and len(rows[0]) == 84
    assert arrays["trial_actions"].shape == (3, 2, 5, 2, 16)
    assert arrays["case_keys"].shape[1] == 9
    assert all(audit.information_checks(cfg, arrays).values())
    assert all(audit.gate_checks(cfg, arrays).values())


def test_complete_current_summary_matches_independent_five_method_summary(producer_fixture):
    from drosolearn.gated_learning import summarize
    cfg, graph, arrays, rows, _, _ = producer_fixture
    summary = summarize(cfg, arrays, rows)
    receipt = audit.verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=True)
    assert receipt["independently_replayed_training_actions"] == 960
    assert receipt["conditions"] == 30
    assert len(summary["families"]) == 30
    assert summary["counts"]["model_initializations"] == 60
