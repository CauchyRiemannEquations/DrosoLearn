"""Handcrafted engineering vectors only; the scientific nine-trial run is deferred."""
import copy
from fractions import Fraction
import json
import pytest

from drosolearn import time_mixture as tm


def case(p=None, q=None, name="joint_drift"):
    return {"id": name, "p0": p or ["1/2"] * 3, "q": q or [["1/2", "1/2"] for _ in range(3)]}


def decoded(value):
    return None if value is None else Fraction(value["numerator"], value["denominator"])


def fixture_config():
    return {"study_id": "time_mixture_v1", "scope": "integration_test", "trials": 3,
            "cases": [case(name=name) for name in tm.CASE_IDS],
            **{key: 0 for key in tm.RESOURCES}, "human_or_species_comparison": False,
            "causal_effect_estimated": False, "metric": tm.METRIC,
            "frozen_files": ["configs/fixture.json", "frozen.py"], "parent_files": ["parent.json"],
            "parent_gate_identity": {"published_commit": "a" * 40, "file_sha256": {"parent.json": "b" * 64}}}


@pytest.mark.parametrize("value,expected", [("0/1", 0), ("1/1", 1), ("1/2", Fraction(1, 2)), ("2/3", Fraction(2, 3))])
def test_exact_input_probabilities(value, expected):
    assert tm.ratio(value) == expected


@pytest.mark.parametrize("value", [0.5, 1, True, None, "0.5", "1", "2/4", "0/3", "2/1", "-1/2", "1/0", "01/2", "1/02", " 1/2", "1/2 ", "nan", "1/2/3"])
def test_reject_approximate_or_noncanonical_probability(value):
    with pytest.raises(ValueError):
        tm.ratio(value)


@pytest.mark.parametrize("path", ["../bad", "a/../b", "a/./b", "a//b", "/tmp/a", "C:/tmp/a", "a\\b", "a:stream", "", None])
def test_reject_noncontained_or_noncanonical_paths(tmp_path, path):
    with pytest.raises(ValueError):
        tm.safe(tmp_path, path)


def test_handcrafted_fair_exact_counts():
    result, events = tm.enumerate_case(case(), 3)
    assert len(events) == 16
    for entry in events:
        assert decoded(entry["weight"]) == Fraction(1, 8)
    for row in result["by_current_choice"].values():
        assert decoded(row["difference_1_minus_0"]) == 0
        for cell in row["by_feedback"].values():
            assert decoded(cell["expected_denominator"]) == Fraction(1, 2)
            assert decoded(cell["expected_stay"]) == decoded(cell["expected_switch"]) == Fraction(1, 4)
            assert decoded(cell["expected_count_ratio"]) == Fraction(1, 2)


def test_original_clock_transition_is_included():
    result, events = tm.enumerate_case(case(["3/4", "1/4"], [["1/2", "1/2"]] * 2), 2)
    assert len(events) == 8 and {event["t"] for event in events} == {1}
    cell = result["by_current_choice"]["0"]["by_feedback"]["1"]
    assert decoded(cell["expected_denominator"]) == Fraction(3, 8)
    assert decoded(cell["expected_stay"]) == Fraction(3, 32)
    assert decoded(cell["expected_count_ratio"]) == Fraction(1, 4)
    assert decoded(result["by_time"][0]["p0_t"]) == Fraction(3, 4)
    assert decoded(result["by_time"][0]["p0_next"]) == Fraction(1, 4)


def test_handcrafted_time_mixture_and_exact_time_conditioning():
    result, _ = tm.enumerate_case(case(["3/4", "3/4", "1/4", "1/4"],
        [["3/4", "1/4"], ["3/4", "1/4"], ["1/4", "1/4"], ["1/4", "1/4"]]), 4)
    row = result["by_current_choice"]["0"]
    assert decoded(row["by_feedback"]["1"]["expected_count_ratio"]) == Fraction(37, 76)
    assert decoded(row["by_feedback"]["0"]["expected_count_ratio"]) == Fraction(5, 12)
    assert decoded(row["difference_1_minus_0"]) == Fraction(4, 57)
    assert all(decoded(row["difference_1_minus_0"]) == 0 for cell in result["by_time"] for row in cell["by_current_choice"].values())


@pytest.mark.parametrize("p,q", [(["0/1", "0/1"], [["1/1", "1/1"]] * 2), (["1/1", "1/1"], [["0/1", "0/1"]] * 2)])
def test_zero_denominators_are_null_not_imputed(p, q):
    result, events = tm.enumerate_case(case(p, q), 2)
    assert len(events) == 8
    assert all(row["difference_1_minus_0"] is None for row in result["by_current_choice"].values())
    for row in result["by_current_choice"].values():
        for cell in row["by_feedback"].values():
            if decoded(cell["expected_denominator"]) == 0:
                assert cell["expected_count_ratio"] is None
                assert cell["expected_stay"] == cell["expected_switch"] == {"numerator": 0, "denominator": 1}


def test_last_feedback_probabilities_are_unused():
    original = case()
    changed = copy.deepcopy(original)
    changed["q"][-1] = ["0/1", "1/1"]
    assert tm.enumerate_case(original, 3) == tm.enumerate_case(changed, 3)


def test_expectation_count_ratio_is_not_mean_of_observed_ratios():
    # Two abstract equally weighted outcomes: (S,D)=(1,1),(0,3).
    assert Fraction(1, 2) != (Fraction(1, 2) / Fraction(4, 2))
    result, _ = tm.enumerate_case(case(), 3)
    cell = result["by_current_choice"]["0"]["by_feedback"]["0"]
    assert decoded(cell["expected_count_ratio"]) == decoded(cell["expected_stay"]) / decoded(cell["expected_denominator"])


@pytest.mark.parametrize("change", [
    lambda cfg: cfg.update(trials=True), lambda cfg: cfg.update(trials=1), lambda cfg: cfg.update(cases=[]),
    lambda cfg: cfg["cases"].reverse(), lambda cfg: cfg["cases"][0].update(p0=["1/2"]),
    lambda cfg: cfg["cases"][0].update(q=[["1/2"]] * 3), lambda cfg: cfg.update(learning_updates=1),
    lambda cfg: cfg.update(rng_draws=False), lambda cfg: cfg.update(human_or_species_comparison=True),
    lambda cfg: cfg.update(causal_effect_estimated=True), lambda cfg: cfg.update(metric="mean_of_ratios"),
    lambda cfg: cfg.update(scope="confirmatory_human_test"), lambda cfg: cfg.update(extra="unexpected"),
    lambda cfg: cfg.update(frozen_files=["a", "a"]), lambda cfg: cfg["parent_gate_identity"].update(file_sha256={}),
    lambda cfg: cfg["parent_gate_identity"].update(published_commit="HEAD"),
])
def test_invalid_config_rejected(change):
    cfg = fixture_config()
    change(cfg)
    with pytest.raises(ValueError):
        tm.validate_config(cfg)


def test_scientific_configuration_structure_only():
    # Reading vectors/parent hashes performs no main scientific enumeration.
    cfg = tm.read_json(tm.ROOT / "configs/time_mixture_v1.json")
    tm.validate_config(cfg)
    assert len(cfg["frozen_files"]) == 12 and len(cfg["parent_files"]) == 17
    assert cfg["trials"] == 9


def test_scientific_vector_or_source_closure_change_rejected():
    cfg = tm.read_json(tm.ROOT / "configs/time_mixture_v1.json")
    cfg["cases"][0]["p0"][0] = "1/2"
    with pytest.raises(ValueError):
        tm.validate_config(cfg)
    cfg = tm.read_json(tm.ROOT / "configs/time_mixture_v1.json")
    cfg["frozen_files"].pop()
    with pytest.raises(ValueError):
        tm.validate_config(cfg)


@pytest.mark.parametrize("trials", [0, 1, True, 2.0, None])
def test_direct_case_requires_two_integer_trials(trials):
    with pytest.raises(ValueError):
        tm.enumerate_case(case(), trials)


@pytest.mark.parametrize("changed", [lambda row: row.update(id="other"), lambda row: row.update(extra=1),
    lambda row: row.update(p0=["1/2"]), lambda row: row.update(q=[["1/2"]] * 3)])
def test_direct_malformed_case_refused(changed):
    row = case()
    changed(row)
    with pytest.raises(ValueError):
        tm.enumerate_case(row, 3)


def prepared(tmp_path, monkeypatch):
    cfg = fixture_config()
    tm.write_new(tmp_path / "parent.json", {"status": "engineering_only"})
    (tmp_path / "frozen.py").write_text("# engineering fixture\n", encoding="utf8")
    cfg["parent_gate_identity"]["file_sha256"]["parent.json"] = tm.digest(tmp_path / "parent.json")
    config = tmp_path / "configs/fixture.json"
    tm.write_new(config, cfg)
    monkeypatch.setattr(tm, "parent_gate", lambda root, config: {"passed": True, "engineering_only": True})
    monkeypatch.setattr(tm.subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    registration = tmp_path / "registration.json"
    tm.register(tmp_path, config, registration)
    return cfg, config, registration


def test_tiny_registered_execution_artifact_identity(tmp_path, monkeypatch):
    cfg, config, registration = prepared(tmp_path, monkeypatch)
    output = tmp_path / "tiny"
    metadata = tm.run(tmp_path, config, registration, output)
    assert metadata["status"] == "completed" and metadata["local_events"] == 64
    assert metadata["original_adjacent_pairs"] == 8 and metadata["cases"] == 4
    assert all(metadata[key] == 0 for key in tm.RESOURCES)
    assert tm.read_json(output / "registration.json") == tm.read_json(registration)
    inventory = tm.read_json(output / "artifact_hashes.json")
    assert set(inventory) == {"registration.json", "metrics.json", "local_events.json", "metadata.json"}
    assert all(tm.digest(output / path) == checksum for path, checksum in inventory.items())
    metrics = tm.read_json(output / "metrics.json")
    assert metrics["interpretation"] == tm.METRIC
    assert metrics["hypotheses"]["time_mixture_counterexample"] is False
    assert metrics["hypotheses"]["exact_time_conditional_zero"] is True


@pytest.mark.parametrize("changed", ["frozen.py", "parent.json", "configs/fixture.json"])
def test_changed_frozen_source_parent_config_prevents_output(tmp_path, monkeypatch, changed):
    _, config, registration = prepared(tmp_path, monkeypatch)
    with (tmp_path / changed).open("a", encoding="utf8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError):
        tm.run(tmp_path, config, registration, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("timestamp", ["2099-01-01T00:00:00+00:00", "2000-01-01T00:00:00"])
def test_future_or_naive_registration_refused(tmp_path, monkeypatch, timestamp):
    _, config, registration = prepared(tmp_path, monkeypatch)
    reg = tm.read_json(registration)
    reg["registered_at_utc"] = timestamp
    registration.write_text(json.dumps(reg), encoding="utf8")
    with pytest.raises(ValueError):
        tm.run(tmp_path, config, registration, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_exclusive_registration_and_result(tmp_path, monkeypatch):
    _, config, registration = prepared(tmp_path, monkeypatch)
    before = registration.read_bytes()
    with pytest.raises(ValueError):
        tm.register(tmp_path, config, registration)
    assert registration.read_bytes() == before
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        tm.run(tmp_path, config, registration, output)
    assert list(output.iterdir()) == []


def test_failure_is_preserved_in_new_folder(tmp_path, monkeypatch):
    _, config, registration = prepared(tmp_path, monkeypatch)
    def failure(cfg):
        raise ArithmeticError("handcrafted injected failure")
    monkeypatch.setattr(tm, "analyse", failure)
    with pytest.raises(ArithmeticError):
        tm.run(tmp_path, config, registration, tmp_path / "failed")
    meta = tm.read_json(tmp_path / "failed/metadata.json")
    assert meta["status"] == "failed" and meta["error_type"] == "ArithmeticError"
    assert set(tm.read_json(tmp_path / "failed/artifact_hashes.json")) == {"registration.json", "metadata.json"}


def test_cli_requires_one_registration_mode(tmp_path):
    with pytest.raises(SystemExit):
        tm.main(["--config", "x", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        tm.main(["--register", "--registration", "x", "--config", "x", "--out", str(tmp_path / "x")])
