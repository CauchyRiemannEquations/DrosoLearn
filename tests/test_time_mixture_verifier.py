"""Invented short timelines only; full registered scientific vectors are not run."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_time_mixture", REPO / "scripts/verify_time_mixture.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def decode(value):
    return None if value is None else Fraction(value["numerator"], value["denominator"])


def tiny_case(p=None, q=None, name="joint_drift"):
    return {"id": name, "p0": p or ["4/5", "4/5", "1/5"],
            "q": q or [["9/10", "2/5"], ["3/5", "1/10"], ["3/5", "1/10"]]}


@pytest.fixture
def fixture(tmp_path):
    """Build both finite short producer results and independent pinned parents."""
    root, out = tmp_path / "invented_repository", tmp_path / "short_result"
    root.mkdir()
    for relative in V.SOURCES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = REPO / relative
        target.write_bytes(source.read_bytes() if source.is_file() else b"Invented fixture source\n")
    for study in ("conditional_transition", "metric_calibration"):
        base = root / f"results/{study}_v1"
        config_path = root / f"configs/{study}_v1.json"
        registration = root / f"results/registration_{study}_v1.json"
        write(config_path, {"study_id": study + "_v1", "synthetic": True})
        write(registration, {"synthetic_parent": study})
        write(base / "registration.json", {"synthetic_parent": study})
        write(base / "metadata.json", {"status": "completed", "registration_sha256": V.digest(registration)})
        write(base / "metrics.json", {"invented_short_parent": True})
        write(base / "ledger.csv", {"invented": True})
        if study == "metric_calibration":
            write(base / "summary.json", {"invented": True})
            write(base / "trials.csv", {"invented": True})
        primary = {p.name: V.digest(p) for p in base.iterdir() if p.is_file()}
        write(base / "artifact_hashes.json", primary)
        write(base / "verification.json", {"status": "passed", "errors": [], "verified_at_utc": "2000-01-01T00:00:00+00:00",
              "registration_sha256": V.digest(registration), "config_sha256": V.digest(config_path)})
    task = root / "results/human_task_structure_v1/audit.json"
    task_reg = root / "results/registration_human_task_structure_v1.json"
    write(task_reg, {"synthetic_parent": "human_structure"})
    write(task, {"status": "pass", "gate": {"source_coded_descriptive_metric_allowed": True,
                "exact_historical_task_execution_version_verified": False, "direct_human_model_similarity_allowed": False}})
    write(root / "results/human_task_structure_v1/verification.json", {"status": "pass", "errors": [],
          "source_coded_descriptive_gate_passed": True, "result_sha256": V.digest(task),
          "registration_sha256": V.digest(task_reg), "verified_at_utc": "2000-01-01T00:00:00+00:00"})
    for name in ("conditional-transition", "metric-calibration"):
        (root / "docs" / (name + "-protocol.md")).write_text("Invented parent protocol\n", encoding="utf8")
    p, q = tiny_case()["p0"], tiny_case()["q"]
    cases = [tiny_case(), tiny_case([p[0]] * 3, q, "p_constant"),
             tiny_case(p, [q[0]] * 3, "q_constant"), tiny_case([p[0]] * 3, [q[0]] * 3, "joint_constant")]
    cfg = {"study_id": V.STUDY, "scope": "integration_test", "trials": 3, "cases": cases, "metric": V.METRIC,
           **{k: 0 for k in V.RESOURCES}, **{k: False for k in V.FLAGS},
           "parent_files": sorted(V.PARENTS), "frozen_files": sorted(V.SOURCES),
           "parent_gate_identity": {"published_commit": "b" * 40,
                                    "file_sha256": {p: V.digest(root / p) for p in sorted(V.PARENTS)}}}
    config_path = root / V.CONFIG
    write(config_path, cfg)
    _, gate, _ = V.parent_gate(root, cfg, V.utc("2000-01-02T00:00:00+00:00"))
    registration = root / "results/registration_time_mixture_v1.json"
    reg = {"study_id": V.STUDY, "registration_type": "exact_no_learning_time_mixture",
           "registered_at_utc": "2000-01-02T00:00:00+00:00", "source_git_commit": "a" * 40,
           "config": cfg, "config_sha256": V.digest(config_path), "frozen_files": {p: V.digest(root / p) for p in sorted(V.SOURCES)},
           "parent_files": cfg["parent_gate_identity"]["file_sha256"], "gate": gate}
    write(registration, reg)
    spec = importlib.util.spec_from_file_location("short_time_mixture_fixture_producer", REPO / "src/drosolearn/time_mixture.py")
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    producer.run(root, config_path, registration, out)
    return root, registration, out


def refresh_artifacts(out):
    write(out / "artifact_hashes.json", {name: V.digest(out / name) for name in sorted(V.PRIMARY)})


def mutate(path, callback):
    obj = V.read_json(path)
    callback(obj)
    write(path, obj)


def test_hand_two_choice_timeline_expected_counts():
    events, metrics, counts = V.exhaustive(tiny_case(["4/5", "1/5"], [["9/10", "2/5"]] * 2))
    assert counts["complete_paths"] == 8 and counts["path_weight_sum"] == V.pack(1)
    assert len(events) == 8
    selected = metrics["by_current_choice"]["0"]["by_feedback"]["1"]
    assert decode(selected["expected_denominator"]) == Fraction(18, 25)
    assert decode(selected["expected_stay"]) == Fraction(18, 125)
    assert decode(selected["expected_count_ratio"]) == Fraction(1, 5)
    assert metrics["by_current_choice"]["0"]["difference_1_minus_0"] == V.pack(0)


def test_short_full_path_calculation_and_all_time_conditioned_differences():
    events, metrics, counts = V.exhaustive(tiny_case())
    assert counts["complete_paths"] == 32 and len(events) == 16
    for row in metrics["by_time"]:
        assert all(c["difference_1_minus_0"] == V.pack(0) for c in row["by_current_choice"].values())
    assert metrics["by_current_choice"]["0"]["difference_1_minus_0"] != V.pack(0)


def test_primary_expected_count_ratio_differs_from_mean_random_ratio():
    _, metrics, _ = V.exhaustive(tiny_case(["1/5"] * 3, [["1/1", "0/1"]] * 3))
    ratio = decode(metrics["by_current_choice"]["0"]["by_feedback"]["1"]["expected_count_ratio"])
    weighted_ratio, support = Fraction(0), Fraction(0)
    for choices in product((0, 1), repeat=3):
        weight = Fraction(1, 5) ** choices.count(0) * Fraction(4, 5) ** choices.count(1)
        denominator = sum(a == 0 for a in choices[:-1])
        numerator = sum(choices[t] == choices[t + 1] == 0 for t in range(2))
        if denominator:
            support += weight
            weighted_ratio += weight * Fraction(numerator, denominator)
    assert ratio == Fraction(1, 5)
    assert weighted_ratio / support == Fraction(7, 45)
    assert ratio != weighted_ratio / support


@pytest.mark.parametrize("p,q", [
    (["2/5"] * 3, [["9/10", "2/5"], ["3/5", "1/10"], ["1/2", "1/2"]]),
    (["4/5", "1/5", "1/2"], [["9/10", "2/5"]] * 3),
    (["2/5"] * 3, [["9/10", "2/5"]] * 3),
])
def test_short_constant_controls_are_exactly_zero(p, q):
    _, metrics, _ = V.exhaustive(tiny_case(p, q))
    assert all(c["difference_1_minus_0"] == V.pack(0) for c in metrics["by_current_choice"].values())


@pytest.mark.parametrize("p,q", [(["0/1"] * 2, [["1/1", "0/1"]] * 2),
                                (["1/1"] * 2, [["0/1", "1/1"]] * 2)])
def test_impossible_events_retained_and_zero_denominators_are_null(p, q):
    events, metrics, counts = V.exhaustive(tiny_case(p, q))
    assert len(events) == 8 and counts["complete_paths"] == 8
    assert sum(decode(e["weight"]) == 0 for e in events) == 7
    empty = [c for a in metrics["by_current_choice"].values() for c in a["by_feedback"].values()
             if decode(c["expected_denominator"]) == 0]
    assert len(empty) == 3 and all(c["expected_count_ratio"] is None for c in empty)


@pytest.mark.parametrize("value", [0.5, True, "2/4", "-1/5", "1/3", "1/0", "1", "0", "11/10", "01/10", "nan"])
def test_noncanonical_nonbase_ten_or_nonprobability_rejected(value):
    with pytest.raises((ValueError, ZeroDivisionError)):
        V.rational(value)


def test_current_reward_and_original_next_time_not_same_time_used():
    _, metrics, _ = V.exhaustive(tiny_case())
    row = metrics["by_time"][1]
    assert row["t"] == 2 and row["p0_t"] == V.pack(Fraction(4, 5))
    assert row["p0_next"] == V.pack(Fraction(1, 5)) and row["q0_t"] == V.pack(Fraction(3, 5))
    assert row["by_current_choice"]["0"]["by_feedback"]["1"]["expected_count_ratio"] == V.pack(Fraction(1, 5))


def test_verifier_has_only_standard_library_imports():
    assert "drosolearn" not in V.source_imports_are_independent()


def test_complete_short_producer_independent_integration(fixture):
    root, reg, out = fixture
    report = V.verify(root, reg, out)
    assert report["passed"], report["errors"]
    assert report["counts"]["complete_paths_total"] == 128
    assert report["counts"]["parent_primary_artifacts"] == 10
    assert report["hypotheses"]["time_mixture_counterexample"] is False
    assert report["counts"]["learning_updates"] == 0


@pytest.mark.parametrize("target,mutation", [
    ("metrics.json", lambda x: x["cases"][0]["by_current_choice"]["0"].update(difference_1_minus_0=V.pack(0))),
    ("metrics.json", lambda x: x["cases"][0]["by_time"][0].update(p0_next=V.pack(Fraction(1, 2)))),
    ("metrics.json", lambda x: x["cases"][0]["by_current_choice"]["0"]["by_feedback"]["1"].update(expected_denominator=V.pack(0))),
    ("metrics.json", lambda x: x["hypotheses"].update(time_mixture_counterexample=True)),
    ("metrics.json", lambda x: x["cases"].pop()),
    ("metrics.json", lambda x: x["cases"][0]["by_time"].reverse()),
    ("local_events.json", lambda x: x["events"][0].update(weight=V.pack(Fraction(1, 2)))),
    ("local_events.json", lambda x: x["events"][0].update(feedback=1)),
    ("local_events.json", lambda x: x["events"].append(deepcopy(x["events"][0]))),
    ("metadata.json", lambda x: x.update(learning_updates=1)),
    ("metadata.json", lambda x: x.update(rng_draws=1)),
    ("metadata.json", lambda x: x.update(human_or_species_comparison=True)),
    ("metadata.json", lambda x: x.update(original_adjacent_pairs=6.0)),
    ("metadata.json", lambda x: x.update(started_at_utc="1900-01-01T00:00:00+00:00")),
    ("metadata.json", lambda x: x.update(source_git_commit="b" * 40)),
])
def test_rehashed_semantic_tamper_is_rejected(fixture, target, mutation):
    root, reg, out = fixture
    mutate(out / target, mutation)
    refresh_artifacts(out)
    report = V.verify(root, reg, out)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("target", ["src/drosolearn/time_mixture.py", "tests/test_time_mixture_verifier.py",
                                   "docs/time-mixture-protocol.md", "requirements-lock.txt"])
def test_frozen_source_change_is_rejected(fixture, target):
    root, reg, out = fixture
    (root / target).write_bytes((root / target).read_bytes() + b"\nChanged source\n")
    assert not V.verify(root, reg, out)["passed"]


@pytest.mark.parametrize("relative", ["results/conditional_transition_v1/metrics.json",
                                     "results/metric_calibration_v1/trials.csv",
                                     "results/human_task_structure_v1/audit.json",
                                     "results/registration_human_task_structure_v1.json"])
def test_all_parent_bytes_are_checked_including_nonselected_artifacts(fixture, relative):
    root, reg, out = fixture
    (root / relative).write_bytes((root / relative).read_bytes() + b"\n")
    assert not V.verify(root, reg, out)["passed"]


def repin_engineering_fixture(root, reg_path, out):
    """Rehash fabricated files so semantic negative tests reach deeper guards."""
    cfg = V.read_json(root / V.CONFIG)
    cfg["parent_gate_identity"]["file_sha256"] = {p: V.digest(root / p) for p in sorted(V.PARENTS)}
    write(root / V.CONFIG, cfg)
    reg = V.read_json(reg_path)
    reg.update(config=cfg, config_sha256=V.digest(root / V.CONFIG),
               frozen_files={p: V.digest(root / p) for p in sorted(V.SOURCES)},
               parent_files=cfg["parent_gate_identity"]["file_sha256"])
    write(reg_path, reg)
    write(out / "registration.json", reg)
    meta = V.read_json(out / "metadata.json")
    for key in ("config", "config_sha256", "frozen_files", "parent_files"):
        meta[key] = reg[key]
    meta["registration_sha256"] = V.digest(reg_path)
    write(out / "metadata.json", meta)
    refresh_artifacts(out)


@pytest.mark.parametrize("attack", ["human_gate", "historical_gate", "parent_reg_binding", "parent_inventory", "executed_verifier"])
def test_rehashed_provenance_semantic_tamper_is_rejected(fixture, attack):
    root, reg, out = fixture
    if attack == "human_gate":
        mutate(root / "results/human_task_structure_v1/audit.json", lambda x: x["gate"].update(direct_human_model_similarity_allowed=True))
    elif attack == "historical_gate":
        mutate(root / "results/human_task_structure_v1/audit.json", lambda x: x["gate"].update(exact_historical_task_execution_version_verified=True))
    elif attack == "parent_reg_binding":
        mutate(root / "results/metric_calibration_v1/verification.json", lambda x: x.update(registration_sha256="0" * 64))
    elif attack == "parent_inventory":
        mutate(root / "results/metric_calibration_v1/artifact_hashes.json", lambda x: x.pop("ledger.csv"))
    else:
        target = root / "scripts/verify_time_mixture.py"
        target.write_bytes(target.read_bytes() + b"\n# fabricated alternate verifier\n")
    repin_engineering_fixture(root, reg, out)
    assert not V.verify(root, reg, out)["passed"]


@pytest.mark.parametrize("which", ["source", "parent", "gate", "config", "original_reg", "reg_copy", "extra_artifact", "artifact_omission"])
def test_registration_and_complete_scope_tampering_rejected(fixture, which):
    root, reg, out = fixture
    if which == "source":
        mutate(reg, lambda x: x["frozen_files"].pop("requirements-lock.txt"))
    elif which == "parent":
        mutate(reg, lambda x: x["parent_files"].pop(next(iter(x["parent_files"]))))
    elif which == "gate":
        mutate(reg, lambda x: x["gate"].update(direct_human_model_similarity_allowed=True))
    elif which == "config":
        mutate(root / V.CONFIG, lambda x: x.update(sampled_trials=1))
    elif which == "original_reg":
        reg.write_bytes(reg.read_bytes() + b"\n")
    elif which == "reg_copy":
        (out / "registration.json").write_bytes((out / "registration.json").read_bytes() + b"\n")
        refresh_artifacts(out)
    elif which == "extra_artifact":
        write(out / "unregistered.json", {"extra": True})
    else:
        mutate(out / "artifact_hashes.json", lambda x: x.pop("local_events.json"))
    assert not V.verify(root, reg, out)["passed"]


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "C:/other", "nested\\file", ""])
def test_untrusted_path_escape_rejected(tmp_path, relative):
    with pytest.raises(V.VerificationError):
        V.safe(tmp_path, relative)


@pytest.mark.parametrize("value", ["2000-01-01T00:00:00", "2000-01-01T00:00:00+01:00", 0])
def test_timestamps_require_explicit_utc(value):
    with pytest.raises(V.VerificationError):
        V.utc(value)


def test_duplicate_keys_and_nonfinite_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for content in ('{"a":1,"a":2}', '{"value":NaN}', '{"value":Infinity}'):
        path.write_text(content, encoding="utf8")
        with pytest.raises(V.VerificationError):
            V.read_json(path)


def test_same_integer_and_boolean_are_not_interchangeable():
    with pytest.raises(V.VerificationError):
        V.same({"value": True}, {"value": 1}, "Strict types")


def test_cli_exclusive_output_and_failed_result(fixture, tmp_path):
    root, reg, out = fixture
    target = tmp_path / "verification.json"
    assert V.main(["--root", str(root), "--registration", str(reg), "--result-dir", str(out), "--out", str(target)]) == 0
    with pytest.raises(V.VerificationError):
        V.main(["--root", str(root), "--registration", str(reg), "--result-dir", str(out), "--out", str(target)])
    mutate(out / "metrics.json", lambda x: x.update(interpretation="other"))
    refresh_artifacts(out)
    assert V.main(["--root", str(root), "--registration", str(reg), "--result-dir", str(out), "--out", str(tmp_path / "failed.json")]) == 1
