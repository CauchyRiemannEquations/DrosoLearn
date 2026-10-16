"""Handwritten-only independent strata tests; no real data or model training."""
from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_conditional_test", ROOT / "scripts/verify_conditional_transition.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def handwritten_records(kind="human"):
    choices, rewards = [0, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 0, 1]
    return [{"source_position": i + 2, "trial": i + 1, "local_trial": i + 1 if kind == "human" else i % 4 + 1,
        "choice": choice, "feedback": reward, "task": "cards" if kind == "human" else "abstract_quantity",
        "phase": (("initial" if i < 4 else "transfer") if kind == "human" else ("phase1" if i < 4 else "phase2")),
        "state": (("1" if i < 4 else "2") if kind == "human" else ("larger" if i < 4 else "smaller")),
        "actual": True, "training": True} for i, (choice, reward) in enumerate(zip(choices, rewards, strict=True))]


@pytest.mark.parametrize("kind,count", [("human", 8), ("model", 4)])
def test_every_predefined_cell_is_retained_and_only_eligible_pairs_are_partitioned(kind, count):
    rows = handwritten_records(kind)
    summary, ledger = VERIFY.measure("handwritten", kind, "absolute", rows)
    assert len(summary["strata"]) == count
    assert summary["trials"] == 8 and summary["candidates"] == 7 and summary["eligible"] == 6 and summary["excluded"] == 1
    assert ledger[3]["reasons"] == "phase_boundary|state_boundary"
    for f in ("0", "1"):
        for key in ("denominator", "stay", "switch"):
            assert sum(s["by_feedback"][f][key] for s in summary["strata"]) == summary["by_feedback"][f][key]


def test_same_current_choice_stratum_exposes_both_feedbacks_and_exact_difference():
    summary, _ = VERIFY.measure("manual", "human", "absolute", handwritten_records())
    cell = next(s for s in summary["strata"] if (s["phase"], s["state"], s["choice"]) == ("initial", "1", 0))
    assert cell["by_feedback"]["1"] == {"denominator": 1, "stay": 1, "switch": 0, "stay_rate": 1., "switch_rate": 0.}
    assert cell["by_feedback"]["0"] == {"denominator": 1, "stay": 0, "switch": 1, "stay_rate": 0., "switch_rate": 1.}
    assert cell["common_support"] is True and cell["support_status"] == "both_feedback"
    assert cell["stay_rate_difference_1_minus_0"] == 1.


@pytest.mark.parametrize("support", ["neither", "feedback_0_only", "feedback_1_only"])
def test_missing_feedback_support_is_not_zero_difference_or_zero_rate(support):
    summary, _ = VERIFY.measure("manual", "human", "absolute", handwritten_records())
    cell = next(s for s in summary["strata"] if s["support_status"] == support)
    assert cell["common_support"] is False and cell["stay_rate_difference_1_minus_0"] is None
    for f in ("0", "1"):
        if cell["by_feedback"][f]["denominator"] == 0:
            assert cell["by_feedback"][f]["stay_rate"] is None and cell["by_feedback"][f]["switch_rate"] is None


@pytest.mark.parametrize("choice,feedback", [("query", "information"), (None, None)])
def test_query_or_missing_trials_stay_in_original_adjacency_and_have_no_binary_stratum(choice, feedback):
    rows = handwritten_records()
    rows[1].update(choice=choice, feedback=feedback)
    summary, ledger = VERIFY.measure("manual", "human", "absolute", rows)
    assert len(ledger) == 7
    assert ledger[0]["eligible"] == ledger[1]["eligible"] == "0"
    assert summary["eligible"] == 4
    assert not any(s["choice"] is None or s["choice"] == "query" for s in summary["strata"])
    assert ledger[0]["next_choice"] == ("" if choice is None else "query")


def test_missing_next_feedback_alone_does_not_remove_current_valid_pair():
    rows = handwritten_records()
    rows[1]["feedback"] = None
    _, ledger = VERIFY.measure("manual", "human", "absolute", rows)
    assert ledger[0]["eligible"] == "1"
    assert ledger[1]["eligible"] == "0" and "current_feedback_missing" in ledger[1]["reasons"]


def test_state_and_phase_boundaries_are_one_exclusion_with_two_preserved_reasons():
    summary, ledger = VERIFY.measure("manual", "human", "absolute", handwritten_records())
    assert summary["reason_counts"] == {"phase_boundary": 1, "state_boundary": 1}
    assert summary["excluded"] == 1
    assert ledger[3]["local_trial"] == "4" and ledger[3]["next_local_trial"] == "5"


def test_model_global_trial_does_not_join_or_relabel_phase_local_coordinates():
    _, ledger = VERIFY.measure("manual", "model", "absolute", handwritten_records("model"))
    assert ledger[3]["trial"] == "4" and ledger[3]["next_trial"] == "5"
    assert ledger[3]["local_trial"] == "4" and ledger[3]["next_local_trial"] == "1"
    assert ledger[3]["eligible"] == "0"


def test_deterministic_reward_has_no_within_absolute_choice_common_support():
    rows = handwritten_records("model")
    for row in rows:
        target = 1 if row["phase"] == "phase1" else 0
        row["feedback"] = int(row["choice"] == target)
    summary, _ = VERIFY.measure("manual", "model", "absolute", rows)
    assert not any(s["common_support"] for s in summary["strata"])
    assert all(s["stay_rate_difference_1_minus_0"] is None for s in summary["strata"])


def test_partitioning_never_changes_canonical_source_records():
    rows = handwritten_records()
    original = copy.deepcopy(rows)
    VERIFY.measure("manual", "human", "absolute", rows)
    assert rows == original


def test_scientific_source_budget_validation_does_not_read_behavior_sources():
    cfg = VERIFY.read_json(ROOT / "configs/conditional_transition_v1.json")
    VERIFY.config_check(cfg)
    assert (cfg["registered_streams"], cfg["registered_input_trials"], cfg["registered_candidate_pairs"], cfg["registered_strata"]) == (265, 217000, 216735, 1160)


@pytest.mark.parametrize("key,value", [("pooling", True), ("strata_averaging", True), ("causal_inference", True),
    ("learning_updates", 1), ("learner_constructions", 1), ("dataset_generations", 1), ("rng_draws", 1),
    ("registered_streams", 264), ("registered_strata", 1159), ("registered_candidate_pairs", 216734)])
def test_scientific_config_rejects_inference_resource_or_denominator_changes(key, value):
    cfg = VERIFY.read_json(ROOT / "configs/conditional_transition_v1.json")
    cfg[key] = value
    with pytest.raises(ValueError):
        VERIFY.config_check(cfg)


def test_runtime_does_not_import_primary_converter_or_learning_code():
    text = (ROOT / "scripts/verify_conditional_transition.py").read_text(encoding="utf8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("drosolearn")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"default_rng", "random", "initialize", "forward", "train", "fit", "simulate"}
    assert "verify_feedback_transition.py" in text


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def handwritten_model_rows():
    rows = []
    for condition in VERIFY.CONDITIONS:
        for environment in VERIFY.ENVIRONMENTS:
            for t in range(8):
                phase, local = ("phase1" if t < 4 else "phase2"), t % 4 + 1
                action = (0, 0, 1, 0, 1, 0, 1, 1)[t]
                left, right = (2, 4) if t % 2 == 0 else (4, 2)
                chosen = (left, right)[action]
                target = 4 if phase == "phase1" else 2
                high, low = {"deterministic": (1., 0.), "probabilistic": (.8, .2), "uninformative": (.5, .5)}[environment]
                q = high if chosen == target else low
                action_u = .1 if action else .9
                reward_u = (.1, .9, .9, .1, .1, .9, .9, .1)[t]
                row = {key: "0" for key in VERIFY.MODEL_FIELDS}
                row.update(stage="network", seed="17", condition=condition, environment=environment, phase=phase,
                    trial=str(local), global_trial=str(t + 1), family_id=f"handwritten:{t}", left_count=str(left), right_count=str(right),
                    chosen_count=str(chosen), action=str(action), reward=str(int(reward_u < q)), p_right="0.5", q_reward=str(q),
                    action_uniform=str(action_u), reward_uniform=str(reward_u))
                rows.append(row)
    return rows


def save_csv(path, fields, rows):
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_model_decoder_maps_observed_screen_side_to_absolute_quantity_and_keeps_all_cells(tmp_path):
    path = tmp_path / "handwritten.csv"
    save_csv(path, VERIFY.MODEL_FIELDS, handwritten_model_rows())
    streams = VERIFY.model_records(path, 17, trials_per_phase=4)
    assert len(streams) == 24 and all(len(rows) == 8 for rows in streams.values())
    first = streams["linear", "deterministic"]
    # First two actions both select left, but counts exchange screen positions.
    assert [r["choice"] for r in first[:2]] == [0, 1]
    assert first[0]["source_position"] == 2
    assert first[4]["local_trial"] == 1 and first[4]["trial"] == 5


def test_probabilistic_decoder_accepts_unexpected_sampled_feedback_for_same_absolute_choice(tmp_path):
    path = tmp_path / "handwritten.csv"
    rows = handwritten_model_rows()
    source = next(r for r in rows if r["environment"] == "probabilistic" and r["phase"] == "phase1" and r["chosen_count"] == "4")
    source["reward"] = "0"
    source["reward_uniform"] = "0.9"
    save_csv(path, VERIFY.MODEL_FIELDS, rows)
    streams = VERIFY.model_records(path, 17, trials_per_phase=4)
    assert any(r["feedback"] == 0 and r["choice"] == 1 for r in streams["linear", "probabilistic"][:4])


@pytest.mark.parametrize("column,value", [("global_trial", "2"), ("chosen_count", "4"), ("action", "2"), ("reward", "2"),
    ("p_right", "nan"), ("q_reward", "0.3"), ("action_uniform", "0.1"), ("reward_uniform", "1.0"),
    ("stage", "linear"), ("trial", "1.0"), ("family_id", ""), ("internal_step_l2", "inf")])
def test_model_decoder_rejects_corrupted_logged_choice_feedback_coordinates_and_schema(tmp_path, column, value):
    rows = handwritten_model_rows()
    rows[0][column] = value
    path = tmp_path / "bad.csv"
    save_csv(path, VERIFY.MODEL_FIELDS, rows)
    with pytest.raises(ValueError):
        VERIFY.model_records(path, 17, trials_per_phase=4)


@pytest.mark.parametrize("mode", ["drop", "duplicate", "reorder", "extra_field"])
def test_model_decoder_refuses_incomplete_duplicate_reordered_or_new_schema_streams(tmp_path, mode):
    rows, fields = handwritten_model_rows(), VERIFY.MODEL_FIELDS
    if mode == "drop":
        rows.pop(3)
    elif mode == "duplicate":
        rows.insert(3, dict(rows[3]))
    elif mode == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        fields = (*fields, "unregistered")
    path = tmp_path / "bad.csv"
    save_csv(path, fields, rows)
    with pytest.raises(ValueError):
        VERIFY.model_records(path, 17, trials_per_phase=4)


@pytest.fixture(scope="module")
def audited_fixture(tmp_path_factory):
    from test_conditional_transition_run import make_fixture
    from drosolearn import conditional_transition as producer
    monkeypatch = pytest.MonkeyPatch()
    try:
        fixture = make_fixture(tmp_path_factory.mktemp("independent-conditional"), monkeypatch)
        producer.register(fixture["root"], fixture["config"], fixture["cache"], fixture["registration"])
        producer.run(fixture["root"], fixture["config"], fixture["registration"], fixture["cache"], fixture["out"])
        report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"], fixture["cache"])
        assert report["status"] == "passed", report["errors"]
        dump(fixture["out"] / "verification.json", report)
        return fixture
    finally:
        monkeypatch.undo()


def test_handcrafted_full_pipeline_passes_with_exact_fixed_denominators_and_byte_bindings(audited_fixture):
    f = audited_fixture
    report = VERIFY.read_json(f["out"] / "verification.json")
    counts = report["counts"]
    assert counts["streams"] == f["cfg"]["registered_streams"]
    assert counts["strata"] == f["cfg"]["registered_strata"]
    assert counts["feedback_cells"] == 2 * counts["strata"]
    assert counts["source_files"] == 16 and counts["parent_files"] == 17
    assert counts["human_records"] == 2 and counts["input_files"] == 2 + len(f["cfg"]["model"]["seeds"])
    assert all(counts[key] == 0 for key in VERIFY.RESOURCE_FIELDS)
    assert report["metrics_sha256"] == VERIFY.digest(f["out"] / "metrics.json")
    assert report["artifact_hashes_sha256"] == VERIFY.digest(f["out"] / "artifact_hashes.json")
    assert report["input_manifest_sha256"] == VERIFY.canonical_digest(VERIFY.read_json(f["registration"])["input_files"])


@pytest.fixture
def mutated(audited_fixture, tmp_path):
    # Copy only handwritten fixtures. No original participant/model source is read.
    root, cache = tmp_path / "source", tmp_path / "human_cache"
    shutil.copytree(audited_fixture["root"], root)
    shutil.copytree(audited_fixture["cache"], cache)
    return {**audited_fixture, "root": root, "cache": cache,
        "registration": root / audited_fixture["registration"].relative_to(audited_fixture["root"]),
        "config": root / audited_fixture["config"].relative_to(audited_fixture["root"]),
        "out": root / audited_fixture["out"].relative_to(audited_fixture["root"])}


def rehash(fixture):
    out = fixture["out"]
    hashes = VERIFY.read_json(out / "artifact_hashes.json")
    for name in hashes:
        hashes[name] = VERIFY.digest(out / name)
    dump(out / "artifact_hashes.json", hashes)


def failed(fixture):
    report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"], fixture["cache"])
    assert report["status"] == "failed", report
    return report


@pytest.mark.parametrize("mutation", ["count", "rate", "support", "status", "difference", "empty_rate", "drop_stratum", "pool", "average", "causal"])
def test_rehashed_output_metric_semantic_mutations_fail_independent_reconstruction(mutated, mutation):
    path = mutated["out"] / "metrics.json"
    value = VERIFY.read_json(path)
    strata = value["streams"][0]["strata"]
    if mutation == "count":
        strata[0]["by_feedback"]["0"]["denominator"] += 1
    elif mutation == "rate":
        strata[0]["by_feedback"]["0"]["stay_rate"] = .123
    elif mutation == "support":
        strata[0]["common_support"] = not strata[0]["common_support"]
    elif mutation == "status":
        strata[0]["support_status"] = "invented"
    elif mutation == "difference":
        strata[0]["stay_rate_difference_1_minus_0"] = .321
    elif mutation == "empty_rate":
        empty = next(c for s in strata for c in s["by_feedback"].values() if c["denominator"] == 0)
        empty["stay_rate"] = 0.
    elif mutation == "drop_stratum":
        strata.pop()
    else:
        key = {"pool": "pool_group_or_species_estimates", "average": "averaged_stratum_estimates", "causal": "causal_effect_estimated"}[mutation]
        value[key] = True
    dump(path, value)
    rehash(mutated)
    report = failed(mutated)
    assert any("recordwise count" in e for e in report["errors"]), report["errors"]


@pytest.mark.parametrize("field", ["choice", "next_choice", "state", "phase", "local_trial", "next_local_trial", "source_position", "reasons"])
def test_rehashed_ledger_changes_cannot_invent_adjacent_or_absolute_choice_transitions(mutated, field):
    path = mutated["out"] / "ledger.csv"
    with path.open(encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    row = rows[0]
    row[field] = "invented" if field in {"state", "phase", "reasons"} else str(int(row[field] or 0) + 1)
    save_csv(path, fields, rows)
    rehash(mutated)
    report = failed(mutated)
    assert any("adjacency" in e for e in report["errors"]), report["errors"]


@pytest.mark.parametrize("field", [*VERIFY.RESOURCE_FIELDS, "streams", "input_trials", "candidates", "eligible", "excluded", "strata"])
def test_rehashed_metadata_resource_and_coverage_mutations_are_rejected(mutated, field):
    path = mutated["out"] / "metadata.json"
    value = VERIFY.read_json(path)
    value[field] += 1
    dump(path, value)
    rehash(mutated)
    failed(mutated)


def test_rehashed_missing_ledger_pair_is_not_a_shorter_valid_stream(mutated):
    path = mutated["out"] / "ledger.csv"
    lines = path.read_text(encoding="utf8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf8", newline="\n")
    rehash(mutated)
    report = failed(mutated)
    assert any("adjacency" in e for e in report["errors"]), report["errors"]


def test_registered_original_input_byte_change_is_rejected_without_changing_old_results(mutated):
    reg = VERIFY.read_json(mutated["registration"])
    entry = next(e for e in reg["input_files"] if e["role"] == "human_raw")
    path = mutated["cache"] / entry["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    report = failed(mutated)
    assert any("input bytes" in e for e in report["errors"]), report["errors"]


def test_rehashed_parent_passing_status_is_not_reused_for_replacement_metrics(mutated):
    path = mutated["root"] / mutated["cfg"]["human"]["prior_metrics"]
    value = VERIFY.read_json(path)
    value["streams"][0]["trials"] += 1
    dump(path, value)
    report = failed(mutated)
    assert any("parent byte" in e or "parent evidence" in e for e in report["errors"]), report["errors"]


def test_exact_old_human_base_fields_are_preserved_in_every_new_human_stream(audited_fixture):
    f = audited_fixture
    prior = VERIFY.read_json(f["root"] / f["cfg"]["human"]["prior_metrics"])
    expected = {r["stream_id"]: r for r in prior["streams"] if r["kind"] == "human"}
    new = VERIFY.read_json(f["out"] / "metrics.json")
    for row in new["streams"]:
        if row["kind"] == "human":
            assert {k: v for k, v in row.items() if k != "strata"} == expected[row["stream_id"]]


def test_cli_writes_exclusive_report_and_binds_all_input_outputs(audited_fixture, tmp_path):
    f, out = audited_fixture, tmp_path / "new_report.json"
    args = ["--root", str(f["root"]), "--human-cache", str(f["cache"]), "--registration", str(f["registration"]),
            "--result-dir", str(f["out"]), "--out", str(out)]
    assert VERIFY.main(args) == 0
    content = out.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        VERIFY.main(args)
    assert out.read_bytes() == content
