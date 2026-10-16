"""Entirely handwritten MAT/CSV parents; no real human/model rows are copied."""
from datetime import datetime, timedelta, timezone
import builtins
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.io import savemat

from drosolearn import conditional_transition as study
from drosolearn.feedback_transition import human_rows, measure_stream
from test_conditional_transition import model_record, synthetic_config


REPO = Path(__file__).resolve().parents[1]
PAST = "2000-01-01T00:00:00+00:00"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def handcrafted_parents(root, cache, cfg):
    entries, human_inputs, human_base, old_ledger = [], [], [], []
    for index, actions in enumerate(([1., 1., 2., 3., 2., np.nan, 1., 2.], [2., 2., 1., np.nan, 1., 3., 2., 1.])):
        a = np.asarray([actions])
        s = np.asarray([[1], [1], [2], [2], [2], [1], [1], [2]], dtype=np.int64)
        r = np.asarray([[1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 1]], dtype=np.int64)
        c = np.full((1, 8), np.nan)
        for trial in range(8):
            if a[0, trial] in (1, 2):
                c[0, trial] = int(a[0, trial] == s[trial, 0])
        path = f"osf/handcrafted_{index}.mat"
        (cache / path).parent.mkdir(parents=True, exist_ok=True)
        savemat(cache / path, {"A": a, "R": r, "S": s, "C": c})
        entry = {"role": "osf_raw", "path": path, "bytes": (cache / path).stat().st_size, "sha256": study.digest(cache / path),
                 "derived_group": "main", "derived_row_index": index, "osf_file_id": f"handcrafted_{index}"}
        entries.append(entry)
        human_inputs.append({"base": "human_cache", "role": "human_raw", "path": path, "bytes": entry["bytes"], "sha256": entry["sha256"],
                             "osf_file_id": entry["osf_file_id"], "row_index": index})
        rows = human_rows({"A": a, "R": r, "S": s, "C": c}, stream_id=f"human/main/{index}", trials=8, regime_boundary_after_trial=4)
        base, ledger = measure_stream(rows)
        human_base.append(base)
        old_ledger.extend(ledger)
    manifest = {"trials_per_row": 8, "source_files": entries, "handcrafted_engineering_only": True}
    save_json(root / cfg["human"]["origin_manifest"], manifest)
    task_reg = root / "results/registration_human_task_structure_v1.json"
    save_json(task_reg, {"handcrafted_engineering_only": True})
    task = {"status": "pass", "inputs": [{key: entry[key] for key in ("path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")}
                                          for entry in entries],
            "gate": {"source_coded_descriptive_metric_allowed": True, "exact_historical_task_execution_version_verified": False,
                     "direct_human_model_similarity_allowed": False}, "handcrafted_engineering_only": True}
    save_json(root / cfg["human"]["structure_result"], task)
    save_json(root / cfg["human"]["structure_verification"], {"status": "pass", "source_coded_descriptive_gate_passed": True,
        "result_sha256": study.digest(root / cfg["human"]["structure_result"]), "registration_sha256": study.digest(task_reg),
        "verified_at_utc": PAST, "handcrafted_engineering_only": True})
    save_json(root / "results/human_osf_origin_v1/verification.json", {"status": "passed", "source_identity_gate_passed": True,
        "manifest_sha256": study.digest(root / cfg["human"]["origin_manifest"]), "verified_at_utc": PAST, "handcrafted_engineering_only": True})
    old_config = {"human": {"trials": 8, "regime_boundary_after_trial": 4}, "handcrafted_engineering_only": True}
    save_json(root / "configs/feedback_transition_v1.json", old_config)
    old_config_sha = study.digest(root / "configs/feedback_transition_v1.json")
    old_reg = root / "results/registration_feedback_transition_v1.json"
    save_json(old_reg, {"inputs": human_inputs, "config": old_config, "config_sha256": old_config_sha,
                       "source_git_commit": "a" * 40, "registered_at_utc": PAST, "handcrafted_engineering_only": True})
    old_base = root / "results/feedback_transition_v1"
    save_json(old_base / "metrics.json", {"study_id": "feedback_transition_v1", "streams": human_base, "pooled_species_or_group_estimates": False})
    save_json(old_base / "metadata.json", {"status": "completed", "config_sha256": old_config_sha, "inputs": human_inputs,
        "registration_sha256": study.digest(old_reg), "handcrafted_engineering_only": True})
    write_csv(old_base / "ledger.csv", study.LEDGER_FIELDS[:9], old_ledger)
    save_json(old_base / "artifact_hashes.json", {name: study.digest(old_base / name) for name in ("metadata.json", "ledger.csv", "metrics.json")})
    save_json(old_base / "verification.json", {"status": "passed", "errors": [], "config_sha256": old_config_sha,
        "registration_sha256": study.digest(old_reg), "verified_at_utc": PAST, "handcrafted_engineering_only": True})
    cconfig = read_json(REPO / "configs/stochastic_feedback_network_v1.json")
    cconfig.update(scope="integration_test", seeds=cfg["model"]["seeds"], trials_per_phase=4, eval_trials=4, checkpoint_every=2, probe_rows=2)
    count = len(cconfig["seeds"]) * 8 * 3 * 8
    cconfig.update(actual_choice_feedback_trials=count, expected_head_update_calls=count * 7 // 8,
                   expected_internal_update_calls=count * 3 // 8)
    save_json(root / cfg["model"]["parent_config"], cconfig)
    cconfig_sha = study.digest(root / cfg["model"]["parent_config"])
    c_reg = root / "results/registration_stochastic_feedback_network_v1.json"
    save_json(c_reg, {"config": cconfig, "config_sha256": cconfig_sha, "source_git_commit": "b" * 40,
                      "registered_at_utc": PAST, "handcrafted_engineering_only": True})
    cbase = root / cfg["model"]["root"]
    for seed in cconfig["seeds"]:
        rows = []
        for condition in cconfig["conditions"]:
            for environment in cconfig["environments"]:
                for phase in ("phase1", "phase2"):
                    for local in range(1, 5):
                        left, right = (2, 4) if local % 2 else (4, 2)
                        choice = int(local % 3 != 0)
                        chosen = 4 if choice else 2
                        action = int(right == chosen)
                        p, au, ru = (.75 if action else .25), (.1 if action else .9), (.1 if local % 2 else .9)
                        q = environment["q_optimal"] if choice == (1 if phase == "phase1" else 0) else environment["q_other"]
                        reward = int(ru < q)
                        global_trial = local + (4 if phase == "phase2" else 0)
                        row = model_record(seed=str(seed), condition=condition, environment=environment["id"], phase=phase,
                            trial=str(local), global_trial=str(global_trial), family_id=f"handwritten:{phase}:{local}",
                            left_count=str(left), right_count=str(right), chosen_count=str(chosen), action=str(action), reward=str(reward),
                            p_right=repr(p), q_reward=repr(q), action_uniform=repr(au), reward_uniform=repr(ru),
                            head_update_called=str(int(condition != "frozen")), internal_update_called=str(int(condition in ("plastic", "rewired_plastic", "ann_plastic"))),
                            optimizer_factor=repr((action - p) * (reward - .5)), head_updates=str(global_trial if condition != "frozen" else 0),
                            internal_updates=str(global_trial if condition in ("plastic", "rewired_plastic", "ann_plastic") else 0))
                        rows.append(row)
        write_csv(cbase / f"seed_{seed}/trials.csv", study.MODEL_FIELDS, rows)
    save_json(cbase / "summary.json", {"study_id": "stochastic_feedback_network_v1", "stage": "network", "status": "completed", "handcrafted_engineering_only": True})
    save_json(cbase / "metadata.json", {"stage": "network", "status": "completed", "config": cconfig, "config_sha256": cconfig_sha,
        "registration_sha256": study.digest(c_reg), "source_git_commit": "b" * 40, "handcrafted_engineering_only": True})
    shutil.copyfile(c_reg, cbase / "registration.json")
    artifacts = {name: study.digest(cbase / name) for name in ("metadata.json", "summary.json", "registration.json")}
    artifacts.update({f"seed_{seed}/trials.csv": study.digest(cbase / f"seed_{seed}/trials.csv") for seed in cconfig["seeds"]})
    save_json(cbase / "artifact_hashes.json", artifacts)
    save_json(cbase / "verification.json", {"status": "passed", "passed": True, "errors": [], "verified_at_utc": PAST,
        "config_sha256": cconfig_sha, "registration_sha256": study.digest(c_reg), "summary_sha256": study.digest(cbase / "summary.json"),
        "counts": {"actual_choice_feedback_trials": count}, "handcrafted_engineering_only": True})


def make_fixture(tmp_path, monkeypatch):
    root, cache = tmp_path / "synthetic_repo", tmp_path / "synthetic_human_cache"
    root.mkdir()
    cache.mkdir()
    cfg = synthetic_config()
    for name in cfg["frozen_files"]:
        if name == "configs/conditional_transition_v1.json":
            continue
        source, target = REPO / name, root / name
        assert source.is_file(), f"Wait for new source files before integration: {name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    def forbidden(*args, **kwargs):
        pytest.fail("Handwritten transition fixtures must not invoke RNG or learning")
    actual_import = builtins.__import__
    def no_learning_import(name, globals=None, locals=None, fromlist=(), level=0):
        resolved = importlib.util.resolve_name("." * level + name, (globals or {}).get("__package__")) if level else name
        requested = [resolved, *(f"{resolved}.{item}" for item in (fromlist or ()))]
        if any(module == blocked or module.startswith(blocked + ".") for module in requested
               for blocked in ("drosolearn.reward_linear", "drosolearn.data")):
            forbidden()
        return actual_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", no_learning_import)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    handcrafted_parents(root, cache, cfg)
    cfg["parent_gate_identity"] = {"published_commit": "c" * 40, "file_sha256": {path: study.digest(root / path) for path in cfg["parent_files"]}}
    config = root / "configs/conditional_transition_v1.json"
    save_json(config, cfg)
    actual_git = study.subprocess.check_output
    def synthetic_git(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "d" * 40 + "\n"
        return actual_git(command, **kwargs)
    monkeypatch.setattr(study.subprocess, "check_output", synthetic_git)
    return {"root": root, "cache": cache, "cfg": cfg, "config": config, "registration": root / "registration.json", "out": root / "measurement"}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return make_fixture(tmp_path, monkeypatch)


def register(fixture):
    return study.register(fixture["root"], fixture["config"], fixture["cache"], fixture["registration"])


def run(fixture):
    return study.run(fixture["root"], fixture["config"], fixture["registration"], fixture["cache"], fixture["out"])


@pytest.fixture
def completed(fixture):
    register(fixture)
    assert run(fixture)["status"] == "completed"
    return fixture


def test_registration_freezes_handwritten_bytes_without_decoding_or_measuring(fixture, monkeypatch):
    monkeypatch.setattr(study, "streams", lambda *a, **k: pytest.fail("Registration measured transitions"))
    record = register(fixture)
    assert len(record["input_files"]) == 4
    assert len(record["parent_files"]) == 17
    assert record["source_git_commit"] == "d" * 40
    assert not fixture["out"].exists()


@pytest.mark.parametrize("name,package,fromlist,level", [
    ("drosolearn.reward_linear", None, (), 0),
    ("drosolearn.data", None, (), 0),
    ("drosolearn", None, ("reward_linear",), 0),
    ("drosolearn", None, ("data",), 0),
    ("reward_linear", "drosolearn", (), 1),
    ("", "drosolearn", ("data",), 1),
])
def test_fixture_blocks_learning_and_data_imports_before_loading(fixture, name, package, fromlist, level):
    with pytest.raises(pytest.fail.Exception, match="must not invoke RNG or learning"):
        builtins.__import__(name, {"__package__": package}, None, fromlist, level)


@pytest.mark.parametrize("target", ["source", "config", "human_raw", "model_csv", "parent"])
def test_registered_source_data_or_gate_changes_are_rejected_before_output(fixture, target):
    record = register(fixture)
    path = {"source": fixture["root"] / "src/drosolearn/conditional_transition.py", "config": fixture["config"],
        "human_raw": fixture["cache"] / record["input_files"][0]["path"],
        "model_csv": fixture["root"] / next(row["path"] for row in record["input_files"] if row["role"] == "model_actual"),
        "parent": fixture["root"] / fixture["cfg"]["human"]["prior_metrics"]}[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_existing_registration_and_result_directory_are_preserved(fixture):
    register(fixture)
    before = fixture["registration"].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        register(fixture)
    assert fixture["registration"].read_bytes() == before
    fixture["out"].mkdir()
    keep = fixture["out"] / "keep.bin"
    keep.write_bytes(b"previous result")
    with pytest.raises((ValueError, FileExistsError)):
        run(fixture)
    assert list(fixture["out"].iterdir()) == [keep] and keep.read_bytes() == b"previous result"


@pytest.mark.parametrize("target", ["future", "naive", "git"])
def test_bad_registration_chronology_or_source_identity_cannot_execute(fixture, target):
    record = register(fixture)
    if target == "git":
        record["source_git_commit"] = "missing"
    else:
        record["registered_at_utc"] = ((datetime.now(timezone.utc) + timedelta(days=1)) if target == "future" else datetime(2000, 1, 1)).isoformat()
    save_json(fixture["registration"], record)
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_complete_handwritten_measurement_has_every_stream_cell_and_original_pair(completed):
    meta = read_json(completed["out"] / "metadata.json")
    assert (meta["streams"], meta["input_trials"], meta["candidates"], meta["strata"]) == (50, 400, 350, 208)
    assert meta["eligible"] + meta["excluded"] == 350
    assert meta["human_base_metrics_exact_match"] is True
    assert all(meta[key] == 0 for key in ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws"))
    metrics = read_json(completed["out"] / "metrics.json")
    assert len(metrics["streams"]) == 50
    assert sum(len(row["strata"]) for row in metrics["streams"]) == 208
    assert all(len(row["strata"]) == (8 if row["kind"] == "human" else 4) for row in metrics["streams"])
    ledger = read_csv(completed["out"] / "ledger.csv")
    assert len(ledger) == 350
    model_edges = [row for row in ledger if row["stream_id"].startswith("model/") and row["trial"] == "4"]
    assert len(model_edges) == 48
    assert all(row["reasons"] == "phase_boundary|state_boundary" and row["local_trial"] == "4" and row["next_local_trial"] == "1" for row in model_edges)


def test_original_human_base_metrics_match_exactly_and_conditional_counts_recombine(completed):
    metrics = read_json(completed["out"] / "metrics.json")
    prior = read_json(completed["root"] / completed["cfg"]["human"]["prior_metrics"])
    old = {row["stream_id"]: row for row in prior["streams"]}
    for stream in metrics["streams"]:
        if stream["kind"] == "human":
            assert {key: stream[key] for key in study.BASE_SUMMARY_FIELDS} == old[stream["stream_id"]]
        for feedback in ("0", "1"):
            for field in ("denominator", "stay", "switch"):
                assert sum(row["by_feedback"][feedback][field] for row in stream["strata"]) == stream["by_feedback"][feedback][field]
        for row in stream["strata"]:
            d0, d1 = row["by_feedback"]["0"]["denominator"], row["by_feedback"]["1"]["denominator"]
            assert row["common_support"] is bool(d0 > 0 and d1 > 0)
            if not row["common_support"]:
                assert row["stay_rate_difference_1_minus_0"] is None


def test_failed_partial_measurement_keeps_completed_records_and_ledger(fixture, monkeypatch):
    register(fixture)
    original = study.streams
    def fail_after_human(*args, **kwargs):
        iterator = original(*args, **kwargs)
        yield next(iterator)
        raise RuntimeError("deliberate post-record failure")
    monkeypatch.setattr(study, "streams", fail_after_human)
    with pytest.raises(RuntimeError, match="deliberate post-record failure"):
        run(fixture)
    metadata = read_json(fixture["out"] / "metadata.json")
    assert metadata["status"] == "failed" and metadata["streams"] == 1
    assert len(read_csv(fixture["out"] / "ledger.csv")) == 7
    assert len(read_json(fixture["out"] / "partial_metrics.json")["streams"]) == 1
    assert all(study.digest(fixture["out"] / path) == sha for path, sha in read_json(fixture["out"] / "artifact_hashes.json").items())


def test_every_final_artifact_and_external_input_retains_its_registered_digest(completed):
    output = completed["out"]
    hashes = read_json(output / "artifact_hashes.json")
    assert set(hashes) == {"registration.json", "metadata.json", "metrics.json", "ledger.csv"}
    assert all(study.digest(output / path) == sha for path, sha in hashes.items())
    record = read_json(completed["registration"])
    assert all(study.digest((completed["cache"] if row["base"] == "human_cache" else completed["root"]) / row["path"]) == row["sha256"]
               for row in record["input_files"])
