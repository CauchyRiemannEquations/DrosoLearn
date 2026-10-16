"""수작업 집계로만 비연속 원 행 번호와 별도 표현 등록을 검사한다."""
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("conditional_plot_v2", REPO / "scripts/plot_conditional_transition_v2.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def feedback(d, s):
    return {"denominator": d, "stay": s, "switch": d - s,
            "stay_rate": s / d if d else None, "switch_rate": (d - s) / d if d else None}


def stream(identifier, human, pattern):
    rows = []
    for (phase, state, choice), (d0, d1) in zip(plot.HUMAN_CELLS if human else plot.MODEL_CELLS, pattern):
        zero, one = feedback(d0, d0 // 2), feedback(d1, (d1 + 1) // 2)
        common = d0 > 0 and d1 > 0
        rows.append({"phase": phase, "state": state, "choice": choice, "by_feedback": {"0": zero, "1": one},
                     "common_support": common, "support_status": "both_feedback" if common else "feedback_0_only" if d0 else
                     "feedback_1_only" if d1 else "neither", "stay_rate_difference_1_minus_0": one["stay_rate"] - zero["stay_rate"] if common else None})
    overall = {f: feedback(sum(row["by_feedback"][f]["denominator"] for row in rows),
                           sum(row["by_feedback"][f]["stay"] for row in rows)) for f in ("0", "1")}
    eligible = sum(row["denominator"] for row in overall.values())
    return {"stream_id": identifier, "kind": "human" if human else "model", "coordinate": "source_coded_card_identity" if human else
            "smaller_larger_quantity_category", "trials": 8, "candidates": 7, "eligible": eligible, "excluded": 7 - eligible,
            "reason_counts": {}, "by_feedback": overall, "strata": rows}


def refresh(data):
    root, reg, meta, metrics = (data[key] for key in ("root", "reg", "meta", "metrics"))
    save(root / "registration.json", reg)
    meta["registration_sha256"] = plot.digest(root / "registration.json")
    meta["input_files"] = copy.deepcopy(reg["input_files"])
    save(root / "metadata.json", meta)
    save(root / "metrics.json", metrics)
    artifacts = {name: plot.digest(root / name) for name in ("registration.json", "metadata.json", "metrics.json", "ledger.csv")}
    save(root / "artifact_hashes.json", artifacts)
    inventory = json.dumps(reg["input_files"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    save(root / "verification.json", {"study_id": plot.STUDY, "status": "passed", "errors": [],
         "verified_at_utc": "2000-01-01T00:00:00+00:00", "metrics_sha256": artifacts["metrics.json"],
         "registration_sha256": artifacts["registration.json"], "config_sha256": reg["config_sha256"],
         "artifact_hashes_sha256": plot.digest(root / "artifact_hashes.json"),
         "input_manifest_sha256": hashlib.sha256(inventory).hexdigest(),
         "handcrafted_plot_contract_only": True})


@pytest.fixture
def example(tmp_path, monkeypatch):
    monkeypatch.setattr(plot.subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    frozen_names = json.loads((REPO / "configs/conditional_transition_v1.json").read_text(encoding="utf8"))["frozen_files"]
    cfg = {"study_id": plot.STUDY, "scope": "integration_test", "human": {"record_count": 2, "trials": 8,
           "coordinate": "source_coded_card_identity"}, "model": {"seeds": [17], "trials_per_phase": 4,
           "conditions": list(plot.CONDITIONS), "environments": [{"id": value} for value in plot.ENVIRONMENTS],
           "coordinate": "smaller_larger_quantity_category"}, "learning_updates": 0, "learner_constructions": 0,
           "dataset_generations": 0, "rng_draws": 0, "pooling": False, "strata_averaging": False, "causal_inference": False,
           "registered_streams": 26, "registered_input_trials": 208, "registered_candidate_pairs": 182,
           "registered_strata": 112, "frozen_files": frozen_names}
    rows = [stream(f"human/main/{index}", True, [(1, 1), (1, 0), (0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]) for index in (0, 27)]
    pattern = {"deterministic": [(2, 0), (0, 1), (0, 2), (1, 0)],
               "probabilistic": [(1, 1), (1, 0), (0, 2), (0, 1)],
               "uninformative": [(0, 0), (2, 1), (1, 1), (0, 1)]}
    rows += [stream(f"model/17/{condition}/{env}", False, pattern[env]) for condition in plot.CONDITIONS for env in plot.ENVIRONMENTS]
    reg = {"study_id": plot.STUDY, "config": cfg, "config_sha256": "c" * 64,
           "frozen_files": {name: plot.digest(REPO / name) for name in frozen_names},
           "input_files": [{"role": "human_raw", "row_index": index, "path": f"osf/handcrafted_{index}.mat",
                            "osf_file_id": f"handcrafted_{index}"} for index in (0, 27)]}
    metrics = {"study_id": plot.STUDY, "streams": rows, "pool_group_or_species_estimates": False,
               "averaged_stratum_estimates": False, "causal_effect_estimated": False, "handcrafted_plot_contract_only": True}
    meta = {"study_id": plot.STUDY, "status": "completed", "config": cfg, "config_sha256": reg["config_sha256"],
            "frozen_files": reg["frozen_files"], "learning_updates": 0, "learner_constructions": 0, "dataset_generations": 0,
            "rng_draws": 0, "pooled_group_or_species_estimates": False, "averaged_stratum_estimates": False,
            "causal_effect_estimated": False, "human_base_metrics_exact_match": True, "streams": len(rows),
            "input_trials": sum(row["trials"] for row in rows), "candidates": sum(row["candidates"] for row in rows),
            "eligible": sum(row["eligible"] for row in rows), "excluded": sum(row["excluded"] for row in rows),
            "strata": sum(len(row["strata"]) for row in rows), "handcrafted_plot_contract_only": True}
    root = tmp_path / "measurement"
    root.mkdir()
    (root / "ledger.csv").write_text("handcrafted_plot_contract_only\n", encoding="utf8")
    data = {"root": root, "reg": reg, "meta": meta, "metrics": metrics, "presentation": tmp_path / "presentation.json"}
    refresh(data)
    plot.register_presentation(root, data["presentation"])
    return data


def test_accepts_nonconsecutive_0_and_27(example):
    bundle = plot.validate_inputs(example["root"], example["presentation"])
    assert bundle["human_ids"] == ["human/main/0", "human/main/27"]


def test_preserves_input_order_without_sorting(example):
    example["reg"]["input_files"].reverse()
    refresh(example)
    path = example["root"].parent / "reordered_presentation.json"
    plot.register_presentation(example["root"], path)
    assert plot.validate_inputs(example["root"], path)["human_ids"] == ["human/main/27", "human/main/0"]


@pytest.mark.parametrize("problem", ["missing", "duplicate_index", "duplicate_path", "duplicate_file_id", "negative", "boolean"])
def test_refuses_missing_or_duplicate_raw_identity(example, problem):
    entries = example["reg"]["input_files"]
    if problem == "missing":
        entries.pop()
    elif problem == "duplicate_index":
        entries[1]["row_index"] = entries[0]["row_index"]
    elif problem == "duplicate_path":
        entries[1]["path"] = entries[0]["path"]
    elif problem == "duplicate_file_id":
        entries[1]["osf_file_id"] = entries[0]["osf_file_id"]
    else:
        entries[1]["row_index"] = -1 if problem == "negative" else True
    refresh(example)
    with pytest.raises(ValueError):
        plot.validate_inputs(example["root"], example["presentation"])


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unregistered"])
def test_refuses_missing_or_duplicate_measured_stream(example, problem):
    rows = example["metrics"]["streams"]
    if problem == "missing":
        rows.pop(1)
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[1]["stream_id"] = "human/main/1"
    refresh(example)
    with pytest.raises(ValueError):
        plot.validate_inputs(example["root"], example["presentation"])


@pytest.mark.parametrize("problem", ["new_source", "main_input", "old_source", "missing_input", "future", "before_verification"])
def test_refuses_unbound_presentation_registration(example, problem):
    path = example["presentation"]
    record = json.loads(path.read_text(encoding="utf8"))
    if problem == "new_source":
        record["frozen_files"][plot.SCRIPT] = "0" * 64
    elif problem == "main_input":
        record["input_hashes"]["metrics.json"] = "0" * 64
    elif problem == "old_source":
        record["original_plot_sha256"] = "0" * 64
    elif problem == "missing_input":
        record["input_hashes"].pop("metadata.json")
    else:
        record["registered_at_utc"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat() if problem == "future" else "1999-01-01T00:00:00+00:00"
    save(path, record)
    with pytest.raises(ValueError):
        plot.validate_inputs(example["root"], path)


def test_registration_mode_has_no_png_and_refuses_overwrite(example):
    path = example["root"].parent / "another_presentation.json"
    assert plot.main(["--register", "--result-dir", str(example["root"]), "--out", str(path)]) == 0
    record = json.loads(path.read_text(encoding="utf8"))
    assert set(record["frozen_files"]) == set(plot.PRESENTATION_FILES)
    assert set(record["input_hashes"]) == set(plot.INPUTS)
    assert record["scientific_reanalysis"] is False and record["source_git_commit"] == "a" * 40
    assert not list(example["root"].parent.rglob("*.png"))
    with pytest.raises(FileExistsError):
        plot.register_presentation(example["root"], path)


def test_registration_requires_unchanged_old_source_bytes(example):
    old = plot.OLD_SCRIPT
    example["reg"]["frozen_files"][old] = "0" * 64
    refresh(example)
    with pytest.raises(ValueError, match="원 연구의 고정 소스"):
        plot.register_presentation(example["root"], example["root"].parent / "invalid.json")


def test_original_plot_failure_is_preserved_on_handwritten_rows(example):
    original_spec = importlib.util.spec_from_file_location("original_conditional_plot", REPO / plot.OLD_SCRIPT)
    original = importlib.util.module_from_spec(original_spec)
    original_spec.loader.exec_module(original)
    with pytest.raises(ValueError, match="기록이 중복되거나 등록 범위"):
        original.validate_inputs(example["root"])


def test_renders_handwritten_nonconsecutive_rows_and_keeps_exclusive_output(example):
    out = example["root"].parent / "figure"
    record = plot.create_figure(example["root"], out, example["presentation"])
    assert record["status"] == "completed"
    assert record["selection"]["human_stream_order"] == ["human/main/0", "human/main/27"]
    assert record["presentation_registration_sha256"] == plot.digest(example["presentation"])
    assert record["source_git_commit"] == "a" * 40 and record["scientific_reanalysis"] is False
    assert set(path.name for path in out.iterdir()) == {"feedback_support.png", "metadata.json"}
    with pytest.raises(FileExistsError):
        plot.create_figure(example["root"], out, example["presentation"])


def test_render_cli_requires_separate_registration(example):
    with pytest.raises(ValueError, match="별도 표현 등록"):
        plot.main(["--result-dir", str(example["root"]), "--out", str(example["root"].parent / "unregistered")])
