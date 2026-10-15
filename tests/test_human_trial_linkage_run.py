"""Frozen-run tests using invented records only; no original data loaded."""
from pathlib import Path
from copy import deepcopy
import csv
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import savemat
from drosolearn.human_trial_linkage import audit, load_mat_records, register, run, validate_config

REPOSITORY = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def audit_fixture(tmp_path):
    root = tmp_path / "synthetic_repository"
    root.mkdir()
    selected = ["src/drosolearn/human_trial_linkage.py", "src/drosolearn/human_trial_data.py",
                "scripts/verify_human_trial_linkage.py", "tests/test_human_trial_linkage.py",
                "tests/test_human_trial_linkage_run.py", "tests/test_human_trial_linkage_verify.py",
                "docs/human-trial-linkage-protocol.md", "docs/human-trial-linkage-source-review.md",
                "requirements-lock.txt", "requirements-human-candidate-audit.txt", "pyproject.toml",
                "configs/human_trial_linkage.json"]
    for relative in selected:
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / relative, dest)
    source = root / "data/human/stolz_2022_openneuro"
    source.mkdir(parents=True)
    pairs = [([0, 1, 0, 1], [1, 1, 0, 0]), ([1, 0, 1, 0], [0, 0, 1, 1]), ([0, 0, 0, 0], [1, 1, 1, 1])]
    for number, pair in enumerate(pairs, 1):
        markers = []
        for task_index, sequence in enumerate(pair):
            markers += ["S100" if task_index == 0 else "S200"]
            for trial, flag in enumerate(sequence):
                offset = 0 if task_index == 0 else 20
                markers += [f"S{51 + offset + trial % 2}", f"S{53 + offset + trial % 4}", f"S{60 + offset + flag}"]
            markers += ["S101" if task_index == 0 else "S201"]
        path = source / f"sub-s{number}/eeg/sub-s{number}_task-task_events.tsv"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["onset", "duration", "sample", "value"])
            writer.writerows((index, 0, index * 1000, code) for index, code in enumerate(markers))
    arrays = {}
    for task_index, task in enumerate(("REW", "PUN")):
        for key in ("feedback", "action", "expect"):
            cells = np.empty(2, dtype=object)
            for index in range(2):
                cells[index] = np.asarray(pairs[index][task_index] if key == "feedback" else ([1, 2, 1, 2] if key == "action" else [3, 4, 5, 6]))
            arrays[task + "_" + key] = cells
    mat_path = source / "code/Data/RevEx1_behavioralData.mat"
    mat_path.parent.mkdir(parents=True)
    savemat(mat_path, arrays)
    save_json(source / "dataset_description.json", {"License": "CC0", "DatasetDOI": "doi:10.18112/openneuro.ds004295.v1.0.0"})
    manifest_items = [{"path": path.relative_to(source).as_posix(), "size_bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(source.rglob("*")) if path.is_file()]
    save_json(source / "source-manifest.json", {"files": manifest_items})
    evidence = root / "data/human/stolz_2022_linkage_evidence"
    evidence.mkdir(parents=True)
    raw = b"% invented source evidence only\n"
    (evidence / "invented.m").write_bytes(raw)
    save_json(evidence / "source-manifest.json", {"files": [{"path": "invented.m", "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]})
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Synthetic Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-q", "-m", "synthetic"], cwd=root, check=True)
    cfg = root / "configs/human_trial_linkage.json"
    registration = root / "results/registration.json"
    register(root, cfg, registration)
    return root, cfg, registration, root / "results/new_linkage"


def test_frozen_run_preserves_all_candidates_and_ordinal_provenance(tmp_path):
    root, cfg, reg, out = audit_fixture(tmp_path)
    report = run(root, cfg, reg, out)
    assert report["matrix_pairs"] == 6
    assert report["exact_feedback_candidate_pairs"] == 2
    assert report["unmatched_event_labels"] == ["sub-s3"]
    assert report["corroboration_counts"]["strict_comparable"] == 16
    assert report["corroboration_counts"]["action_numeric_agreements"] == 16
    assert report["corroboration_counts"]["expectation_numeric_agreements"] == 16
    assert len(json.loads((out / "artifact_hashes.json").read_text())) == 7
    assert json.loads((out / "metadata.json").read_text())["source_root"] == "../.."
    assert report["readiness"]["reversal_boundaries_or_correct_shape"] == "not_reconstructed"
    assert report["readiness"]["performance_or_human_model_similarity"] == "not_tested"


@pytest.mark.parametrize("relative", ["docs/human-trial-linkage-protocol.md", "data/human/stolz_2022_openneuro/sub-s1/eeg/sub-s1_task-task_events.tsv", "data/human/stolz_2022_linkage_evidence/invented.m"])
def test_any_frozen_source_change_is_rejected_before_output(tmp_path, relative):
    root, cfg, reg, out = audit_fixture(tmp_path)
    with (root / relative).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="after freeze"):
        run(root, cfg, reg, out)
    assert not out.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    root, cfg, reg, out = audit_fixture(tmp_path)
    run(root, cfg, reg, out)
    before = (out / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(root, cfg, reg, out)
    assert (out / "metadata.json").read_bytes() == before


def test_runtime_failure_preserves_metadata_and_registration(tmp_path, monkeypatch):
    import drosolearn.human_trial_linkage as module
    root, cfg, reg, out = audit_fixture(tmp_path)
    def fail(*args):
        raise RuntimeError("synthetic audit failure")
    monkeypatch.setattr(module, "audit", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        module.run(root, cfg, reg, out)
    assert json.loads((out / "metadata.json").read_text())["status"] == "failed"
    assert (out / "registration.json").read_bytes() == reg.read_bytes()
    assert not (out / "artifact_hashes.json").exists()


def test_registration_is_exclusive(tmp_path):
    root, cfg, reg, _ = audit_fixture(tmp_path)
    with pytest.raises(FileExistsError):
        register(root, cfg, reg)


@pytest.mark.parametrize("change", [{"performance_analysis": True}, {"performance_analysis": 0}, {"approximate_matching": 0},
                                     {"action_map": {"left": True, "right": 2}}, {"action_map": {"left": 1.0, "right": 2}},
                                     {"expectation_offset": 3.0}, {"expectation_offset": 4}, {"unexpected": 1}])
def test_config_cannot_enable_analysis_or_change_fixed_coding(change):
    config = json.loads((REPOSITORY / "configs/human_trial_linkage.json").read_text())
    config.update(change)
    with pytest.raises(ValueError, match="Fixed"):
        validate_config(config)


def test_evidence_manifest_path_escape_rejected(tmp_path):
    root, cfg, reg, _ = audit_fixture(tmp_path)
    path = root / "data/human/stolz_2022_linkage_evidence/source-manifest.json"
    data = json.loads(path.read_text())
    data["files"][0]["path"] = "../../../../requirements-lock.txt"
    save_json(path, data)
    with pytest.raises(ValueError, match="Unsafe"):
        register(root, cfg, root / "new-reg.json")


def test_numeric_disagreements_are_diagnostics_not_exclusion(tmp_path):
    root, cfg, _, _ = audit_fixture(tmp_path)
    path = root / "data/human/stolz_2022_openneuro/sub-s1/eeg/sub-s1_task-task_events.tsv"
    text = path.read_text().replace("S51", "S52").replace("S53", "S54")
    path.write_text(text, encoding="utf-8")
    manifest_path = root / "data/human/stolz_2022_openneuro/source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    item = next(item for item in manifest["files"] if item["path"].startswith("sub-s1/"))
    item.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save_json(manifest_path, manifest)
    config = json.loads(cfg.read_text())
    report, _, _, rows = audit(root, config)
    assert report["exact_feedback_candidate_pairs"] == 2
    # Only the REW marker codes were changed; the PUN codes remain untouched.
    assert report["corroboration_counts"]["action_numeric_disagreements"] == 2
    assert report["corroboration_counts"]["expectation_numeric_disagreements"] == 1
    assert any(not row["action_numeric_agreement"] for row in rows)
