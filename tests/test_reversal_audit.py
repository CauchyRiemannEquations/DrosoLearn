"""Accept complete histories and reject altered/rehashed scientific artifacts."""
from pathlib import Path
import ast
import copy
import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

from drosolearn.reversal import register, run
from drosolearn.reversal_audit import verify


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audited_fixture(tmp_path_factory):
    # This short integration fixture lives outside research results and never
    # uses the new study's registered 2001..2010 seeds.
    root = tmp_path_factory.mktemp("reversal-audit-fixture")
    shutil.copytree(ROOT / "src", root / "src")
    graph = "data/flywire_783_mb_left_kc512_s701"
    shutil.copytree(ROOT / graph, root / graph)
    for relative in ("docs/reversal-protocol.md", "pyproject.toml", "requirements-lock.txt",
                     "data/source-manifest.json", "data/LICENSE.md"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    config = json.loads((ROOT / "configs/reversal_diagnostic.json").read_text(encoding="utf-8"))
    config.update(scope="integration_test", seeds=[31], trials_per_phase=8,
                  checkpoint_every=4, eval_trials=8, probe_rows=4)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    registration = root / "registration.json"
    register(root, config_path, registration)
    output = root / "fixture-output"
    run(root, config_path, registration, output)
    return output


def _clone(fixture, tmp_path):
    output = tmp_path / "tampered-output"
    shutil.copytree(fixture, output)
    return output


def _save_rehash(output, relative, contents):
    path = output / relative
    if isinstance(contents, dict):
        path.write_text(json.dumps(contents), encoding="utf-8")
    elif isinstance(contents, str):
        path.write_text(contents, encoding="utf-8")
    manifest_path = output / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_rows(output, relative, rows):
    path = output / relative
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _save_rehash(output, relative, None)


def test_independent_auditor_accepts_all_families_and_histories(audited_fixture):
    report = verify(audited_fixture)
    assert report["passed"], report["errors"]
    assert report["counts"]["trials"] == 6 * (4 * 16 + 8) == 432
    assert report["counts"]["histories"] == 5
    assert report["counts"]["curve_rows"] == 6 * 9 * 3 * 2
    assert report["counts"]["evaluation_cases"] == 6 * 9 * 2 * 2 * 8
    assert report["counts"]["boundaries"] == 30
    assert report["counts"]["checkpoints"] == 6 * (4 * 6 + 4)
    assert any("counterfactual" in check for check in report["checks"])


def test_auditor_has_no_experiment_model_or_data_imports():
    tree = ast.parse((ROOT / "src/drosolearn/reversal_audit.py").read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == ["datetime", "pathlib"]
    assert all(node.level == 0 for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))


def test_audit_never_mutates_primary_artifacts(audited_fixture):
    manifest = json.loads((audited_fixture / "artifact_hashes.json").read_text())
    before = {key: hashlib.sha256((audited_fixture / key).read_bytes()).hexdigest() for key in manifest}
    assert verify(audited_fixture)["passed"]
    after = {key: hashlib.sha256((audited_fixture / key).read_bytes()).hexdigest() for key in manifest}
    assert before == after == manifest


@pytest.mark.parametrize("section,field", [
    ("outcomes", "phase2_iid_auc"), ("outcomes", "phase1_acquired"),
    ("outcomes", "phase2_acquired"), ("outcomes", "qualified_reversal_criterion_met"),
    ("paired_contrasts", "history_auc_penalty"), ("paired_contrasts", "reset_auc_gain"),
    ("family_verdicts", "history_cost_criterion_met"),
    ("family_verdicts", "positive_seed_count"), ("family_verdicts", "sd_history_auc_penalty"),
])
def test_summary_recalculation_rejects_rehashed_claims(audited_fixture, tmp_path, section, field):
    output = _clone(audited_fixture, tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    value = summary[section][0][field]
    summary[section][0][field] = not value if type(value) is bool else (0.123 if value is None else value + 0.123)
    _save_rehash(output, "summary.json", summary)
    report = verify(output)
    assert not report["passed"]
    assert any(label in report["errors"][0] for label in ("Summary", "Paired", "Family"))


@pytest.mark.parametrize("field", ["action", "reward", "optimizer_factor", "p_right", "internal_step_l2"])
def test_manual_trial_replay_rejects_rehashed_trial_edits(audited_fixture, tmp_path, field):
    output = _clone(audited_fixture, tmp_path)
    relative = "seed_31/trials.csv"
    with (output / relative).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = str(1 - int(rows[0][field])) if field in ("action", "reward") else str(float(rows[0][field]) + 0.01)
    _rewrite_rows(output, relative, rows)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["head_reset_l2", "internal_reset_l2", "head_updates_before_phase2", "final_internal_updates"])
def test_rehashed_boundary_counterfactual_edits_are_rejected(audited_fixture, tmp_path, field):
    output = _clone(audited_fixture, tmp_path)
    relative = "seed_31/plastic/reset_head_after_high/boundary.json"
    boundary = json.loads((output / relative).read_text())
    boundary[field] += 1
    _save_rehash(output, relative, boundary)
    report = verify(output)
    assert not report["passed"] and "Boundary" in report["errors"][0]


@pytest.mark.parametrize("tamper", ["missing_trial", "fresh_phase1", "duplicate_curve", "raw_bytes"])
def test_design_coverage_and_hash_failure_are_detected(audited_fixture, tmp_path, tamper):
    output = _clone(audited_fixture, tmp_path)
    if tamper in ("missing_trial", "fresh_phase1"):
        relative = "seed_31/trials.csv"
        with (output / relative).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if tamper == "missing_trial": rows.pop()
        else:
            extra = copy.copy(rows[0]); extra["history"] = "fresh_low"; rows.append(extra)
        _rewrite_rows(output, relative, rows)
    elif tamper == "duplicate_curve":
        relative = "learning_curves.csv"
        with (output / relative).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows.append(rows[0].copy()); _rewrite_rows(output, relative, rows)
    else:
        with (output / "seed_31/trials.csv").open("a", encoding="utf-8") as stream: stream.write("corruption\n")
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("tamper", ["phase2_uniform", "joint_crossing", "input_rng", "topology_counts", "snapshot_head"])
def test_saved_inputs_topology_and_snapshot_edits_are_rejected(audited_fixture, tmp_path, tamper):
    output = _clone(audited_fixture, tmp_path)
    relative = ("seed_31/ann_topology.npz" if tamper == "topology_counts" else
                "seed_31/fixed/reset_head_after_high/phase2_0000.npz" if tamper == "snapshot_head" else "seed_31/stimuli.npz")
    path = output / relative
    with np.load(path, allow_pickle=False) as saved: arrays = {key: saved[key].copy() for key in saved.files}
    if tamper == "phase2_uniform": arrays["phase2_uniforms"][0] = 0.12345
    elif tamper in ("joint_crossing", "input_rng"):
        if tamper == "joint_crossing": arrays["iid_nuisance"][0, 0, 0] += 0.001
        else:
            # Preserve crossing and marginal chance, but alter the registered
            # input RNG stream and its numerical identity.
            arrays["iid_nuisance"][:, :, 0] = 0.25 + (0.75 - arrays["iid_nuisance"][:, :, 0])
    elif tamper == "topology_counts": arrays["counts"][0] += 1
    else: arrays["readout_weights"][0] += 0.1
    np.savez_compressed(path, **arrays); _save_rehash(output, relative, None)
    report = verify(output)
    assert not report["passed"] and report["errors"]
