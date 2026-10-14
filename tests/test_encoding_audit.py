"""Auditor acceptance and semantic tampering after primary-file rehashing."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

from drosolearn.encoding import run
from drosolearn.encoding_audit import verify
from test_encoding import prepare_encoding_fixture


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audited_encoding_fixture(tmp_path_factory):
    # Engineering fixture seed41 is separate from scientific seeds3001..3010.
    root, config, registration = prepare_encoding_fixture(tmp_path_factory.mktemp("encoding-audit"))
    output = root / "fixture-output"
    run(root, config, registration, output)
    report = verify(output)
    assert report["passed"], report["errors"]
    return output


def _clone(fixture, tmp_path):
    output = tmp_path / "tampered-output"
    shutil.copytree(fixture, output)
    return output


def _rehash(output, relative):
    path = output / "artifact_hashes.json"
    hashes = json.loads(path.read_text(encoding="utf-8"))
    hashes[relative] = hashlib.sha256((output / relative).read_bytes()).hexdigest()
    path.write_text(json.dumps(hashes), encoding="utf-8")


def _save_json(output, relative, value):
    (output / relative).write_text(json.dumps(value), encoding="utf-8")
    _rehash(output, relative)


def _rows(output, relative):
    with (output / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save_rows(output, relative, rows):
    with (output / relative).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _rehash(output, relative)


def _arrays(output, relative):
    with np.load(output / relative, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _save_arrays(output, relative, arrays):
    np.savez_compressed(output / relative, **arrays)
    _rehash(output, relative)


def test_auditor_accepts_all_six_families_and_two_actual_fits(audited_encoding_fixture):
    report = verify(audited_encoding_fixture)
    assert report["passed"], report["errors"]
    assert report["counts"]["trials"] == 6 * 2 * 8 == 96
    assert report["counts"]["actual_fits"] == 12
    assert report["counts"]["checkpoints"] == 6 * 2 * 3
    assert report["counts"]["curve_rows"] == 6 * 2 * 3 * 2
    assert report["counts"]["transfer_files"] == 6 * 2 * 2
    assert report["counts"]["transfer_probability_entries"] == 6 * 2 * (720 * 27 + 27)
    assert report["counts"]["primary_unseen_pairs"] == 15
    assert report["counts"]["anchored_pairs"] == 12
    assert any("not reported as discoveries" in check for check in report["checks"])


def test_auditor_has_no_learner_data_experiment_or_encoding_helper_imports():
    tree = ast.parse((ROOT / "src/drosolearn/encoding_audit.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert [node.module for node in imports] == ["datetime", "itertools", "pathlib"]
    assert all(node.level == 0 for node in imports)
    assert {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)} == {"csv", "hashlib", "json", "numpy"}


def test_audit_preserves_all_primary_files(audited_encoding_fixture):
    manifest = json.loads((audited_encoding_fixture / "artifact_hashes.json").read_text())
    before = {key: hashlib.sha256((audited_encoding_fixture / key).read_bytes()).hexdigest() for key in manifest}
    assert verify(audited_encoding_fixture)["passed"]
    after = {key: hashlib.sha256((audited_encoding_fixture / key).read_bytes()).hexdigest() for key in manifest}
    assert before == after == manifest


@pytest.mark.parametrize("section,field", [
    ("outcomes", "after_accuracy"), ("outcomes", "acquired"),
    ("outcomes", "qualified_ordered_transfer"), ("outcomes", "qualified_permuted_transfer"),
    ("outcomes", "permuted_anchored"), ("outcomes", "ordered_unseen"),
    ("contrasts", "representation_gap"), ("contrasts", "acquired"),
    ("family_verdicts", "criterion_met"), ("family_verdicts", "qualified_seed_count"),
])
def test_rehashed_summary_claims_are_independently_rejected(audited_encoding_fixture, tmp_path, section, field):
    output = _clone(audited_encoding_fixture, tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    old = summary[section][0][field]
    summary[section][0][field] = not old if type(old) is bool else float(old) + .013
    _save_json(output, "summary.json", summary)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("tamper", ["training_count", "permutation_as_fit"])
def test_panel_evaluations_cannot_be_reported_as_actual_training(audited_encoding_fixture, tmp_path, tamper):
    output = _clone(audited_encoding_fixture, tmp_path)
    if tamper == "training_count":
        summary = json.loads((output / "summary.json").read_text())
        summary["actual_training_trials"] *= 720
        _save_json(output, "summary.json", summary)
    else:
        rows = _rows(output, "seed_41/trials.csv")
        extra = rows[0].copy()
        extra["fit"] = "permuted_1"
        rows.append(extra)
        _save_rows(output, "seed_41/trials.csv", rows)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2", "action", "reward"])
def test_manual_all_trial_replay_rejects_rehashed_training_edits(audited_encoding_fixture, tmp_path, field):
    output = _clone(audited_encoding_fixture, tmp_path)
    rows = _rows(output, "seed_41/trials.csv")
    rows[0][field] = str(1 - int(rows[0][field])) if field in ("action", "reward") else str(float(rows[0][field]) + .001)
    _save_rows(output, "seed_41/trials.csv", rows)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("tamper", ["reorder", "anchor", "repeat"])
def test_exact_720_panel_coverage_rejects_rehashed_mapping_changes(audited_encoding_fixture, tmp_path, tamper):
    output = _clone(audited_encoding_fixture, tmp_path)
    panel = np.load(output / "permutations.npy", allow_pickle=False)
    if tamper == "reorder":
        panel[[1, 2]] = panel[[2, 1]]
    elif tamper == "anchor":
        panel[0, [0, 1]] = panel[0, [1, 0]]
    else:
        panel[1] = panel[2]
    np.save(output / "permutations.npy", panel)
    _rehash(output, "permutations.npy")
    report = verify(output)
    assert not report["passed"] and "anchored panel" in report["errors"][0]


@pytest.mark.parametrize("tamper", ["cache", "preserved_mean_matrix", "greedy", "case_probability", "removed"])
def test_every_frozen_entry_recomputed_even_if_column_mean_is_unchanged(audited_encoding_fixture, tmp_path, tamper):
    output = _clone(audited_encoding_fixture, tmp_path)
    fit = "quantity_removed" if tamper == "removed" else "quantity_present"
    relative = f"seed_41/fixed/{fit}/transfer_final.npz"
    arrays = _arrays(output, relative)
    if tamper == "cache":
        arrays["score_cache"][0, 0, 0, 0] += .01
    elif tamper == "preserved_mean_matrix":
        # A mean-only auditor would accept this semantic edit.
        arrays["map_p_target"][0, 0] += .001
        arrays["map_p_target"][1, 0] -= .001
    elif tamper == "greedy":
        arrays["map_greedy_accuracy"][0, 0] += .001
    elif tamper == "case_probability":
        arrays["ordered_p_right"][0, 0] += .01
    else:
        arrays["p_target"][0] += .01
    _save_arrays(output, relative, arrays)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("tamper", ["uniforms", "joint_crossing", "rng_stream", "head", "encoder", "topology"])
def test_source_randomness_and_frozen_learning_scope_are_checked(audited_encoding_fixture, tmp_path, tamper):
    output = _clone(audited_encoding_fixture, tmp_path)
    relative = ("seed_41/ann_topology.npz" if tamper == "topology" else
                "seed_41/plastic/quantity_present/checkpoint_0004.npz" if tamper in ("head", "encoder") else "seed_41/stimuli.npz")
    arrays = _arrays(output, relative)
    if tamper == "uniforms":
        arrays["action_uniforms"][0] = .123
    elif tamper == "joint_crossing":
        arrays["iid_nuisance"][0, 0, 0] += .001
    elif tamper == "rng_stream":
        arrays["iid_nuisance"][:, :, 0] = 1 - arrays["iid_nuisance"][:, :, 0]
    elif tamper == "head":
        arrays["readout_weights"][0] += .01
    elif tamper == "encoder":
        arrays["encoder"][0, 0] += .01
    else:
        arrays["counts"][0] += 1
    _save_arrays(output, relative, arrays)
    report = verify(output)
    assert not report["passed"] and report["errors"]


def test_privileged_inverse_lookup_cannot_be_mislabeled_as_learned_rule(audited_encoding_fixture, tmp_path):
    output = _clone(audited_encoding_fixture, tmp_path)
    code_audit = json.loads((output / "code_audit.json").read_text())
    code_audit["oracle_is_privileged"] = False
    _save_json(output, "code_audit.json", code_audit)
    report = verify(output)
    assert not report["passed"] and "privileged" in report["errors"][0]


@pytest.mark.parametrize("field", ["quantity_ratios", "quantity_differences", "code_gaps", "code_order_agreement"])
def test_true_quantity_geometry_is_distinct_from_supplied_code_geometry(audited_encoding_fixture, tmp_path, field):
    output = _clone(audited_encoding_fixture, tmp_path)
    arrays = _arrays(output, "code_geometry.npz")
    arrays[field].flat[0] += 1
    _save_arrays(output, "code_geometry.npz", arrays)
    report = verify(output)
    assert not report["passed"] and "Code geometry" in report["errors"][0]


def test_raw_byte_corruption_is_detected_before_semantic_replay(audited_encoding_fixture, tmp_path):
    output = _clone(audited_encoding_fixture, tmp_path)
    with (output / "seed_41/trials.csv").open("a", encoding="utf-8") as stream:
        stream.write("broken\n")
    report = verify(output)
    assert not report["passed"] and "hash mismatch" in report["errors"][0]
