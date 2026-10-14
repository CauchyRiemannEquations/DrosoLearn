"""Independent audit acceptance and semantic tampering after rehashing."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

from drosolearn.linearity import run
from drosolearn.linearity_audit import verify
from test_linearity import prepare_linearity_fixture


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audited_linearity_fixture(tmp_path_factory):
    # Seed71 is an engineering fixture, separate from planned seeds5001..5010.
    root, config, registration = prepare_linearity_fixture(tmp_path_factory.mktemp("linearity-audit"))
    output = root / "fixture-output"
    run(root, config, registration, output)
    report = verify(output)
    assert report["passed"], report["errors"]
    return output


def _clone(fixture, tmp_path):
    # Keep the recorded relative source-root path valid in a relocatable copy.
    root = tmp_path / "tampered-study"
    shutil.copytree(fixture.parent, root)
    return root / fixture.name


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
        writer.writeheader(); writer.writerows(rows)
    _rehash(output, relative)


def _arrays(output, relative):
    with np.load(output / relative, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _save_arrays(output, relative, arrays):
    np.savez_compressed(output / relative, **arrays)
    _rehash(output, relative)


def test_auditor_accepts_all_six_families_and_one_actual_fit(audited_linearity_fixture):
    report = verify(audited_linearity_fixture)
    assert report["passed"], report["errors"]
    assert report["counts"]["trials"] == 6 * 8 == 48
    assert report["counts"]["actual_fits"] == 6
    assert report["counts"]["frozen_evaluation_maps"] == 4
    assert report["counts"]["checkpoints"] == 6 * 3
    assert report["counts"]["curve_rows"] == 6 * 3 * 2
    assert report["counts"]["transfer_files"] == 6 * 2
    assert report["counts"]["reference_files"] == 6 * 2
    assert report["counts"]["reference_models"] == 2
    assert report["counts"]["calibration_candidate_score_entries"] == 6 * 2 * 32 * 2
    assert report["counts"]["transfer_probability_entries"] == 6 * 2 * 4 * 27
    assert report["counts"]["case_probability_entries"] == 6 * 2 * 4 * 27 * 8
    assert report["counts"]["candidate_score_entries"] == 6 * 2 * 27 * 8 * 2 * 14
    assert report["counts"]["reference_case_probability_entries"] == 6 * 2 * 4 * 27 * 8 * 2
    assert report["counts"]["raw_margin_entries"] == 6 * 2 * 4 * 27 * 8 * 3
    assert report["counts"]["focal_pairs"] == 1
    assert report["counts"]["secondary_unseen_pairs"] == 15


def test_auditor_imports_only_stdlib_numpy_and_earlier_independent_helpers():
    tree = ast.parse((ROOT / "src/drosolearn/linearity_audit.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert {node.module for node in imports} == {"datetime", "itertools", "pathlib", "encoding_audit"}
    helper_import = next(node for node in imports if node.module == "encoding_audit")
    assert helper_import.level == 1
    assert {alias.name for alias in helper_import.names} == {
        "_require", "_near", "_integer", "_number", "_json", "_hash", "_csv", "_npz",
        "_regenerate_dataset", "_dataset", "_inputs", "_mbon", "_forward", "_metrics", "_topologies", "_cache"}
    assert {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)} == {"csv", "json", "numpy"}
    old_tree = ast.parse((ROOT / "src/drosolearn/encoding_audit.py").read_text(encoding="utf-8"))
    assert all(node.level == 0 for node in ast.walk(old_tree) if isinstance(node, ast.ImportFrom))


def test_audit_preserves_primary_and_registered_source_files(audited_linearity_fixture):
    manifest = json.loads((audited_linearity_fixture / "artifact_hashes.json").read_text())
    registration = json.loads((audited_linearity_fixture / "registration.json").read_text())
    root = audited_linearity_fixture.parent
    before = {name: hashlib.sha256((audited_linearity_fixture / name).read_bytes()).hexdigest() for name in manifest}
    source_before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in registration["source_hashes"]}
    assert verify(audited_linearity_fixture)["passed"]
    assert before == manifest == {name: hashlib.sha256((audited_linearity_fixture / name).read_bytes()).hexdigest() for name in manifest}
    assert source_before == registration["source_hashes"] == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_before}


def test_independent_auditor_uses_qr_without_producer_lstsq(audited_linearity_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Producer least-squares solver called by independent audit")
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    report = verify(audited_linearity_fixture)
    assert report["passed"], report["errors"]


@pytest.mark.parametrize("section,field", [
    ("outcomes", "before_accuracy"), ("outcomes", "after_accuracy"), ("outcomes", "gain"), ("outcomes", "acquired"),
    ("outcomes", "fidelity_passed"), ("outcomes", "qualified_fidelity"), ("outcomes", "head_updates"),
    ("outcomes", "focal_model_before"), ("outcomes", "focal_model_after"), ("outcomes", "focal_model_greedy_after"),
    ("family_verdicts", "criterion_met"), ("family_verdicts", "mean_affine_probability_mae"),
    ("family_verdicts", "sd_affine_probability_mae"), ("family_verdicts", "passing_seed_count"),
    ("family_verdicts", "acquisition_count"), ("family_verdicts", "qualified_count"),
    ("family_verdicts", "mean_scalar_probability_mae"), ("family_verdicts", "mean_affine_margin_rmse"),
    ("family_verdicts", "mean_scalar_margin_rmse"), ("family_verdicts", "mean_affine_greedy_disagreement"),
    ("family_verdicts", "mean_scalar_greedy_disagreement"),
])
def test_rehashed_summary_claims_independently_rejected(audited_linearity_fixture, tmp_path, section, field):
    output = _clone(audited_linearity_fixture, tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    old = summary[section][0][field]
    if isinstance(old, list):
        old[0] = not old[0] if type(old[0]) is bool else float(old[0]) + .013
    else:
        summary[section][0][field] = not old if type(old) is bool else float(old) + .013
    _save_json(output, "summary.json", summary)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("stage,kind", [(stage, kind) for stage in ("baseline", "final") for kind in ("affine", "scalar")])
@pytest.mark.parametrize("field", ["unseen_probability_mae", "unseen_margin_mae", "unseen_margin_rmse",
                                  "unseen_score_rmse", "unseen_greedy_disagreement", "anchored_probability_mae",
                                  "all_probability_mae", "focal_probability_mae", "focal_reference_p_correct",
                                  "focal_reference_greedy_accuracy", "unseen_reference_p_correct"])
def test_every_reference_residual_summary_recomputed(audited_linearity_fixture, tmp_path, stage, kind, field):
    output = _clone(audited_linearity_fixture, tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    entry = summary["outcomes"][0][stage][kind]
    if isinstance(entry[field], list):
        entry[field][0] += .013
    else:
        entry[field] += .013
    _save_json(output, "summary.json", summary)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("stage", ["baseline", "final"])
@pytest.mark.parametrize("field", ["affine_coefficients", "affine_rank", "affine_singular_values", "affine_calibration_score_rmse",
                                  "scalar_coefficients", "scalar_anchor_means", "scalar_calibration_score_rmse",
                                  "calibration_inputs", "calibration_scores"])
def test_privileged_reference_fit_recomputed_from_calibration_only(audited_linearity_fixture, tmp_path, stage, field):
    output = _clone(audited_linearity_fixture, tmp_path)
    relative = f"seed_71/fixed/reference_{stage}.npz"
    arrays = _arrays(output, relative)
    arrays[field].flat[0] += 1 if field == "affine_rank" else .013
    _save_arrays(output, relative, arrays)
    report = verify(output)
    assert not report["passed"] and ("calibration" in report["errors"][0] or "reference fit" in report["errors"][0])


@pytest.mark.parametrize("field", ["teacher_map_scores", "teacher_map_margin",
                                  *(f"{kind}_{field}" for kind in ("affine", "scalar") for field in
                                    ("map_scores", "map_margin", "map_p_right", "map_p_target", "map_greedy_accuracy", "probability_mae"))])
def test_every_teacher_and_distilled_reference_case_recomputed(audited_linearity_fixture, tmp_path, field):
    output = _clone(audited_linearity_fixture, tmp_path)
    relative = "seed_71/fixed/transfer_final.npz"
    arrays = _arrays(output, relative); arrays[field].flat[0] += .013
    _save_arrays(output, relative, arrays)
    report = verify(output)
    assert not report["passed"] and report["errors"]


def test_calibration_canonical_leakage_checked_after_rehash(audited_linearity_fixture, tmp_path):
    output = _clone(audited_linearity_fixture, tmp_path)
    relative = "seed_71/stimuli.npz"
    arrays = _arrays(output, relative)
    cal = np.flatnonzero(arrays["calibration_family_ids"] == "calibration:0")
    iid = np.flatnonzero(arrays["iid_family_ids"] == "iid:0")
    for field in ("counts", "nuisance"):
        arrays[f"calibration_{field}"][cal] = arrays[f"iid_{field}"][iid]
    _save_arrays(output, relative, arrays)
    report = verify(output)
    assert not report["passed"] and "leakage" in report["errors"][0]


def test_calibration_rng_checked_even_after_joint_crossing_preserved(audited_linearity_fixture, tmp_path):
    output = _clone(audited_linearity_fixture, tmp_path)
    relative = "seed_71/stimuli.npz"
    arrays = _arrays(output, relative)
    arrays["calibration_nuisance"][..., 0] = 1 - arrays["calibration_nuisance"][..., 0]
    _save_arrays(output, relative, arrays)
    report = verify(output)
    assert not report["passed"] and "data RNG" in report["errors"][0]


def test_reference_models_not_claimed_as_additional_reward_learners(audited_linearity_fixture, tmp_path):
    output = _clone(audited_linearity_fixture, tmp_path)
    metadata = json.loads((output / "metadata.json").read_text())
    metadata["reference_scope"] = "two independent reward learners"
    _save_json(output, "metadata.json", metadata)
    report = verify(output)
    assert not report["passed"] and "reference calibration scope" in report["errors"][0]


@pytest.mark.parametrize("tamper", ["count_maps_as_fits", "extra_trial"])
def test_frozen_maps_cannot_be_claimed_as_extra_training(audited_linearity_fixture, tmp_path, tamper):
    output = _clone(audited_linearity_fixture, tmp_path)
    if tamper == "count_maps_as_fits":
        summary = json.loads((output / "summary.json").read_text())
        summary["actual_training_trials"] *= 4
        _save_json(output, "summary.json", summary)
    else:
        rows = _rows(output, "seed_71/trials.csv")
        extra = rows[0].copy(); extra["condition"] = "tight_extra_fit"; rows.append(extra)
        _save_rows(output, "seed_71/trials.csv", rows)
    assert not verify(output)["passed"]


@pytest.mark.parametrize("field", ["p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2", "action", "reward", "family_id"])
def test_every_training_update_replayed_after_rehash(audited_linearity_fixture, tmp_path, field):
    output = _clone(audited_linearity_fixture, tmp_path)
    rows = _rows(output, "seed_71/trials.csv")
    old = rows[0][field]
    rows[0][field] = "unregistered_stimulus" if field == "family_id" else str(1 - int(old)) if field in ("action", "reward") else str(float(old) + .001)
    _save_rows(output, "seed_71/trials.csv", rows)
    assert not verify(output)["passed"]


@pytest.mark.parametrize("tamper", ["reorder", "anchor", "range", "gap", "midpoint"])
def test_exact_linearity_map_construction_is_checked_after_rehash(audited_linearity_fixture, tmp_path, tamper):
    output = _clone(audited_linearity_fixture, tmp_path)
    panel = np.load(output / "spacing_codes.npy", allow_pickle=False)
    if tamper == "reorder":
        panel[[0, 1]] = panel[[1, 0]]
    elif tamper == "anchor":
        panel[0, 1] += .1
    elif tamper == "range":
        panel[0, 0] += .1
    elif tamper == "gap":
        panel[0, 4] += .1
    else:
        panel[0, [4, 6]] += .1
    np.save(output / "spacing_codes.npy", panel); _rehash(output, "spacing_codes.npy")
    report = verify(output)
    assert not report["passed"] and "spacing panel" in report["errors"][0]


@pytest.mark.parametrize("field", ["quantity_ratios", "quantity_differences", "code_gaps", "normalized_code_gaps", "code_midpoints", "normalized_code_midpoints", "code_order_agreement"])
def test_original_quantity_and_mapped_geometry_are_separately_recomputed(audited_linearity_fixture, tmp_path, field):
    output = _clone(audited_linearity_fixture, tmp_path)
    arrays = _arrays(output, "code_geometry.npz"); arrays[field].flat[0] += 1
    _save_arrays(output, "code_geometry.npz", arrays)
    report = verify(output)
    assert not report["passed"] and "code geometry" in report["errors"][0]


@pytest.mark.parametrize("tamper", ["cache", "preserved_mean_matrix", "greedy", "case_probability", "baseline_case", "code_values"])
def test_every_baseline_and_final_transfer_entry_checked(audited_linearity_fixture, tmp_path, tamper):
    output = _clone(audited_linearity_fixture, tmp_path)
    stage = "baseline" if tamper == "baseline_case" else "final"
    relative = f"seed_71/fixed/transfer_{stage}.npz"
    arrays = _arrays(output, relative)
    if tamper == "cache":
        arrays["score_cache"][0, 0, 0, 0] += .01
    elif tamper == "preserved_mean_matrix":
        arrays["map_p_target"][0, 0] += .001; arrays["map_p_target"][0, 1] -= .001
    elif tamper == "greedy":
        arrays["map_greedy_accuracy"][0, 0] += .001
    elif tamper == "code_values":
        arrays["code_values"][0] += .01
    else:
        arrays["map_p_right"][0, 0, 0] += .01
    _save_arrays(output, relative, arrays)
    assert not verify(output)["passed"]


@pytest.mark.parametrize("tamper", ["uniforms", "joint_crossing", "rng_stream", "head", "encoder", "topology"])
def test_randomness_source_and_checkpoint_scope_checked(audited_linearity_fixture, tmp_path, tamper):
    output = _clone(audited_linearity_fixture, tmp_path)
    relative = "seed_71/ann_topology.npz" if tamper == "topology" else "seed_71/plastic/checkpoint_0004.npz" if tamper in ("head", "encoder") else "seed_71/stimuli.npz"
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
    assert not verify(output)["passed"]


@pytest.mark.parametrize("tamper", ["source_bytes", "extra_source", "missing_source", "config_bytes", "config_content"])
def test_source_and_configuration_freeze_are_independently_checked(audited_linearity_fixture, tmp_path, tamper):
    output = _clone(audited_linearity_fixture, tmp_path); root = output.parent
    if tamper == "source_bytes":
        with (root / "src/drosolearn/model.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# post-registration edit\n")
    elif tamper == "extra_source":
        (root / "src/drosolearn/unregistered.py").write_text("x = 1\n", encoding="utf-8")
    elif tamper == "missing_source":
        (root / "src/drosolearn/model.py").unlink()
    elif tamper == "config_bytes":
        with (root / "config.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        config = json.loads((root / "config.json").read_text()); config["head_lr"] += .001
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    report = verify(output)
    assert not report["passed"] and ("Registered source" in report["errors"][0] or "Registered config" in report["errors"][0])


def test_registration_and_chronology_not_replaceable_by_artifact_rehash(audited_linearity_fixture, tmp_path):
    output = _clone(audited_linearity_fixture, tmp_path)
    registration = json.loads((output / "registration.json").read_text())
    registration["registered_at_utc"] = "2100-01-01T00:00:00+00:00"
    _save_json(output, "registration.json", registration)
    metadata = json.loads((output / "metadata.json").read_text())
    metadata["registration_sha256"] = hashlib.sha256((output / "registration.json").read_bytes()).hexdigest()
    _save_json(output, "metadata.json", metadata)
    report = verify(output)
    assert not report["passed"] and "chronology" in report["errors"][0]


def test_raw_primary_corruption_detected_before_replay(audited_linearity_fixture, tmp_path):
    output = _clone(audited_linearity_fixture, tmp_path)
    with (output / "seed_71/trials.csv").open("a", encoding="utf-8") as stream:
        stream.write("broken\n")
    report = verify(output)
    assert not report["passed"] and "hash mismatch" in report["errors"][0]
