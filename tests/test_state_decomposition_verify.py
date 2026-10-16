"""Static engineering tests reuse only already completed tiny source states."""
from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_static_decomposition_test", ROOT / "scripts/verify_state_decomposition.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def data():
    return {"counts": np.asarray([[2, 4], [4, 2], [2, 4], [4, 2]]),
            "nuisance": np.full((4, 2, 6), .5), "family_ids": np.asarray(["a", "b", "c", "d"])}


def snapshot(linear=True, *, head=None, weights=None):
    if linear:
        h = np.asarray([1., 0., 0., 0., 0., 0., 0.] if head is None else head, dtype=float)
        w, encoder, bias, pre, post = np.empty(0), np.empty((0, 7)), np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int)
        original, normalizer = np.empty(0), np.empty(0)
    else:
        h = np.asarray([1., -.4] if head is None else head, dtype=float)
        w = np.asarray([.8, 1.5] if weights is None else weights, dtype=float)
        encoder = np.zeros((2, 7))
        encoder[:, 0] = [1., 2.]
        bias, pre, post = np.asarray([.2, .4]), np.asarray([0, 1]), np.asarray([0, 1])
        original, normalizer = np.asarray([1., 2.]), np.asarray([2., 3.])
    source = {"readout_weights": h, "internal_weights": w, "encoder": encoder, "bias": bias, "pre": pre, "post": post,
        "feature_scale": np.asarray(1.), "initial_feature_gap_rms": np.asarray(1.), "initial_internal_weights": original,
        "internal_upper_bounds": 2 * original, "incoming_normalizer": normalizer,
        **{key: np.asarray(0) for key in ("head_updates", "internal_updates", "phase_head_updates", "phase_internal_updates")}}
    x = VERIFY.inputs(data(), np.arange(1, 9))
    kc, mb, decision, p = VERIFY._OLD.state(x[:2], encoder, bias, pre, post, w, h, 1., linear)
    all_p = VERIFY._OLD.state(x, encoder, bias, pre, post, w, h, 1., linear)[-1]
    source.update(kc_activity=kc, mbon_activity=mb, decision_activity=decision, choice_probability=p,
                  iid_p_right=all_p.copy(), ood_p_right=all_p.copy())
    return source


@pytest.mark.parametrize("linear", [True, False])
def test_four_crossed_states_obey_casewise_bilinear_identity(linear):
    prefix = snapshot(linear)
    current = snapshot(linear, head=[-.5, .8] if not linear else [2., 0., 0., 0., 0., 0., 0.],
                       weights=[1., 1.1] if not linear else None)
    values = VERIFY.static_cases(prefix, current, data(), linear=linear)
    for key in ("bilinear_residual", "raw_interaction_residual"):
        assert np.max(np.abs(values[key])) < 1e-14
    np.testing.assert_allclose(values["total_raw_gap_change"], values["head_term"] + values["internal_term"] + values["cross_term"], atol=1e-14)
    assert set(values) == set(VERIFY.VALUE_FIELDS)


def test_orientation_alignment_avoids_cancellation_and_all_crosses_use_smaller_target():
    prefix, current = snapshot(), snapshot(head=[2., 0., 0., 0., 0., 0., 0.])
    values = VERIFY.static_cases(prefix, current, data(), linear=True)
    assert np.mean(values["raw_gap_00"]) == 0
    np.testing.assert_array_equal(values["target_action"], [0, 1, 0, 1])
    np.testing.assert_array_equal(values["target_sign"], [-1, 1, -1, 1])
    np.testing.assert_array_equal(values["aligned_raw_gap_00"], [-.25] * 4)
    for cross in VERIFY.CROSSES:
        assert (values["p_rule_" + cross] < .5).all()
        np.testing.assert_allclose(values["p_rule_" + cross], np.where(values["target_action"], values["p_right_" + cross], 1 - values["p_right_" + cross]))


@pytest.mark.parametrize("coefficient,clipped", [(160., 0), (-160., 0), (160.01, 1), (-160.01, 1)])
def test_clipping_is_strictly_beyond_positive_and_negative_forty(coefficient, clipped):
    source = snapshot(head=[coefficient, 0., 0., 0., 0., 0., 0.])
    values = VERIFY.static_cases(source, source, data(), linear=True)
    for cross in VERIFY.CROSSES:
        np.testing.assert_array_equal(values["clipped_" + cross], [clipped] * 4)
        assert np.isfinite(values["p_right_" + cross]).all()


@pytest.mark.parametrize("field", VERIFY.FIXED_FIELDS)
def test_fixed_encoder_topology_original_bounds_and_scale_cannot_change(field):
    a, b = snapshot(False), snapshot(False)
    b[field] = b[field].copy()
    b[field].flat[0] += 1
    with pytest.raises(ValueError, match="Fixed source initialization"):
        VERIFY.static_cases(a, b, data(), linear=False)


@pytest.mark.parametrize("branch", VERIFY.BRANCHES)
def test_masked_component_terms_are_exact_zero(branch):
    prefix = snapshot(False)
    head = prefix["readout_weights"] if branch in {"internal_only", "frozen"} else [-.5, .8]
    weights = prefix["internal_weights"] if branch in {"head_only", "frozen"} else [1., 1.1]
    values = VERIFY.static_cases(prefix, snapshot(False, head=head, weights=weights), data(), linear=False)
    zero = {"head_only": ("internal_term", "cross_term"), "internal_only": ("head_term", "cross_term"),
            "frozen": ("head_term", "internal_term", "cross_term", "total_raw_gap_change"), "full": ()}[branch]
    for key in zero:
        np.testing.assert_array_equal(values[key], np.zeros(4))


def test_policy_interaction_is_distinct_from_raw_cross_term():
    prefix, current = snapshot(False), snapshot(False, head=[-.5, .8], weights=[1., 1.1])
    values = VERIFY.static_cases(prefix, current, data(), linear=False)
    assert not np.allclose(values["cross_term"], values["policy_interaction_rule"])


def test_pure_arithmetic_does_not_change_saved_arrays_or_draw_random_numbers(monkeypatch):
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("No RNG in static diagnostics"))
    prefix, current = snapshot(False), snapshot(False, head=[-.5, .8], weights=[1., 1.1])
    originals = copy.deepcopy((prefix, current))
    VERIFY.static_cases(prefix, current, data(), linear=False)
    for before, after in zip(originals, (prefix, current), strict=True):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


def test_runtime_only_imports_explicit_immutable_independent_helpers():
    text = (ROOT / "scripts/verify_state_decomposition.py").read_text(encoding="utf8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("drosolearn")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"default_rng", "random", "forward", "initialize", "summarize", "run", "register"}
            if node.func.attr == "update":
                assert isinstance(node.func.value, ast.Name) and node.func.value.id == "report"
    assert "verify_learning_ablation.py" in text


def test_registered_scientific_denominators_are_checked_without_reading_scientific_states():
    cfg = VERIFY.read_json(ROOT / "configs/state_decomposition_v1.json")
    VERIFY.config_check(cfg)
    assert len(VERIFY.selected_paths(cfg)) == 850
    assert cfg["cases"] == 204800 and cfg["crossed_policy_values"] == 819200


@pytest.mark.parametrize("key,value", [("cases", 204799), ("crossed_policy_values", 819199), ("selected_input_files", 849),
    ("learning_updates", 1), ("learner_constructions", 1), ("dataset_generations", 1), ("rng_draws", 1),
    ("target", "larger_quantity"), ("interpretation", "causal_learning_effect"), ("all_registered_inputs_retained", False),
    ("raw_residual_tolerance", 2e-10), ("policy_match_tolerance", 2e-12), ("policy_clip", [-40., 40.00000001])])
def test_registration_rejects_incomplete_denominators_resources_and_causal_scope(key, value):
    cfg = VERIFY.read_json(ROOT / "configs/state_decomposition_v1.json")
    cfg[key] = value
    with pytest.raises(ValueError):
        VERIFY.config_check(cfg)


@pytest.mark.parametrize("name", ["../escape.npz", "/absolute.npz", "folder\\bad.npz"])
def test_source_manifest_rejects_escaping_paths(tmp_path, name):
    with pytest.raises(ValueError):
        VERIFY.file_manifest_check(tmp_path, [{"path": name, "bytes": 1, "sha256": "0" * 64}], [name])


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")


@pytest.fixture(scope="module")
def audited_fixture(tmp_path_factory):
    from test_state_decomposition_run import make_fixture
    from drosolearn import state_decomposition as producer
    monkeypatch = pytest.MonkeyPatch()
    try:
        fixture = make_fixture(tmp_path_factory.mktemp("independent-static"), monkeypatch)
        monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("No new data or training RNG"))
        producer.register(fixture["root"], fixture["config"], fixture["registration"])
        producer.run(fixture["root"], fixture["config"], fixture["registration"], fixture["out"])
        report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"])
        assert report["status"] == "passed", report["errors"]
        dump(fixture["out"] / "verification.json", report)
        return fixture
    finally:
        monkeypatch.undo()


def test_all_static_cases_independently_pass_with_no_new_training(audited_fixture):
    fixture = audited_fixture
    report = VERIFY.read_json(fixture["out"] / "verification.json")
    counts = report["counts"]
    nseed = len(fixture["cfg"]["seeds"])
    assert counts["cases"] == nseed * 768
    assert counts["crossed_policy_values"] == nseed * 3072
    assert counts["selected_input_files"] == nseed * 53
    assert counts["source_hashes"] == len(fixture["cfg"]["frozen_files"])
    assert all(counts[k] == 0 for k in VERIFY.RESOURCE_FIELDS)
    assert counts["linear_equivalent_checkpoint_pairs"] == nseed * 6
    assert report["input_manifest_sha256"] == VERIFY.canonical_input_digest(VERIFY.read_json(fixture["registration"])["input_files"])
    assert report["artifact_hashes_sha256"] == VERIFY.digest(fixture["out"] / "artifact_hashes.json")


@pytest.fixture
def mutated(audited_fixture, tmp_path):
    destination = tmp_path / "source"
    shutil.copytree(audited_fixture["root"], destination)
    return {**audited_fixture, "root": destination, "config": destination / "configs/state_decomposition_v1.json",
            "registration": destination / "registration.json", "out": destination / audited_fixture["out"].relative_to(audited_fixture["root"])}


def rehash(fixture):
    out = fixture["out"]
    meta_path = out / "metadata.json"
    meta = VERIFY.read_json(meta_path)
    meta["maximum_case_csv_bytes"] = max(path.stat().st_size for path in out.glob("seed_*/cases.csv"))
    dump(meta_path, meta)
    hashes = VERIFY.read_json(out / "artifact_hashes.json")
    for name in hashes:
        hashes[name] = VERIFY.digest(out / name)
    dump(out / "artifact_hashes.json", hashes)


def rebind_registered_sources(fixture):
    """Rehash all links so a source-state mutation reaches semantic validation."""
    root, output = fixture["root"], fixture["out"]
    cfg, reg = VERIFY.read_json(fixture["config"]), VERIFY.read_json(fixture["registration"])
    base = root / cfg["parent_root"]
    parent_hashes = VERIFY.read_json(base / "artifact_hashes.json")
    for name in parent_hashes:
        parent_hashes[name] = VERIFY.digest(base / name)
    dump(base / "artifact_hashes.json", parent_hashes)
    checked = VERIFY.read_json(base / "verification.json")
    checked["artifact_hashes_sha256"] = VERIFY.digest(base / "artifact_hashes.json")
    dump(base / "verification.json", checked)
    correction_path = root / "results/learning_ablation_documentation_correction_v1.json"
    correction = VERIFY.read_json(correction_path)
    correction["verification_sha256"] = VERIFY.digest(base / "verification.json")
    dump(correction_path, correction)
    final_path = root / "results/learning_ablation_final_validation_v1.json"
    final = VERIFY.read_json(final_path)
    final["documentation_correction"] = correction
    dump(final_path, final)
    parents = {name: VERIFY.digest(root / name) for name in cfg["parent_files"]}
    cfg["parent_gate_identity"]["file_sha256"] = parents
    dump(fixture["config"], cfg)
    reg.update(config=cfg, config_sha256=VERIFY.digest(fixture["config"]), parent_files=parents)
    reg["frozen_files"] = {name: VERIFY.digest(root / name) for name in cfg["frozen_files"]}
    reg["input_files"] = [{"path": name, "bytes": (root / name).stat().st_size, "sha256": VERIFY.digest(root / name)}
                          for name in VERIFY.selected_paths(cfg)]
    reg["gate"] = {"name": "verified_learning_ablation_before_static_decomposition", "passed": True,
                   **cfg["parent_gate_identity"], "parent_primary_artifacts_verified": len(parent_hashes)}
    dump(fixture["registration"], reg)
    dump(output / "registration.json", reg)
    metadata = VERIFY.read_json(output / "metadata.json")
    metadata.update({key: reg[key] for key in ("config", "config_sha256", "frozen_files", "parent_files", "input_files", "gate")})
    metadata["registration_sha256"] = VERIFY.digest(fixture["registration"])
    dump(output / "metadata.json", metadata)
    rehash(fixture)


@pytest.mark.parametrize("field", ["encoder", "feature_scale", "internal_upper_bounds", "internal_weights", "readout_weights"])
def test_fully_rehashed_source_state_changes_cannot_bypass_arithmetic_and_mask_checks(mutated, field):
    cfg, root = mutated["cfg"], mutated["root"]
    path = root / f"{cfg['parent_root']}/seed_{cfg['seeds'][0]}/real/head_only/phase2_0004.npz"
    saved = VERIFY.npz(path, VERIFY.SNAPSHOT_FIELDS)
    if field in {"feature_scale", "internal_upper_bounds"}:
        saved[field] = saved[field] + .01
    else:
        saved[field] = saved[field].copy()
        saved[field].flat[0] += .01
    np.savez_compressed(path, **saved)
    rebind_registered_sources(mutated)
    report = VERIFY.verify(root, mutated["registration"], mutated["out"])
    assert report["status"] == "failed" and not any("bytes" in e or "hash" in e.lower() for e in report["errors"]), report["errors"]


@pytest.mark.parametrize("mode", ["manifest_drop", "manifest_bytes", "artifact_omit", "missing_parent_pass", "registration_future", "phase1_goal"])
def test_provenance_coverage_and_prior_gate_cannot_be_relabelled_after_rehash(mutated, mode):
    root, output = mutated["root"], mutated["out"]
    reg, metadata = VERIFY.read_json(mutated["registration"]), VERIFY.read_json(output / "metadata.json")
    if mode.startswith("manifest"):
        if mode == "manifest_drop":
            reg["input_files"].pop()
        else:
            reg["input_files"][0]["bytes"] += 1
        metadata["input_files"] = reg["input_files"]
    elif mode == "artifact_omit":
        hashes = VERIFY.read_json(output / "artifact_hashes.json")
        hashes.pop("controls.json")
        dump(output / "artifact_hashes.json", hashes)
    elif mode == "missing_parent_pass":
        checked = root / reg["config"]["parent_root"] / "verification.json"
        content = VERIFY.read_json(checked)
        content["status"] = "failed"
        dump(checked, content)
        rebind_registered_sources(mutated)
        reg, metadata = VERIFY.read_json(mutated["registration"]), VERIFY.read_json(output / "metadata.json")
    elif mode == "registration_future":
        reg["registered_at_utc"] = "2099-01-01T00:00:00+00:00"
    else:
        reg["config"]["target"] = "larger_quantity_phase1_baseline"
        dump(mutated["config"], reg["config"])
        reg["config_sha256"] = VERIFY.digest(mutated["config"])
        metadata["config"] = reg["config"]
        metadata["config_sha256"] = reg["config_sha256"]
    dump(mutated["registration"], reg)
    dump(output / "registration.json", reg)
    metadata["registration_sha256"] = VERIFY.digest(mutated["registration"])
    dump(output / "metadata.json", metadata)
    rehash(mutated)
    report = VERIFY.verify(root, mutated["registration"], output)
    assert report["status"] == "failed", report


@pytest.mark.parametrize("column", ["raw_gap_10", "aligned_head_term", "p_rule_00", "policy_interaction_rule", "target_sign", "clipped_00", "family_id"])
def test_rehashed_case_semantic_mutations_are_rejected(mutated, column):
    fixture = mutated
    path = fixture["out"] / f"seed_{fixture['cfg']['seeds'][0]}/cases.csv"
    with path.open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    row = rows[0]
    if column == "family_id":
        row[column] = "invented"
    elif column == "target_sign":
        row[column] = str(-int(row[column]))
    elif column == "clipped_00":
        row[column] = str(1 - int(row[column]))
    else:
        row[column] = str(float(row[column]) + .02)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rehash(fixture)
    report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"])
    assert report["status"] == "failed"
    assert any("case" in e or "policy" in e or "cross" in e for e in report["errors"]), report["errors"]


@pytest.mark.parametrize("field", ["cases", "behavior_hypotheses_tested", "arithmetic_identity_passed", "mean", "sample_sd", "aligned"])
def test_rehashed_summary_mutations_are_rejected(mutated, field):
    summary_path = mutated["out"] / "summary.json"
    summary = VERIFY.read_json(summary_path)
    if field == "mean":
        summary["cells"][0]["statistics"]["p_rule_00"]["mean"] += .02
    elif field == "sample_sd":
        summary["cells"][0]["statistics"]["p_rule_00"]["sample_sd_across_seed_means"] = .4
    elif field == "aligned":
        summary["seed_summaries"][0]["means"]["aligned_raw_gap_00"] += .02
    elif field == "arithmetic_identity_passed":
        summary[field] = False
    else:
        summary[field] += 1
    dump(summary_path, summary)
    rehash(mutated)
    report = VERIFY.verify(mutated["root"], mutated["registration"], mutated["out"])
    assert report["status"] == "failed" and any("summary" in e for e in report["errors"]), report["errors"]


@pytest.mark.parametrize("key", VERIFY.RESOURCE_FIELDS)
def test_rehashed_metadata_cannot_claim_nonzero_new_resources(mutated, key):
    path = mutated["out"] / "metadata.json"
    meta = VERIFY.read_json(path)
    meta[key] = 1
    dump(path, meta)
    rehash(mutated)
    report = VERIFY.verify(mutated["root"], mutated["registration"], mutated["out"])
    assert report["status"] == "failed" and any(key in e for e in report["errors"]), report["errors"]


def test_rehashed_missing_case_is_rejected(mutated):
    path = mutated["out"] / f"seed_{mutated['cfg']['seeds'][0]}/cases.csv"
    lines = path.read_text(encoding="utf8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf8", newline="\n")
    rehash(mutated)
    report = VERIFY.verify(mutated["root"], mutated["registration"], mutated["out"])
    assert report["status"] == "failed" and any("case" in e for e in report["errors"]), report["errors"]


def test_copied_source_files_and_parent_bytes_are_immutable(audited_fixture):
    reg = VERIFY.read_json(audited_fixture["registration"])
    for key in ("frozen_files", "parent_files"):
        for name, sha in reg[key].items():
            assert VERIFY.digest(audited_fixture["root"] / name) == sha
    for row in reg["input_files"]:
        assert VERIFY.digest(audited_fixture["root"] / row["path"]) == row["sha256"]


def test_cli_writes_one_new_report_and_refuses_overwrite(audited_fixture, tmp_path):
    fixture, target = audited_fixture, tmp_path / "report.json"
    args = ["--root", str(fixture["root"]), "--registration", str(fixture["registration"]),
            "--result-dir", str(fixture["out"]), "--out", str(target)]
    assert VERIFY.main(args) == 0
    assert VERIFY.read_json(target)["status"] == "passed"
    before = target.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        VERIFY.main(args)
    assert target.read_bytes() == before
