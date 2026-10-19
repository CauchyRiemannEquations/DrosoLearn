"""Invented evidence only: no scientific trial or random draw is executed."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_phase5_report.py"
SPEC = importlib.util.spec_from_file_location("independent_phase5_fixture", SCRIPT)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


@pytest.fixture
def evidence(tmp_path):
    (tmp_path / "audit.json").write_text('{"passed":true,"errors":[]}', encoding="utf8")
    (tmp_path / "source.json").write_text(json.dumps({"items": [{"gain": .125}], "a/b": {"~": False},
                                                     "int": 1, "zero": 0, "list": [0, True, None], "": "empty key"}), encoding="utf8")
    (tmp_path / "report.md").write_text("Invented retrospective report. No experiment was run.", encoding="utf8")
    claims = [{"claim_id": f"claim_{q}", "question": q, "level": 0, "status": "bounded",
               "text_ko": "합성 근거의 기록", "limits_ko": "신규 실험이 아님",
               "observations": [{"file": "source.json", "pointer": "/items/0/gain", "value": .125}]}
              for q in range(1, 6)]
    cfg = {"study_id": "phase5_synthesis_v1", "scope": "retrospective_bounded_evidence_synthesis",
           "studies": [{"study_id": sid, "tag": "invented_fixture", "verification": "audit.json",
                        "report": "report.md", "primary": "source.json", "role": "synthetic_only"} for sid in v.STUDY_IDS],
           "claims": claims, "questions": [{"question": q, "text_ko": "합성 질문", "claim_ids": [f"claim_{q}"]}
                                          for q in range(1, 6)],
           "input_files": ["audit.json", "source.json", "report.md"], "parent_files": [],
           "synthesis_interpretation": copy.deepcopy(v.INTERPRETATION)}
    return tmp_path, cfg


def evaluate_producer(root, cfg):
    # Producer is deliberately imported only by this integration test module.
    # The independent verification implementation imports no producer code.
    from drosolearn.phase5_report import evaluate
    return evaluate(cfg, root)


def test_producer_and_independent_ledger_agree_without_rng(evidence, monkeypatch):
    root, cfg = evidence
    def forbidden(*args, **kwargs):
        raise AssertionError("No RNG may be used for retrospective evidence")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    arrays, rows, summary = evaluate_producer(root, cfg)
    checks = v.verify_science(cfg, root, arrays, rows, summary)
    assert arrays["study_checks"].shape == (28, 2)
    assert arrays["claim_codes"].shape == (5, 4)
    assert checks["verified_historical_studies"] == 28
    assert checks["literal_json_observations"] == 5
    assert all(checks[key] == 0 for key in ("new_learning_updates", "new_rng_draws", "new_human_experiments", "new_effect_tests"))


def test_csv_roundtrip_and_evidence_hashes(evidence):
    root, cfg = evidence
    arrays, rows, summary = evaluate_producer(root, cfg)
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    handle.seek(0)
    csv_rows = list(csv.DictReader(handle))
    assert v.verify_science(cfg, root, arrays, csv_rows, summary)["exact_int64_ledgers"] == 2
    record = summary["studies"][0]
    assert record["verification_sha256"] == v.digest(root / "audit.json")
    assert record["report_sha256"] == v.digest(root / "report.md")
    assert record["primary_sha256"] == v.digest(root / "source.json")


@pytest.mark.parametrize("status", v.STATUSES)
def test_all_outcome_statuses_preserved(evidence, status):
    root, cfg = evidence
    cfg["claims"][0]["status"] = status
    arrays, rows, summary = evaluate_producer(root, cfg)
    v.verify_science(cfg, root, arrays, rows, summary)
    assert rows[0]["status"] == status
    assert arrays["claim_codes"][0, 2] == v.STATUSES.index(status)


@pytest.mark.parametrize("audit", [{"status": "pass"}, {"status": "passed"}, {"passed": True},
                                    {"passed": True, "status": "passed", "errors": []}])
def test_historical_verification_formats_accepted(evidence, audit):
    root, cfg = evidence
    (root / "audit.json").write_text(json.dumps(audit), encoding="utf8")
    arrays, rows, summary = evaluate_producer(root, cfg)
    v.verify_science(cfg, root, arrays, rows, summary)


@pytest.mark.parametrize("audit", [{}, {"passed": False}, {"status": "failed"}, {"status": "completed"},
                                    {"status": "passed", "passed": False}, {"passed": True, "errors": ["unresolved"]},
                                    {"status": "passed", "errors": None}, {"passed": 1}])
def test_unverified_or_conflicting_study_rejected(evidence, audit):
    root, cfg = evidence
    (root / "audit.json").write_text(json.dumps(audit), encoding="utf8")
    with pytest.raises(ValueError):
        v.expected_evidence(cfg, root)


@pytest.mark.parametrize("declared", ["smoke_v1", "reversal_v1", None, "human_trial_data_v1"])
def test_prior_verification_identity_when_present(evidence, declared):
    root, cfg = evidence
    name = "identity_audit.json"
    (root / name).write_text(json.dumps({"passed": True, "study_id": declared}), encoding="utf8")
    cfg["input_files"].append(name)
    cfg["studies"][0]["verification"] = name
    if declared == "smoke_v1":
        arrays, rows, summary = evaluate_producer(root, cfg)
        v.verify_science(cfg, root, arrays, rows, summary)
    else:
        with pytest.raises(ValueError, match="different study"):
            v.expected_evidence(cfg, root)


@pytest.mark.parametrize("change", ["value", "float_type", "missing_key", "scalar_traversal", "bad_index", "outside", "unfrozen", "missing_file"])
def test_evidence_link_tampering_rejected(evidence, change):
    root, cfg = evidence
    obs = cfg["claims"][0]["observations"][0]
    if change == "value": obs["value"] = .126
    if change == "float_type": obs.update(pointer="/int", value=1.)
    if change == "missing_key": obs["pointer"] = "/absent"
    if change == "scalar_traversal": obs["pointer"] = "/int/a"
    if change == "bad_index": obs["pointer"] = "/items/2"
    if change == "outside": obs["file"] = "../report.md"; cfg["input_files"].append(obs["file"])
    if change == "unfrozen": cfg["input_files"].remove("source.json")
    if change == "missing_file": (root / "source.json").unlink()
    with pytest.raises(ValueError):
        v.expected_evidence(cfg, root)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'])
def test_invalid_json_rejected(tmp_path, text):
    path = tmp_path / "evidence.json"; path.write_text(text, encoding="utf8")
    with pytest.raises(ValueError): v.read_json(path)


@pytest.mark.parametrize("pointer", ["items", "/items/00", "/items/-1", "/items/+0", "/items/٠", "/items/-", "/items/1", "/a~2b", "/a~"])
def test_noncanonical_or_invalid_pointer_rejected(pointer):
    with pytest.raises(ValueError):
        v.resolve_pointer({"items": [3], "a~2b": 9, "a~": 8}, pointer)


def test_rfc6901_literal_keys_root_and_structural_values(evidence):
    root, cfg = evidence
    document = v.read_json(root / "source.json")
    assert v.resolve_pointer(document, "") == document
    assert v.resolve_pointer(document, "/") == "empty key"
    assert v.resolve_pointer(document, "/a~1b/~0") is False
    cfg["claims"][0]["observations"] = [{"file": "source.json", "pointer": "/list", "value": [0, True, None]},
                                         {"file": "source.json", "pointer": "/a~1b", "value": {"~": False}}]
    arrays, rows, summary = evaluate_producer(root, cfg)
    assert v.verify_science(cfg, root, arrays, rows, summary)["literal_json_observations"] == 6


@pytest.mark.parametrize("change", ["level5", "level_bool", "question_bool", "positive4", "status", "empty_text", "empty_limits", "empty_observations", "duplicate_claim", "claim_extra", "missing_study", "study_order", "question_order", "question_links", "question_empty", "inflated_interpretation", "scope", "duplicate_input"])
def test_invalid_design_or_claim_scope_rejected(evidence, change):
    _, cfg = evidence
    claim = cfg["claims"][0]
    if change == "level5": claim["level"] = 5
    if change == "level_bool": claim["level"] = True
    if change == "question_bool": claim["question"] = True
    if change == "positive4": claim.update(level=4, status="supported")
    if change == "status": claim["status"] = "success_somehow"
    if change == "empty_text": claim["text_ko"] = ""
    if change == "empty_limits": claim["limits_ko"] = ""
    if change == "empty_observations": claim["observations"] = []
    if change == "duplicate_claim": cfg["claims"][1]["claim_id"] = claim["claim_id"]
    if change == "claim_extra": claim["new_learning"] = True
    if change == "missing_study": cfg["studies"].pop()
    if change == "study_order": cfg["studies"].reverse()
    if change == "question_order": cfg["questions"].reverse()
    if change == "question_links": cfg["questions"][0]["claim_ids"] = ["claim_2"]
    if change == "question_empty": cfg["questions"][0]["claim_ids"] = []
    if change == "inflated_interpretation": cfg["synthesis_interpretation"]["level4_similarity_supported"] = True
    if change == "scope": cfg["scope"] = "new_learning_experiment"
    if change == "duplicate_input": cfg["input_files"].append("source.json")
    with pytest.raises(ValueError): v.validate_science(cfg)


@pytest.mark.parametrize("which", ["study_value", "claim_value", "study_dtype", "claim_dtype", "extra_array", "missing_array", "csv_value", "csv_extra", "csv_missing", "summary_value", "summary_hash", "summary_effect", "summary_extra"])
def test_output_tampering_rejected(evidence, which):
    root, cfg = evidence
    arrays, rows, summary = evaluate_producer(root, cfg)
    summary = copy.deepcopy(summary)
    if which == "study_value": arrays["study_checks"][0, 1] = 0
    if which == "claim_value": arrays["claim_codes"][0, 1] = 4
    if which == "study_dtype": arrays["study_checks"] = arrays["study_checks"].astype(float)
    if which == "claim_dtype": arrays["claim_codes"] = arrays["claim_codes"].astype(np.int32)
    if which == "extra_array": arrays["new_effect"] = np.asarray([1])
    if which == "missing_array": arrays.pop("claim_codes")
    if which == "csv_value": rows[0]["status"] = "supported"
    if which == "csv_extra": rows[0]["extra"] = "bad"
    if which == "csv_missing": rows.pop()
    if which == "summary_value": summary["claims"][0] = dict(summary["claims"][0], status="untested")
    if which == "summary_hash": summary["studies"][0]["report_sha256"] = "0" * 64
    if which == "summary_effect": summary["counts"]["new_effect_tests"] = 1
    if which == "summary_extra": summary["effect_size"] = .1
    with pytest.raises(ValueError): v.verify_science(cfg, root, arrays, rows, summary)


def test_report_bytes_tampering_changes_evidence_receipt(evidence):
    root, cfg = evidence
    arrays, rows, summary = evaluate_producer(root, cfg)
    (root / "report.md").write_text("Changed after original ledger creation", encoding="utf8")
    with pytest.raises(ValueError): v.verify_science(cfg, root, arrays, rows, summary)


def test_verified_primary_hash_is_checked(evidence):
    root, cfg = evidence
    (root / "summary.json").write_text('{"invented":true}', encoding="utf8")
    cfg["input_files"].append("summary.json")
    cfg["studies"][0]["primary"] = "summary.json"
    audit = {"status": "passed", "summary_sha256": v.digest(root / "summary.json")}
    (root / "audit.json").write_text(json.dumps(audit), encoding="utf8")
    arrays, rows, summary = evaluate_producer(root, cfg)
    v.verify_science(cfg, root, arrays, rows, summary)
    (root / "summary.json").write_text('{"invented":false}', encoding="utf8")
    with pytest.raises(ValueError): v.expected_evidence(cfg, root)


def test_no_primary_is_preserved_and_parent_frozen_path_accepted(evidence):
    root, cfg = evidence
    cfg["studies"][0]["primary"] = None
    cfg["studies"][1].pop("primary")
    cfg["input_files"].remove("report.md"); cfg["parent_files"].append("report.md")
    arrays, rows, summary = evaluate_producer(root, cfg)
    v.verify_science(cfg, root, arrays, rows, summary)
    assert summary["studies"][0]["primary_sha256"] is None


def test_verifier_source_has_no_scientific_producer_import():
    source = SCRIPT.read_text(encoding="utf8")
    assert "from drosolearn.phase5_report" not in source
    assert "import drosolearn.phase5_report" not in source
    assert "np.random" not in source


def commit_fixture(root):
    subprocess.run(["git", "add", "--all"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Invented fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--quiet", "--message", "Invented lossless-storage fixture"], cwd=root, check=True, capture_output=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()


@pytest.fixture
def parts_repo(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True)
    payload = b"Invented first lossless bytes\x00" + b"Invented second bytes\xff"
    parts = (payload[:17], payload[17:])
    target = tmp_path / v.SPLIT_RAW; target.parent.mkdir(parents=True)
    (target.parent / ".gitignore").write_text("arrays.npz\n", encoding="utf8")
    target.write_bytes(payload)
    (tmp_path / "ordinary.json").write_text('{"ordinary":true}', encoding="utf8")
    records = []
    for name, data in zip(v.SPLIT_PARTS, parts):
        path = tmp_path / name; path.write_bytes(data)
        records.append({"path": name, "bytes": len(data), "sha256": v.digest(path)})
    manifest = {"format": "drosolearn.lossless_file_parts.v1", "original_path": v.SPLIT_RAW,
                "original_bytes": len(payload), "original_sha256": v.digest(target), "max_part_bytes": 67108864, "parts": records}
    (tmp_path / v.SPLIT_MANIFEST).write_text(json.dumps(manifest), encoding="utf8")
    (tmp_path / v.SPLIT_ARTIFACTS).write_text(json.dumps({"arrays.npz": v.digest(target)}), encoding="utf8")
    commit = commit_fixture(tmp_path)
    return tmp_path, payload, manifest, commit


def test_independent_parts_match_producer_and_exact_original(parts_repo):
    root, payload, manifest, commit = parts_repo
    from drosolearn.phase5_report import split_git_payload
    expected = manifest["original_sha256"]
    assert v.joined_parent_from_git(root, commit, expected) == payload
    assert split_git_payload(root, commit, expected) == payload
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root).decode().splitlines()
    assert v.SPLIT_RAW not in tracked


@pytest.mark.parametrize("change", ["extra_field", "wrong_path", "format", "byte_total", "original_hash", "part_order", "part_name", "part_count", "part_bytes", "part_hash", "part_limit", "part_extra", "wrong_artifact"])
def test_committed_part_schema_or_original_hash_tamper_rejected(parts_repo, change):
    root, _, manifest, _ = parts_repo
    expected = manifest["original_sha256"]
    if change == "extra_field": manifest["extra"] = 1
    if change == "wrong_path": manifest["original_path"] = "results/other/arrays.npz"
    if change == "format": manifest["format"] = "some_other_format"
    if change == "byte_total": manifest["original_bytes"] += 1
    if change == "original_hash": manifest["original_sha256"] = "0" * 64
    if change == "part_order": manifest["parts"].reverse()
    if change == "part_name": manifest["parts"][0]["path"] = "../other"
    if change == "part_count": manifest["parts"].pop()
    if change == "part_bytes": manifest["parts"][0]["bytes"] += 1
    if change == "part_hash": manifest["parts"][0]["sha256"] = "0" * 64
    if change == "part_limit": manifest["max_part_bytes"] = 67108865
    if change == "part_extra": manifest["parts"][0]["extra"] = 1
    if change == "wrong_artifact": (root / v.SPLIT_ARTIFACTS).write_text('{"arrays.npz":"wrong"}', encoding="utf8")
    (root / v.SPLIT_MANIFEST).write_text(json.dumps(manifest), encoding="utf8")
    commit = commit_fixture(root)
    with pytest.raises(ValueError): v.joined_parent_from_git(root, commit, expected)


@pytest.mark.parametrize("name", [v.SPLIT_MANIFEST, *v.SPLIT_PARTS])
def test_uncommitted_local_part_or_manifest_tamper_rejected(parts_repo, name):
    root, _, manifest, commit = parts_repo
    path = root / name; path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError): v.joined_parent_from_git(root, commit, manifest["original_sha256"])


def test_materialization_is_exact_exclusive_and_idempotent(parts_repo):
    root, payload, _, commit = parts_repo
    raw = root / v.SPLIT_RAW
    assert v.materialize_parent_from_git(root, commit) == "already_matches"
    raw.unlink()
    assert v.materialize_parent_from_git(root, commit) == "materialized_exact_original_bytes"
    assert raw.read_bytes() == payload
    assert v.materialize_parent_from_git(root, commit) == "already_matches"
    raw.write_bytes(b"different existing raw result")
    with pytest.raises(ValueError, match="never overwrite"):
        v.materialize_parent_from_git(root, commit)
    assert raw.read_bytes() == b"different existing raw result"


def test_context_adapts_only_exact_parent_and_restores_after_failure(parts_repo):
    from drosolearn import prospective
    root, payload, manifest, commit = parts_repo
    original = prospective.committed
    names = {v.SPLIT_RAW: manifest["original_sha256"], "ordinary.json": v.digest(root / "ordinary.json")}
    with v.independent_parent_parts(root):
        prospective.committed(root, commit, names)
        prospective.committed(root, commit, {"ordinary.json": names["ordinary.json"]})
    assert prospective.committed is original
    with pytest.raises(ValueError):
        with v.independent_parent_parts(root):
            prospective.committed(root, commit, dict(names, **{"ordinary.json": "0" * 64}))
    assert prospective.committed is original
    assert (root / v.SPLIT_RAW).read_bytes() == payload
    with pytest.raises(RuntimeError):
        with v.independent_parent_parts(root):
            raise RuntimeError("fixture interruption")
    assert prospective.committed is original


def test_context_does_not_hide_wrong_existing_raw_or_wrong_root(parts_repo, tmp_path):
    from drosolearn import prospective
    root, _, manifest, commit = parts_repo
    original = prospective.committed
    (root / v.SPLIT_RAW).write_bytes(b"altered original")
    with pytest.raises(ValueError, match="Local raw parent"):
        with v.independent_parent_parts(root):
            prospective.committed(root, commit, {v.SPLIT_RAW: manifest["original_sha256"]})
    assert prospective.committed is original
    with pytest.raises(ValueError, match="Same parent repository"):
        with v.independent_parent_parts(root / "another-root"):
            prospective.committed(root, commit, {v.SPLIT_RAW: manifest["original_sha256"]})
    assert prospective.committed is original


@pytest.mark.parametrize("change", ["manifest", "missing_part", "missing_manifest"])
def test_cli_parent_storage_inventory_cannot_be_relaxed(change):
    cfg = {"chunked_parent_manifest": v.SPLIT_MANIFEST, "input_files": [v.SPLIT_MANIFEST, *v.SPLIT_PARTS]}
    v.validate_part_inventory(cfg)
    if change == "manifest": cfg["chunked_parent_manifest"] = "different.parts.json"
    if change == "missing_part": cfg["input_files"].remove(v.SPLIT_PARTS[0])
    if change == "missing_manifest": cfg["input_files"].remove(v.SPLIT_MANIFEST)
    with pytest.raises(ValueError): v.validate_part_inventory(cfg)
