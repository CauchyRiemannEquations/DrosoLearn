"""Synthetic-only independent source identity replay and corruption tests."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pytest

from test_human_osf_origin import complete_fixture, make_fixture, origin


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_human_osf_origin.py"
SPEC = importlib.util.spec_from_file_location("independent_human_osf_origin", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def invented_arrays():
    return {"A": np.array([[1., 2., 3., np.nan, 3., 1.]]),
            "R": np.array([[1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.], [-1., 1.]]),
            "S": np.array([[1.], [1.], [2.], [1.], [1.], [2.]]),
            "C": np.array([[1., 0., np.nan, np.nan, np.nan, 1.]])}


def test_source_formulas_distinguish_reward_hint_and_missing():
    arrays = invented_arrays()
    before = {key: value.copy() for key, value in arrays.items()}
    result = VERIFIER.transform_raw(arrays, 6)
    assert np.array_equal(result["responses"], [0, 1, 2, np.nan, 2, 0], equal_nan=True)
    assert np.array_equal(result["outcomes"], [1, 0, 3, np.nan, 2, 0], equal_nan=True)
    assert np.array_equal(result["correct_responses"], arrays["C"][0], equal_nan=True)
    assert all(np.array_equal(value, before[key], equal_nan=True) for key, value in arrays.items())


def test_correctness_is_copied_instead_of_recomputed():
    arrays = invented_arrays()
    arrays["C"][0, 0] = 0
    assert VERIFIER.transform_raw(arrays, 6)["correct_responses"][0] == 0


@pytest.mark.parametrize("name", ["A", "R", "S", "C"])
def test_raw_shape_change_is_rejected(name):
    arrays = invented_arrays()
    arrays[name] = arrays[name].ravel()
    with pytest.raises(VERIFIER.VerificationError, match="shape/type"):
        VERIFIER.transform_raw(arrays, 6)


def test_raw_complex_code_is_rejected_without_casting():
    arrays = invented_arrays()
    arrays["A"] = arrays["A"].astype(complex)
    with pytest.raises(VERIFIER.VerificationError, match="shape/type"):
        VERIFIER.transform_raw(arrays, 6)


def test_ast_list_order_and_duplicates_survive_without_execution(tmp_path):
    sentinel = tmp_path / "must_not_exist"
    path = tmp_path / "formatter.py"
    path.write_text("fnames1=['b.mat','b.mat']\nfnames2=['a.mat']\nfnames=fnames1+fnames2\n"
                    f"open({str(sentinel)!r},'w').write('do not execute')\n", encoding="utf-8")
    assert VERIFIER.filename_tokens(path, ["fnames1", "fnames2"]) == ["b.mat", "b.mat", "a.mat"]
    assert not sentinel.exists()


def test_ast_dynamic_filename_expression_is_rejected(tmp_path):
    path = tmp_path / "formatter.py"
    path.write_text("fnames1=list(['invented.mat'])\n", encoding="utf-8")
    with pytest.raises(ValueError):
        VERIFIER.filename_tokens(path, ["fnames1"])


def verify_fixture(fixture):
    return VERIFIER.verify(fixture["result_path"], fixture["registration_path"],
                           fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


def test_full_independent_replay_preserves_all_coverage_and_missing_cells(tmp_path):
    fixture = complete_fixture(tmp_path)
    result = verify_fixture(fixture)
    assert result["status"] == "passed" and result["source_identity_gate_passed"] is True
    assert result["raw_rows_replayed"] == 2 and result["processed_rows_retained"] == 3
    assert result["unmatched_rows_preserved"] == 1 and result["values_replayed"] == 24
    assert result["mismatches_verified"] == 0 and result["source_files_frozen"] == len(VERIFIER.FROZEN)


@pytest.mark.parametrize("mutation", ["value_count", "mismatch_index", "drop_unmatched", "extend_license", "reverse_chronology"])
def test_report_tamper_is_detected_by_independent_replay(tmp_path, mutation):
    fixture = complete_fixture(tmp_path)
    report = fixture["result"]
    if mutation == "value_count":
        report["total_values_compared"] -= 1
    elif mutation == "mismatch_index":
        report["rows"][0]["comparison"]["outcomes"]["first_mismatch_indices"] = [0]
    elif mutation == "drop_unmatched":
        report["coverage"]["groups"]["pilot"]["mapping"] = []
    elif mutation == "extend_license":
        report["license_extended_to_derived_CSV"] = True
    else:
        report["started_at_utc"] = "2000-01-01T00:00:00+00:00"
    write_json(fixture["result_path"], report)
    with pytest.raises(VERIFIER.VerificationError):
        verify_fixture(fixture)


def test_unselected_derived_row_change_cannot_hide_outside_matched_subset(tmp_path):
    fixture = complete_fixture(tmp_path)
    with (fixture["derived_cache"] / "pilot/outcomes.csv").open("a", encoding="utf8") as handle:
        handle.write(" ")
    with pytest.raises(VERIFIER.VerificationError, match="Input size/hash"):
        verify_fixture(fixture)


def test_rehashed_registration_and_report_do_not_hide_changed_coverage(tmp_path):
    fixture = complete_fixture(tmp_path)
    registration = fixture["registration"]
    registration["coverage"]["groups"]["main"]["mapping"][0]["matches"] = 0
    write_json(fixture["registration_path"], registration)
    report = fixture["result"]
    report["registration_sha256"] = VERIFIER.sha(fixture["registration_path"])
    report["coverage"] = registration["coverage"]
    write_json(fixture["result_path"], report)
    with pytest.raises(VERIFIER.VerificationError, match="Registered inputs/coverage"):
        verify_fixture(fixture)


def test_honest_value_mismatch_remains_failed_audit_with_passed_verification(tmp_path):
    fixture = make_fixture(tmp_path)
    path = fixture["derived_cache"] / "main/outcomes.csv"
    path.write_text(path.read_text(encoding="utf8").replace("0,1.0,1.0", "0,0.0,1.0"), encoding="utf8")
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf8"))
    entry = next(item for item in manifest["derived_sources"] if item["group"] == "main" and item["role"] == "outcomes")
    entry.update(bytes=path.stat().st_size, sha256=VERIFIER.sha(path))
    write_json(fixture["manifest_path"], manifest)
    registration = origin.register(fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])
    write_json(fixture["registration_path"], registration)
    report = origin.run(fixture["registration_path"], fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])
    write_json(fixture["result_path"], report)
    verification = verify_fixture(fixture)
    assert report["status"] == "failed" and verification["status"] == "passed"
    assert verification["source_identity_gate_passed"] is False and verification["mismatches_verified"] == 1


def test_cli_writes_only_new_verification_and_preserves_primary_files(tmp_path, monkeypatch, capsys):
    fixture = complete_fixture(tmp_path)
    before = {key: fixture[key].read_bytes() for key in ("result_path", "registration_path", "manifest_path")}
    out = tmp_path / "verification.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--result", str(fixture["result_path"]),
        "--registration", str(fixture["registration_path"]), "--cache", str(fixture["cache"]),
        "--derived-cache", str(fixture["derived_cache"]), "--manifest", str(fixture["manifest_path"]), "--out", str(out)])
    with pytest.raises(SystemExit) as exited:
        VERIFIER.main()
    assert exited.value.code == 0
    assert json.loads(out.read_text(encoding="utf8"))["status"] == "passed"
    assert len(capsys.readouterr().out) < 300
    assert all(fixture[key].read_bytes() == value for key, value in before.items())
    saved = out.read_bytes()
    with pytest.raises(VERIFIER.VerificationError, match="already exists"):
        VERIFIER.main()
    assert out.read_bytes() == saved
