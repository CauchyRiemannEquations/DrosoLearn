"""Derived reports preserve primary bytes and separate privileged drift."""
import json

from test_bandit_reversal import completed_fixture
from drosolearn.bandit_reversal_analysis import analyze
from drosolearn.graph import sha256


def test_derived_analysis_preserves_primary_records_and_separates_reference(completed_fixture):
    _, _, _, output, summary = completed_fixture
    originals = {p.relative_to(output): sha256(p) for p in output.rglob("*") if p.is_file()}
    derived = analyze(output)
    assert all(sha256(output / name) == digest for name, digest in originals.items())
    assert len(derived["aggregate"]) == 7
    assert all(r["condition"] != "expected_linear" for r in derived["aggregate"])
    assert derived["actual_reward_updates"] == summary["actual_training_trials"] == 392
    assert derived["conditional_reference_updates"] == summary["expected_reference_updates"] == 56
    reference = [r for r in summary["outcomes"] if r["condition"] == "expected_linear"]
    assert len(reference) == 4
    for row in reference:
        assert all(row[key] is None for key in ("phase1_acquired", "fresh_acquired", "phase2_target_passed", "qualified_adaptation", "first_observed_75_checkpoint"))
    assert len(derived["conditional_reference"]["histories"]) == 4
    assert derived["raw_margin_reconstruction"]["rows"] == 192
    assert "not ensemble expectation" in derived["conditional_reference"]["scope"]
    assert not any(r["condition"] == "expected_linear" for r in derived["qualified_adaptation_failures"])
    assert len(list(output.glob("*.png"))) == 8
    assert (output / "raw_margin_summary.csv").is_file()
    assert (output / "reference_comparison.csv").is_file()
    saved = json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))
    assert saved == derived
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "소프트웨어 확인용 작은 fixture" in report
    assert "정확한 전체 기대값도 아니다" in report
    assert "학습 능력 자체가 느려졌" in report
    assert "단순 선형에는 KC·MBON이 없어" in report
