"""Derived artifacts preserve bytes and separate immediate effects from gain."""
import json

import numpy as np
import pytest

from test_head_attenuation import completed_fixture, csv_rows
from drosolearn.head_attenuation_analysis import analyze
from drosolearn.graph import sha256


def test_analysis_preserves_primary_records_and_separates_boundary_learning_reference(completed_fixture):
    _, _, _, output, summary = completed_fixture
    originals = {p.relative_to(output): sha256(p) for p in output.rglob("*") if p.is_file()}
    derived = analyze(output)
    assert all(sha256(output / name) == digest for name, digest in originals.items())
    assert len(derived["aggregate"]) == 7
    assert all(r["condition"] != "expected_linear" for r in derived["aggregate"])
    assert derived["actual_reward_updates"] == summary["actual_training_trials"] == 336
    assert derived["conditional_reference_updates"] == summary["expected_reference_updates"] == 48
    assert derived["frozen_boundary_baseline"] == {
        "rows": 240, "reward_updates": 0, "new_learner": False,
        "definition": "each path phase2 trial0 target probability held constant at every checkpoint; endpoint benefit minus initial advantage equals gain difference"}
    assert derived["raw_margin_reconstruction"]["rows"] == 240
    assert derived["boundary_manipulation_checks"]["rows"] == 64
    assert len(derived["conditional_reference"]["histories"]) == 5
    assert "not ensemble expectation" in derived["conditional_reference"]["scope"]
    assert "no success flags" in derived["conditional_reference"]["scope"]
    assert not any(r["condition"] == "expected_linear" for r in derived["qualified_adaptation_failures"])
    assert len(list(output.glob("*.png"))) == 8
    assert json.loads((output / "analysis_summary.json").read_text(encoding="utf-8")) == derived
    contrasts = csv_rows(output / "paired_attenuation_contrasts.csv")
    assert len(contrasts) == 7
    for row in contrasts:
        assert float(row["endpoint_benefit"]) - float(row["initial_advantage"]) == pytest.approx(float(row["gain_difference"]), abs=2e-15)
    frozen = csv_rows(output / "frozen_boundary_baseline.csv")
    assert len(frozen) == 240
    for row in frozen:
        assert int(row["reward_updates"]) == 0
        assert float(row["actual_p_target"]) - float(row["frozen_boundary_p_target"]) == pytest.approx(float(row["actual_minus_frozen"]), abs=2e-15)
        if int(row["phase2_trial"]) == 0:
            assert float(row["actual_minus_frozen"]) == 0
    boundaries = csv_rows(output / "boundary_confidence.csv")
    assert len(boundaries) == 64
    for row in boundaries:
        assert row["confidence_nonincreasing"] == "True"
        assert float(row["mean_confidence_after"]) <= float(row["mean_confidence_before"]) + 2e-12
        if float(row["gamma"]) > 0:
            assert row["raw_sign_preserved"] == "True"
        else:
            assert float(row["mean_confidence_after"]) == .5
    references = csv_rows(output / "reference_comparison.csv")
    for history in {r["history"] for r in references}:
        for split in ("iid", "ood"):
            selected = [r for r in references if r["history"] == history and r["split"] == split]
            abs_error = np.mean([abs(float(r["difference"])) for r in selected])
            item = next(r for r in derived["conditional_reference"]["histories"] if r["history"] == history)
            assert item[split]["per_seed_checkpoint_mae"]["mean"] == pytest.approx(abs_error, abs=2e-15)
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "소프트웨어 확인용 작은 fixture" in report
    assert "정확한 전체 기대값도 아니고" in report
    assert "학습 없는 기준선" in report
    assert "confidence-only 인과 효과" in report
    assert "단순 선형에는 KC·MBON이 없고" in report
    assert "상속 counter를 네 번의 호출로 세지 않는다" in report
    # Every header, separator, and data row has the same table cell count.
    table_width = None
    for line in report.splitlines():
        if line.startswith("|"):
            if table_width is None:
                table_width = line.count("|")
            assert line.count("|") == table_width
        else:
            table_width = None
