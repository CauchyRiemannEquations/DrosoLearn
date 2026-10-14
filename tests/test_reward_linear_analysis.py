"""Derived reports preserve raw artifacts and retain failed seed outcomes."""
import csv
import hashlib
import json

from drosolearn.reward_linear import run
from drosolearn.reward_linear_analysis import analyze
from test_reward_linear import prepare_reward_fixture


def test_derived_analysis_preserves_primary_bytes_and_all_conditions(tmp_path):
    root, config, registration = prepare_reward_fixture(tmp_path)
    output = root / "analysis-fixture"
    run(root, config, registration, output)
    manifest = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    original_manifest = (output / "artifact_hashes.json").read_bytes()
    result = analyze(output)
    assert (output / "artifact_hashes.json").read_bytes() == original_manifest
    assert all(hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest for relative, digest in manifest.items())
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    analysis = json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))
    assert len(result) == 9
    assert analysis["all_seeds_included"]
    assert analysis["hypotheses_from_run_summary"] == summary["hypotheses"]
    assert analysis["acquisition_failures"] == [r for r in summary["outcomes"] if not r["acquired"]]
    assert analysis["joint_identity_transfer_failures"] == [r for r in summary["outcomes"] if not r["qualified_unseen_transfer"][2]]
    assert analysis["pair_summary_rows"] == 9 * 2 * 4 * 27
    with (output / "pair_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == analysis["pair_summary_rows"]
    assert {r["stratum"] for r in rows} == {"fully_unseen", "one_anchor"}
    assert len(list(output.glob("*.png"))) == 6
    assert all(p.stat().st_size > 1000 for p in output.glob("*.png"))
    assert "소프트웨어 확인용 작은 fixture" in (output / "report.md").read_text(encoding="utf-8")
