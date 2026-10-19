"""Synthetic archives only: no downloaded TNU source is used in these tests."""
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from drosolearn.human_shell_source import load_cohort, SourceValidationError


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path, change=None, exclusions=None):
    # Four seasons of nine encounters per shell produce 216 tiny synthetic rows.
    seasons = {1: ("R", "N"), 2: ("N", "R"), 3: ("R", "P"),
               4: ("P", "R"), 5: ("P", "N"), 6: ("N", "P")}
    trace = []
    for encounter in range(36):
        state = 1 if encounter // 9 % 2 == 0 else 2
        for shell in range(1, 7):
            season = seasons[shell][state - 1]
            trace.append([shell, state, {"R": 1, "P": 2, "N": 0}[season]])
    log = [[i + 1, t[0], t[2], i % 2, 100 if i % 2 else 0] for i, t in enumerate(trace)]
    sources = {"A0001": (copy.deepcopy(trace), copy.deepcopy(log)),
               "A0016": (copy.deepcopy(trace), copy.deepcopy(log))}
    if change:
        change(sources)
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as z:
        for pid, (tr, lg) in sources.items():
            z.writestr(f"behavior/{pid}/GONOGO_trace_{pid}.csv", "\r\r\n".join(("," if index % 2 else " ").join(map(str, r)) for index, r in enumerate(tr)))
            z.writestr(f"behavior/{pid}/GONOGO_shortlog_{pid}.txt", "\r\r\n".join(str(r[0]) + ":  " + " ".join(map(str, r[1:])) for r in lg))
        extra = b"synthetic nonstandard file"
        extra_name = "behavior/A0001/unused.txt"
        z.writestr(extra_name, extra)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("A_Data.csv", "ID,VerumPlacebo\r\r\nA0001,0\r\r\nA0016,1\r\r\n")
        z.writestr("A_OCIR.csv", "ID,score\nA0001,0\nA0016,0\n")
        z.writestr("A_Behav.zip", nested.getvalue())
    files = {"Dataset_TNU.zip": outer.getvalue(), "readme.pdf": b"synthetic README test bytes"}
    metadata = {"id": 10066829, "metadata": {"license": {"id": "cc-by-4.0"}},
                "files": [{"key": name, "size": len(data), "checksum": "md5:" + hashlib.md5(data).hexdigest()}
                          for name, data in files.items()]}
    files["metadata.json"] = json.dumps(metadata).encode()
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    excluded = exclusions or {}
    manifest = {"schema": "drosolearn.tnu-shell-source.v1", "record_id": 10066829,
                "expected_trials_per_trace": 216,
                "raw_metadata_ids": ["A0001", "A0016"],
                "allowed_complete_ids": [p for p in ["A0001", "A0016"] if p not in excluded],
                "expected_exclusions": excluded,
                "additional_behavior_members": {extra_name: {"bytes": len(extra), "sha256": _digest(extra)}},
                "files": {name: {"bytes": len(data), "sha256": _digest(data)} for name, data in files.items()}}
    return manifest


def test_complete_synthetic_cohort_and_environment_contents(tmp_path):
    manifest = _fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    records, audit = load_cohort(tmp_path, manifest)
    assert list(records) == ["A0001", "A0016"]
    assert len(records["A0001"]) == 216
    # This actual potential reward was hidden on NoGo, yet must remain available
    # to an environment replay. The learner masking is another module's contract.
    assert records["A0001"][0] == {"trial": 1, "shell": 1, "state": 1, "content": 1, "action": 0}
    assert audit["complete_trials"] == 432
    assert audit["participants"]["A0001"]["shell_input_structure"]["3"]["phase_lengths"] == [9, 9, 9, 9]
    assert not audit["behavioral_performance_aggregated"]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_predefined_truncated_record_is_preserved_in_audit(tmp_path):
    def truncate(sources):
        sources["A0016"][1][:] = sources["A0016"][1][:120]
        sources["A0016"][1][-1][:] = sources["A0016"][1][-1][:3]
    reasons = ["trace_shortlog_row_count_mismatch", "shortlog_not_five_columns",
               "response_rt_domain_invalid_or_short_row"]
    manifest = _fixture(tmp_path, truncate, {"A0016": reasons})
    records, audit = load_cohort(tmp_path, manifest)
    assert list(records) == ["A0001"]
    assert audit["excluded"] == {"A0016": reasons}
    assert audit["participants"]["A0016"]["shortlog_trials"] == 120
    assert audit["participants"]["A0016"]["shortlog_column_counts"] == [3, 5]


@pytest.mark.parametrize("filename", ["Dataset_TNU.zip", "metadata.json", "readme.pdf"])
def test_literal_source_change_is_rejected(tmp_path, filename):
    manifest = _fixture(tmp_path)
    with (tmp_path / filename).open("ab") as f:
        f.write(b"changed")
    with pytest.raises(SourceValidationError, match="source bytes differ"):
        load_cohort(tmp_path, manifest)


@pytest.mark.parametrize("mutation", [
    lambda s: s["A0001"][0][0].__setitem__(0, 7),
    lambda s: s["A0001"][0][0].__setitem__(1, 0),
    lambda s: s["A0001"][0][0].__setitem__(2, 3),
    lambda s: s["A0001"][1][0].__setitem__(0, 2),
    lambda s: s["A0001"][1][0].__setitem__(1, 2),
    lambda s: s["A0001"][1][0].__setitem__(2, 0),
    lambda s: s["A0001"][1][0].__setitem__(3, 2),
    lambda s: s["A0001"][1][0].__setitem__(4, 100),
    lambda s: s["A0001"][1][1].__setitem__(4, 0),
    lambda s: s["A0001"][0].pop(),
    lambda s: s["A0001"][1].pop(),
    lambda s: s["A0001"][0][0].append(0),
    lambda s: s["A0001"][1][0].append(0),
])
def test_new_structural_failure_cannot_silently_change_cohort(tmp_path, mutation):
    manifest = _fixture(tmp_path, mutation)
    with pytest.raises(SourceValidationError, match="complete cohort differs"):
        load_cohort(tmp_path, manifest)


def test_n_season_cannot_contain_a_pearl(tmp_path):
    def wrong_content(sources):
        # Shell 2 starts in N. Keep files mutually consistent while falsifying
        # the documented state/content meaning, so matching alone cannot pass.
        sources["A0001"][0][1][2] = 1
        sources["A0001"][1][1][2] = 1
    manifest = _fixture(tmp_path, wrong_content)
    with pytest.raises(SourceValidationError, match="complete cohort differs"):
        load_cohort(tmp_path, manifest)


def test_manifest_cannot_exclude_a_complete_but_unwanted_person(tmp_path):
    manifest = _fixture(tmp_path, exclusions={"A0016": ["failed_learning"]})
    with pytest.raises(SourceValidationError, match="complete cohort differs"):
        load_cohort(tmp_path, manifest)


def test_exclusion_reason_cannot_be_relabelled(tmp_path):
    def truncate(sources):
        sources["A0016"][1].pop()
    manifest = _fixture(tmp_path, truncate, {"A0016": ["failed_learning"]})
    with pytest.raises(SourceValidationError, match="exclusion reasons differ"):
        load_cohort(tmp_path, manifest)


def test_additional_member_hash_is_checked(tmp_path):
    manifest = _fixture(tmp_path)
    next(iter(manifest["additional_behavior_members"].values()))["sha256"] = "0" * 64
    with pytest.raises(SourceValidationError, match="additional member bytes differ"):
        load_cohort(tmp_path, manifest)


@pytest.mark.parametrize("key,value", [
    ("expected_trials_per_trace", 120),
    ("raw_metadata_ids", ["A0016", "A0001"]),
    ("allowed_complete_ids", ["A0001"]),
    ("schema", "unknown"),
])
def test_manifest_contract_is_validated(tmp_path, key, value):
    manifest = _fixture(tmp_path)
    manifest[key] = value
    with pytest.raises(SourceValidationError):
        load_cohort(tmp_path, manifest)
