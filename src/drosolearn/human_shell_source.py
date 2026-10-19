"""Read the frozen TNU shell traces without extracting or changing source bytes.

This module performs structural validation only. Potential contents are returned
for environment replay; callers must hide them on NoGo and before each choice.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile


class SourceValidationError(ValueError):
    """The supplied sources or cohort differ from the prospective manifest."""


SEASONS = {1: {1: "R", 2: "N"}, 2: {1: "N", 2: "R"},
           3: {1: "R", 2: "P"}, 4: {1: "P", 2: "R"},
           5: {1: "P", 2: "N"}, 6: {1: "N", 2: "P"}}
CONTENT = {"R": {0, 1}, "P": {0, 2}, "N": {0}}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceValidationError(message)


def _rows(data: bytes, *, shortlog: bool) -> list[list[int]]:
    rows = []
    for line in data.decode("utf-8-sig").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append([int(value) for value in re.split(r"[:\s]+" if shortlog else r"[,\s]+", line)])
            except ValueError as exc:
                raise SourceValidationError("noninteger source row") from exc
    return rows


def _unique_zip(archive: zipfile.ZipFile, label: str) -> None:
    names = archive.namelist()
    _require(len(names) == len(set(names)), f"duplicate {label} archive member")


def _review_pair(trace: list[list[int]], log: list[list[int]], expected: int) -> tuple[list[str], dict]:
    issues = []
    if len(trace) != expected:
        issues.append("trace_expected_row_count_mismatch")
    if len(trace) != len(log):
        issues.append("trace_shortlog_row_count_mismatch")
    trace_shape = all(len(row) == 3 for row in trace)
    log_shape = all(len(row) == 5 for row in log)
    if not trace_shape:
        issues.append("trace_not_three_columns")
    if not log_shape:
        issues.append("shortlog_not_five_columns")
    contiguous = [row[0] for row in log if row] == list(range(1, len(log) + 1))
    if not contiguous:
        issues.append("noncontiguous_original_trials")
    matching = all(len(t) >= 3 and len(l) >= 3 and t[0] == l[1] and t[2] == l[2]
                   for t, l in zip(trace, log))
    if not matching:
        issues.append("stimulus_or_content_mismatch")
    valid_trace = trace_shape and all(row[0] in SEASONS and row[1] in {1, 2}
                                    and row[2] in {0, 1, 2} for row in trace)
    if not valid_trace:
        issues.append("trace_domain_invalid")
    valid_rt = log_shape and all(row[3] in {0, 1} and
                                (row[4] == 0 if row[3] == 0 else row[4] > 0) for row in log)
    if not valid_rt:
        issues.append("response_rt_domain_invalid_or_short_row")
    seasonal = valid_trace and all(row[2] in CONTENT[SEASONS[row[0]][row[1]]] for row in trace)
    if not seasonal:
        issues.append("state_content_mapping_invalid")
    shell_inputs = {}
    if valid_trace:
        for shell in range(1, 7):
            own = [row for row in trace if row[0] == shell]
            states, lengths = [], []
            for row in own:
                if not states or states[-1] != row[1]:
                    states.append(row[1])
                    lengths.append(0)
                lengths[-1] += 1
            shell_inputs[str(shell)] = {"input_occurrences": len(own), "state_run_sequence": states,
                                        "phase_lengths": lengths}
            if states != [1, 2, 1, 2]:
                issues.append(f"shell_{shell}_state_sequence_invalid")
    return issues, {"trace_trials": len(trace), "shortlog_trials": len(log),
                    "shortlog_column_counts": sorted({len(row) for row in log}),
                    "input_output_match_on_available_pairs": matching,
                    "original_trial_sequence_contiguous": contiguous,
                    "response_rt_domain_valid": valid_rt,
                    "state_content_mapping_valid": seasonal,
                    "shell_input_structure": shell_inputs}


def load_cohort(cache: Path, manifest: dict) -> tuple[dict[str, list[dict]], dict]:
    """Return the exact complete cohort and a source-only audit.

    The manifest is supplied by the caller and must itself be frozen before the
    scientific run. Any new corruption or changed exclusion causes a hard error.
    An excluded participant is parsed and audited, never silently dropped.
    """
    _require(manifest.get("schema") == "drosolearn.tnu-shell-source.v1", "unknown source manifest schema")
    _require(manifest.get("expected_trials_per_trace") == 216, "source trace length must be 216")
    raw_ids = manifest["raw_metadata_ids"]
    allowed_ids = manifest["allowed_complete_ids"]
    exclusions = manifest["expected_exclusions"]
    _require(raw_ids == sorted(set(raw_ids)), "manifest raw IDs must be unique and sorted")
    _require(allowed_ids == sorted(set(allowed_ids)), "manifest allowed IDs must be unique and sorted")
    _require(set(raw_ids) == set(allowed_ids) | set(exclusions), "manifest cohort partition differs")
    _require(not set(allowed_ids) & set(exclusions), "manifest allowed/excluded overlap")
    _require(all(re.fullmatch(r"A\d{4}", pid) for pid in raw_ids), "invalid manifest ID")
    expected_names = {"Dataset_TNU.zip", "metadata.json", "readme.pdf"}
    _require(set(manifest["files"]) == expected_names, "manifest cache file set differs")
    source_bytes = {}
    file_audit = {}
    for name in sorted(expected_names):
        data = (Path(cache) / name).read_bytes()
        spec = manifest["files"][name]
        _require(len(data) == spec["bytes"] and _sha(data) == spec["sha256"], f"source bytes differ: {name}")
        source_bytes[name] = data
        file_audit[name] = {"bytes": len(data), "sha256": _sha(data)}
    metadata = json.loads(source_bytes["metadata.json"])
    _require(metadata["id"] == manifest["record_id"], "record ID differs")
    _require(metadata["metadata"]["license"]["id"] == "cc-by-4.0", "metadata license differs")
    for entry in metadata["files"]:
        name = entry["key"]
        _require(name in source_bytes, "unexpected metadata file")
        _require(len(source_bytes[name]) == entry["size"], "metadata source size differs")
        _require(entry["checksum"] == "md5:" + hashlib.md5(source_bytes[name]).hexdigest(), "metadata MD5 differs")
    records, participants, observed_exclusions = {}, {}, {}
    member_hashes = {}
    with zipfile.ZipFile(io.BytesIO(source_bytes["Dataset_TNU.zip"])) as outer:
        _unique_zip(outer, "outer")
        _require(set(outer.namelist()) == {"A_Data.csv", "A_OCIR.csv", "A_Behav.zip"}, "outer member set differs")
        member_hashes["outer"] = {name: {"bytes": len(outer.read(name)), "sha256": _sha(outer.read(name))}
                                  for name in sorted(outer.namelist())}
        reader = csv.DictReader(io.StringIO(outer.read("A_Data.csv").decode("utf-8-sig")))
        _require(reader.fieldnames is not None and "ID" in reader.fieldnames, "missing metadata ID column")
        _require(len(reader.fieldnames) == len(set(reader.fieldnames)), "duplicate metadata column")
        rows = [row for row in reader if row.get("ID")]
        observed_ids = [row["ID"] for row in rows]
        _require(len(observed_ids) == len(set(observed_ids)), "duplicate metadata ID")
        _require(sorted(observed_ids) == raw_ids, "raw metadata ID set differs")
        with zipfile.ZipFile(io.BytesIO(outer.read("A_Behav.zip"))) as inner:
            _unique_zip(inner, "behavior")
            standard = {f"behavior/{pid}/GONOGO_{kind}_{pid}.{ext}" for pid in raw_ids
                        for kind, ext in [("trace", "csv"), ("shortlog", "txt")]}
            non_dirs = {info.filename for info in inner.infolist() if not info.is_dir()}
            extras = manifest["additional_behavior_members"]
            _require(non_dirs == standard | set(extras), "behavior member set differs")
            member_hashes["behavior"] = {name: {"bytes": len(inner.read(name)), "sha256": _sha(inner.read(name))}
                                         for name in sorted(non_dirs)}
            for name, spec in extras.items():
                _require(member_hashes["behavior"][name] == spec, "additional member bytes differ")
            for pid in raw_ids:
                trace_name = f"behavior/{pid}/GONOGO_trace_{pid}.csv"
                log_name = f"behavior/{pid}/GONOGO_shortlog_{pid}.txt"
                trace = _rows(inner.read(trace_name), shortlog=False)
                log = _rows(inner.read(log_name), shortlog=True)
                issues, audit = _review_pair(trace, log, 216)
                audit["structural_exclusion_reasons"] = issues
                audit["complete_pair_eligible"] = not issues
                participants[pid] = audit
                if issues:
                    observed_exclusions[pid] = issues
                else:
                    records[pid] = [{"trial": l[0], "shell": t[0], "state": t[1],
                                     "content": t[2], "action": l[3]} for t, l in zip(trace, log)]
    _require(sorted(records) == allowed_ids, "complete cohort differs from manifest")
    _require(observed_exclusions == exclusions, "structural exclusion reasons differ from manifest")
    audit = {"schema": "drosolearn.tnu-shell-source-audit.v1", "record_id": manifest["record_id"],
             "raw_metadata_participants": len(raw_ids), "complete_participants": len(records),
             "complete_trials": sum(len(rows) for rows in records.values()),
             "eligible_ids": sorted(records), "excluded": observed_exclusions,
             "files": file_audit, "member_hashes": member_hashes, "participants": participants,
             "potential_content_is_environment_only": True, "raw_files_modified": False,
             "raw_files_extracted": False, "behavioral_performance_aggregated": False,
             "medication_groups_compared": False, "license_conflict_preserved": True}
    return records, audit
