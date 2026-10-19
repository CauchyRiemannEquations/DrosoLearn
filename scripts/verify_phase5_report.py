"""Independent evidence-ledger audit; imports no synthesis producer or learner."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import numpy as np

STUDY_IDS = (
    "smoke_v1", "reversal_v1", "encoding_v1", "spacing_v1", "linearity_v1",
    "reward_linear_v1", "bandit_reversal_v1", "head_attenuation_v1",
    "human_trial_data_audit_v1", "human_trial_linkage_v1", "human_semantics_v1",
    "human_osf_origin_v1", "human_task_structure_v1", "feedback_transition_v1",
    "metric_calibration_v1", "stochastic_feedback_linear_v1", "stochastic_feedback_network_v1",
    "learning_ablation_v1", "state_decomposition_v1", "conditional_transition_v1",
    "time_mixture_v1", "phase4_synthesis_v1", "human_shell_v1", "human_shell_rng_v2",
    "shell_credit_v1", "shell_behavior_v1", "pattern_rule_v1", "interference_order_v1",
)
STATUSES = ("supported", "not_supported", "bounded", "not_assessable", "design_limited", "untested")
INTERPRETATION = {
    "bounded_synthesis_complete": True,
    "all_possible_phase2_phase3_work_complete": False,
    "level4_similarity_supported": False,
    "shared_neural_mechanism_claim": False,
    "education_effect_claim": False,
}
SPLIT_RAW = "results/interference_order_v1/arrays.npz"
SPLIT_MANIFEST = SPLIT_RAW + ".parts.json"
SPLIT_PARTS = (SPLIT_RAW + ".part001", SPLIT_RAW + ".part002")
SPLIT_ARTIFACTS = "results/interference_order_v1/artifact_hashes.json"


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            demand(key not in result, "Duplicate evidence JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite evidence JSON: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path):
    return decode_json(Path(path).read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed_blob(root, commit, name):
    return subprocess.check_output(["git", "show", str(commit) + ":" + name], cwd=root)


def joined_parent_from_git(root, commit, expected_sha):
    """Independently validate two ordinary Git blobs for exactly one parent.

    The original numerical result is unchanged. Local manifest/parts must equal
    their committed blobs, and the join must match the old artifact hash.
    """
    root = Path(root)
    demand(all((root / name).resolve().is_relative_to(root.resolve())
               for name in (SPLIT_MANIFEST, SPLIT_ARTIFACTS, *SPLIT_PARTS)), "Contained parent storage paths required")
    raw_manifest = committed_blob(root, commit, SPLIT_MANIFEST)
    demand((root / SPLIT_MANIFEST).read_bytes() == raw_manifest, "Parent-part manifest differs from Git")
    manifest = decode_json(raw_manifest)
    demand(type(manifest) is dict and set(manifest) == {"format", "original_path", "original_bytes", "original_sha256", "max_part_bytes", "parts"},
           "Exact lossless-part manifest schema required")
    demand(manifest["format"] == "drosolearn.lossless_file_parts.v1" and manifest["original_path"] == SPLIT_RAW
           and type(manifest["max_part_bytes"]) is int and manifest["max_part_bytes"] == 67108864
           and type(manifest["original_bytes"]) is int and manifest["original_bytes"] > 0
           and type(expected_sha) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None
           and manifest["original_sha256"] == expected_sha, "Exact original parent identity required")
    artifact = decode_json(committed_blob(root, commit, SPLIT_ARTIFACTS))
    demand(type(artifact) is dict and artifact.get("arrays.npz") == expected_sha, "Original scientific artifact hash differs")
    parts = manifest["parts"]
    demand(type(parts) is list and len(parts) == 2
           and all(type(p) is dict for p in parts)
           and [p.get("path") for p in parts] == list(SPLIT_PARTS), "Exactly two ordered parent parts required")
    joined, byte_count = hashlib.sha256(), 0
    pieces = []
    for name, spec in zip(SPLIT_PARTS, parts):
        demand(set(spec) == {"path", "bytes", "sha256"} and type(spec["bytes"]) is int
               and 0 < spec["bytes"] <= 67108864 and type(spec["sha256"]) is str
               and re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is not None, "Valid bounded parent-part schema required")
        data = committed_blob(root, commit, name)
        demand((root / name).read_bytes() == data and len(data) == spec["bytes"]
               and hashlib.sha256(data).hexdigest() == spec["sha256"], "Parent part differs from literal Git bytes")
        joined.update(data); byte_count += len(data); pieces.append(data)
    demand(byte_count == manifest["original_bytes"] and joined.hexdigest() == expected_sha,
           "Joined bytes differ from immutable original parent")
    return b"".join(pieces)


@contextmanager
def independent_parent_parts(root):
    """Only the exact large raw-parent Git check is adapted; restore on exit."""
    from drosolearn import prospective
    original = prospective.committed
    def checked(check_root, commit, names):
        if SPLIT_RAW not in names:
            return original(check_root, commit, names)
        demand(Path(check_root).resolve() == Path(root).resolve(), "Same parent repository root required")
        ordinary = {name: checksum for name, checksum in names.items() if name != SPLIT_RAW}
        original(check_root, commit, ordinary)
        raw = joined_parent_from_git(check_root, commit, names[SPLIT_RAW])
        demand((Path(check_root) / SPLIT_RAW).resolve().is_relative_to(Path(root).resolve()), "Contained raw parent path required")
        demand((Path(check_root) / SPLIT_RAW).read_bytes() == raw, "Local raw parent differs from its lossless committed join")
    prospective.committed = checked
    try:
        yield
    finally:
        prospective.committed = original


def materialize_parent_from_git(root, commit="HEAD"):
    """Restore missing ignored raw bytes exclusively, without overwriting files."""
    root = Path(root)
    artifact = decode_json(committed_blob(root, commit, SPLIT_ARTIFACTS))
    expected_sha = artifact["arrays.npz"]
    payload = joined_parent_from_git(root, commit, expected_sha)
    target = root / SPLIT_RAW
    demand(target.resolve().is_relative_to(root.resolve()), "Contained materialized parent path required")
    if target.exists():
        demand(target.is_file() and target.read_bytes() == payload, "Existing original parent differs; never overwrite")
        return "already_matches"
    with target.open("xb") as handle:
        handle.write(payload)
    demand(digest(target) == expected_sha, "Restored original parent hash differs")
    return "materialized_exact_original_bytes"


def validate_part_inventory(cfg):
    demand(cfg.get("chunked_parent_manifest") == SPLIT_MANIFEST
           and {SPLIT_MANIFEST, *SPLIT_PARTS} <= set(cfg.get("input_files", [])), "Frozen exact parent-part storage inventory required")


def evidence_path(root, name, frozen):
    demand(type(name) is str and bool(name) and name in frozen, "Unfrozen evidence path")
    demand("\\" not in name and ":" not in name and not PurePosixPath(name).is_absolute()
           and ".." not in PurePosixPath(name).parts, "Contained POSIX evidence path required")
    path = (Path(root) / name).resolve()
    demand(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Missing or escaping evidence file: " + name)
    return path


def resolve_pointer(document, address):
    """RFC6901 JSON pointer, with canonical nonnegative array subscripts."""
    demand(type(address) is str and (address == "" or address.startswith("/")), "RFC6901 pointer required")
    current = document
    if address == "":
        return current
    for raw in address[1:].split("/"):
        demand(re.search(r"~(?![01])", raw) is None, "Invalid RFC6901 escape")
        key = raw.replace("~1", "/").replace("~0", "~")
        if type(current) is list:
            demand(re.fullmatch(r"0|[1-9][0-9]*", key) is not None, "Canonical array index required")
            index = int(key)
            demand(index < len(current), "Evidence array index out of range")
            current = current[index]
        elif type(current) is dict:
            demand(key in current, "Missing evidence key")
            current = current[key]
        else:
            raise ValueError("Evidence pointer traverses a scalar")
    return current


def validate_science(cfg):
    demand(cfg.get("study_id") == "phase5_synthesis_v1"
           and cfg.get("scope") == "retrospective_bounded_evidence_synthesis", "Bounded retrospective scope required")
    canonical(cfg)
    studies = cfg.get("studies")
    demand(type(studies) is list and [s.get("study_id") for s in studies] == list(STUDY_IDS),
           "All 28 historical studies required in order")
    for study in studies:
        demand(set(study) in ({"study_id", "tag", "verification", "report", "role"},
                             {"study_id", "tag", "verification", "report", "primary", "role"}),
               "Explicit study evidence schema required")
        demand(all(type(study[k]) is str and bool(study[k]) for k in ("tag", "verification", "report", "role")),
               "Nonempty study identity and evidence paths required")
        demand(study.get("primary") is None or type(study["primary"]) is str and bool(study["primary"]),
               "Optional primary evidence path required")
    demand(canonical(cfg.get("synthesis_interpretation")) == canonical(INTERPRETATION), "Fixed interpretation limits required")
    frozen = set()
    for key in ("input_files", "parent_files"):
        names = cfg.get(key)
        demand(type(names) is list and all(type(n) is str and n for n in names)
               and len(names) == len(set(names)), "Unique explicit evidence inventory required")
        frozen.update(names)
    claims = cfg.get("claims")
    demand(type(claims) is list and bool(claims), "Nonempty claim ledger required")
    identifiers = []
    for claim in claims:
        demand(set(claim) == {"claim_id", "question", "level", "status", "text_ko", "observations", "limits_ko"},
               "Explicit claim schema required")
        demand(type(claim["claim_id"]) is str and bool(claim["claim_id"]), "Nonempty claim identity required")
        identifiers.append(claim["claim_id"])
        demand(type(claim["question"]) is int and 1 <= claim["question"] <= 5
               and type(claim["level"]) is int and 0 <= claim["level"] <= 4
               and claim["status"] in STATUSES, "Valid claim coordinates required")
        demand(not (claim["level"] == 4 and claim["status"] == "supported"), "Positive Level4 claim unsupported by this package")
        demand(all(type(claim[k]) is str and bool(claim[k]) for k in ("text_ko", "limits_ko")), "Claim text and limits required")
        demand(type(claim["observations"]) is list and bool(claim["observations"]), "Every claim needs frozen observations")
        for observation in claim["observations"]:
            demand(type(observation) is dict and set(observation) == {"file", "pointer", "value"}, "Explicit observation schema required")
            demand(type(observation["file"]) is str and observation["file"] in frozen, "Observation file not frozen")
            demand(type(observation["pointer"]) is str, "Evidence pointer required")
    demand(len(identifiers) == len(set(identifiers)), "Duplicate claim identity")
    questions = cfg.get("questions")
    demand(type(questions) is list and [q.get("question") for q in questions] == list(range(1, 6)), "Five original questions required in order")
    for question in questions:
        demand(set(question) == {"question", "text_ko", "claim_ids"}
               and type(question["question"]) is int
               and type(question["text_ko"]) is str and bool(question["text_ko"]), "Explicit research question required")
        expected = [c["claim_id"] for c in claims if c["question"] == question["question"]]
        demand(bool(expected) and question["claim_ids"] == expected, "Question must link all aligned claims exactly once")


def expected_evidence(cfg, root):
    """Rebuild the ledger from literal prior evidence, not producer functions."""
    validate_science(cfg)
    frozen = set(cfg["input_files"]) | set(cfg["parent_files"])
    studies = []
    for study in cfg["studies"]:
        verification = evidence_path(root, study["verification"], frozen)
        audit = read_json(verification)
        demand(type(audit) is dict and audit.get("passed") is not False
               and (audit.get("passed") is True or audit.get("status") in ("passed", "pass")), "Study verification did not pass")
        if "study_id" in audit:
            demand(audit["study_id"] == study["study_id"], "Prior verification belongs to a different study")
        demand(audit.get("errors", []) == [], "Prior verification has unresolved errors")
        report = evidence_path(root, study["report"], frozen)
        primary = evidence_path(root, study["primary"], frozen) if study.get("primary") is not None else None
        if primary is not None and primary.name == "summary.json" and "summary_sha256" in audit:
            demand(digest(primary) == audit["summary_sha256"], "Prior summary differs from its verification")
        studies.append(dict(study, verified=True, verification_sha256=digest(verification),
                            report_sha256=digest(report), primary_sha256=digest(primary) if primary else None))
    rows, codes = [], []
    for claim in cfg["claims"]:
        for observation in claim["observations"]:
            document = read_json(evidence_path(root, observation["file"], frozen))
            actual = resolve_pointer(document, observation["pointer"])
            demand(canonical(actual) == canonical(observation["value"]), "Declared evidence value differs: " + claim["claim_id"])
        row = {key: claim[key] for key in ("claim_id", "question", "level", "status", "text_ko", "limits_ko")}
        row["observations_json"] = canonical(claim["observations"])
        rows.append(row)
        codes.append([claim["question"], claim["level"], STATUSES.index(claim["status"]), len(claim["observations"])])
    arrays = {"study_checks": np.column_stack((np.arange(28, dtype=np.int64), np.ones(28, dtype=np.int64))),
              "claim_codes": np.asarray(codes, dtype=np.int64)}
    summary = {
        "status": "completed", "study_id": cfg["study_id"], "scope": cfg["scope"],
        "studies": studies, "claims": cfg["claims"], "questions": cfg["questions"],
        "counts": {"studies": 28, "claims": len(cfg["claims"]),
                   "observations": sum(len(c["observations"]) for c in cfg["claims"]), "questions": 5,
                   "new_learning_updates": 0, "new_rng_draws": 0, "new_human_experiments": 0, "new_effect_tests": 0},
        "interpretation": dict(INTERPRETATION),
    }
    return arrays, rows, summary


def verify_science(cfg, root, arrays, rows, summary, cache=None):
    expected_arrays, expected_rows, expected_summary = expected_evidence(cfg, root)
    demand(set(arrays) == set(expected_arrays), "Exactly two retrospective numeric ledgers required")
    for key, expected in expected_arrays.items():
        actual = np.asarray(arrays[key])
        demand(actual.dtype == np.dtype(np.int64) and np.array_equal(actual, expected), "Exact int64 evidence ledger differs: " + key)
    demand(len(rows) == len(expected_rows), "All claims need one CSV record")
    for actual, expected in zip(rows, expected_rows):
        demand(set(actual) == set(expected), "Claim CSV columns differ")
        if all(type(value) is str for value in actual.values()):
            demand(actual == {key: str(value) for key, value in expected.items()}, "CSV claim evidence differs")
        else:
            demand(canonical(actual) == canonical(expected), "Typed claim evidence differs")
    demand(canonical(summary) == canonical(expected_summary), "Synthesis claims/counts/hashes differ from frozen evidence")
    return {
        "verified_historical_studies": 28, "independently_checked_claims": len(expected_rows),
        "literal_json_observations": expected_summary["counts"]["observations"],
        "original_research_questions": 5, "exact_int64_ledgers": 2,
        "all_study_report_primary_and_verification_hashes_checked": True,
        "all_observations_resolved_from_frozen_files": True,
        "negative_and_design_limited_evidence_preserved": True,
        "claims_are_bounded_synthesis_not_new_effect_tests": True,
        "new_learning_updates": 0, "new_rng_draws": 0, "new_human_experiments": 0, "new_effect_tests": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from drosolearn.prospective import verify_envelope, write_new
    preliminary_cfg = read_json(args.config)
    validate_part_inventory(preliminary_cfg)
    with independent_parent_parts(args.root):
        cfg, reg, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.cache)
    timestamps = [datetime.fromisoformat(value) for value in
                  (reg["registered_at_utc"], meta["started_at_utc"], meta["completed_at_utc"])]
    demand(timestamps[0] < timestamps[1] <= timestamps[2] <= datetime.now(timezone.utc), "Synthesis chronology must follow registration")
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with (args.result_dir / "measurements.csv").open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Unique synthesis CSV columns required")
        rows = list(reader)
    summary = read_json(args.result_dir / "summary.json")
    checks = verify_science(cfg, args.root, arrays, rows, summary, args.cache)
    report = {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": digest(args.registration), "summary_sha256": digest(args.result_dir / "summary.json"),
              "scientific_producer_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
