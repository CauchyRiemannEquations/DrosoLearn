"""Independent fixed-code contingency counts; select no mapping or performance metric."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import re
import subprocess


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact(actual[k], value) for k, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def safe(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Relative file path required")
    target = (Path(root) / relative).resolve()
    require(target.is_relative_to(Path(root).resolve()) and target.is_file(), "Unsafe or missing file")
    return target


PARENT = "results/human_trial_linkage_v1/"
INPUTS = tuple(PARENT + name for name in ("corroboration.csv", "record-links.csv", "linkage-report.json",
                                        "metadata.json", "registration.json", "artifact_hashes.json", "verification.json"))
CONFIG = {"audit_id": "human_semantics_v1", "input_tag": "human-trial-linkage-v1",
          "input_directory": PARENT.rstrip("/"), "scope": "fixed_code_cross_tabs_no_mapping_search",
          "select_mapping": False, "performance_analysis": False}
FROZEN = set(INPUTS) | {"src/drosolearn/human_semantics.py", "src/drosolearn/human_trial_data.py",
                       "scripts/verify_human_semantics.py", "tests/test_human_semantics.py",
                       "tests/test_human_semantics_run.py", "docs/human-semantics-protocol.md",
                       "docs/human-semantics-source-review.md", "requirements-lock.txt", "pyproject.toml",
                       "data/human/stolz_2022_semantics_evidence/source-manifest.json"}
FIELDS = ("event_label", "mat_row", "task", "feedback_ordinal", "feedback_event_row", "strict_comparable",
          "event_choice_side_code", "mat_action_code", "event_expectation_plus3", "mat_expect_code",
          "action_numeric_agreement", "expectation_numeric_agreement")


def source_csv(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == list(fields), "Parent CSV schema differs")
        rows = []
        for values in reader:
            require(len(values) == len(fields), "Malformed parent CSV row")
            rows.append(dict(zip(fields, values)))
    require(rows, "Empty parent CSV")
    return rows


def integer(text, allowed=None):
    require(isinstance(text, str) and re.fullmatch(r"0|[1-9][0-9]*", text), "Noncanonical integer code")
    value = int(text)
    require(allowed is None or value in allowed, "Code outside fixed alphabet")
    return value


def count_table(rows):
    selected = [row for row in rows if row["strict_comparable"] == "True"]
    action = Counter((integer(row["event_choice_side_code"]), integer(row["mat_action_code"])) for row in selected)
    expectation = Counter((integer(row["event_expectation_plus3"]), integer(row["mat_expect_code"])) for row in selected)
    return {"total_rows": len(rows), "strict_rows": len(selected), "not_strict_rows": len(rows) - len(selected),
            "action_matrix": [[action[(left, right)] for right in (1, 2)] for left in (1, 2)],
            "expectation_matrix": [[expectation[(left, right)] for right in (3, 4, 5, 6)] for left in (3, 4, 5, 6)],
            "action_disagreements": sum(count for (left, right), count in action.items() if left != right),
            "expectation_disagreements": sum(count for (left, right), count in expectation.items() if left != right)}


def independent_counts(rows, links):
    require(len({row["event_label"] for row in links}) == len(links) and
            len({row["mat_row"] for row in links}) == len(links), "Parent links are not one-to-one")
    require(all(row["event_label"] and row["mat_row"] for row in links), "Empty parent link label")
    pairs = {(row["event_label"], row["mat_row"]) for row in links}
    seen, positions, groups = set(), set(), defaultdict(list)
    for row in rows:
        pair, task = (row["event_label"], row["mat_row"]), row["task"]
        require(pair in pairs and task in ("REW", "PUN"), "Unknown pair/context")
        strict = row["strict_comparable"]
        require(strict in ("True", "False"), "Invalid strict-comparable flag")
        ordinal, event_row = integer(row["feedback_ordinal"]), integer(row["feedback_event_row"])
        require(ordinal >= 1 and event_row >= 2, "Invalid original position")
        key = (*pair, task, ordinal)
        position = (pair[0], event_row)
        require(key not in seen and position not in positions, "Duplicate parent diagnostic position")
        seen.add(key)
        positions.add(position)
        integer(row["mat_action_code"], {1, 2})
        integer(row["mat_expect_code"], {3, 4, 5, 6})
        for field, alphabet in (("event_choice_side_code", {1, 2}), ("event_expectation_plus3", {3, 4, 5, 6})):
            if row[field] or strict == "True":
                integer(row[field], alphabet)
        for flag, left, right in (("action_numeric_agreement", "event_choice_side_code", "mat_action_code"),
                                  ("expectation_numeric_agreement", "event_expectation_plus3", "mat_expect_code")):
            expected = str(row[left] == row[right]) if strict == "True" else ""
            require(row[flag] == expected, "Stored agreement flag disagrees with original codes/strictness")
        groups[(*pair, task)].append(row)
    require(set(groups) == {(*pair, task) for pair in pairs for task in ("REW", "PUN")},
            "Missing linked pair/context group")
    for group in groups.values():
        require(sorted(integer(row["feedback_ordinal"]) for row in group) == list(range(1, len(group) + 1)),
                "Parent feedback ordinal gaps")
    total = count_table(rows)
    total.update(groups=[{"event_label": key[0], "mat_row": key[1], "task": key[2], **count_table(groups[key])}
                         for key in sorted(groups)], mapping_selected=False, behavioral_performance_computed=False)
    require(sum(group["total_rows"] for group in total["groups"]) == total["total_rows"], "Group totals differ")
    require(sum(sum(line) for line in total["action_matrix"]) == total["strict_rows"] and
            sum(sum(line) for line in total["expectation_matrix"]) == total["strict_rows"], "Cross-tab denominator differs")
    return total


def verify(output):
    output = Path(output).resolve()
    names = {"metadata.json", "registration.json", "code-cross-tabs.json"}
    artifacts = read_json(output / "artifact_hashes.json")
    require(set(artifacts) == names, "Primary artifact set differs")
    for name, value in artifacts.items():
        require(sha(safe(output, name)) == value, f"Artifact hash differs: {name}")
    meta, reg = read_json(output / "metadata.json"), read_json(output / "registration.json")
    require(set(meta) == {"started_at_utc", "python", "status", "registration_sha256", "source_root", "completed_at_utc"},
            "Metadata fields/scope differ")
    require(set(reg) == {"config", "config_sha256", "source_hashes", "input_tag_commit", "git_commit",
                         "registered_at_utc", "notice"}, "Registration fields differ")
    require(meta["status"] == "completed" and isinstance(meta["python"], str) and meta["python"], "Execution incomplete")
    require(meta["registration_sha256"] == sha(output / "registration.json"), "Registration copy hash differs")
    stamps = [datetime.fromisoformat(reg["registered_at_utc"]), datetime.fromisoformat(meta["started_at_utc"]),
              datetime.fromisoformat(meta["completed_at_utc"])]
    require(all(stamp.tzinfo is not None for stamp in stamps) and stamps[0] <= stamps[1] <= stamps[2], "Audit chronology differs")
    require(reg["notice"] == "descriptive follow-up to already observed code disagreements; no blinded hypothesis preregistration",
            "Descriptive scope notice differs")
    require(all(re.fullmatch(r"[a-f0-9]{40}", reg[key]) for key in ("git_commit", "input_tag_commit")), "Invalid source commit")
    root = (output / meta["source_root"]).resolve()
    config_path = safe(root, "configs/human_semantics.json")
    require(exact(read_json(config_path), CONFIG) and exact(reg["config"], CONFIG), "Fixed scope/config differs")
    require(sha(config_path) == reg["config_sha256"], "Config hash differs")
    required_frozen = set(FROZEN)
    evidence_path = root / "data/human/stolz_2022_semantics_evidence/source-manifest.json"
    evidence = read_json(evidence_path)
    evidence_files = [item["path"] for item in evidence["files"]]
    require(len(evidence_files) == len(set(evidence_files)), "Duplicate evidence file paths")
    for item in evidence["files"]:
        path = safe(evidence_path.parent, item["path"])
        require(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"], "Evidence file bytes differ")
        required_frozen.add(path.relative_to(root).as_posix())
    frozen = reg["source_hashes"]
    require(set(frozen) == required_frozen, "Frozen file coverage differs")
    for name, value in frozen.items():
        require(sha(safe(root, name)) == value, f"Frozen source hash differs: {name}")
    tag_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", CONFIG["input_tag"]], text=True).strip()
    require(tag_commit == reg["input_tag_commit"], "Immutable parent tag commit differs")
    for name in INPUTS:
        original = subprocess.check_output(["git", "-C", str(root), "show", f"{tag_commit}:{name}"])
        require(original == safe(root, name).read_bytes(), f"Parent input differs from tag bytes: {name}")
    prior_verification = read_json(root / PARENT / "verification.json")
    require(prior_verification["status"] == "passed", "Parent verification did not pass")
    parent_artifacts = read_json(root / PARENT / "artifact_hashes.json")
    # Parent files actually consumed here must still agree with its pinned index.
    for filename in ("corroboration.csv", "record-links.csv", "linkage-report.json", "metadata.json", "registration.json"):
        require(parent_artifacts.get(filename) == sha(root / PARENT / filename), "Parent artifact hash differs")
    rows = source_csv(root / PARENT / "corroboration.csv", FIELDS)
    links = source_csv(root / PARENT / "record-links.csv", ("event_label", "mat_row"))
    expected = independent_counts(rows, links)
    parent_counts = read_json(root / PARENT / "linkage-report.json")["corroboration_counts"]
    for key, parent_key in (("total_rows", "feedback_anchored_pairs"), ("strict_rows", "strict_comparable"),
                            ("action_disagreements", "action_numeric_disagreements"),
                            ("expectation_disagreements", "expectation_numeric_disagreements")):
        require(expected[key] == parent_counts.get(parent_key, 0), "Parent diagnostic counts differ")
    expected.update(audit_id=CONFIG["audit_id"],
                    code_axes={"action_rows": [1, 2], "action_columns": [1, 2],
                               "expectation_rows_plus3": [3, 4, 5, 6], "expectation_columns_mat": [3, 4, 5, 6]},
                    interpretation="cross-tabs diagnose codes; no selected permutation, cause attribution, original trial identity or human/model comparison")
    require(exact(read_json(output / "code-cross-tabs.json"), expected), "Cross-tab arithmetic/axes/scope differ")
    return {"status": "passed", "scope": "independent_fixed_code_counts_no_mapping_selection_or_performance_analysis",
            "primary_artifacts_verified": len(names), "parent_tag_commit": tag_commit,
            "immutable_parent_files_verified": len(INPUTS), "frozen_files_verified": len(frozen),
            "evidence_files_hash_verified": len(evidence_files), "diagnostic_rows_verified": len(rows),
            "strict_rows_counted": expected["strict_rows"], "non_strict_rows_preserved": expected["not_strict_rows"],
            "linked_pair_context_groups_verified": len(expected["groups"]), "errors": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.output)
    except Exception as error:
        result = {"status": "failed", "scope": "independent_fixed_code_counts_no_mapping_selection_or_performance_analysis",
                  "errors": [f"{type(error).__name__}: {error}"]}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "errors")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
