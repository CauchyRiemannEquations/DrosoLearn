"""Fixed code cross-tabs from an immutable prior linkage; select no new mapping."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import os
import platform
import subprocess

from .human_trial_data import digest, write_json

FIELDS = ("event_label", "mat_row", "task", "feedback_ordinal", "feedback_event_row", "strict_comparable",
          "event_choice_side_code", "mat_action_code", "event_expectation_plus3", "mat_expect_code",
          "action_numeric_agreement", "expectation_numeric_agreement")
PARENT = "results/human_trial_linkage_v1/"
INPUTS = tuple(PARENT + name for name in ("corroboration.csv", "record-links.csv", "linkage-report.json", "metadata.json", "registration.json", "artifact_hashes.json", "verification.json"))


def _int(value, allowed=None):
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or str(int(value)) != value:
        raise ValueError("Canonical integer code required")
    number = int(value)
    if allowed is not None and number not in allowed:
        raise ValueError("Code outside the fixed alphabet")
    return number


def _counts(rows):
    action = [[0, 0], [0, 0]]
    expectation = [[0] * 4 for _ in range(4)]
    strict = 0
    for row in rows:
        if row["strict_comparable"] == "True":
            strict += 1
            action[int(row["event_choice_side_code"]) - 1][int(row["mat_action_code"]) - 1] += 1
            expectation[int(row["event_expectation_plus3"]) - 3][int(row["mat_expect_code"]) - 3] += 1
    return {"total_rows": len(rows), "strict_rows": strict, "not_strict_rows": len(rows) - strict,
            "action_matrix": action, "expectation_matrix": expectation,
            "action_disagreements": strict - sum(action[i][i] for i in range(2)),
            "expectation_disagreements": strict - sum(expectation[i][i] for i in range(4))}


def summarize(rows, links):
    if not rows or not links:
        raise ValueError("Nonempty fixed linkage inputs required")
    if any(set(link) != {"event_label", "mat_row"} or not all(isinstance(v, str) and v for v in link.values()) for link in links):
        raise ValueError("Exact nonempty link fields required")
    if len({r["event_label"] for r in links}) != len(links) or len({r["mat_row"] for r in links}) != len(links):
        raise ValueError("Links must be one-to-one")
    pairs = {(r["event_label"], r["mat_row"]) for r in links}
    groups, keys, source_rows = {}, set(), set()
    for row in rows:
        if set(row) != set(FIELDS) or any(not isinstance(v, str) for v in row.values()):
            raise ValueError("Exact string-valued diagnostic schema required")
        pair = row["event_label"], row["mat_row"]
        if pair not in pairs or row["task"] not in ("REW", "PUN") or row["strict_comparable"] not in ("True", "False"):
            raise ValueError("Unknown pair, context or strict flag")
        ordinal, source_row = _int(row["feedback_ordinal"]), _int(row["feedback_event_row"])
        if ordinal < 1 or source_row < 2:
            raise ValueError("Original positions must be positive")
        key = pair + (row["task"], ordinal)
        source_key = pair[0], source_row
        if key in keys or source_key in source_rows:
            raise ValueError("Duplicate diagnostic position")
        keys.add(key)
        source_rows.add(source_key)
        for field, alphabet in (("mat_action_code", {1, 2}), ("mat_expect_code", {3, 4, 5, 6})):
            _int(row[field], alphabet)
        for field, alphabet in (("event_choice_side_code", {1, 2}), ("event_expectation_plus3", {3, 4, 5, 6})):
            if row[field] or row["strict_comparable"] == "True":
                _int(row[field], alphabet)
        if row["strict_comparable"] == "True":
            for field, left, right in (("action_numeric_agreement", "event_choice_side_code", "mat_action_code"),
                                       ("expectation_numeric_agreement", "event_expectation_plus3", "mat_expect_code")):
                if row[field] != str(row[left] == row[right]):
                    raise ValueError("Stored numeric agreement differs from codes")
        elif row["action_numeric_agreement"] or row["expectation_numeric_agreement"]:
            raise ValueError("Non-strict records must not be scored")
        groups.setdefault(pair + (row["task"],), []).append(row)
    expected_groups = {pair + (task,) for pair in pairs for task in ("REW", "PUN")}
    if set(groups) != expected_groups:
        raise ValueError("Both contexts for every linked pair required")
    for group in groups.values():
        if sorted(int(r["feedback_ordinal"]) for r in group) != list(range(1, len(group) + 1)):
            raise ValueError("Diagnostic ordinals have gaps")
    return {**_counts(rows), "groups": [{"event_label": key[0], "mat_row": key[1], "task": key[2], **_counts(groups[key])} for key in sorted(groups)],
            "mapping_selected": False, "behavioral_performance_computed": False}


def read_csv(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError("Input CSV fields differ")
        return list(reader)


def validate_config(config):
    expected = {"audit_id": "human_semantics_v1", "input_tag": "human-trial-linkage-v1", "input_directory": PARENT.rstrip("/"),
                "scope": "fixed_code_cross_tabs_no_mapping_search", "select_mapping": False, "performance_analysis": False}
    if config != expected or type(config.get("select_mapping")) is not bool or type(config.get("performance_analysis")) is not bool:
        raise ValueError("Fixed descriptive-only configuration required")


def study_hashes(root):
    files = list(INPUTS) + ["src/drosolearn/human_semantics.py", "src/drosolearn/human_trial_data.py",
                            "scripts/verify_human_semantics.py", "tests/test_human_semantics.py", "tests/test_human_semantics_run.py",
                            "docs/human-semantics-protocol.md", "docs/human-semantics-source-review.md", "requirements-lock.txt", "pyproject.toml",
                            "data/human/stolz_2022_semantics_evidence/source-manifest.json"]
    manifest_path = Path(root) / files[-1]
    for item in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]:
        path = (manifest_path.parent / item["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Unsafe evidence path")
        if path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
            raise ValueError("Evidence bytes differ from manifest")
        files.append(path.relative_to(Path(root).resolve()).as_posix())
    return {name: digest(Path(root) / name) for name in sorted(files)}


def check_parent(root, config):
    root = Path(root)
    tag_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", config["input_tag"]], text=True).strip()
    for relative in INPUTS:
        original = subprocess.check_output(["git", "-C", str(root), "show", f"{tag_commit}:{relative}"])
        if original != (root / relative).read_bytes():
            raise ValueError("Parent input differs from immutable tag")
    verification = json.loads((root / PARENT / "verification.json").read_text())
    if verification["status"] != "passed":
        raise ValueError("Parent independent verification did not pass")
    return tag_commit


def register(root, config_path, out):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    record = {"config": config, "config_sha256": digest(config_path), "source_hashes": study_hashes(root),
              "input_tag_commit": check_parent(root, config),
              "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "notice": "descriptive follow-up to already observed code disagreements; no blinded hypothesis preregistration"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_json(out, record)
    return record


def run(root, config_path, registration_path, out):
    root, config_path, registration_path, out = map(lambda p: Path(p).resolve(), (root, config_path, registration_path, out))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reg = json.loads(registration_path.read_text(encoding="utf-8"))
    validate_config(config)
    if config != reg["config"] or digest(config_path) != reg["config_sha256"] or study_hashes(root) != reg["source_hashes"] or check_parent(root, config) != reg["input_tag_commit"]:
        raise ValueError("Inputs or code changed after freeze")
    out.mkdir(parents=True, exist_ok=False)
    meta = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "status": "started",
            "registration_sha256": digest(registration_path), "source_root": Path(os.path.relpath(root, out)).as_posix()}
    write_json(out / "registration.json", reg)
    write_json(out / "metadata.json", meta)
    try:
        rows = read_csv(root / PARENT / "corroboration.csv", FIELDS)
        links = read_csv(root / PARENT / "record-links.csv", ("event_label", "mat_row"))
        report = summarize(rows, links)
        prior = json.loads((root / PARENT / "linkage-report.json").read_text())
        counts = prior["corroboration_counts"]
        if (report["total_rows"], report["strict_rows"], report["action_disagreements"], report["expectation_disagreements"]) != (counts["feedback_anchored_pairs"], counts["strict_comparable"], counts["action_numeric_disagreements"], counts["expectation_numeric_disagreements"]):
            raise ValueError("Cross-tabs do not preserve parent counts")
        report.update(audit_id=config["audit_id"], code_axes={"action_rows": [1, 2], "action_columns": [1, 2],
                                                            "expectation_rows_plus3": [3, 4, 5, 6], "expectation_columns_mat": [3, 4, 5, 6]},
                      interpretation="cross-tabs diagnose codes; no selected permutation, cause attribution, original trial identity or human/model comparison")
        write_json(out / "code-cross-tabs.json", report)
        meta.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        meta.update(status="failed", failure=repr(error), completed_at_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_json(out / "artifact_hashes.json", {path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        record = register(args.root, args.config, args.out)
        print(json.dumps({"registered_at_utc": record["registered_at_utc"], "frozen_files": len(record["source_hashes"])}))
    else:
        if args.registration is None:
            parser.error("--registration required")
        report = run(args.root, args.config, args.registration, args.out)
        print(json.dumps({key: report[key] for key in ("total_rows", "strict_rows", "action_disagreements", "expectation_disagreements")}))


if __name__ == "__main__":
    main()
