"""Synthetic parent artifacts and tamper checks, never original human records."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import pytest
from drosolearn.human_semantics import FIELDS, INPUTS, PARENT, register, run

REPO = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def synthetic_rows():
    rows = []
    values = [("1", "1", "3", "3", "True"), ("2", "1", "4", "5", "True"), ("", "1", "", "6", "False"),
              ("1", "2", "5", "4", "True"), ("2", "2", "6", "6", "True"), ("1", "1", "3", "3", "False")]
    for index, (ea, ma, ee, me, strict) in enumerate(values):
        row = dict(zip(FIELDS, ["s-a", "m-a", "REW" if index < 3 else "PUN", str(index % 3 + 1), str(index * 3 + 5), strict,
                                ea, ma, ee, me, str(ea == ma) if strict == "True" else "", str(ee == me) if strict == "True" else ""]))
        rows.append(row)
    return rows


def audit_fixture(tmp_path):
    root = tmp_path / "invented_repo"
    root.mkdir()
    (root / ".gitattributes").write_bytes(b"* -text\n")
    selected = ["src/drosolearn/human_semantics.py", "src/drosolearn/human_trial_data.py", "scripts/verify_human_semantics.py",
                "tests/test_human_semantics.py", "tests/test_human_semantics_run.py", "docs/human-semantics-protocol.md",
                "docs/human-semantics-source-review.md", "requirements-lock.txt", "pyproject.toml", "configs/human_semantics.json"]
    for relative in selected:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    parent = root / PARENT
    parent.mkdir(parents=True)
    with (parent / "corroboration.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(synthetic_rows())
    (parent / "record-links.csv").write_text("event_label,mat_row\ns-a,m-a\n", encoding="utf-8")
    save_json(parent / "linkage-report.json", {"corroboration_counts": {"feedback_anchored_pairs": 6, "strict_comparable": 4,
                                                                       "action_numeric_disagreements": 2, "expectation_numeric_disagreements": 2}})
    for name, content in (("metadata.json", {"status": "completed"}), ("registration.json", {}),
                          ("artifact_hashes.json", {}), ("verification.json", {"status": "passed"})):
        save_json(parent / name, content)
    save_json(parent / "artifact_hashes.json", {name: hashlib.sha256((parent / name).read_bytes()).hexdigest()
              for name in ("corroboration.csv", "record-links.csv", "linkage-report.json", "metadata.json", "registration.json")})
    evidence = root / "data/human/stolz_2022_semantics_evidence"
    save_json(evidence / "invented.json", {"synthetic": True})
    raw = (evidence / "invented.json").read_bytes()
    save_json(evidence / "source-manifest.json", {"files": [{"path": "invented.json", "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]})
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Synthetic", "-c", "user.email=test@example.invalid", "commit", "-qm", "synthetic parent"], cwd=root, check=True)
    subprocess.run(["git", "tag", "human-trial-linkage-v1"], cwd=root, check=True)
    cfg, reg, out = root / "configs/human_semantics.json", root / "results/registration.json", root / "results/cross_tabs"
    register(root, cfg, reg)
    return root, cfg, reg, out


def verifier():
    spec = importlib.util.spec_from_file_location("independent_human_semantics", REPO / "scripts/verify_human_semantics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rehash(out):
    hashes = json.loads((out / "artifact_hashes.json").read_text())
    for name in hashes:
        hashes[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    save_json(out / "artifact_hashes.json", hashes)


def test_run_and_independent_replay(tmp_path):
    root, cfg, reg, out = audit_fixture(tmp_path)
    report = run(root, cfg, reg, out)
    assert report["total_rows"] == 6 and report["strict_rows"] == 4
    assert report["action_disagreements"] == report["expectation_disagreements"] == 2
    assert not report["mapping_selected"] and not report["behavioral_performance_computed"]
    assert verifier().verify(out)["status"] == "passed"


def test_existing_output_and_registration_not_overwritten(tmp_path):
    root, cfg, reg, out = audit_fixture(tmp_path)
    run(root, cfg, reg, out)
    with pytest.raises(FileExistsError):
        run(root, cfg, reg, out)
    with pytest.raises(FileExistsError):
        register(root, cfg, reg)


@pytest.mark.parametrize("relative", ["docs/human-semantics-protocol.md", PARENT + "corroboration.csv"])
def test_source_change_rejected_before_output(tmp_path, relative):
    root, cfg, reg, out = audit_fixture(tmp_path)
    with (root / relative).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed after freeze"):
        run(root, cfg, reg, out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["matrix", "group", "scope", "axis", "count"])
def test_rehashed_result_tampering_rejected(tmp_path, change):
    root, cfg, reg, out = audit_fixture(tmp_path)
    run(root, cfg, reg, out)
    path = out / "code-cross-tabs.json"
    report = json.loads(path.read_text())
    if change == "matrix":
        report["action_matrix"][0][0] += 1
    elif change == "group":
        report["groups"].pop()
    elif change == "scope":
        report["mapping_selected"] = True
    elif change == "axis":
        report["code_axes"]["action_rows"].reverse()
    else:
        report["strict_rows"] = True
    save_json(path, report)
    rehash(out)
    with pytest.raises((ValueError, AssertionError)):
        verifier().verify(out)


def test_parent_tag_retarget_rejected(tmp_path):
    root, cfg, reg, out = audit_fixture(tmp_path)
    subprocess.run(["git", "-c", "user.name=Synthetic", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "later"], cwd=root, check=True)
    subprocess.run(["git", "tag", "-f", "human-trial-linkage-v1"], cwd=root, check=True, capture_output=True)
    with pytest.raises(ValueError, match="changed after freeze"):
        run(root, cfg, reg, out)
