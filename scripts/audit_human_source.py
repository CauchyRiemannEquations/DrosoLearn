"""Read-only audit of two licensed PLOS supporting XLSX files.

Uses only Python's standard library; never edits the original workbooks.
Run from the DrosoLearn repository: python scripts/audit_human_source.py
Use --output for a new audit JSON. Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "human" / "honore_noel_2016"


def read_xlsx(path: Path) -> dict:
    """Read literal and cached cell values plus source formulas, not execute them."""
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        sheets = {}
        for sheet in workbook.find("m:sheets", NS):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = targets[rid]
            xml_path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            xml = ET.fromstring(archive.read(xml_path))
            cells, formulas = {}, {}
            for c in xml.findall(".//m:sheetData/m:row/m:c", NS):
                address = c.attrib["r"]
                typ = c.attrib.get("t")
                value_node = c.find("m:v", NS)
                raw = value_node.text if value_node is not None else None
                if typ == "s" and raw is not None:
                    value = shared[int(raw)]
                elif typ == "inlineStr":
                    value = "".join(c.find("m:is", NS).itertext())
                elif raw is None:
                    value = None
                elif typ in ("str", "e"):
                    value = raw
                else:
                    value = float(raw)
                    if value.is_integer():
                        value = int(value)
                cells[address] = value
                formula = c.find("m:f", NS)
                if formula is not None:
                    formulas[address] = formula.text
            sheets[sheet.attrib["name"]] = {"range": xml.find("m:dimension", NS).attrib["ref"], "cells": cells, "formulas": formulas}
        return sheets


def audit() -> dict:
    manifest = json.loads((DATA / "source-manifest.json").read_text(encoding="utf-8"))
    results = {
        "scope": "Source readiness and published exclusion-count check. No cross-species comparison or full ANOVA reproduction.",
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inclusion_for_raw_descriptives": "All 56 source rows; source-coded group labels retained without inferred training-label mapping.",
        "published_exclusion_rule": "One pass: exclude participant if overall score is below own group mean minus two sample SD at T1 OR T2, separately per task.",
        "published_exclusion_rule_source": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0166685 (Results, Analyses)",
        "datasets": [],
    }
    participant_group_keys = None
    for item in manifest["sources"]:
        path = DATA / item["original_filename"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {path.name}")
        workbook = read_xlsx(path)
        if set(workbook) != {"T1", "T2"}:
            raise ValueError("Unexpected source sheets")
        is_comparison = path.name.endswith("s002.xlsx")
        cols = list("ABCDEFGHIJKLMNO" if is_comparison else "ABCDEFGHI")
        measured_cols = list("DEFGHIJK" if is_comparison else "DE")
        zcol, mean_col, sdcol = ("M", "N", "O") if is_comparison else ("G", "H", "I")
        dataset = {"original_filename": path.name, "sha256": digest, "doi": item["doi"], "granularity": "participant x measurement time x aggregate condition", "sheets": [], "original_rule_excluded_ids": []}
        prior_ids = None
        excluded_ids = set()
        keys = None
        for name in ("T1", "T2"):
            source = workbook[name]
            cell = source["cells"]
            rows = [{c: cell.get(f"{c}{r}") for c in cols} for r in range(2, 58)]
            ids = [r["A"] for r in rows]
            if None in ids or len(set(ids)) != 56:
                raise ValueError("Missing or duplicate participant IDs")
            if prior_ids is not None and set(prior_ids) != set(ids):
                raise ValueError("T1/T2 participant ID mismatch")
            prior_ids = ids
            current_keys = {(r["A"], r["B"]) for r in rows}
            if keys is not None and keys != current_keys:
                raise ValueError("T1/T2 group assignment mismatch")
            keys = current_keys
            score_error = max(abs(r["C"] - mean(r[c] for c in measured_cols)) for r in rows)
            if score_error > 1e-12:
                raise ValueError("Overall score does not match balanced condition means")
            raw_descriptives, cached_mean_error, cached_sd_error, cached_z_error = {}, 0.0, 0.0, 0.0
            for code in sorted(set(r["B"] for r in rows)):
                group = [r for r in rows if r["B"] == code]
                scores = [r["C"] for r in group]
                m, sd = mean(scores), stdev(scores)
                raw_descriptives[str(code)] = {"n": len(group), "mean_accuracy_all_rows": m, "sample_sd_all_rows": sd}
                for row in group:
                    z = (row["C"] - m) / sd
                    cached_mean_error = max(cached_mean_error, abs(m - row[mean_col]))
                    cached_sd_error = max(cached_sd_error, abs(sd - row[sdcol]))
                    cached_z_error = max(cached_z_error, abs(z - row[zcol]))
                    if (z < -2) != (row[zcol] < -2):
                        raise ValueError("Source and independent exclusion classifications disagree")
                    if z < -2:
                        excluded_ids.add(row["A"])
            if max(cached_mean_error, cached_sd_error, cached_z_error) > 1e-12:
                raise ValueError("Source cached summary formula values disagree with independent arithmetic")
            schema = []
            for c in cols:
                values = [r[c] for r in rows]
                numeric = [v for v in values if isinstance(v, (int, float))]
                schema.append({"column": c, "header": cell.get(f"{c}1"), "missing_values": sum(v is None for v in values), "numeric_values": len(numeric), "minimum": min(numeric) if numeric else None, "maximum": max(numeric) if numeric else None, "formula_cells": sum(f"{c}{r}" in source["formulas"] for r in range(2, 58))})
            dataset["sheets"].append({"sheet": name, "used_range": source["range"], "participants": 56, "group_code_counts": dict(Counter(str(r["B"]) for r in rows)), "column_schema": schema, "raw_descriptives": raw_descriptives, "overall_score_condition_mean_max_abs_error": score_error, "source_formula_mean_max_abs_error": cached_mean_error, "source_formula_sd_max_abs_error": cached_sd_error, "source_formula_z_max_abs_error": cached_z_error})
        if participant_group_keys is not None and keys != participant_group_keys:
            raise ValueError("Participant/group keys differ between S1/S6 tasks")
        participant_group_keys = keys
        excluded_by_group = Counter(str(group) for identifier, group in keys if identifier in excluded_ids)
        expected_count = 3 if is_comparison else 1
        dataset["original_rule_excluded_ids"] = sorted(excluded_ids)
        dataset["original_rule_excluded_count"] = len(excluded_ids)
        dataset["original_rule_excluded_by_source_group_code"] = dict(excluded_by_group)
        dataset["paper_reported_excluded_count"] = expected_count
        dataset["excluded_count_matches_paper"] = len(excluded_ids) == expected_count
        dataset["original_rule_retained_count"] = 56 - len(excluded_ids)
        if not dataset["excluded_count_matches_paper"]:
            raise ValueError("Published excluded participant count not reproduced")
        results["datasets"].append(dataset)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON output path; refuses to overwrite")
    args = parser.parse_args()
    report = audit()
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        print(f"Audit saved: {args.output}")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
