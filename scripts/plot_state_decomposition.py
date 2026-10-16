"""Illustrate the fixed static decomposition without importing any learner.

The figures show registered full-branch/final-checkpoint/IID summaries only.
All inputs are checked against the independent report and artifact inventory;
outputs are exclusive files in a new sibling directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


MODELS = ["linear", "real", "rewired", "ann"]
BRANCHES = ["full", "head_only", "internal_only", "frozen"]
CROSSES = ("00", "10", "01", "11")
RAW_FIELDS = ("aligned_head_term", "aligned_internal_term", "aligned_cross_term", "aligned_total_raw_gap_change")
PLOT_FIELDS = (*RAW_FIELDS, *("p_rule_" + cross for cross in CROSSES))
SCRIPT_RELATIVE = "scripts/plot_state_decomposition.py"
INPUTS = ("metadata.json", "registration.json", "summary.json", "verification.json", "artifact_hashes.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "Duplicate plot-input JSON key")
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Nonfinite plot-input JSON value")))


def safe_child(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Safe relative artifact name required")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes result directory")
    return path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(first, second, tolerance=1e-12):
    return finite(first) and finite(second) and math.isclose(first, second, rel_tol=0, abs_tol=tolerance)


def validate_inputs(result_dir):
    result_dir = Path(result_dir).resolve()
    meta, reg, summary, checked, artifacts = (read_json(result_dir / name) for name in INPUTS)
    require(meta["status"] == "completed" and checked["status"] == "passed" and checked["errors"] == [],
            "A completed, independently passed static diagnostic is required")
    cfg = meta["config"]
    require(cfg["study_id"] == meta["study_id"] == reg["study_id"] == summary["study_id"] == checked["study_id"]
            == "state_decomposition_v1" and reg["config"] == cfg, "Exact static diagnostic identity/configuration required")
    require(cfg["models"] == MODELS and cfg["branches"] == BRANCHES
            and cfg["scope"] in ("registered_descriptive_diagnostic", "integration_test")
            and type(cfg["seeds"]) is list and cfg["seeds"] and len(set(cfg["seeds"])) == len(cfg["seeds"]),
            "All original models, branches and registered seeds are required")
    if cfg["scope"] == "registered_descriptive_diagnostic":
        require(cfg["seeds"] == list(range(12001, 12011)), "All ten original computational seeds are required")
    for key in ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws"):
        require(type(cfg[key]) is int and cfg[key] == 0 and meta[key] == 0, "This static analysis cannot add learning or generated data")
    require(meta["behavior_hypotheses_tested"] == summary["behavior_hypotheses_tested"] == 0
            and summary["arithmetic_identity_passed"] is True, "Arithmetic verification, not a new behavior hypothesis, is required")
    require(checked["summary_sha256"] == digest(result_dir / "summary.json")
            and checked["artifact_hashes_sha256"] == digest(result_dir / "artifact_hashes.json")
            and checked["registration_sha256"] == meta["registration_sha256"] == digest(result_dir / "registration.json")
            and checked["config_sha256"] == meta["config_sha256"] == reg["config_sha256"],
            "Independent verification is not bound to these exact static-input bytes")
    manifest_bytes = json.dumps(reg["input_files"], ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf8")
    require(meta["input_files"] == reg["input_files"]
            and checked["input_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
            "The independently checked selected NPZ inventory differs")
    require(meta["frozen_files"] == reg["frozen_files"]
            and meta["frozen_files"][SCRIPT_RELATIVE] == digest(__file__), "The plotting source differs from its prospective freeze")
    timestamp = datetime.fromisoformat(checked["verified_at_utc"])
    require(timestamp.tzinfo is not None and timestamp.utcoffset().total_seconds() == 0
            and timestamp <= datetime.now(timezone.utc), "Prior independent UTC verification required")
    for relative, expected in artifacts.items():
        require(type(expected) is str and re.fullmatch(r"[a-f0-9]{64}", expected), "Valid primary artifact hash required")
        require(digest(safe_child(result_dir, relative)) == expected, "Static primary artifact changed after verification")
    require(artifacts["summary.json"] == checked["summary_sha256"]
            and artifacts["registration.json"] == checked["registration_sha256"], "Missing primary plot-input binding")

    expected_keys = {(seed, model, branch, trial, split) for seed in cfg["seeds"] for model in MODELS
                     for branch in BRANCHES for trial in cfg["checkpoints"] for split in ("iid", "ood")}
    by_seed = {}
    for row in summary["seed_summaries"]:
        key = (row["seed"], row["model"], row["branch"], row["trial"], row["split"])
        require(key not in by_seed and row["cases"] == cfg["eval_trials"], "Duplicated or incomplete per-seed summary")
        require(all(finite(row["means"][field]) for field in PLOT_FIELDS), "Finite plot terms and crossed probabilities required")
        require(all(0 <= row["means"]["p_rule_" + cross] <= 1 for cross in CROSSES), "Crossed p_rule must remain in [0,1]")
        require(close(sum(row["means"][field] for field in RAW_FIELDS[:3]), row["means"][RAW_FIELDS[3]], cfg["raw_residual_tolerance"]),
                "Target-aligned raw terms do not sum to the total change")
        require(row["maximum_absolute_bilinear_residual"] <= cfg["raw_residual_tolerance"]
                and row["maximum_absolute_raw_interaction_residual"] <= cfg["raw_residual_tolerance"], "Recorded case residual exceeds its fixed tolerance")
        by_seed[key] = row
    require(set(by_seed) == expected_keys and sum(row["cases"] for row in by_seed.values()) == cfg["cases"] == summary["cases"] == meta["cases"],
            "All original cases and contexts must remain in the static summary")
    require(summary["crossed_policy_values"] == meta["crossed_policy_values"] == cfg["crossed_policy_values"], "Crossed-state coverage differs")
    grouped = {}
    for row in summary["cells"]:
        key = (row["model"], row["branch"], row["trial"], row["split"])
        require(key not in grouped and row["n_seeds"] == len(cfg["seeds"])
                and row["cases"] == len(cfg["seeds"]) * cfg["eval_trials"], "Complete all-seed grouped summaries required")
        values = [by_seed[seed, *key] for seed in cfg["seeds"]]
        for field in PLOT_FIELDS:
            data = np.array([value["means"][field] for value in values])
            require(close(row["statistics"][field]["mean"], float(data.mean())), "Grouped plot mean differs from the all-seed values")
            sd = row["statistics"][field]["sample_sd_across_seed_means"]
            require(sd is None if len(data) == 1 else close(sd, float(data.std(ddof=1))), "Grouped sample SD differs from the all-seed values")
        grouped[key] = row
    require(set(grouped) == {key[1:] for key in expected_keys}, "Every group, including failed-learning contexts, must be retained")
    return {"root": result_dir, "config": cfg, "by_seed": by_seed, "grouped": grouped,
            "input_hashes": {name: digest(result_dir / name) for name in INPUTS}}


def labels():
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    korean = next((name for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic") if name in available), None)
    plt.rcParams.update({"font.family": korean or "DejaVu Sans", "axes.unicode_minus": False, "font.size": 10})
    if korean:
        return {"language": "ko", "models": {"linear": "단순 선형 · 내부 연결 없음", "real": "실제 부분 배선",
            "rewired": "차수 보존 재배선", "ann": "일반 희소망"},
            "raw_title": "이미 저장된 계산 점수 변화를 세 항으로 표현하기",
            "selection": "둘 다 학습한 분기 · 반전 최종 상태 · IID · 모든 기존 시드",
            "raw_fields": ["출력 항", "내부 항", "교차 항", "전체 변화"], "raw_ylabel": "작은 수량 방향으로 정렬한 원점수 변화",
            "raw_footer": "점: 시드별 사례 평균 · 막대/큰 표식: 전체 시드 평균 · 앞 세 항의 합 = 전체 변화\n원점수의 정적 산술항이며 확률 향상의 기여율·학습의 인과 효과가 아님",
            "policy_title": "공통·최종 상태를 네 가지로 조합한 선택 확률",
            "head0": "공통 출력 h0", "head1": "최종 출력 ht", "internal_axis": ["공통 내부 W0", "최종 내부 Wt"],
            "policy_ylabel": "같은 작은 수량 기준의 선택 확률", "policy_footer": "얇은 선: 같은 시드의 정적 조합 · 굵은 선: 전체 시드 평균\n10·01은 실제 출력만/내부만 학습한 분기의 결과가 아님 · 새 학습 0회\n네 확률의 교차 차이는 sigmoid/clip 뒤의 값이며 원점수 교차항과 다른 양"}
    return {"language": "en", "models": {"linear": "Linear: no internal synapses", "real": "Real subgraph",
        "rewired": "Degree-preserving rewiring", "ann": "Sparse ANN"},
        "raw_title": "Static algebra of changes in saved decision scores",
        "selection": "Full branch; final reversal checkpoint; IID; every original seed",
        "raw_fields": ["Head term", "Internal term", "Cross term", "Total change"], "raw_ylabel": "Smaller-target-aligned raw score change",
        "raw_footer": "Dots: seed-level case means; bars/large marker: all-seed means; first three terms sum to total\nStatic arithmetic terms, not causal learning contributions or fractions of probability improvement.",
        "policy_title": "Choice probabilities from four static parameter combinations",
        "head0": "Common head h0", "head1": "Final head ht", "internal_axis": ["Common internal W0", "Final internal Wt"],
        "policy_ylabel": "Probability of choosing the same smaller target",
        "policy_footer": "Thin lines: paired static combinations; thick lines: all-seed means\n10/01 are not trained head-only/internal-only branches; no new learning\nThe probability interaction after sigmoid/clip differs from the raw bilinear cross term."}


def save_new(fig, path):
    with Path(path).open("xb") as handle:
        fig.savefig(handle, format="png", dpi=160, facecolor="white")
    plt.close(fig)


def endpoint_rows(bundle, model):
    cfg = bundle["config"]
    return [bundle["by_seed"][seed, model, "full", cfg["checkpoints"][-1], "iid"]["means"] for seed in cfg["seeds"]]


def raw_terms_figure(bundle, text, path):
    cfg = bundle["config"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), sharey=True)
    palette = ["#2377a4", "#368770", "#d08336"]
    jitter = np.linspace(-.08, .08, len(cfg["seeds"]))
    all_values = np.array([[row[field] for field in RAW_FIELDS] for model in MODELS for row in endpoint_rows(bundle, model)])
    low, high = min(0., float(all_values.min())), max(0., float(all_values.max()))
    padding = .12 * max(high - low, .1)
    for ax, model in zip(axes.flat, MODELS):
        values = np.array([[row[field] for field in RAW_FIELDS] for row in endpoint_rows(bundle, model)])
        means = values.mean(axis=0)
        ax.bar(np.arange(3), means[:3], color=palette, alpha=.78, width=.58)
        for index in range(4):
            ax.scatter(index + jitter, values[:, index], s=15, color="#353b45", alpha=.48, zorder=3)
        ax.scatter([3], [means[3]], color="#10151d", marker="D", s=60, zorder=4)
        ax.axhline(0, color="#535963", lw=.8)
        ax.set_xticks(range(4), text["raw_fields"])
        ax.set_ylim(low - padding, high + padding)
        ax.set_title(text["models"][model])
        ax.grid(axis="y", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(text["raw_title"], fontsize=15, y=.985)
    fig.text(.5, .94, text["selection"] + f" (n={len(cfg['seeds'])})", ha="center", fontsize=10)
    fig.supylabel(text["raw_ylabel"], x=.012)
    fig.text(.5, .035, text["raw_footer"], ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=.87, bottom=.18, left=.08, right=.98, hspace=.35, wspace=.18)
    save_new(fig, path)


def crossed_probabilities_figure(bundle, text, path):
    cfg = bundle["config"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.1), sharey=True)
    head_levels = [(False, ("00", "01"), "#7b818b", "s", text["head0"]),
                   (True, ("10", "11"), "#2377a4", "o", text["head1"])]
    for ax, model in zip(axes.flat, MODELS):
        rows = endpoint_rows(bundle, model)
        for current, crosses, color, marker, name in head_levels:
            x = np.array([0., 1.]) + (.025 if current else -.025)
            data = np.array([[row["p_rule_" + cross] for cross in crosses] for row in rows])
            for values in data:
                ax.plot(x, values, color=color, alpha=.25, lw=.75, marker=marker, ms=3)
            ax.plot(x, data.mean(axis=0), color=color, lw=2.8, marker=marker, ms=7, label=name)
        ax.set_title(text["models"][model])
        ax.set_xticks([0, 1], text["internal_axis"])
        ax.set_xlim(-.25, 1.25)
        ax.set_ylim(0, 1)
        ax.axhline(.5, color="#cbd0d5", lw=.6)
        ax.grid(axis="y", alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(text["policy_title"], fontsize=15, y=.985)
    fig.text(.5, .94, text["selection"] + f" (n={len(cfg['seeds'])})", ha="center", fontsize=10)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .14), ncol=2, frameon=False)
    fig.supylabel(text["policy_ylabel"], x=.012)
    fig.text(.5, .025, text["policy_footer"], ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=.87, bottom=.24, left=.08, right=.98, hspace=.35, wspace=.18)
    save_new(fig, path)


def create_figures(result_dir, out):
    result_dir, out = Path(result_dir).resolve(), Path(out).resolve()
    require(out.parent == result_dir.parent and out != result_dir, "Figures require a new separate sibling directory")
    require(not out.exists(), "Figure output already exists; refusing overwrite")
    bundle, text = validate_inputs(result_dir), labels()
    metadata = {"study_id": "state_decomposition_v1", "input_hashes": bundle["input_hashes"],
        "plot_script_sha256": digest(__file__), "figure_language": text["language"],
        "selection": {"models": MODELS, "branch": "full", "trial": bundle["config"]["checkpoints"][-1],
                      "split": "iid", "seeds": bundle["config"]["seeds"], "all_original_seeds": True},
        "scope": "Target-aligned raw terms and four static crossed-state probabilities; seed case means and all-seed means; no new learning, hypothesis test or causal contribution claim"}
    out.mkdir(parents=True, exist_ok=False)
    try:
        raw_terms_figure(bundle, text, out / "raw_gap_terms.png")
        crossed_probabilities_figure(bundle, text, out / "crossed_probabilities.png")
        require(bundle["input_hashes"] == {name: digest(result_dir / name) for name in INPUTS}, "Static input changed during plotting")
        metadata.update(status="completed", created_at_utc=datetime.now(timezone.utc).isoformat(),
            figure_hashes={name: digest(out / name) for name in ("raw_gap_terms.png", "crossed_probabilities.png")})
    except Exception as error:
        plt.close("all")
        metadata.update(status="failed", created_at_utc=datetime.now(timezone.utc).isoformat(),
                        error_type=type(error).__name__, error_message=str(error),
                        figure_hashes={path.name: digest(path) for path in out.glob("*.png") if path.is_file()})
        with (out / "metadata.json").open("x", encoding="utf8", newline="\n") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        raise
    with (out / "metadata.json").open("x", encoding="utf8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = create_figures(args.result_dir, args.out)
    print(json.dumps({"status": result["status"], "figures": list(result["figure_hashes"]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
