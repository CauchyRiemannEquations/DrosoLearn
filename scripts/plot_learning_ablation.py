"""Plot the registered 2x2 ablation after exact independent-verification binding.

No learner is imported or run. Existing inputs/results are read only; two PNGs
and their provenance are created exclusively in a new sibling directory.
"""
from __future__ import annotations

import argparse
import csv
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
INPUTS = ["metadata.json", "registration.json", "artifact_hashes.json", "summary.json",
          "learning_curves.csv", "verification.json"]
SCRIPT_RELATIVE = "scripts/plot_learning_ablation.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "Duplicate JSON key in plot input")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite plot JSON input")))


def safe_child(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Safe relative artifact path required")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes result directory")
    return path


def integer(value):
    require(isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value), "Original integer checkpoint required")
    return int(value)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def validate_inputs(result_dir):
    root = Path(result_dir).resolve()
    meta, registration, checked, summary, artifacts = (
        read_json(root / name) for name in ("metadata.json", "registration.json", "verification.json", "summary.json", "artifact_hashes.json"))
    require(meta["status"] == "completed" and checked["status"] == "passed"
            and checked["errors"] == [], "Completed, independently passed study required")
    cfg = meta["config"]
    require(cfg["study_id"] == registration["study_id"] == meta["study_id"] == summary["study_id"]
            == checked["study_id"] == "learning_ablation_v1", "Only the registered ablation study is supported")
    require(registration["config"] == cfg and cfg["models"] == MODELS and cfg["branches"] == BRANCHES,
            "Exact registered model/branch inventory required")
    require(cfg["scope"] in ("preregistered_diagnostic", "integration_test")
            and type(cfg["seeds"]) is list and cfg["seeds"] and len(set(cfg["seeds"])) == len(cfg["seeds"]),
            "All original registered seeds required")
    if cfg["scope"] == "preregistered_diagnostic":
        require(cfg["seeds"] == list(range(12001, 12011)), "Fixed ten scientific seeds required")
    require(checked["summary_sha256"] == digest(root / "summary.json")
            and checked["learning_curves_sha256"] == digest(root / "learning_curves.csv")
            and checked["artifact_hashes_sha256"] == digest(root / "artifact_hashes.json")
            and checked["registration_sha256"] == meta["registration_sha256"] == digest(root / "registration.json")
            and checked["config_sha256"] == meta["config_sha256"] == registration["config_sha256"],
            "Independent verification is not bound to these exact plot-input bytes")
    require(meta["frozen_files"][SCRIPT_RELATIVE] == registration["frozen_files"][SCRIPT_RELATIVE] == digest(__file__),
            "This plotting source differs from the prospectively frozen version")
    for relative, expected in artifacts.items():
        require(re.fullmatch(r"[a-f0-9]{64}", expected), "Invalid primary artifact hash")
        require(digest(safe_child(root, relative)) == expected, "Primary artifact changed after verification")
    require(artifacts["summary.json"] == checked["summary_sha256"]
            and artifacts["learning_curves.csv"] == checked["learning_curves_sha256"]
            and artifacts["registration.json"] == checked["registration_sha256"], "Missing primary plot-input bindings")
    stamp = datetime.fromisoformat(checked["verified_at_utc"])
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
            and stamp <= datetime.now(timezone.utc), "Prior independent UTC verification required")

    n, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    require(type(n) is int and n > 0 and type(every) is int and every > 0 and n % every == 0,
            "Complete registered checkpoint budget required")
    times = list(range(0, n + 1, every))
    with (root / "learning_curves.csv").open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    curves = {}
    for row in rows:
        key = (integer(row["seed"]), row["model"], row["branch"], row["phase"], integer(row["trial"]), row["split"])
        require(key not in curves, "Duplicate original curve checkpoint")
        probability = float(row["p_rule"])
        require(math.isfinite(probability) and 0 <= probability <= 1, "Finite p_rule in [0,1] required")
        curves[key] = probability
    expected = {(seed, model, branch, phase, t, split) for seed in cfg["seeds"] for model in MODELS
                for branch, phase in [("prefix", "phase1"), *[(branch, "phase2") for branch in BRANCHES]]
                for t in times for split in ("iid", "ood")}
    require(set(curves) == expected, "Every seed/model/prefix/branch/checkpoint/split is required")
    outcomes = {}
    for row in summary["outcomes"]:
        key = (row["seed"], row["model"], row["branch"], row["split"])
        require(key not in outcomes, "Duplicate original endpoint outcome")
        outcomes[key] = row
    require(set(outcomes) == {(seed, model, branch, split) for seed in cfg["seeds"] for model in MODELS
                             for branch in BRANCHES for split in ("iid", "ood")}, "Full endpoint coverage required")
    for (seed, model, branch, split), outcome in outcomes.items():
        first = [curves[seed, model, "prefix", "phase1", t, split] for t in times]
        second = [curves[seed, model, branch, "phase2", t, split] for t in times]
        area = sum((a + b) * .5 * every for a, b in zip(second, second[1:])) / n
        require(all(close(outcome[field], value) for field, value in (
            ("prefix_start", first[0]), ("prefix_end", first[-1]), ("prefix_gain", first[-1] - first[0]),
            ("phase2_start", second[0]), ("phase2_end", second[-1]), ("phase2_gain", second[-1] - second[0]),
            ("phase2_raw_auc", area), ("phase2_baseline_adjusted_auc", area - second[0]))),
            "Plot curves do not match the independently verified summary")
        require(close(first[-1] + second[0], 1), "Cloned boundary must only reverse the evaluation direction")
    return {"root": root, "config": cfg, "curves": curves, "outcomes": outcomes,
            "input_hashes": {name: digest(root / name) for name in INPUTS}}


def labels():
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    korean_font = next((name for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic") if name in available), None)
    plt.rcParams.update({"font.family": korean_font or "DejaVu Sans", "axes.unicode_minus": False, "font.size": 10})
    if korean_font:
        return {"language": "ko", "models": {"linear": "단순 선형 · 내부 연결 없음", "real": "실제 부분 배선",
            "rewired": "차수 보존 재배선", "ann": "일반 희소망"},
            "branches": {"full": "출력+내부 갱신", "head_only": "출력만 갱신", "internal_only": "내부만 갱신", "frozen": "둘 다 중지"},
            "endpoint_title": "같은 출발 상태에서 학습 성분을 켜고 끈 결과",
            "endpoint_description": "반전 학습 종료 · 고정된 IID 평가 · 모든 계산 시드를 짝지어 표시",
            "off": "출력 갱신 중지", "on": "출력 갱신 허용", "internal_axis": ["내부 갱신 중지", "내부 갱신 허용"],
            "probability": "작은 수량을 선택할 확률", "endpoint_footer": "얇은 선: 같은 시드의 짝 · 굵은 선: 전체 시드 평균\n내부 갱신이 없는 선형 모델은 두 내부 조건이 같은 대조 · 사람·실제 초파리 비교가 아님",
            "curve_title": "공통 학습 뒤 네 분기의 선택 변화", "curve_ylabel": "해당 단계의 기준 수량 선택 확률",
            "prefix": "공통 앞부분", "curve_axis": "큰 수량 공통 학습 → 작은 수량 반전 학습의 진행 위치",
            "curve_footer": "선: 전체 계산 시드 평균 · 띠: 표준편차 (신뢰구간 아님)\n경계에서는 기준 방향만 바뀌며 가중치를 초기화하지 않음 · 공통 앞부분은 한 번만 실행·집계\n입력에 수량 순서가 이미 주어진 계산 과제이며 공통 생물학적 기작의 증거가 아님"}
    return {"language": "en", "models": {"linear": "Linear: no internal synapses", "real": "Real subgraph",
        "rewired": "Degree-preserving rewiring", "ann": "Sparse ANN"},
        "branches": {"full": "Head + internal", "head_only": "Head only", "internal_only": "Internal only", "frozen": "Frozen"},
        "endpoint_title": "Learning-component interventions from the same trained state",
        "endpoint_description": "Final reversal IID probability; every registered seed shown as paired values",
        "off": "Head updates off", "on": "Head updates on", "internal_axis": ["Internal updates off", "Internal updates on"],
        "probability": "Probability of choosing smaller", "endpoint_footer": "Thin lines: paired values from one seed; thick lines: all-seed means\nLinear model has no internal synapses. Computational intervention, not a human/animal comparison.",
        "curve_title": "One shared training prefix and four reversal branches", "curve_ylabel": "Current designated-choice probability",
        "prefix": "Shared prefix", "curve_axis": "Shared larger-choice training, then smaller-choice reversal",
        "curve_footer": "Lines: all-seed means; bands: sample SD, not confidence intervals\nAt the boundary only the reference direction changes; no weight reset; shared prefix counted once\nQuantity order is supplied in the input. These results do not establish a common biological mechanism."}


def style_axis(ax):
    ax.set_ylim(0, 1)
    ax.axhline(.5, color="#c9ced4", lw=.6)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def save_new(fig, path):
    with Path(path).open("xb") as handle:
        fig.savefig(handle, format="png", dpi=160, facecolor="white")
    plt.close(fig)


def paired_endpoints(bundle, text, path):
    cfg, outcomes = bundle["config"], bundle["outcomes"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), sharey=True)
    head_levels = [(False, ("frozen", "internal_only"), "#7b818b", "s", text["off"]),
                   (True, ("head_only", "full"), "#2377a4", "o", text["on"])]
    for ax, model in zip(axes.flat, MODELS):
        for head_on, branches, color, marker, title in head_levels:
            x = np.array([0., 1.]) + (.025 if head_on else -.025)
            values = np.array([[outcomes[seed, model, branch, "iid"]["phase2_end"] for branch in branches]
                               for seed in cfg["seeds"]])
            for seed_values in values:
                ax.plot(x, seed_values, color=color, lw=.7, alpha=.25, marker=marker, ms=3)
            ax.plot(x, values.mean(axis=0), color=color, lw=2.8, marker=marker, ms=7, label=title)
        style_axis(ax)
        ax.set_title(text["models"][model])
        ax.set_xticks([0, 1], text["internal_axis"])
        ax.set_xlim(-.25, 1.25)
    fig.suptitle(text["endpoint_title"], fontsize=15, y=.985)
    fig.text(.5, .94, text["endpoint_description"] + f" (n={len(cfg['seeds'])})", ha="center", fontsize=10)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .105), ncol=2, frameon=False)
    fig.supylabel(text["probability"], x=.012)
    fig.text(.5, .025, text["endpoint_footer"], ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=.87, bottom=.20, left=.075, right=.98, hspace=.35, wspace=.18)
    save_new(fig, path)


def learning_curves(bundle, text, path):
    cfg, curves = bundle["config"], bundle["curves"]
    n, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    times = list(range(0, n + 1, every))
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.3), sharex=True, sharey=True)
    colors = {"prefix": "#242b34", "full": "#2377a4", "head_only": "#d08336", "internal_only": "#368770", "frozen": "#9699a0"}
    for ax, model in zip(axes.flat, MODELS):
        for branch, phase in [("prefix", "phase1"), *[(branch, "phase2") for branch in BRANCHES]]:
            values = np.array([[curves[seed, model, branch, phase, t, "iid"] for t in times] for seed in cfg["seeds"]])
            mean = values.mean(axis=0)
            x = np.array(times) + (0 if phase == "phase1" else n)
            label = text["prefix"] if branch == "prefix" else text["branches"][branch]
            ax.plot(x, mean, lw=2 if branch in ("prefix", "full") else 1.65, color=colors[branch], label=label)
            if len(cfg["seeds"]) > 1:
                sd = values.std(axis=0, ddof=1)
                ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color=colors[branch], alpha=.10)
        style_axis(ax)
        ax.set_title(text["models"][model])
        ax.axvline(n, color="#454a52", lw=.9, ls="--")
        ax.axhline(cfg["criteria"]["adaptation_probability"], color="#aaaeb4", lw=.8, ls=":")
        ax.set_xlim(0, 2 * n)
        ax.set_xticks([0, n // 2, n, n + n // 2, 2 * n])
    fig.suptitle(text["curve_title"] + f" (n={len(cfg['seeds'])})", fontsize=15, y=.985)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .13), ncol=5, frameon=False, fontsize=9)
    fig.supxlabel(text["curve_axis"], y=.19)
    fig.supylabel(text["curve_ylabel"], x=.012)
    fig.text(.5, .025, text["curve_footer"], ha="center", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(top=.91, bottom=.25, left=.075, right=.98, hspace=.29, wspace=.15)
    save_new(fig, path)


def create_figures(result_dir, out):
    result_dir, out = Path(result_dir).resolve(), Path(out).resolve()
    require(out.parent == result_dir.parent and out != result_dir, "New figures must use a separate sibling directory")
    require(not out.exists(), "Figure output already exists; refusing overwrite")
    bundle = validate_inputs(result_dir)
    text = labels()
    provenance = {"study_id": "learning_ablation_v1", "input_hashes": bundle["input_hashes"],
        "plot_script_sha256": digest(__file__), "figure_language": text["language"],
        "seeds": bundle["config"]["seeds"], "models": MODELS, "branches": BRANCHES, "split": "iid",
        "scope": "All registered seeds, paired 2x2 endpoint and fixed-checkpoint means; sample SD, no confidence intervals or new hypothesis test; prefix displayed and counted once"}
    out.mkdir(parents=True, exist_ok=False)
    try:
        paired_endpoints(bundle, text, out / "paired_endpoints.png")
        learning_curves(bundle, text, out / "learning_curves.png")
        require(bundle["input_hashes"] == {name: digest(result_dir / name) for name in INPUTS},
                "Input bytes changed during plotting")
        provenance.update(status="completed", created_at_utc=datetime.now(timezone.utc).isoformat(),
            figure_hashes={name: digest(out / name) for name in ("paired_endpoints.png", "learning_curves.png")})
    except Exception as error:
        plt.close("all")
        provenance.update(status="failed", created_at_utc=datetime.now(timezone.utc).isoformat(),
                          error_type=type(error).__name__, error_message=str(error),
                          figure_hashes={path.name: digest(path) for path in out.glob("*.png") if path.is_file()})
        with (out / "metadata.json").open("x", encoding="utf8", newline="\n") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        raise
    with (out / "metadata.json").open("x", encoding="utf8", newline="\n") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = create_figures(args.result_dir, args.out)
    print(json.dumps({"status": record["status"], "figures": list(record["figure_hashes"]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
