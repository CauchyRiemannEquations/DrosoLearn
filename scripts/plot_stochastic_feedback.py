"""Plot fixed-checkpoint IID curves; bars/bands are descriptive seed variation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.result_dir / "metadata.json").read_text(encoding="utf8"))
    if meta["status"] != "completed":
        raise ValueError("Only a completed, separately verified run can be illustrated")
    checked = json.loads((args.result_dir / "verification.json").read_text(encoding="utf8"))
    if checked["status"] != "passed":
        raise ValueError("Independent verification is required")
    cfg = meta["config"]
    with (args.result_dir / "learning_curves.csv").open(encoding="utf8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "iid"]
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    conditions = cfg["conditions"]
    nrows = (len(conditions) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12.5, 3.2 * nrows + 2.0), sharex=True, sharey=True, squeeze=False)
    names = {"linear": "단순 선형", "frozen": "학습 중지 선형", "fixed": "실제 배선 · 내부 고정",
             "plastic": "실제 배선 · 내부 가소성", "rewired_fixed": "재배선 · 내부 고정",
             "rewired_plastic": "재배선 · 내부 가소성", "ann_fixed": "일반망 · 내부 고정", "ann_plastic": "일반망 · 내부 가소성"}
    envs = [("deterministic", "결정적 1/0", "#397ca6"), ("probabilistic", "확률적 .8/.2", "#d28042"),
            ("uninformative", "무정보 .5/.5", "#73757c")]
    for ax, condition in zip(axes.flat, conditions):
        for env, label, color in envs:
            for phase in ("phase1", "phase2"):
                times = list(range(0, cfg["trials_per_phase"] + 1, cfg["checkpoint_every"]))
                values = np.array([[float(next(r["p_rule"] for r in rows if
                          (r["condition"], r["environment"], r["phase"], int(r["trial"]), int(r["seed"])) ==
                          (condition, env, phase, t, seed))) for t in times] for seed in cfg["seeds"]])
                mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                x = np.array(times) + (cfg["trials_per_phase"] if phase == "phase2" else 0)
                ax.plot(x, mean, color=color, lw=1.9, label=label if phase == "phase1" else None)
                ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color=color, alpha=.13)
        ax.axvline(cfg["trials_per_phase"], color="#363d41", ls="--", lw=.8)
        ax.axhline(.75, color="#aab0b4", ls=":", lw=.8)
        ax.axhline(.5, color="#cad0d3", lw=.5)
        ax.set_title(names[condition])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 2 * cfg["trials_per_phase"])
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .15 if nrows == 1 else .09), ncol=3, frameon=False)
    fig.suptitle("보상 방식에 따른 학습 과정 · 1단계 큰 수량 → 2단계 작은 수량", fontsize=15, y=.99)
    fig.supxlabel("누적 선택·피드백 횟수", y=.24 if nrows == 1 else .15)
    fig.supylabel("해당 단계의 기준 수량을 선택할 확률", x=.012)
    fig.text(.5, .02, "선: 계산시드10개 평균 · 띠: 표준편차 (신뢰구간 아님)\n무정보 환경에는 더 유리한 선택이 없음 · 400회 경계에서는 기준 방향만 반전, 가중치 즉시 변경 없음\n입력에 수량 순서가 이미 주어지는 계산 과제이며 사람·실제 초파리와의 비교가 아님", ha="center", fontsize=9)
    fig.subplots_adjust(top=.87 if nrows == 1 else .91, bottom=.34 if nrows == 1 else .2, left=.07, right=.98, hspace=.3, wspace=.13)
    image = args.out / "learning_curves.png"
    fig.savefig(image, dpi=160, facecolor="white")
    plt.close(fig)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": cfg["stage"],
              "learning_curves_sha256": digest(args.result_dir / "learning_curves.csv"),
              "verification_sha256": digest(args.result_dir / "verification.json"),
              "plot_script_sha256": digest(__file__), "figure_sha256": digest(image),
              "scope": "IID fixed checkpoints, all ten seeds, descriptive SD; no added hypothesis test"}
    (args.out / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"figure": str(image)}))


if __name__ == "__main__":
    main()
