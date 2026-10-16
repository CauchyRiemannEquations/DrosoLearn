"""Render separately hashed descriptive calibration figures, without new estimates."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summary = json.loads(args.summary.read_text(encoding="utf8"))
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharey=True)
    colors = ["#da8655", "#377da4"]
    def case(schedule, policy, stratum):
        return next(r["by_feedback"] for r in summary["cases"] if
                    (r["schedule"], r["policy"], r["environment"], r["stratum"]) ==
                    (schedule, policy, "probabilistic", stratum))
    panels = [([case("fixed", policy, "all") for policy in ("iid_fair", "iid_biased", "sticky")],
               ["균등 무작위", "B를 80% 선택", "직전 선택을 90% 반복"], "학습 없는 세 가지 선택 규칙"),
              ([case("reversal", "iid_biased", label) for label in ("phase1", "phase2", "all")],
               ["반전 전", "반전 후", "두 시기 합산"], "같은 편향 규칙: 시기를 합치면 차이를 가릴 수 있음")]
    for ax, (cells, labels, title) in zip(axes, panels):
        x = np.arange(3)
        for f, offset in (("0", -.19), ("1", .19)):
            theory = [c[f]["ratio_of_expected_counts"] for c in cells]
            actual = [c[f]["ratio_of_summed_empirical_counts"] for c in cells]
            ax.bar(x + offset, theory, width=.34, color=colors[int(f)], label="무보상 뒤" if f == "0" else "보상 뒤", zorder=2)
            ax.scatter(x + offset, actual, color="#18252b", marker="x", s=32, zorder=4)
            for position, value in zip(x + offset, theory):
                ax.text(position, value + .025, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.04)
        ax.set_yticks(np.arange(0, 1.01, .2), ["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_title(title, fontsize=12, pad=16)
        ax.grid(axis="y", alpha=.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("바로 다음에 같은 선택을 한 비율")
    axes[0].legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -.15))
    fig.suptitle("유지율이 높거나 보상별로 달라도, 그것만으로 학습의 증거는 아니다", fontsize=16, y=.98)
    fig.text(.5, .035, "모든 조건의 학습 갱신 0회 · 막대: 기대 횟수의 비 · ×: 실제 횟수의 합산 비 (10시드)\n인공 두 선택 과제의 대조 실험이며 사람·초파리의 학습 비교가 아님", ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(top=.81, bottom=.26, left=.07, right=.98, wspace=.12)
    destination = args.out / "calibration.png"
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "summary_sha256": digest(args.summary),
              "plot_script_sha256": digest(__file__), "figure_sha256": digest(destination),
              "scope": "predefined H1-H3 cases; descriptive plotting only; no new hypothesis tests"}
    (args.out / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"figure": str(destination)}))


if __name__ == "__main__":
    main()
