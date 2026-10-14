"""Descriptive plots and a plain-language report for the frozen spacing study."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .spacing import ALPHAS, FOCAL_INDEX, PAIRS, PRIMARY, spacing_panel
from .experiment import write_json

NAMES = {"fixed": "실제 배선·출력만 학습", "plastic": "실제 배선·내부도 학습",
         "rewired_fixed": "재배선·출력만 학습", "rewired_plastic": "재배선·내부도 학습",
         "ann_fixed": "일반 신경망·출력만 학습", "ann_plastic": "일반 신경망·내부도 학습"}
MAP_NAMES = ("1/4 간격", "1/2 간격", "기존 간격", "1.5배 간격")


def analyze(output):
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    cfg, outcomes = metadata["config"], summary["outcomes"]
    with (output / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    try:
        font_manager.findfont("Malgun Gothic", fallback_to_default=False)
        plt.rcParams["font.family"] = "Malgun Gothic"
        korean = True
    except ValueError:
        korean = False
    plt.rcParams["axes.unicode_minus"] = False
    def label(ko, en):
        return ko if korean else en
    conditions, seeds = cfg["conditions"], cfg["seeds"]
    aggregate = []
    for condition in conditions:
        rows = [r for r in outcomes if r["condition"] == condition]
        verdict = next(v for v in summary["family_verdicts"] if v["condition"] == condition)
        aggregate.append({"condition": condition, "acquisition_passes": sum(r["acquired"] for r in rows),
                          "after_accuracy": float(np.mean([r["after_accuracy"] for r in rows])),
                          **{key: np.mean([r[key] for r in rows], axis=0).tolist()
                             for key in ("focal_before", "focal_after", "focal_gain", "focal_greedy_after", "unseen_after", "anchored_after", "all_after")},
                          **{key: verdict[key] for key in ("mean_sensitivity_increase", "sd_sensitivity_increase", "positive_seed_count",
                                                         "criterion_met", "rank_robust_count", "focal_transfer_counts", "unseen_transfer_counts")}})
    write_json(output / "analysis_summary.json", {"aggregate": aggregate, "alphas": list(ALPHAS),
               "actual_updates": summary["actual_training_trials"], "primary_pair": [5, 7],
               "uncertainty": "seed means and sample SD; no biological independence or significance test"})

    panel = spacing_panel()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in enumerate(panel):
        ax.plot(np.arange(1, 9), row / 8, "o-", label=MAP_NAMES[i] if korean else f"gap scale {ALPHAS[i]}")
    ax.scatter([2, 4], [.25, .5], color="black", marker="s", s=55, zorder=5,
               label=label("연습한 두 입력값은 고정", "Both trained inputs fixed"))
    ax.set(xlabel=label("실제 수량", "True quantity"), ylabel=label("모델에 전달한 입력값", "Supplied scalar input"),
           xticks=np.arange(1, 9), title=label("수량의 순서는 유지하고 5와 7의 입력 간격을 조절", "Order preserved; focal spacing varied"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "spacing_map.png", dpi=150)
    plt.close(fig)

    selected = [c for c in ("fixed", "plastic") if c in conditions] or conditions[:2]
    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 4.5), squeeze=False)
    for ax, condition in zip(axes[0], selected):
        row = next(r for r in aggregate if r["condition"] == condition)
        ax.plot(ALPHAS, row["focal_before"], "o:", label=label("학습 전 정답 선택 확률", "Before: correct probability"))
        ax.plot(ALPHAS, row["focal_after"], "o-", label=label("학습 후 정답 선택 확률", "After: correct probability"))
        ax.plot(ALPHAS, row["focal_greedy_after"], "s--", label=label("확률이 높은 쪽만 고른 성적", "After: greedy accuracy"))
        ax.axhline(.5, color="gray", linestyle=":")
        ax.set(xticks=ALPHAS, ylim=(0, 1.04), xlabel=label("기존 간격 대비 배율", "Focal gap / original gap"),
               ylabel=label("정답 선택 확률 / 결정적 선택 성적", "Probability / greedy accuracy"),
               title=NAMES[condition] if korean else condition)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(label("같은 5대7 문제: 확률과 선택 방향을 구분", "Same5-vs-7: probability and decision rank distinguished"))
    fig.tight_layout()
    fig.savefig(output / "focal_response.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for condition in conditions:
        data = np.asarray([[float(r["p_target"]) for r in curves if int(r["seed"]) == seed
                            and r["condition"] == condition and r["split"] == "iid"] for seed in seeds])
        ax.plot(np.arange(data.shape[1]) * cfg["checkpoint_every"], data.mean(axis=0), label=condition)
    ax.axhline(.5, color="gray", linestyle=":")
    ax.set(ylim=(0, 1.02), xlabel=label("보상 학습 시행", "Rewarded trials"),
           ylabel=label("연습한 2대4의 정답 선택 확률", "Correct probability on trained2-vs-4"),
           title=label("한 번의 학습 경로를 네 입력 간격 검사에 공유", "One training trajectory shared by all four code evaluations"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "acquisition.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(aggregate))
    ax.bar(x, [100 * r["mean_sensitivity_increase"] for r in aggregate],
           yerr=[100 * r["sd_sensitivity_increase"] for r in aggregate], capsize=4)
    ax.axhline(100 * cfg["criteria"]["sensitivity_increase"], color="black", linestyle=":",
               label=label("사전 평균 차이 기준", "Registered mean criterion"))
    ax.set(xticks=x, xticklabels=conditions, ylabel=label("학습 전을 보정한 간격 효과 (%포인트)", "Baseline-adjusted spacing effect (percentage points)"),
           title=label("모든 시드 포함: 오차 막대는 시드 표준편차", "All seeds included; bars show sample SD"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "condition_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = (("internal_change_l2", label("내부 연결 가중치", "Internal weights")),
               ("readout_change_l2", label("마지막 선택 가중치", "Readout weights")),
               ("probe_mbon_change_l2", label("같은 입력의 MBON 활성", "Fixed-input MBON activity")))
    for ax, (key, title) in zip(axes, metrics):
        for condition in selected:
            data = np.asarray([[float(r[key]) for r in curves if int(r["seed"]) == seed
                                and r["condition"] == condition and r["split"] == "iid"] for seed in seeds])
            ax.plot(np.arange(data.shape[1]) * cfg["checkpoint_every"], data.mean(axis=0),
                    label=NAMES[condition] if korean else condition)
        ax.set(title=title, xlabel=label("학습 시행", "Training trials"),
               ylabel=label("시작 상태와의 거리 (L2)", "L2 distance from initial state"))
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "changes.png", dpi=150)
    plt.close(fig)

    pair_rows = []
    for condition in conditions:
        matrices = []
        for seed in seeds:
            with np.load(output / f"seed_{seed}" / condition / "transfer_final.npz", allow_pickle=False) as saved:
                matrices.append(saved["map_p_target"].copy())
        means = np.mean(matrices, axis=0)
        for i, alpha in enumerate(ALPHAS):
            for j, (low, high) in enumerate(PAIRS):
                pair_rows.append({"condition": condition, "alpha": alpha, "low": low, "high": high,
                                  "ratio": low / high, "difference": high - low, "focal": j == FOCAL_INDEX,
                                  "stratum": "fully_unseen" if j in set(PRIMARY) else "one_anchor",
                                  "input_gap": (panel[i, high - 1] - panel[i, low - 1]) / 8,
                                  "input_midpoint": (panel[i, high - 1] + panel[i, low - 1]) / 16,
                                  "p_target": float(means[i, j])})
    with (output / "pair_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    p_rows, g_rows, e_rows = [], [], []
    for r in aggregate:
        name = NAMES[r["condition"]]
        p_rows.append("| " + name + " | " + " | ".join(f"{100 * v:.1f}%" for v in r["focal_after"]) + " |")
        g_rows.append("| " + name + " | " + " | ".join(f"{100 * v:.1f}%" for v in r["focal_greedy_after"]) + f" | {r['rank_robust_count']}/{len(seeds)} |")
        e_rows.append(f"| {name} | {r['acquisition_passes']}/{len(seeds)} | {100*r['mean_sensitivity_increase']:.2f}±{100*r['sd_sensitivity_increase']:.2f} | {r['positive_seed_count']}/{len(seeds)} | {'통과' if r['criterion_met'] else '미달'} | {' / '.join(map(str,r['focal_transfer_counts']))} |")
    p_table = "\n".join(["| 모델 | 1/4 간격 | 1/2 간격 | 기존 간격 | 1.5배 간격 |", "|---|---:|---:|---:|---:|", *p_rows])
    g_table = "\n".join(["| 모델 | 1/4 간격 | 1/2 간격 | 기존 간격 | 1.5배 간격 | 습득 후 네 간격 모두90% |", "|---|---:|---:|---:|---:|---:|", *g_rows])
    e_table = "\n".join(["| 모델 | 습득 | 주 효과 평균±SD (%포인트) | 양수 시드 | H-S | 5대7 절대 전이 통과 수(네 간격 순) |", "|---|---:|---:|---:|---|---:|", *e_rows])
    passed = sum(r["criterion_met"] for r in aggregate)
    fixture_notice = "\n이 보고서는 소프트웨어 점검용 작은 fixture이며 연구 결과가 아니다.\n" if cfg["scope"] == "integration_test" else ""
    text = f"""# 순서를 유지한 입력 간격 진단
{fixture_notice}

## 먼저 쉬운 설명

연습은 2대4에서 큰 쪽을 고르는 {cfg['trials']}시행이다. 배운 모델은 그대로 두고, 새 5대7 문제를 전달하는 숫자 간격만 줄이거나 늘렸다. 모든 코드에서 5보다 7의 입력값이 크고 두 값의 중간 지점은 같다. 실제 개수와 비율·자료·모델·보상 경험을 유지한다. 물체 그림을 보거나 수량 의미를 스스로 발견하는 과제는 아니다.

이번에는 정답 선택 **확률**과, 확률이 높은 쪽만 고르는 **결정적 선택 성적**을 나눠 본다. 확률은 줄어도 결정적 선택은 유지될 수 있다. 인간의 주관적인 확신을 측정한 값은 아니다.

## 같은5대7의 정답 선택 확률

아래는{len(seeds)}개 계산 시드 모두의 학습 후 평균이다. 간격은 기존 입력값 사이 거리에 대한 배율이다. 훈련 수량2·4, 전체 범위·끝점, 새5·7의 중간값은 고정했다. 5·7은 훈련 입력 구간 밖이다.

{p_table}

## 확률이 높은 쪽만 고른 경우

{g_table}

마지막 열은 연습 습득을 통과하고, 네 간격 모두에서 사전 greedy90% 기준을 만족한 시드 수다. 그 개수에 사후 가족 성공 기준을 추가하지 않았다. 동률은0.5로 처리했다.

## 미리 정한 판정

주 효과는 `(학습후 기존간격 확률−학습후1/4간격 확률)−(학습전 같은 차이)`다. 시작 상태에서도 있던 간격 차이를 보정했다. 전체 시드 평균≥{100*cfg['criteria']['sensitivity_increase']:.0f}%포인트와 양수 시드≥{100*cfg['criteria']['positive_seed_fraction']:.0f}%라는 사전 H-S 기준은{passed}/{len(conditions)}모델이 통과했다. 다음 표의 평균에는 습득 실패 시드도 포함했다. 습득 통과 시드만의 보조 평균은 [원자료 판정](summary.json)에 별도로 있다.

{e_table}

연습 습득은 최종 정답 확률≥75% 및 시작 대비≥15%포인트다. 각 코드의 절대 전이는 습득 통과 및5대7 정답 확률≥75%를 모두 요구한다. 전체15개 완전 새 수량쌍의 macro 전이 기준은 별도 보조 지표이며 위5대7 기준과 섞지 않았다. ±는 시드 표본 표준편차로 생물학적 표본의 신뢰구간이나 유의성 검정이 아니다.

## 무엇을 설명할 수 있는가

같은5대7과 같은 비율5/7에서 코드 간격에 따라 선택 반응이 달라지는지를 측정한다. 현재 신경망 점수는 인공 입력에 대해 구간별 선형이며, 점수 차이를sigmoid 선택 확률로 바꾼다. **단순 선형 점수와 같은 선택 함수만으로도 확률 변화와 선택 방향 보존이 함께 나타날 수 있다.** 따라서 결과가 그런 방향이어도 새로운 수량 계산 원리나 Weber 법칙을 발견했다고 주장하지 않는다. 확률 지표가 입력 기하와 선택 함수에 얼마나 의존하는지 좁힌 결과다.

단순히 큰 코드를 고르는 분석적 기준은 모든 네 코드·27새 쌍에서100%다. 이100%는 신경망 결과가 아니다. 비수량 단서만으로는 교차 설계상50%다. 신경망에는5대7의 대응표나 평가 정답을 전달하지 않았다. 다른26쌍에서는 간격과 중간값이 함께 바뀔 수 있으므로 [쌍별 성적](pair_summary.csv)을 기술적으로 보고하며 간격만의 인과 효과라고 말하지 않는다.

## 실행·검산·한계

실제 학습은 {summary['actual_training_trials']:,}시행이다. 네 코드는 같은 학습 경로의 고정 평가이며 4배의 추가 학습이나 독립 초파리 개체가 아니다. 같은 시드·모델 조건의 네 코드 평가는 배선·초기값·학습 경험을 공유했다. 내부 학습 모델은 내부 가중치 1,474개를 추가 갱신하므로 출력 전용 모델과 같은 학습 자원이라고 주장하지 않는다.

[사전 프로토콜](../../docs/spacing-protocol.md) · [설계의 수학적 확인](../../docs/spacing-background.md) · [실행 정보](metadata.json) · [모든 시드 판정](summary.json) · [독립 검산](verification.json)

각 seed에 모든 시행·입력·선택 난수·보상·배선·checkpoint가 있다. 내부·출력·활성의 변화량을 구분해 기록했다. 후보별cache와 네코드×27쌍×평가행 확률을 보존하며 독립 산술로 검산한다. 기존 실패와 옛 프로토콜을 바꾸지 않는다. 인간 공개자료와 이번 곡선의 직접 비교, 실제LC11/DAN, 수학 기호 의미, Level4/5는 검증하지 않았다.

![입력 간격](spacing_map.png)

![확률과 선택](focal_response.png)

![연습 곡선](acquisition.png)

![조건 비교](condition_comparison.png)

![가중치와 활성 변화의 구분](changes.png)
"""
    (output / "report.md").write_text(text, encoding="utf-8")
    return aggregate
