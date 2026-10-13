"""Scientific plots and a Korean report with a nonexpert explanation."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .experiment import write_json

HISTORY_LABELS = {"high_then_low": "Higher -> lower", "low_then_low": "Lower -> lower",
                  "high_then_high": "Higher -> higher", "fresh_low": "Fresh lower",
                  "reset_head_after_high": "Higher -> reset head -> lower"}
FAMILY_KO = {"fixed": "실제 배선·출력만 학습", "plastic": "실제 배선·내부도 학습",
             "rewired_fixed": "재배선·출력만 학습", "rewired_plastic": "재배선·내부도 학습",
             "ann_fixed": "일반 신경망·출력만 학습", "ann_plastic": "일반 신경망·내부도 학습"}

def analyze(output):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8")); cfg = meta["config"]
    with (output / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    seeds, conditions, histories = cfg["seeds"], cfg["conditions"], cfg["histories"]
    tmax, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    x = np.arange(0, tmax + 1, every)
    colors = {h: c for h, c in zip(histories, ("#d1495b", "#2a9d8f", "#777777", "#247ba0", "#f4a261"))}
    def series(condition, history, metric):
        return np.array([[float(r[metric]) for r in curves if int(r["seed"]) == seed and r["condition"] == condition
                          and r["history"] == history and r["phase"] == "phase2" and r["split"] == "iid"] for seed in seeds])

    ncol = 3; nrow = (len(conditions) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.7 * nrow), squeeze=False, sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, conditions):
        for history in histories:
            values = series(condition, history, "p_target")
            ax.plot(x, values.mean(axis=0), label=HISTORY_LABELS[history], color=colors[history])
            ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[history], alpha=.09)
        ax.axhline(.5, color="black", linestyle=":", linewidth=1)
        ax.axhline(cfg["criteria"]["adaptation_accuracy"], color="gray", linestyle="--", linewidth=1)
        ax.set(title=condition, ylim=(0, 1.02), xlabel="Phase 2 training trials", ylabel="Correct-choice probability")
    for ax in list(axes.flat)[len(conditions):]: ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Matched learning histories: {len(seeds)} computational seeds; bands = seed range\nHigher->higher targets higher; other phase 2 histories target lower")
    fig.tight_layout(rect=(0,.07,1,.93)); fig.savefig(output / "history_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, condition in enumerate(conditions):
        values = [r["history_auc_penalty"] for r in summary["paired_contrasts"] if r["condition"] == condition]
        ax.scatter(np.full(len(values), i) + np.linspace(-.13,.13,len(values)), values, color="#247ba0", alpha=.65, s=25)
        ax.plot([i-.18,i+.18], [np.mean(values)]*2, color="black", linewidth=2)
    ax.axhline(0, color="gray", linestyle=":")
    ax.axhline(cfg["criteria"]["history_auc_penalty"], color="#d1495b", linestyle="--", label="Registered mean-penalty threshold")
    ax.set(xticks=np.arange(len(conditions)), xticklabels=conditions, ylabel="Fresh lower AUC - higher->lower AUC",
           title="Cost of conflicting prior learning: dots are paired computational seeds")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "history_cost.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=True)
    for ax, condition in zip(axes, ("fixed", "plastic")):
        if condition not in conditions:
            ax.set_visible(False); continue
        for history in ("high_then_low", "fresh_low", "reset_head_after_high"):
            values = series(condition, history, "mean_policy_sensitivity")
            ax.plot(x, values.mean(axis=0), color=colors[history], label=HISTORY_LABELS[history])
        ax.set(title=condition, xlabel="Phase 2 trials", ylabel="Mean p(1-p)", ylim=(0,.255))
    axes[-1].legend(fontsize=7)
    fig.suptitle("Algorithmic expected-update factor; not a dopamine measurement")
    fig.tight_layout(); fig.savefig(output / "policy_sensitivity.png", dpi=150); plt.close(fig)

    rows, aggregate = [], []
    for condition in conditions:
        result = {h: [r for r in summary["outcomes"] if r["condition"] == condition and r["history"] == h] for h in histories}
        means = {h: {key: float(np.mean([r[key] for r in selected])) for key in ("phase2_before", "phase2_after", "phase2_iid_auc")}
                 for h, selected in result.items()}
        verdict = next(v for v in summary["family_verdicts"] if v["condition"] == condition)
        count = sum(r["endpoint_criterion_met"] for r in result["high_then_low"])
        qualified = sum(bool(r["qualified_reversal_criterion_met"]) for r in result["high_then_low"])
        aggregate.append({"condition": condition, "means": means, "history_verdict": verdict,
                          "reversal_endpoint_passes": count, "qualified_reversal_passes": qualified})
        rows.append(f"| {FAMILY_KO[condition]} | {100*means['fresh_low']['phase2_after']:.1f}% | {100*means['high_then_low']['phase2_after']:.1f}% | {100*means['reset_head_after_high']['phase2_after']:.1f}% | {100*verdict['mean_history_auc_penalty']:+.1f}점 | {qualified}/{len(seeds)} |")
    write_json(output / "analysis_summary.json", {"scope": cfg["scope"], "aggregate": aggregate,
                "uncertainty": "seed ranges and descriptive means; no biological sample inference or significance test"})
    met = sum(v["history_cost_criterion_met"] for v in summary["family_verdicts"])
    table = "\n".join(["| 모델 | 처음부터 작은 쪽: 마지막 성적 | 큰 쪽→작은 쪽 | 선택부분 초기화 후 | 과정 점수의 이전학습 비용 | 습득 후 반전 기준 통과 |",
                        "|---|---:|---:|---:|---:|---:|", *rows])
    report = f"""# 반전 실패를 이해하기 위한 대조 실험

## 먼저 쉬운 설명

모델에게 2와4 중 큰 쪽을 고르면 보상을 주다가 작은 쪽에 보상을 주면, 이전 답을 바꾸기 어려울 수 있습니다. 하지만 지난 실패만으로 그 원인은 알 수 없습니다. 이번에는 **처음부터 작은 쪽을 배우는 모델**과 **큰 쪽을 배운 다음 작은 쪽으로 바꾸는 모델**을 같은 문제와 같은 {tmax}번의 새 학습 기회로 비교했습니다. 또 마지막 선택을 정하는 가중치만 초기화하는 실험으로, 이전 선택 습관의 영향을 시험했습니다. 내부 연결은 이 초기화에서 그대로 남깁니다.

이때 ‘선택 습관’은 계산 모델의 가중치를 이해하기 위한 비유입니다. 살아 있는 초파리나 사람의 심리 상태를 측정한 것은 아닙니다. 수량 n/8은 미리 제공되므로 물체 개수를 알아내는 능력을 배우는 실험도 아닙니다.

## 무엇을 바꾸고 무엇을 맞췄나

- 큰 쪽→작은 쪽: {tmax}번 큰 쪽 학습 후 {tmax}번 반전.
- 작은 쪽→작은 쪽: 같은 총 학습량에서 익숙한 규칙 유지.
- 큰 쪽→큰 쪽: 같은 총 학습량에서 초기 학습 유지 확인. 마지막 성적의 정답은 큰 쪽.
- 처음부터 작은 쪽: 사전 학습 없이 작은 쪽 {tmax}번. 의도적으로 사전 업데이트가 없습니다.
- 큰 쪽→선택부분 초기화→작은 쪽: 내부 가중치는 보존하고 출력 가중치만 초기값으로 돌려 재학습.

뉴런·입력 표현·학습률·평가 문제·두 번째 단계의 난수 추첨을 조건별로 맞췄습니다. 초기화는 학습 업데이트와 따로 기록했습니다. 각 모델은 {len(seeds)}개의 다른 초기 난수 조건으로 반복했습니다. 이 숫자는 초파리 개체 수가 아닙니다. 기존 실험보다 더 오래 돌려 성공을 찾은 것이 아니라, 같은 예산에 원인을 구분할 대조군을 추가한 새 실험입니다.

## 결과를 읽는 법

성적은 정답을 고를 확률의 평균입니다. 50%는 동전 던지기 수준입니다. ‘과정 점수’는 중간 평가를 이어 계산한 AUC를0–100으로 바꾼 값으로, 마지막 성적이 같더라도 얼마나 빨리 배웠는지 반영합니다. **이전학습 비용**은 처음부터 작은 쪽을 배운 모델의 과정 점수에서 반전 모델 점수를 뺀 값입니다. 양수면 이번 조건에서는 과거의 반대 학습을 가진 쪽이 과정 전체에서 불리했습니다.

{table}

사전 기준은 평균 과정 비용 ≥{100*cfg['criteria']['history_auc_penalty']:.0f}점 및 양의 차이를 보인 시드 비율 ≥{100*cfg['criteria']['positive_seed_fraction']:.0f}%였습니다. **{met}/{len(conditions)}모델이 이 기준을 충족**했습니다. 모든 시드를 포함한 판정이며, 과거 규칙을 충분히 습득한 시드만의 결과는 이차 분석으로 따로 저장했습니다. 마지막 반전 성적 기준은{100*cfg['criteria']['adaptation_accuracy']:.0f}%이며 습득 실패를 숨기지 않습니다.

## 어디까지 설명할 수 있나

출력만 학습하는 모델에서 선택부분 초기화와 처음부터 작은 쪽 학습이 같아지는 것은 구조상 예상되는 검증 항목입니다. 내부 상태가 바뀌지 않기 때문입니다. 이를 새로운 생물학적 발견으로 세지 않습니다. 내부도 바뀐 모델에서 초기화 이후 차이가 남거나 좋아지더라도 내부 표현·활성 크기·갱신 경로·clipping 등이 함께 작용하므로 한 종류의 기억 장소를 특정할 수 없습니다.

선택이 매우 한쪽으로 치우치면 이 학습식의 평균 갱신 계수 p(1−p)는 작아집니다. 이는 현 구현에서 직접 도출한 수학적 성질이며 인간의 고집이나 실제 DAN 반응을 발견한 것이 아닙니다. 따라서 이번 결과는 추상 입력을 쓰는 알고리즘의 학습 이력과 초기화 효과에 대한 진단입니다. 커넥톰 특이성·시각 수량 인지·인간과 동일한 기작·교육 효과는 별도 증거가 필요합니다.

## 기록과 재현

[프로토콜](../../docs/reversal-protocol.md) · [실행 정보](metadata.json) · [사전 동결](registration.json) · [모든 시드 판정](summary.json) · [별도 검증](verification.json)

각 seed 디렉터리의 trials.csv, evaluation_cases.csv와 NPZ에 모든 시행·입력·가중치·활동·초기화 경계를 보존했습니다. 그림 band는 시드 최솟값–최댓값이며 신뢰구간이 아닙니다. 평가 건수는 독립 참가자 수가 아닙니다.

![두 번째 단계의 학습](history_curves.png)

![이전 학습의 과정 비용](history_cost.png)

![갱신 계수 진단](policy_sensitivity.png)
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return aggregate
