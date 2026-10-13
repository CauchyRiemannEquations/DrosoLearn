"""Automatic static scientific plots and a bounded result interpretation."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .experiment import write_json

def analyze(output):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    cfg, outcomes = metadata["config"], summary["outcomes"]
    with (output / "learning_curves.csv").open(encoding="utf-8", newline="") as f:
        curves = list(csv.DictReader(f))
    conditions = cfg["conditions"]
    duration = cfg["trials_per_phase"]
    seeds = cfg["seeds"]
    cmap = plt.get_cmap("tab10")
    palette = {c: cmap(i % 10) for i, c in enumerate(conditions)}

    def matrix(condition, phase, split, metric):
        return np.array([[float(r[metric]) for r in curves if int(r["seed"]) == s
                          and r["condition"] == condition and r["phase"] == phase and r["split"] == split]
                         for s in seeds])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, phase in zip(axes, ("acquisition", "reversal")):
        for condition in conditions:
            values = matrix(condition, phase, "iid", "expected_reward")
            x = np.arange(values.shape[1]) * cfg["checkpoint_every"]
            ax.plot(x, values.mean(axis=0), label=condition, color=palette[condition])
            ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=palette[condition], alpha=0.08)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.axhline(0.75, color="gray", linestyle="--", linewidth=1)
        ax.set(title=phase.capitalize(), xlabel="Rewarded training trials", ylim=(0, 1.02))
    axes[0].set_ylabel("Exact stochastic choice accuracy (IID probe)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Abstract quantity association: {len(seeds)} computational seeds; bands = seed range")
    fig.tight_layout(); fig.savefig(output / "learning_curves.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ("internal_change_l2", "readout_change_l2", "probe_mbon_change_l2"),
                               ("Internal synaptic change", "Readout change", "Fixed-probe MBON activity change")):
        for condition in conditions:
            values = np.concatenate([matrix(condition, p, "iid", metric) for p in ("acquisition", "reversal")], axis=1)
            x = np.concatenate([np.arange(duration // cfg["checkpoint_every"] + 1) * cfg["checkpoint_every"],
                                duration + np.arange(duration // cfg["checkpoint_every"] + 1) * cfg["checkpoint_every"]])
            ax.plot(x, values.mean(axis=0), color=palette[condition], label=condition)
        ax.axvline(duration, color="gray", linestyle=":")
        ax.set(title=title, xlabel="Cumulative trials", ylabel="L2 distance from initial checkpoint")
    axes[-1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(output / "changes.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = ["iid", "ood", *["transfer_" + "_".join(map(str, p)) for p in cfg["transfer_pairs"]]]
    x = np.arange(len(names))
    for i, condition in enumerate(conditions):
        values = np.array([matrix(condition, "acquisition", name, "expected_reward")[:, -1].mean() for name in names])
        ax.plot(x, values, "o-", color=palette[condition], label=condition)
    ax.axhline(0.5, color="black", linestyle=":")
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1.02), ylabel="Exact stochastic choice accuracy",
           title="Final acquisition probe: artificial nuisance shift and untrained quantity pairs")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(output / "transfer.png", dpi=160); plt.close(fig)

    aggregate = []
    for condition in conditions:
        rows = [r for r in outcomes if r["condition"] == condition]
        verdict = None if cfg["scope"] == "smoke" else {
            k: sum(r[k] for r in rows) / len(rows) >= cfg["criteria"]["confirmatory_seed_fraction"]
            for k in ("acquisition_criterion_met", "generalization_criterion_met", "reversal_criterion_met")}
        aggregate.append({"condition": condition, "seeds": len(rows), "confirmatory_verdict": verdict,
            **{key: float(np.mean([r[key] for r in rows])) for key in
               ("before_accuracy", "after_accuracy", "gain", "ood_accuracy", "reversal_after_accuracy")},
            **{key: sum(r[key] for r in rows) for key in
               ("acquisition_criterion_met", "generalization_criterion_met", "reversal_criterion_met")}})
    contrasts = {}
    for a, b in (("plastic", "fixed"), ("rewired_plastic", "rewired_fixed"),
                 ("fixed", "rewired_fixed"), ("plastic", "rewired_plastic"), ("plastic", "ann_plastic")):
        if a in conditions and b in conditions:
            values = [next(r for r in outcomes if r["seed"] == s and r["condition"] == a)["after_accuracy"]
                      - next(r for r in outcomes if r["seed"] == s and r["condition"] == b)["after_accuracy"] for s in seeds]
            contrasts[f"{a}_minus_{b}"] = {"per_seed": values, "mean": float(np.mean(values))}
    write_json(output / "analysis_summary.json", {"scope": cfg["scope"], "aggregate": aggregate,
               "paired_descriptive_contrasts": contrasts, "inferential_tests": "none; smoke only unless separately preregistered",
               "bands": "min/max seed range; not a confidence interval"})
    table = ["| 조건 | 전 | 습득 후 | 변화 | OOD | 반전 후 | 습득/일반화/반전 기준 시드 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in aggregate:
        table.append(f"| {row['condition']} | {row['before_accuracy']:.3f} | {row['after_accuracy']:.3f} | {row['gain']:+.3f} | {row['ood_accuracy']:.3f} | {row['reversal_after_accuracy']:.3f} | {row['acquisition_criterion_met']}/{row['seeds']} · {row['generalization_criterion_met']}/{row['seeds']} · {row['reversal_criterion_met']}/{row['seeds']} |")
    criterion = cfg["criteria"]
    train_low, train_high = cfg["train_pair"]
    text = f"""# DrosoLearn 초기 실행 보고서

범위: **{cfg['scope']}**, {len(seeds)}개 계산 시드. 정확도는 stochastic policy의 정답 선택 확률 평균이며 표의 값은 시드 평균이다. 결정론적 greedy 정확도는 별도 원자료에 보존했다. 이 결과를 살아 있는 초파리, 시각 수량 인지 또는 인간 학습 결과로 해석하지 않는다.

## 설계와 사전 기준

{train_low} 대 {train_high}에서 큰 수량에 보상하는 {duration}시행 후 같은 모델에서 적은 수량에 보상하는 {duration}시행을 수행했다. 평가 데이터는 학습하지 않는다. 습득은 마지막 IID probe의 확률 정확도 ≥{criterion['acquisition_accuracy']} 및 초기 대비 ≥{criterion['acquisition_gain']} 향상, 인공 nuisance OOD 일반화는 습득 통과와 OOD ≥{criterion['generalization_accuracy']}, 반전은 습득 통과와 반전 후 IID ≥{criterion['reversal_accuracy']}다. 모든 시드와 실패를 포함했다. smoke는 확증 연구가 아니다. confirmatory 실행은 기준 통과 시드 비율 ≥{criterion['confirmatory_seed_fraction']}을 추가 판정하며 analysis_summary.json에 보존한다.

{chr(10).join(table)}

## 해석

명시적 n/8 입력이 이미 수량과 순서를 제공한다. 성공은 이 표현에 대한 보상 연합과 선택 교정을 보여주며, 개수를 시각적으로 추출했거나 수학 규칙을 발견했다는 증거가 아니다. OOD는 실제 물체 배치가 아닌 인공 descriptor 범위 변화다. 새 수량쌍 probe는 입력 표현이 제공한 순서의 사용 가능성을 보는 이차 진단이다. 원자료의 습득 gate를 통과하지 못한 조건의 전이를 일반화 성공으로 해석하지 않는다.

실제 구조는 512 KC·48 MBON 중 489 KC→29 MBON의 1,474쌍을 사용한다. 19 MBON에는 보존된 KC 입력이 없다. DAN/APL/반복 회로는 실행하지 않는다. 고정 조건은 출력 48파라미터만, 가소성 조건은 추가 내부 1,474파라미터를 갱신한다. 동일 입력·시행·출력 자원을 맞추었지만 고정 대 가소성의 총 학습 자원은 같지 않다. 내부만 학습하는 진단과 무학습·무정보 보상 대조도 함께 보존한다.

실제 배선의 우수성은 연구 전제가 아니며 작은 시드 평균 차이로 주장하지 않는다. 내부 가중치 변화만으로 생물학적 가소성을 주장하지 않는다. 학습/반전 중 활동 변화, 내부 변화와 readout 변화는 따로 기록했다. 초기 모델에는 시간 감쇠가 없어 휴지 시간의 생물학적 망각을 평가하지 않는다. 인간 공개 자료 비교와 Level 4/5 검증은 수행하지 않았다.

## 재현·검증·그림

- [실행 메타데이터](metadata.json), [사전 동결 기록](registration.json), [모든 시드 판정](summary.json)
- [시행별 기록](trials.csv), [평가 시행](evaluation_cases.csv), [학습 곡선](learning_curves.csv)
- [독립 검증](verification.json): 별도 코드로 저장된 활동·확률·판정을 재계산한 검증 범위를 확인할 것.
- 각 seed 디렉터리에 입력, 재배선과 전 checkpoint 가중치/encoder/probe 활동이 있다.

![학습과 반전](learning_curves.png)

![변화의 구분](changes.png)

![미학습 조건](transfer.png)

그림의 band는 시드 최솟값–최댓값이며 신뢰구간이 아니다. 재현 명령은 metadata.json의 reproduce에 있다. 원본 디렉터리에 재실행하면 거부되므로 새 출력 경로를 사용한다.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
    return aggregate
