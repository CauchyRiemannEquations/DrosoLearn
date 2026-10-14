"""Descriptive all-seed analysis of independent reward learners; no refitting."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from .reward_linear import ALPHAS, ANCHORED, FOCAL_INDEX, IDENTITY, PAIRS, PRIMARY, spacing_panel
from .experiment import write_json


NAMES = {
    "linear_raw": "선형·원시 입력",
    "linear_normalized": "선형·척도 정규화",
    "linear_random_reward": "선형·무작위 보상",
    "fixed": "실제 배선·출력만 학습", "plastic": "실제 배선·내부도 학습",
    "rewired_fixed": "재배선·출력만 학습", "rewired_plastic": "재배선·내부도 학습",
    "ann_fixed": "일반 신경망·출력만 학습", "ann_plastic": "일반 신경망·내부도 학습",
}
SHORT_NAMES = {
    "linear_raw": "선형\n원시", "linear_normalized": "선형\n정규화",
    "linear_random_reward": "선형\n무작위 보상", "fixed": "실제 배선\n출력만",
    "plastic": "실제 배선\n내부도", "rewired_fixed": "재배선\n출력만",
    "rewired_plastic": "재배선\n내부도", "ann_fixed": "일반망\n출력만",
    "ann_plastic": "일반망\n내부도",
}


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty seed statistics required")
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.}


def _curve_matrix(curves, seeds, condition, split, key):
    selected = {}
    for row in curves:
        if row["condition"] == condition and row["split"] == split:
            identifier = int(row["seed"]), int(row["trial"])
            if identifier in selected:
                raise ValueError("Duplicate curve entry")
            selected[identifier] = float(row[key])
    times = sorted({trial for seed, trial in selected if seed in seeds})
    matrix = np.asarray([[selected[seed, trial] for trial in times] for seed in seeds])
    if not len(times) or not np.isfinite(matrix).all():
        raise ValueError("Complete finite learning curves required")
    return np.asarray(times), matrix


def analyze(output):
    """Read immutable run records; only write explicitly derived files."""
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    resources = json.loads((output / "model_resources.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed" or summary["status"] != "completed":
        raise ValueError("Analysis requires a completed run")
    cfg, outcomes = metadata["config"], summary["outcomes"]
    conditions, seeds = cfg["conditions"], cfg["seeds"]
    with (output / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    verdicts = {r["condition"]: r for r in summary["family_verdicts"]}
    by_seed = {(r["seed"], r["condition"]): r for r in outcomes}
    if len(by_seed) != len(outcomes) or set(by_seed) != {(s, c) for s in seeds for c in conditions}:
        raise ValueError("All registered seed/condition outcomes required")
    resource_index = {(r["seed"], r["condition"]): r for r in resources}
    if len(resource_index) != len(resources) or set(resource_index) != set(by_seed):
        raise ValueError("All registered resource records required")

    aggregate = []
    first_checkpoints, coefficients = [], []
    for condition in conditions:
        rows = [by_seed[s, condition] for s in seeds]
        resource_rows = [resource_index[s, condition] for s in seeds]
        values = {key: _statistics([r[key] for r in rows])
                  for key in ("before_accuracy", "after_accuracy", "gain", "ood_accuracy")}
        maps = {key: [_statistics([r[key][i] for r in rows]) for i in range(len(ALPHAS))]
                for key in ("unseen_before", "unseen_after", "anchored_after", "all_after", "focal_before", "focal_after", "focal_greedy_after")}
        scale_values = {key: _statistics([r[key] for r in resource_rows])
                        for key in ("feature_scale", "initial_feature_gap_rms")}
        aggregate.append({"condition": condition, "n_seeds": len(seeds), **values, **maps,
                          "resources": {**scale_values,
                                        "readout_parameters": sorted({r["readout_parameters"] for r in resource_rows}),
                                        "internal_parameters": sorted({r["internal_parameters"] for r in resource_rows})},
                          "verdict": verdicts[condition]})
        times, probability = _curve_matrix(curves, seeds, condition, "iid", "p_target")
        for i, seed in enumerate(seeds):
            passing = times[(probability[i] >= cfg["criteria"]["acquisition_accuracy"])
                            & (probability[i] - probability[i, 0] >= cfg["criteria"]["acquisition_gain"])]
            first_checkpoints.append({"seed": seed, "condition": condition,
                                      "first_observed_acquisition_checkpoint": int(passing[0]) if len(passing) else None})
            if condition.startswith("linear_"):
                with np.load(output / f"seed_{seed}" / condition / f"checkpoint_{cfg['trials']:04d}.npz", allow_pickle=False) as saved:
                    coefficients.append({"seed": seed, "condition": condition,
                                         "coefficient_order": ["q", "nuisance1", "nuisance2", "nuisance3", "nuisance4", "nuisance5", "nuisance6"],
                                         "normalized_feature_coefficients": saved["readout_weights"].tolist(),
                                         "raw_input_coefficients": (saved["readout_weights"] / float(saved["feature_scale"])).tolist()})

    primary = set(PRIMARY.tolist())
    panel, pair_rows = spacing_panel(), []
    for condition in conditions:
        for stage in ("baseline", "final"):
            scores, greedy = [], []
            for seed in seeds:
                with np.load(output / f"seed_{seed}" / condition / f"transfer_{stage}.npz", allow_pickle=False) as saved:
                    scores.append(saved["map_p_target"].copy())
                    greedy.append(saved["map_greedy_accuracy"].copy())
            probability, greedy = np.asarray(scores), np.asarray(greedy)
            if probability.shape != (len(seeds), len(ALPHAS), len(PAIRS)) or not np.isfinite(probability).all():
                raise ValueError("Complete finite transfer maps required")
            for i, alpha in enumerate(ALPHAS):
                for j, (low, high) in enumerate(PAIRS):
                    p_stats, g_stats = _statistics(probability[:, i, j]), _statistics(greedy[:, i, j])
                    pair_rows.append({"condition": condition, "stage": stage, "alpha": float(alpha),
                                      "low": low, "high": high, "ratio": low / high, "difference": high - low,
                                      "stratum": "fully_unseen" if j in primary else "one_anchor", "focal": j == FOCAL_INDEX,
                                      "input_gap": float((panel[i, high - 1] - panel[i, low - 1]) / 8),
                                      "input_midpoint": float((panel[i, high - 1] + panel[i, low - 1]) / 16),
                                      "p_correct_mean": p_stats["mean"], "p_correct_sample_sd": p_stats["sample_sd"],
                                      "greedy_accuracy_mean": g_stats["mean"], "greedy_accuracy_sample_sd": g_stats["sample_sd"]})
    with (output / "pair_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    contrasts = []
    for seed in seeds:
        normalized = by_seed[seed, "linear_normalized"]
        random = by_seed[seed, "linear_random_reward"]
        raw = by_seed[seed, "linear_raw"]
        contrasts.append({"seed": seed,
                          "iid_correct_minus_random": normalized["after_accuracy"] - random["after_accuracy"],
                          "identity_unseen_correct_minus_random": normalized["unseen_after"][IDENTITY] - random["unseen_after"][IDENTITY],
                          "iid_normalized_minus_raw": normalized["after_accuracy"] - raw["after_accuracy"],
                          "identity_unseen_normalized_minus_raw": normalized["unseen_after"][IDENTITY] - raw["unseen_after"][IDENTITY]})
    contrast_statistics = {key: _statistics([r[key] for r in contrasts]) for key in contrasts[0] if key != "seed"}
    failed = [r for r in outcomes if not r["acquired"]]
    joint_failed = [r for r in outcomes if not r["qualified_unseen_transfer"][IDENTITY]]
    derived = {"scope": cfg["scope"], "n_seeds": len(seeds), "all_seeds_included": True,
               "conditions": aggregate, "hypotheses_from_run_summary": summary["hypotheses"],
               "paired_seed_contrasts": contrasts, "paired_contrast_statistics": contrast_statistics,
               "first_observed_acquisition_checkpoints": first_checkpoints, "linear_coefficients": coefficients,
               "acquisition_failures": failed, "joint_identity_transfer_failures": joint_failed,
               "pair_summary_rows": len(pair_rows),
               "notice": "Descriptive derived analysis; no retraining, model selection or replacement of frozen gates. SD is computational seed variation."}
    write_json(output / "analysis_summary.json", derived)

    try:
        font_manager.findfont("Malgun Gothic", fallback_to_default=False)
        plt.rcParams["font.family"] = "Malgun Gothic"
        korean = True
    except ValueError:
        korean = False
    plt.rcParams["axes.unicode_minus"] = False
    def label(ko, en):
        return ko if korean else en
    def name(condition, short=False):
        return (SHORT_NAMES if short else NAMES)[condition] if korean else condition.replace("_", "\n") if short else condition
    def save(fig, filename):
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    def plot_curve(ax, chosen):
        for condition in chosen:
            times, data = _curve_matrix(curves, seeds, condition, "iid", "p_target")
            mean = data.mean(axis=0)
            sd = data.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(times))
            line, = ax.plot(times, 100 * mean, marker="o", label=name(condition))
            ax.fill_between(times, 100 * (mean - sd), 100 * (mean + sd), alpha=.1, color=line.get_color())
        ax.axhline(100 * cfg["criteria"]["acquisition_accuracy"], color="gray", linestyle=":")
        ax.set(xlabel=label("학습 시행", "Training trials"),
               ylabel=label("연습한 2대4의 정답 선택 확률 (%)", "Correct-choice probability on trained2-vs-4 (%)"), ylim=(0, 105))
        ax.legend(fontsize=8)

    groups = (conditions[:3], [c for c in conditions if c in ("fixed", "rewired_fixed", "ann_fixed")],
              [c for c in conditions if c in ("plastic", "rewired_plastic", "ann_plastic")])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, group, title in zip(axes, groups, (label("독립 선형 학습자", "Independent linear learners"), label("네트워크·출력만 학습", "Network readout learning"), label("네트워크·내부도 학습", "Network internal and readout learning"))):
        plot_curve(ax, group)
        ax.set_title(title)
    fig.suptitle(label("새 연구: 모든 초기 출력은 0, 모든 시드 평균±표본 표준편차", "New study: zero initial heads, all-seed mean ± sample SD"))
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, "learning_curves.png")

    x = np.arange(len(conditions))
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for ax, transfer in zip(axes, (False, True)):
        stats = [r["unseen_after"][IDENTITY] if transfer else r["after_accuracy"] for r in aggregate]
        ax.bar(x, [100 * d["mean"] for d in stats], yerr=[100 * d["sample_sd"] for d in stats], capsize=3)
        ax.axhline(100 * cfg["criteria"]["transfer_accuracy" if transfer else "acquisition_accuracy"], color="gray", linestyle=":")
        ax.set(xticks=x, xticklabels=[name(c, True) for c in conditions], ylim=(0, 116),
               ylabel=label("정답 선택 확률 (%)", "Correct-choice probability (%)"),
               title=label("identity 완전 새15쌍·습득과 동시 통과 수", "Identity fully unseen15; joint passing seeds") if transfer else label("연습한 2대4·습득 통과 수", "Trained2-vs-4; acquisition passing seeds"))
        ax.tick_params(axis="x", labelsize=8)
        for i, row in enumerate(aggregate):
            count = row["verdict"]["joint_identity_transfer_count" if transfer else "acquisition_count"]
            ax.text(i, 107, f"{count}/{len(seeds)}", ha="center", fontsize=9)
    fig.suptitle(label("평균±표본 SD; 숫자는 각 시드의 사전 gate 통과 수", "Mean ± sample SD; labels count seedwise frozen gates"))
    fig.text(.5, .01, label("평균 막대가 75%를 넘는 것과 8/10시드 gate는 다르다. 일반화는 습득도 요구한다.", "A mean above75% differs from the8/10 seed gate; transfer also requires acquisition."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    save(fig, "acquisition_unseen.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    index = {r["condition"]: r for r in aggregate}
    for ax, chosen, title in zip(axes, groups, (label("선형 세 조건", "Linear controls"), label("출력만 학습", "Readout-only"), label("내부도 학습", "Internal and readout"))):
        for condition in chosen:
            stats = index[condition]["focal_after"]
            ax.errorbar(ALPHAS, [100 * d["mean"] for d in stats], yerr=[100 * d["sample_sd"] for d in stats], marker="o", capsize=3, label=name(condition))
        ax.axhline(50, color="gray", linestyle=":")
        ax.set(xscale="log", xticks=ALPHAS, xticklabels=["1/4", "1/2", "1", "1.5"], ylim=(0, 105),
               xlabel=label("수량 코드의 간격 배율", "Input-code gap scale"), title=title,
               ylabel=label("새 5대7의 정답 선택 확률 (%)", "Correct-choice probability on unseen5-vs-7 (%)"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8)
    fig.suptitle(label("같은 학습 상태를 네 코드로 평가: 모든 시드 평균±표본 SD", "One trained state evaluated under four maps: all-seed mean ± sample SD"))
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, "focal_response.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_curve(axes[0], conditions[:3])
    axes[0].set_title(label("정답과 관련된 보상·척도·무작위 보상", "Relevant reward, scaling and random reward"))
    keys = ("iid_correct_minus_random", "identity_unseen_correct_minus_random", "iid_normalized_minus_raw", "identity_unseen_normalized_minus_raw")
    labels = [label("보상 차이\n2대4", "Reward gap\n2-vs-4"), label("보상 차이\n새15쌍", "Reward gap\nunseen15"), label("척도 차이\n2대4", "Scale gap\n2-vs-4"), label("척도 차이\n새15쌍", "Scale gap\nunseen15")]
    positions = np.arange(4)
    for i, key in enumerate(keys):
        values = np.asarray([r[key] for r in contrasts])
        offsets = np.linspace(-.13, .13, len(seeds))
        axes[1].scatter(i + offsets, 100 * values, s=24, alpha=.6)
        stats = contrast_statistics[key]
        axes[1].errorbar(i, 100 * stats["mean"], yerr=100 * stats["sample_sd"], color="black", marker="s", capsize=4)
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set(xticks=positions, xticklabels=labels, ylabel=label("같은 시드의 정답 확률 차이 (%포인트)", "Paired-seed probability difference (percentage points)"),
                title=label("점: 모든 시드; 검은 표시: 평균±표본 SD", "Dots: all seeds; black: mean ± sample SD"))
    fig.tight_layout()
    save(fig, "linear_controls.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.3))
    for ax, key, title in zip(axes, ("initial_feature_gap_rms", "feature_scale"), (label("초기 raw 특징 차이의 RMS", "Initial raw feature-gap RMS"), label("고정한 나눗값", "Fixed divisor"))):
        stats = [r["resources"][key] for r in aggregate]
        ax.bar(x, [d["mean"] for d in stats], yerr=[d["sample_sd"] for d in stats], capsize=3)
        ax.set(xticks=x, xticklabels=[name(c, True) for c in conditions], yscale="log", title=title,
               ylabel=label("척도 값 (로그 축)", "Scale value (log axis)"))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(label("훈련 입력·초기 특징만으로 정한 척도: 전체 시드 평균±표본 SD", "Scale determined from TRAIN inputs and initial features only: all-seed mean ± sample SD"))
    fig.text(.5, .01, label("원시 선형만 나누지 않는다. 초기 전체 크기를 맞춰도 자원·특징 방향·학습 후 크기는 다르다.", "Raw linear is unscaled; equal initial RMS does not equalize capacity, feature directions or later scale."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    save(fig, "feature_scales.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    changes = (("internal_change_l2", label("내부 가중치", "Internal weights")), ("readout_change_l2", label("출력 계수", "Readout weights")),
               ("probe_kc_change_l2", label("고정 입력 KC 활성", "KC activity for fixed inputs")), ("probe_mbon_change_l2", label("고정 입력 raw MBON 활성", "Raw MBON activity for fixed inputs")),
               ("probe_decision_change_l2", label("고정 입력 선택 특징 활성", "Decision-feature activity for fixed inputs")))
    for ax, (key, title) in zip(axes.flat, changes):
        for condition in conditions:
            times, data = _curve_matrix(curves, seeds, condition, "iid", key)
            ax.plot(times, data.mean(axis=0), label=name(condition))
        ax.set(title=title, xlabel=label("학습 시행", "Training trials"), ylabel=label("시작 상태와의 거리 (L2)", "L2 distance from initial state"))
    axes.flat[-1].axis("off")
    handles, texts = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, texts, loc="center", fontsize=8)
    fig.suptitle(label("가중치와 활성 변화는 서로 다른 기록이며 크기가 학습의 질을 뜻하지 않는다", "Weight and activity changes are distinct; magnitude does not measure learning quality"))
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, "changes.png")

    table_rows, resource_rows, focal_rows = [], [], []
    for row in aggregate:
        verdict = row["verdict"]
        def percent(stat):
            return f"{100*stat['mean']:.2f}±{100*stat['sample_sd']:.2f}%"
        table_rows.append(f"| {NAMES[row['condition']]} | {percent(row['after_accuracy'])} | {percent(row['unseen_after'][IDENTITY])} | {verdict['acquisition_count']}/{len(seeds)} | {verdict['joint_identity_transfer_count']}/{len(seeds)} |")
        res = row["resources"]
        resource_rows.append(f"| {NAMES[row['condition']]} | {res['readout_parameters']} | {res['internal_parameters']} | {res['initial_feature_gap_rms']['mean']:.6f} | {res['feature_scale']['mean']:.6f} |")
        focal_rows.append("| " + NAMES[row["condition"]] + " | " + " | ".join(percent(d) for d in row["focal_after"]) + " |")
    performance_table = "\n".join(["| 조건 | 최종2대4 | identity 완전 새15쌍 | 습득 gate | 습득+새15쌍 gate |", "|---|---:|---:|---:|---:|", *table_rows])
    resource_table = "\n".join(["| 조건 | 출력 계수 수 | 학습 내부 연결 수 | 초기 gap RMS 평균 | 고정 나눗값 평균 |", "|---|---:|---:|---:|---:|", *resource_rows])
    focal_table = "\n".join(["| 조건 | 1/4 간격 | 1/2 간격 | 기존 간격 | 1.5배 간격 |", "|---|---:|---:|---:|---:|", *focal_rows])
    failure_table = "모든 조건·시드가 습득 gate를 통과했다." if not failed else "\n".join([
        "| 습득 미달 조건 | 시드 | 최종2대4 | 초기 대비 향상 |", "|---|---:|---:|---:|",
        *[f"| {NAMES[r['condition']]} | {r['seed']} | {100*r['after_accuracy']:.4f}% | {100*r['gain']:.4f}%포인트 |" for r in failed]])
    h_r, h_g = summary["hypotheses"]["H_R"], summary["hypotheses"]["H_G"]
    fixture_notice = "**이 실행은 소프트웨어 확인용 작은 fixture이며 주요 연구 결과가 아니다.**\n" if cfg["scope"] == "integration_test" else ""
    report = f"""# 독립 보상 선형 학습 — 결과
{fixture_notice}

단순한 모델을 아무 계수도 배우지 않은 상태에서 시작하고, 2개와 4개 중 고른 선택의 보상으로 {cfg['trials']}번 학습했다. 이전 모델의 점수나 가중치는 제공하지 않았다. 학습하지 않은 수량에서의 반응은 별도로 평가했다. 수량의 숫자 순서는 이미 입력에 있으며 실제 물체를 세는 실험은 아니다.

## 주 판정

**H-R: {'통과' if h_r['criterion_met'] else '미달'}.** 정규화 선형의 습득은 {h_r['acquisition_count']}/{len(seeds)}시드다. 같은 시드의 최종2대4 확률에서 정답 선택 보상−무작위 보상 차이는 전체 평균 {100*h_r['mean_iid_reward_gap']:.3f}%포인트, 양의 차이는 {h_r['positive_seed_count']}/{len(seeds)}시드다. 사전 H-R은 습득8/10 이상, 전체 쌍 차이 평균15%포인트 이상, 양의 차이8/10 이상을 모두 요구한다.

**H-G: {'통과' if h_g['criterion_met'] else '미달'}.** 정규화 선형의 습득과 identity 완전 새15쌍 정답 선택 확률75% 이상을 동시에 충족한 수는 {h_g['joint_identity_transfer_count']}/{len(seeds)}다. 사전 기준은8/10 이상이다. 습득 미달의 높은 새 입력 성적을 습득 후 일반화로 세지 않았다. 현재 fixture의 시드 수가 작으면 분율 gate는 구현 확인이며 연구용8/10 판단을 대신하지 않는다.

## 모든 시드의 성적

아래는 전체 시드 평균±표본 표준편차다. 확률 80%는 주어진 사례에서 정답을 고를 평균 확률이며, 확률이 큰 쪽만 고르는 greedy 정확도와 다르다. 마지막 두 열은 평균 막대 기준이 아니라 각 시드의 사전 gate 통과 수다. 계산 시드는 초파리 개체나 인간 참가자가 아니다.

{performance_table}

모든 초기 head는0이고 학습 전 정답 확률은0.5다. 습득 gate는 최종2대4 확률75% 이상과 초기 대비15%포인트 이상 향상을 모두 요구한다. 새15쌍 gate는 습득도 함께 요구한다. [쌍별 기록](pair_summary.csv)에 15완전 미학습쌍·12anchor 포함쌍, 네 코드, 학습 전후 확률과 greedy 방향을 남겼다. 조건을 다시 고르거나 성공 기준을 바꾸지 않았다.

## 보상과 척도의 구분

정답과 무관한 보상을 주는 `linear_random_reward`는 같은 입력과 초기 척도를 쓰는 대조다. 같은 행동 uniform을 사용해도 정책의 확률이 변하면 선택은 달라질 수 있다. 유한한 무작위 보상 표본에서 우연 편향이 생길 수 있으므로 결과를 자동으로50%라고 만들지 않았다.

두 선택에서 결정적인0/1 보상을 받으면 선택과 보상으로 정답쪽을 복원할 수 있다. optimizer에는 정답 라벨을 전달하지 않지만 보상 정보량이 라벨과 동등하다. 드문·지연 보상에서 스스로 수량 개념을 발견한 결과로 확대하지 않는다.

초기 전체 훈련입력의 후보 특징 차이 RMS를 무보상으로 계산해 고정했다. raw 선형만 나누지 않았다. 사후 성능을 보고 척도를 고르거나 학습 중 다시 맞추지 않았다. 이는 훈련입력 전체를 사전 관찰하는 batch 전처리이며 미래 입력을 전혀 모르는 pure-online 설정이 아니다. 아래 수치는 시드 평균이며 파라미터 수 목록은 조건 내 모든 시드의 값이다.

{resource_table}

같은 400시행·출력 학습률·초기 gap RMS는 파라미터 수, 각 특징 방향, 내부 가소성 자원까지 같게 만들지 않는다. 이번 네트워크는 초기 head와 정규화가 이전 연구와 다르다. 이전 성적을 같은 실험의 개선값으로 직접 비교하지 않는다. [분석 원자료](analysis_summary.json)에 정규화−원시 선형과 정답 보상−무작위 보상의 시드별 쌍 차이, 전체 평균·SD, 최초 관측 습득 checkpoint와 일곱 선형 계수를 보존했다.

## 새5대7의 입력 간격

5·7은 둘 다 학습하지 않은 수량이다. 네 코드 모두 훈련2·4와 순서를 유지하며5·7의 중간값은 같고 코드 간격만 달라진다. 주 H-G에는 identity 간격만 사용했다. 아래 네 간격 결과는 보조이고 새로운 규칙 전이 또는 실제 수량 비율 효과가 아니다.

{focal_table}

## 습득 실패의 보존

{failure_table}

이 실패도 전체 평균·주 보상 쌍 비교에서 삭제하지 않았다. 새 자료에 맞춰 계수를 다시 적합하거나 시행을 늘리지 않았다. 습득+identity 전이 미달 전체 목록은 [분석 원자료](analysis_summary.json)에 있다. 실패는 이 입력·척도·갱신·예산 범위의 결과이며 모든 선형 학습의 불가능성 또는 실제 배선의 필요성을 뜻하지 않는다.

## 검증 범위와 연구 목표

총 실제 갱신 호출은 {summary['actual_training_trials']:,}번이며 네 코드 평가가 한 학습 상태를 공유한다. [사전 프로토콜](../../docs/reward-linear-protocol.md) · [계산적 배경과 문헌](../../docs/reward-linear-background.md) · [모든 시드·gate](summary.json) · [실행 정보](metadata.json) · [독립 검산](verification.json)을 함께 보존한다. 검산의 통과 여부는 별도 기록에서 확인한다. 이 보고서는 저장된 판정을 시각화한 파생 산출물이며 원자료 해시 목록을 고치지 않는다.

선형 계수와 네트워크 출력 계수의 변화, 내부 가중치 변화, 고정 입력의 raw 활성과 정규화 특징 활성은 각각 기록한다. 내부 학습은 단순화한 계산 규칙이며 실제DAN 신호, LC11 입력 경로, 재귀 회로와 물체 그림을 구현하지 않았다. 경험 의존 변화는Level2, 습득 이후 이 순서 제공 표현에서 확인된 새 입력 반응은 제한된Level3다. 인간·실제 초파리와 정량 비교한Level4 또는 관련 신경 기작Level5는 검증하지 않았다.

인간 기초 수량 학습과 비교할 장기 목표는 유지한다. 현재 공개 인간 자료는 참가자별 전후 요약이며 동일한 입력 코드·시행별 보상 기록이 없다. 이번 보상·척도·자원 차이를 구분한 결과는 후속 비교 가설의 재료이며 인간의 수학 이해나 교육 효과의 결론이 아니다.

![모델 조건별 학습 곡선](learning_curves.png)

![습득과 새입력 반응](acquisition_unseen.png)

![선형 보상·척도 대조](linear_controls.png)

![새5대7의 간격 반응](focal_response.png)

![초기 특징 척도](feature_scales.png)

![가중치와 활성 변화의 구분](changes.png)
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return aggregate
