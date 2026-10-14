"""All-seed descriptive reversal analysis; immutable inputs and no refitting."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .experiment import write_json


NAMES = {
    "linear_normalized": "단순 선형·척도 정규화",
    "fixed": "실제 배선·출력만 학습", "plastic": "실제 배선·내부도 학습",
    "rewired_fixed": "재배선·출력만 학습", "rewired_plastic": "재배선·내부도 학습",
    "ann_fixed": "일반 신경망·출력만 학습", "ann_plastic": "일반 신경망·내부도 학습",
    "expected_linear": "조건부 행동 평균 참조",
}
SHORT_NAMES = {
    "linear_normalized": "단순 선형\n정규화", "fixed": "실제 배선\n출력만",
    "plastic": "실제 배선\n내부도", "rewired_fixed": "재배선\n출력만",
    "rewired_plastic": "재배선\n내부도", "ann_fixed": "일반망\n출력만",
    "ann_plastic": "일반망\n내부도",
}
HISTORY_NAMES = {
    "fresh_low": "처음부터 작은 쪽", "high_to_low": "큰 쪽 → 작은 쪽",
    "high_to_low_reset_head": "큰 쪽 → 선택계수 0 → 작은 쪽",
    "low_to_low": "작은 쪽 → 작은 쪽",
}
HISTORY_SHORT = {
    "fresh_low": "처음부터\n작은 쪽", "high_to_low": "큰 쪽 →\n작은 쪽",
    "high_to_low_reset_head": "큰 쪽 →\n계수 0", "low_to_low": "작은 쪽 →\n작은 쪽",
}
COLORS = {"fresh_low": "#247ba0", "high_to_low": "#d1495b",
          "high_to_low_reset_head": "#8b5fbf", "low_to_low": "#2a9d8f"}


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty seed statistics required")
    return {"mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.}


def _matrix(index, seeds, condition, history, times, split, metric):
    values = np.asarray([[float(index[seed, condition, history, "phase2", int(t), split][metric])
                          for t in times] for seed in seeds])
    if values.shape != (len(seeds), len(times)) or not np.isfinite(values).all():
        raise ValueError("Complete finite phase2 curve required")
    return values


def _raw_margin_records(output, seeds, conditions, histories, times):
    """Reconstruct unclipped margins from archived inputs and checkpoint state."""
    rows = []
    for seed in seeds:
        with np.load(output / f"seed_{seed}" / "stimuli.npz", allow_pickle=False) as stimuli:
            inputs = {split: np.concatenate((stimuli[f"{split}_counts"][..., None] / 8.,
                                             .1 * (stimuli[f"{split}_nuisance"] - .5)), axis=-1)
                      for split in ("iid", "ood")}
            signs = {split: 2 * np.argmin(stimuli[f"{split}_counts"], axis=-1) - 1
                     for split in ("iid", "ood")}
        for condition in conditions:
            linear = condition in ("linear_normalized", "expected_linear")
            path = output / f"seed_{seed}" / condition
            kc = {}
            if not linear:
                with np.load(path / histories[0] / "phase2_0000.npz", allow_pickle=False) as initial:
                    kc = {split: np.maximum(x @ initial["encoder"].T + initial["bias"], 0)
                          for split, x in inputs.items()}
            for history in histories:
                for trial in times:
                    with np.load(path / history / f"phase2_{trial:04d}.npz", allow_pickle=False) as saved:
                        divisor, head = float(saved["feature_scale"]), saved["readout_weights"]
                        if not np.isfinite(divisor) or divisor <= 0:
                            raise ValueError("Positive archived feature scale required")
                        if not linear:
                            gain = np.bincount(saved["pre"], weights=saved["internal_weights"] * head[saved["post"]], minlength=kc["iid"].shape[-1]) / divisor
                        for split, x in inputs.items():
                            scores = x @ (head / divisor) if linear else kc[split] @ gain
                            margin = scores[:, 1] - scores[:, 0]
                            p_right = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
                            if not np.isfinite(margin).all() or not np.allclose(p_right, saved[f"{split}_p_right"], atol=2e-12, rtol=0):
                                raise ValueError("Raw margin reconstruction disagrees with archived cases")
                            rows.append({"seed": seed, "condition": condition, "history": history,
                                         "phase2_trial": int(trial), "split": split,
                                         "mean_target_raw_margin": float(np.mean(signs[split] * margin)),
                                         "raw_margin_rms": float(np.sqrt(np.mean(margin ** 2))),
                                         "max_abs_raw_margin": float(np.max(np.abs(margin))),
                                         "score_clip_fraction": float(np.mean(np.abs(margin) > 40))})
    return rows


def analyze(output):
    """Read completed records and write only declared derived reports and plots."""
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    resources = json.loads((output / "model_resources.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed" or summary["status"] != "completed":
        raise ValueError("Analysis requires a completed run")
    cfg = metadata["config"]
    conditions, histories, seeds = cfg["conditions"], cfg["histories"], cfg["seeds"]
    all_conditions = [*conditions, "expected_linear"]
    times = np.arange(0, cfg["trials_per_phase"] + 1, cfg["checkpoint_every"])
    outcomes = summary["outcomes"]
    keyed = {(int(r["seed"]), r["condition"], r["history"]): r for r in outcomes}
    expected_outcomes = {(s, c, h) for s in seeds for c in all_conditions for h in histories}
    if len(keyed) != len(outcomes) or set(keyed) != expected_outcomes:
        raise ValueError("All registered seed/condition/history outcomes required")
    resource_index = {(int(r["seed"]), r["condition"], r["history"]): r for r in resources}
    if len(resource_index) != len(resources) or set(resource_index) != expected_outcomes:
        raise ValueError("All model resource records required")
    with (output / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    index = {(int(r["seed"]), r["condition"], r["history"], r["phase"], int(r["trial"]), r["split"]): r for r in curves}
    expected_curves = {(s, c, h, phase, int(t), split)
                       for s in seeds for c in all_conditions for h in histories
                       for phase in ("phase1", "phase2") if phase != "phase1" or h != "fresh_low"
                       for t in times for split in ("iid", "ood")}
    if len(index) != len(curves) or set(index) != expected_curves:
        raise ValueError("Complete unique registered curves required")
    verdicts = {r["condition"]: r for r in summary["family_verdicts"]}
    if len(verdicts) != len(conditions) or set(verdicts) != set(conditions):
        raise ValueError("All actual-condition verdicts required")
    contrast_index = {(int(r["seed"]), r["condition"]): r for r in summary["paired_contrasts"]}
    if len(contrast_index) != len(summary["paired_contrasts"]) or set(contrast_index) != {(s, c) for s in seeds for c in conditions}:
        raise ValueError("All paired seed contrasts required")
    def matrix(condition, history, metric="p_target", split="iid"):
        return _matrix(index, seeds, condition, history, times, split, metric)

    metric_keys = ("phase2_before", "phase2_after", "phase2_gain", "phase2_auc",
                   "phase2_baseline_adjusted_auc", "phase2_ood_after", "phase2_greedy_after")
    contrast_keys = ("endpoint_cost", "raw_auc_cost", "baseline_adjusted_auc_cost", "gain_cost",
                     "reset_minus_reverse_endpoint", "reset_minus_fresh_endpoint", "repeated_minus_reverse_endpoint")
    aggregate, linear_coefficients = [], []
    for condition in conditions:
        history_rows = {}
        for history in histories:
            rows = [keyed[seed, condition, history] for seed in seeds]
            stats = {key: _statistics([r[key] for r in rows]) for key in metric_keys}
            stats.update({"n_seeds": len(seeds),
                          "phase1_acquisition_count": None if history == "fresh_low" else sum(bool(r["phase1_acquired"]) for r in rows),
                          "fresh_acquisition_count": sum(bool(r["fresh_acquired"]) for r in rows) if history == "fresh_low" else None,
                          "phase2_target_pass_count": sum(bool(r["phase2_target_passed"]) for r in rows),
                          "qualified_adaptation_count": sum(bool(r["qualified_adaptation"]) for r in rows),
                          "first_observed_75_checkpoints": [{"seed": r["seed"], "trial": r["first_observed_75_checkpoint"]} for r in rows]})
            if history != "fresh_low":
                stats.update({key: _statistics([r[key] for r in rows]) for key in ("phase1_before", "phase1_after")})
            history_rows[history] = stats
        scales = [resource_index[seed, condition, "fresh_low"] for seed in seeds]
        aggregate.append({"condition": condition, "n_seeds": len(seeds), "histories": history_rows,
                          "paired_contrasts": {key: _statistics([contrast_index[seed, condition][key] for seed in seeds]) for key in contrast_keys},
                          "resources": {key: _statistics([r[key] for r in scales]) for key in ("feature_scale", "initial_feature_gap_rms")},
                          "readout_parameters": sorted({r["readout_parameters"] for r in scales}),
                          "internal_parameters": sorted({r["internal_parameters"] for r in scales}),
                          "verdict": verdicts[condition]})
    for condition in ("linear_normalized", "expected_linear"):
        for seed in seeds:
            for history in histories:
                with np.load(output / f"seed_{seed}" / condition / history / f"phase2_{cfg['trials_per_phase']:04d}.npz", allow_pickle=False) as saved:
                    if saved["readout_weights"].shape != (7,):
                        raise ValueError("Seven linear coefficients required")
                    linear_coefficients.append({"seed": seed, "condition": condition, "history": history,
                                                "coefficient_order": ["q", "nuisance1", "nuisance2", "nuisance3", "nuisance4", "nuisance5", "nuisance6"],
                                                "normalized_feature_coefficients": saved["readout_weights"].tolist(),
                                                "raw_input_coefficients": (saved["readout_weights"] / float(saved["feature_scale"])).tolist()})

    reference_aggregate, reference_rows = [], []
    for history in histories:
        reference_stats = {}
        for split in ("iid", "ood"):
            actual = matrix("linear_normalized", history, split=split)
            reference = matrix("expected_linear", history, split=split)
            difference = actual - reference
            per_seed_mae = np.abs(difference).mean(axis=1)
            reference_stats[split] = {"per_seed_checkpoint_mae": _statistics(per_seed_mae),
                                      "per_seed_checkpoint_mae_values": [{"seed": s, "mae": float(v)} for s, v in zip(seeds, per_seed_mae)],
                                      "mean_trajectory_mae": float(np.abs(difference.mean(axis=0)).mean()),
                                      "max_individual_checkpoint_abs_difference": float(np.abs(difference).max()),
                                      "stochastic_mean_trajectory": actual.mean(axis=0).tolist(),
                                      "reference_mean_trajectory": reference.mean(axis=0).tolist(),
                                      "mean_trajectory_difference": difference.mean(axis=0).tolist(),
                                      "reference_endpoint": _statistics(reference[:, -1])}
            for i, seed in enumerate(seeds):
                for j, trial in enumerate(times):
                    reference_rows.append({"seed": seed, "history": history, "split": split, "phase2_trial": int(trial),
                                           "stochastic_p_target": float(actual[i, j]), "reference_p_target": float(reference[i, j]),
                                           "difference": float(difference[i, j]), "absolute_difference": float(abs(difference[i, j]))})
        reference_aggregate.append({"history": history, "n_seeds": len(seeds), "n_checkpoints": len(times), **reference_stats})
    with (output / "reference_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)
    actual_outcomes = [r for r in outcomes if r["condition"] in conditions]
    qualified_failures = [r for r in actual_outcomes if not r["qualified_adaptation"]]
    phase1_failures = [r for r in actual_outcomes if r["history"] != "fresh_low" and not r["phase1_acquired"]]
    fresh_failures = [r for r in actual_outcomes if r["history"] == "fresh_low" and not r["fresh_acquired"]]
    target_failures = [r for r in actual_outcomes if not r["phase2_target_passed"]]
    raw_margin_rows = _raw_margin_records(output, seeds, all_conditions, histories, times)
    with (output / "raw_margin_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(raw_margin_rows[0]))
        writer.writeheader()
        writer.writerows(raw_margin_rows)
    derived = {"scope": cfg["scope"], "n_seeds": len(seeds), "all_seeds_included": True,
               "aggregate": aggregate, "primary_hypothesis_from_run": summary["primary_hypothesis"],
               "conditional_reference": {"scope": "descriptive label-privileged conditional-action drift; not ensemble expectation or reward-learning control",
                                          "phase2_trials": times.tolist(), "histories": reference_aggregate,
                                          "pooled_individual_checkpoint_mae": float(np.mean([r["absolute_difference"] for r in reference_rows]))},
               "linear_coefficients": linear_coefficients, "qualified_adaptation_failures": qualified_failures,
               "phase1_acquisition_failures": phase1_failures, "fresh_acquisition_failures": fresh_failures,
               "phase2_target_failures": target_failures, "actual_reward_updates": summary["actual_training_trials"],
               "conditional_reference_updates": summary["expected_reference_updates"],
               "raw_margin_reconstruction": {"rows": len(raw_margin_rows), "source": "archived whole IID/OOD inputs and checkpoint encoder, topology, internal/readout weights and fixed feature scale",
                                             "max_case_replay_tolerance": 2e-12, "retraining": False},
               "notice": "Derived all-seed statistics only; no retraining, threshold changes, inference of intrinsic slower learning, p-value, biological replicates or common neural mechanism."}
    write_json(output / "analysis_summary.json", derived)

    try:
        font_manager.findfont("Malgun Gothic", fallback_to_default=False)
        plt.rcParams["font.family"] = "Malgun Gothic"
        korean = True
    except ValueError:
        korean = False
    plt.rcParams["axes.unicode_minus"] = False
    def label(ko, en):
        return (ko if korean else en).replace("\u2212", "-")
    def name(condition, short=False):
        return (SHORT_NAMES if short else NAMES)[condition] if korean else condition.replace("_", "\n") if short else condition
    def history_name(history, short=False):
        return (HISTORY_SHORT if short else HISTORY_NAMES)[history] if korean else history.replace("_", "\n") if short else history
    def save(fig, filename):
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    def draw_curve(ax, condition, history, metric="p_target", split="iid", percent=True):
        values = matrix(condition, history, metric, split)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(times))
        factor = 100 if percent else 1
        ax.plot(times, factor * mean, marker="o", color=COLORS[history], label=history_name(history))
        ax.fill_between(times, factor * (mean - sd), factor * (mean + sd), color=COLORS[history], alpha=.1)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, conditions):
        for history in histories:
            draw_curve(ax, condition, history)
        ax.axhline(75, color="gray", linestyle=":", linewidth=1)
        ax.set(title=name(condition), xlabel=label("2단계 연습 시행", "Phase2 training trials"),
               ylabel=label("작은 쪽 선택 확률 (%)", "Lower-choice probability (%)"), ylim=(-2, 103))
        ax.tick_params(axis="both", labelsize=8)
    for ax in list(axes.flat)[len(conditions):]:
        ax.axis("off")
    handles, texts = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(label("2단계는 모두 작은 쪽 보상: 전체 시드 평균±표본 SD", "Phase2 rewards lower in all histories: all-seed mean ± sample SD"))
    fig.text(.5, .065, label("출발 성적·회복량·끝점을 구분한다. 곡선 높이만으로 학습 능력이 느려졌다고 결론내리지 않는다.", "Separate starting performance, gain and endpoint; curve height does not prove an intrinsic learning-rate deficit."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .96))
    save(fig, "history_curves.png")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, condition in enumerate(conditions):
        values = np.asarray([contrast_index[seed, condition]["endpoint_cost"] for seed in seeds])
        ax.scatter(i + np.linspace(-.13, .13, len(seeds)), 100 * values, alpha=.6, s=24, color="#247ba0")
        stats = _statistics(values)
        ax.errorbar(i, 100 * stats["mean"], yerr=100 * stats["sample_sd"], marker="s", color="black", capsize=4)
    ax.axhline(0, color="gray", linestyle=":")
    ax.axhline(100 * cfg["criteria"]["endpoint_cost"], color="#d1495b", linestyle="--",
               label=label("평균 끝점 차이 기준 (H-B는 습득·시드 수도 요구)", "Mean endpoint threshold (H-B also requires acquisition and seed counts)"))
    ax.set(xticks=np.arange(len(conditions)), xticklabels=[name(c, True) for c in conditions],
           ylabel=label("처음부터 − 반전: 최종 확률 차이 (%포인트)", "Fresh minus reversal final probability (percentage points)"),
           title=label("같은 새 규칙 예산 뒤 끝점 비용: 점은 모든 시드, 검정은 평균±표본 SD", "Endpoint cost after matched new-rule exposure: all seeds; black mean ± sample SD"))
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=8)
    fig.text(.5, .01, label("단순 선형이 주 대상이다. 다른 여섯 조건은 보조 판정이며 전체 학습 자원은 같지 않다.", "Linear is primary; six networks are secondary, and total learning resources differ."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "endpoint_cost.png")

    agg_index = {r["condition"]: r for r in aggregate}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    positions = np.arange(len(conditions))
    for ax, key, title in zip(axes, ("phase2_before", "phase2_gain", "phase2_after"),
                              (label("2단계 출발점", "Phase2 starting performance"), label("2단계 회복·향상량", "Phase2 gain"), label("2단계 마지막 성적", "Phase2 endpoint"))):
        for i, history in enumerate(histories):
            stats = [agg_index[c]["histories"][history][key] for c in conditions]
            ax.bar(positions + (i - 1.5) * .19, [100 * s["mean"] for s in stats], width=.18,
                   yerr=[100 * s["sample_sd"] for s in stats], capsize=2, color=COLORS[history], label=history_name(history))
        ax.set(xticks=positions, xticklabels=[name(c, True) for c in conditions], title=title,
               ylabel=label("확률 변화 (%포인트)", "Probability gain (percentage points)") if key == "phase2_gain" else label("작은 쪽 선택 확률 (%)", "Lower-choice probability (%)"))
        if key != "phase2_gain":
            ax.set_ylim(-2, 104)
        ax.tick_params(axis="x", labelsize=8)
    handles, texts = axes[0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle(label("전체 시드 평균±표본 SD: 낮은 출발점에서는 회복량이 더 클 수도 있다", "All-seed mean ± sample SD: a lower baseline can yield a larger gain"))
    fig.tight_layout(rect=(0, .13, 1, .94))
    save(fig, "initial_gain_endpoint.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, history in zip(axes.flat, histories):
        for condition, color in (("linear_normalized", "#247ba0"), ("expected_linear", "#e28f20")):
            values = matrix(condition, history)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(times))
            ax.plot(times, 100 * mean, marker="o", color=color,
                    label=label("실제 선택·보상 학습", "Stochastic choice-reward learning") if condition == "linear_normalized" else label("조건부 행동 평균 참조", "Conditional-action drift reference"))
            ax.fill_between(times, 100 * (mean - sd), 100 * (mean + sd), color=color, alpha=.12)
        ax.axhline(75, color="gray", linestyle=":")
        ax.set(title=history_name(history), xlabel=label("2단계 시행", "Phase2 trials"),
               ylabel=label("작은 쪽 선택 확률 (%)", "Lower-choice probability (%)"), ylim=(-2, 103))
        ax.legend(fontsize=8)
    fig.suptitle(label("단순 선형만 비교: 전체 시드 평균±표본 SD", "Linear only: all-seed mean ± sample SD"))
    fig.text(.5, .012, label("참조는 정답 정보를 직접 받는 기술 계산이며 확률적 궤적의 정확한 기대값이나 독립 보상 대조가 아니다.", "The label-privileged reference is neither an exact stochastic expectation nor an independent reward-learning control."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    save(fig, "conditional_reference.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for i, row in enumerate(reference_aggregate):
        values = np.asarray([r["mae"] for r in row["iid"]["per_seed_checkpoint_mae_values"]])
        axes[0].scatter(i + np.linspace(-.1, .1, len(seeds)), 100 * values, color=COLORS[row["history"]], s=24, alpha=.65)
        stats = row["iid"]["per_seed_checkpoint_mae"]
        axes[0].errorbar(i, 100 * stats["mean"], yerr=100 * stats["sample_sd"], color="black", marker="s", capsize=4)
        axes[1].plot(times, 100 * np.asarray(row["iid"]["mean_trajectory_difference"]), marker="o", color=COLORS[row["history"]], label=history_name(row["history"]))
    axes[0].set(xticks=np.arange(len(histories)), xticklabels=[history_name(h, True) for h in histories],
                ylabel=label("개별 경로의 checkpoint 평균 절대오차 (%포인트)", "Per-seed checkpoint MAE (percentage points)"),
                title=label("전체 시드의 개별 오차: 점, 평균±표본 SD", "Individual errors: all seeds, mean ± sample SD"))
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set(xlabel=label("2단계 시행", "Phase2 trials"),
                ylabel=label("전체 평균 확률의 실제 − 참조 (%포인트)", "Actual minus reference seed-mean probability (percentage points)"),
                title=label("시드 평균 곡선의 차이: 개별 절대오차와 다르다", "Seed-mean curve difference differs from individual MAE"))
    axes[1].legend(fontsize=8)
    fig.text(.5, .01, label("IID 기술 비교이며 성공 임계값은 없다. OOD도 CSV/JSON에 별도 보존한다.", "Descriptive IID comparison with no success threshold; OOD is preserved separately in CSV/JSON."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, "reference_errors.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, condition in zip(axes, ("linear_normalized", "fixed", "plastic")):
        for history in histories[:3]:
            draw_curve(ax, condition, history, "mean_policy_sensitivity", percent=False)
        ax.set(title=name(condition), xlabel=label("2단계 시행", "Phase2 trials"),
               ylabel=label("평균 p(1−p): 조건부 갱신 factor 크기", "Mean p(1-p): conditional-update factor size"), ylim=(-.005, .26))
        ax.legend(fontsize=7)
    fig.suptitle(label("이 갱신식의 계산 지표: dopamine 측정이나 전체 변화량이 아니다", "An algorithmic quantity; neither dopamine measurement nor full update magnitude"))
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, "policy_sensitivity.png")

    raw_index = {(r["seed"], r["condition"], r["history"], r["phase2_trial"], r["split"]): r for r in raw_margin_rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, condition in zip(axes, ("linear_normalized", "fixed", "plastic")):
        for history in histories:
            values = np.asarray([[raw_index[seed, condition, history, int(t), "iid"]["mean_target_raw_margin"] for t in times] for seed in seeds])
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(times))
            ax.plot(times, mean, marker="o", color=COLORS[history], label=history_name(history))
            ax.fill_between(times, mean - sd, mean + sd, color=COLORS[history], alpha=.1)
        ax.axhline(0, color="gray", linestyle=":")
        ax.set(title=name(condition), xlabel=label("2단계 시행", "Phase2 trials"),
               ylabel=label("작은 수량 방향의 raw 점수 차이 평균", "Mean raw score margin toward lower choice"))
        ax.legend(fontsize=7)
    fig.suptitle(label("입력과 저장한 가중치에서 복원한 unclipped 점수 차이: 전체 시드 평균±표본 SD", "Unclipped margins reconstructed from archived inputs and weights: all-seed mean ± sample SD"))
    fig.text(.5, .01, label("양수는 작은 쪽 방향이다. 확률로 바꾸기 전 점수이며 정답률·생물학적 활성 단위가 아니다.", "Positive favors lower; these pre-probability scores are not accuracy or biological activity units."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    save(fig, "raw_margin.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    changes = (("internal_change_l2", label("내부 가중치", "Internal weights")),
               ("readout_change_l2", label("출력 계수", "Readout weights")),
               ("probe_kc_change_l2", label("고정 입력의 KC 활성", "KC activity for fixed input")),
               ("probe_mbon_change_l2", label("고정 입력의 raw MBON 활성", "Raw MBON activity for fixed input")),
               ("probe_decision_change_l2", label("고정 입력의 정규화 특징 활성", "Normalized-feature activity for fixed input")))
    for ax, (metric, title) in zip(axes.flat, changes):
        for condition in conditions:
            values = matrix(condition, "high_to_low", metric)
            ax.plot(times, values.mean(axis=0), label=name(condition))
        ax.set(title=title, xlabel=label("반전 단계 시행", "Reversal phase trials"),
               ylabel=label("학습 전 최초 상태와의 거리 (L2)", "L2 distance from the untrained initial state"))
    axes.flat[-1].axis("off")
    handles, texts = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, texts, loc="center", fontsize=8)
    fig.suptitle(label("큰 쪽 → 작은 쪽 이력의 상태 변화; 선형의 KC/MBON 배열은 비어 있다", "High-to-low state changes; linear KC/MBON arrays are empty"))
    fig.text(.5, .01, label("2단계 시작도 최초 미학습 상태와 비교한다. 내부 변화·출력 변화·활성 변화의 단위와 의미는 다르다.", "Phase2 start is compared with the untrained state; internal, readout and activity changes have different meanings."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, "changes.png")

    def percent(stat):
        return f"{100*stat['mean']:.2f}±{100*stat['sample_sd']:.2f}%"
    endpoint_rows, gate_rows, contrast_rows, baseline_rows, reference_table_rows = [], [], [], [], []
    for row in aggregate:
        h = row["histories"]
        endpoint_rows.append("| " + NAMES[row["condition"]] + " | " + " | ".join(percent(h[history]["phase2_after"]) for history in histories) + " |")
        gate_rows.append("| " + NAMES[row["condition"]] + " | " + " | ".join(f"{h[history]['qualified_adaptation_count']}/{len(seeds)}" for history in histories) + " |")
        v = row["verdict"]
        cost = row["paired_contrasts"]["endpoint_cost"]
        contrast_rows.append(f"| {NAMES[row['condition']]} | {100*cost['mean']:.2f}±{100*cost['sample_sd']:.2f} | {v['positive_seed_count']}/{len(seeds)} | {v['fresh_acquisition_count']}/{len(seeds)} | {v['high_prefix_acquisition_count']}/{len(seeds)} | {'통과' if v['criterion_met'] else '미달'} |")
        for history in histories:
            baseline_rows.append(f"| {NAMES[row['condition']]} | {HISTORY_NAMES[history]} | {percent(h[history]['phase2_before'])} | {100*h[history]['phase2_gain']['mean']:.2f} | {100*h[history]['phase2_auc']['mean']:.2f} |")
    for row in reference_aggregate:
        ref_iid, ref_ood = row["iid"], row["ood"]
        reference_table_rows.append(f"| {HISTORY_NAMES[row['history']]} | {100*ref_iid['per_seed_checkpoint_mae']['mean']:.3f}±{100*ref_iid['per_seed_checkpoint_mae']['sample_sd']:.3f} | {100*ref_iid['mean_trajectory_mae']:.3f} | {100*ref_ood['per_seed_checkpoint_mae']['mean']:.3f} | {percent(ref_iid['reference_endpoint'])} |")
    endpoint_table = "\n".join(["| 모델 | 처음부터 작은 쪽 | 큰 쪽→작은 쪽 | 큰 쪽→선택계수0→작은 쪽 | 작은 쪽→작은 쪽 |", "|---|---:|---:|---:|---:|", *endpoint_rows])
    gate_table = "\n".join(["| 모델 | fresh 습득 | 습득 후 반전 | 습득 후 reset 수정 | 습득 후 같은 규칙 유지 |", "|---|---:|---:|---:|---:|", *gate_rows])
    contrast_table = "\n".join(["| 모델 | 끝점 비용 평균±표본SD (%포인트) | 양수 시드 | fresh 습득 | high 습득 | 사전 비용 기준 |", "|---|---:|---:|---:|---:|---|", *contrast_rows])
    baseline_table = "\n".join(["| 모델 | 이력 | 2단계 시작 확률 | 2단계 gain 평균 (%포인트) | raw AUC 평균×100 |", "|---|---|---:|---:|", *baseline_rows])
    reference_table = "\n".join(["| 이력 | IID 개별경로 MAE 평균±SD (%포인트) | IID 평균곡선 MAE (%포인트) | OOD 개별경로 MAE 평균 (%포인트) | 참조 IID 마지막 확률 |", "|---|---:|---:|---:|---:|", *reference_table_rows])
    if qualified_failures:
        failure_table = "\n".join(["| 자격을 확인한 수정·유지 기준 미달 모델 | 이력 | 시드 | 1단계 습득 | 2단계 최종 IID 확률 |", "|---|---|---:|---|---:|",
                                    *[f"| {NAMES[r['condition']]} | {HISTORY_NAMES[r['history']]} | {r['seed']} | {'해당없음' if r['phase1_acquired'] is None else r['phase1_acquired']} | {100*r['phase2_after']:.4f}% |" for r in qualified_failures]])
    else:
        failure_table = "실제 학습자의 모든 시드·이력이 사전 습득을 확인한 2단계 기준을 통과했다."
    h_b = summary["primary_hypothesis"]
    fixture_notice = "**소프트웨어 확인용 작은 fixture이며 주요 연구 결과가 아니다.**\n" if cfg["scope"] == "integration_test" else ""
    report = f"""# 독립 보상 학습자의 반전 — 결과
{fixture_notice}

처음부터 작은 수량을 선택하도록 배우는 경우와 큰 수량을 배운 뒤 작은 수량으로 바꾸는 경우를 같은 새 규칙 {cfg['trials_per_phase']}시행으로 비교했다. 선택 담당 계수만0으로 돌린 경우와 작은 수량을 계속 연습한 경우도 함께 실행했다. 수량의 순서는 숫자 입력이 이미 제공한다. 실제 그림에서 수량을 알아내는 실험이나 인간의 수학 이해를 시험한 것이 아니다.

## 주 H-B 판정

**단순 선형의 H-B: {'통과' if h_b['criterion_met'] else '미달'}.** 모든 시드의 처음부터−반전 끝점 차이는 평균 {100*h_b['mean_endpoint_cost']:.3f}%포인트, 양수 {h_b['positive_seed_count']}/{len(seeds)}, fresh 습득 {h_b['fresh_acquisition_count']}/{len(seeds)}, 먼저 high 습득 {h_b['high_prefix_acquisition_count']}/{len(seeds)}였다. 사전 주 기준은 전체 평균10%포인트 이상, 양수·fresh 습득·high 습득이 각각8/10 이상인 네 조건을 모두 요구한다. 유의성 검정이나 생물학적 임계값이 아니다. fixture는 시드 분율과 작은 예산의 구현 검사이며 연구용10시드 판정을 대신하지 않는다.

{contrast_table}

단순 선형만 주 대상이고 나머지 여섯 조건은 사전 보조 판정이다. 습득 미달 시드를 제외하지 않았다. 작은 모델 차이로 실제 배선의 우수성·동등성을 주장하지 않는다.

## 마지막 성적과 성공 기준을 구분한다

전체 시드 평균±표본 표준편차다. 성적은 평가 사례에서 작은 쪽을 선택할 확률의 평균이며, 확률이 높은 쪽만 고르는 greedy 성적과 다르다.

{endpoint_table}

아래 숫자는 전체 평균 막대가75%를 넘는지가 아니라 각 시드의 사전 gate를 통과한 수다. 사전 학습이 있는 이력은 1단계 습득(75%와15%포인트 gain) 및 2단계 최종 작은 쪽 확률75%를 함께 요구한다. fresh는2단계75%와 시작 대비15%포인트 gain을 요구한다. 같은 규칙을 계속 연습한 경우에 추가15%포인트 gain을 요구하지 않는다.

{gate_table}

## 출발점·회복량·곡선 높이

{baseline_table}

반전 직후에는 기존에 잘하던 답이 새 규칙에서 틀린 답이 된다. 낮은 출발점 때문에 raw AUC가 낮아질 수 있고, 같은 이유로 회복량이 클 수도 있다. AUC는 저장 checkpoint를 사다리꼴 적분하여 새 규칙 예산으로 나눈 요약이다. 이 표의 지표 한 개로 학습 능력 자체가 느려졌거나 내부 간섭이 독립적으로 증명됐다고 해석하지 않는다. 최초75% 도달은 저장 checkpoint에서만 관측하며 정확한 최초 시행·지속 성공으로 부르지 않는다.

![새 규칙 단계의 모든 이력](history_curves.png)

![전체 시드의 끝점 비용](endpoint_cost.png)

![출발점과 회복량과 끝점](initial_gain_endpoint.png)

## 조건부 행동 평균 참조

별도 참조는 현재 참조 상태에서 행동을 평균낸 factor `(2y−1)p(1−p)`로 일곱 선형 계수를 갱신한다. 정답쪽 정보를 직접 받는 기술 계산이며 실제 보상 학습자와 같은 정보·자원을 갖는 독립 대조가 아니다. 비선형 확률 때문에 이 식을 반복한 참조는 확률적 학습 궤적의 정확한 전체 기대값도 아니다. 별도 성공 기준은 없다.

{reference_table}

개별경로 MAE는 각 시드의 phase2 저장 checkpoint에서 실제−참조 차이의 절댓값을 먼저 평균한다. 평균곡선 MAE는 같은 checkpoint의 전체 시드 평균을 먼저 구한 뒤 차이의 절댓값을 평균한다. 서로 다른 지표이며 상쇄가 생길 수 있다. 단위는 확률의 %포인트이고 정답률 손실이나 독립 보상 학습 성적이 아니다. [시행별 비교 CSV](reference_comparison.csv)에 IID/OOD를 따로 남겼다. 시드마다 입력과 척도도 달라 동일 입력에서 행동만 반복한 ensemble가 아니다.

![선형의 실제 학습과 조건부 행동 평균 참조](conditional_reference.png)

![개별 오차와 평균 곡선 차이는 다르다](reference_errors.png)

## 변화의 기록과 reset

선형과 고정 내부 모델에서 head를0으로 돌리면 같은 새 입력·행동 uniform·척도 아래 fresh와 같은 선택·보상 경로여야 한다. 이것은 구현 무결성 검사이며 새 과학적 성공이 아니다. 누적 갱신 counter는 앞서 학습한400회를 포함하므로 다를 수 있다. 가소성의 reset 뒤 차이는 바뀐 내부와0head의 상호작용이며 뇌의 기억 삭제나 시냅스 기억 기작의 증명이 아니다.

아래 상태 변화는2단계 시작도 **최초 미학습 상태**와 비교한다. 내부 가중치·출력 계수·고정 입력 활성 변화는 서로 다른 기록이다. 단순 선형에는 KC·MBON이 없어 해당 배열이 비어 있으며 활성 변화0을 그 회로의 측정값으로 부르지 않는다. `p(1−p)`는 이 알고리즘의 조건부 factor이지 실제 dopamine 측정이나 전체 가중치 변화량이 아니다.

![확률 함수의 조건부 factor](policy_sensitivity.png)

![저장한 가중치와 입력에서 복원한 raw 점수 차이](raw_margin.png)

raw margin은 primary checkpoint의 내부·출력 가중치·encoder·척도와 전체 IID/OOD 입력에서 학습 없이 복원했다. primary case 파일의 확률과 일치하는지 확인한 파생 기술값이다. [raw margin CSV](raw_margin_summary.csv)에 방향 있는 평균, RMS, 최대 절댓값과 clipping 비율을 남겼다. raw 점수를 직접 저장한 독립 primary field로 오해하지 않는다.

![내부와 출력과 활성의 별도 변화](changes.png)

## 미달 기록

실제 조건의 사전 자격을 확인한2단계 기준 미달은 {len(qualified_failures)}건,1단계 습득 미달은 {len(phase1_failures)}건,fresh 습득 미달은 {len(fresh_failures)}건,2단계 끝점75% 자체 미달은 {len(target_failures)}건이다. 이 집계는 같은 행이 여러 종류에 포함될 수 있으며 합쳐 독립 실패 수로 부풀리지 않는다. 기술 참조는 성공·실패 gate를 적용하지 않고 이 집계에서 제외한다.

{failure_table}

## 기록·재현과 인간 비교

실제 선택·보상 학습은 {summary['actual_training_trials']:,}번, 별도 결정적 기술 참조는 {summary['expected_reference_updates']:,}번이다. 평가·척도 전처리·reset은 실제 보상 갱신 수에 더하지 않았다. fresh는 총400회, 나머지는총800회(주 과학 설정)의 보상 경험으로 전체 경험까지 같지 않다. 출력 계수7개와 네트워크48개 슬롯·내부 추가1474개 연결 역시 같은 자원이라고 주장하지 않는다. 단순 시드는 생물학적 개체나 인간 참가자가 아니다.

[사전 계획](../../docs/bandit-reversal-protocol.md) · [계산적 배경](../../docs/bandit-reversal-background.md) · [모든 원값과 판정](summary.json) · [실행 정보](metadata.json) · [전향적 동결](registration.json) · [독립 검산](verification.json) · [파생 기술 통계](analysis_summary.json)

이번 결과는 인공 입력의 경험 의존 변화 Level2 범위다. 실제 시각 수량 인지·LC11 경로·DAN 구획·자연 망각·수학 규칙 발견을 재현하지 않았다. 새 수량 일반화는이번에실행하지않았다. 사람의 기초 수량·수학 학습과 비교할 장기 목표에서 출발 편향·고정 예산 끝점 비용·회복량·기준 도달·오류 방향이 후보 지표로 남는다. 대응하는 인간 시행별 피드백 원자료가 아직 없어 정량적 인간 유사성 Level4나 공통 신경 기작 Level5를 주장하지 않는다. 새 인간 실험·수학교육 효과도 현재 범위 밖이다. 같은 행동 형태는 같은 신경 원인의 증거가 아니다.

재현은 `study-bandit-reversal-v1` 태그의 별도 checkout과 잠금 환경에서 존재하지 않는 새 결과 경로로 실행한다. 이전 실패나 등록 해시를 최신 소스에 맞춰 고치지 않는다. 모든 그림 band와errorbar는 계산 시드의 표본SD이며 신뢰구간이 아니다.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return derived
