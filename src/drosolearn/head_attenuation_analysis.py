"""All-seed attenuation descriptions; primary files remain immutable.

Endpoint benefit, immediate boundary advantage, and later gain are separate.
The frozen boundary baseline performs no learning and is not a new learner.
"""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .experiment import write_json
from .bandit_reversal_analysis import NAMES, SHORT_NAMES


HISTORY_NAMES = {"fresh_low": "처음부터 작은 쪽", "gamma_0": "γ=0",
                 "gamma_025": "γ=.25", "gamma_05": "γ=.5", "gamma_1": "γ=1"}
GAMMAS = {"gamma_0": 0., "gamma_025": .25, "gamma_05": .5, "gamma_1": 1.}
COLORS = {"fresh_low": "#247ba0", "gamma_0": "#8b5fbf", "gamma_025": "#2a9d8f",
          "gamma_05": "#e09f3e", "gamma_1": "#d1495b"}


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Complete finite seed statistics required")
    return {"mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.}


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(output, seed):
    with np.load(output / f"seed_{seed}" / "stimuli.npz", allow_pickle=False) as saved:
        inputs = {split: np.concatenate((saved[f"{split}_counts"][..., None] / 8.,
                                         .1 * (saved[f"{split}_nuisance"] - .5)), axis=-1)
                  for split in ("iid", "ood")}
        signs = {split: 2 * np.argmin(saved[f"{split}_counts"], axis=-1) - 1
                 for split in ("iid", "ood")}
    return inputs, signs


def _raw_margin(path, inputs):
    """Reconstruct scores without producer learner or any state mutation."""
    with np.load(path, allow_pickle=False) as saved:
        head, divisor = saved["readout_weights"], float(saved["feature_scale"])
        if not np.isfinite(divisor) or divisor <= 0 or not np.isfinite(head).all():
            raise ValueError("Finite head and positive frozen feature scale required")
        if saved["encoder"].shape[0] == 0:
            scores = inputs @ (head / divisor)
        else:
            kc = np.maximum(inputs @ saved["encoder"].T + saved["bias"], 0)
            gain = np.bincount(saved["pre"],
                               weights=saved["internal_weights"] * head[saved["post"]],
                               minlength=kc.shape[-1]) / divisor
            scores = kc @ gain
        margin = scores[:, 1] - scores[:, 0]
        probability = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
        if not np.isfinite(margin).all():
            raise ValueError("Finite reconstructed raw margins required")
        cases = {key: saved[key].copy() for key in saved.files
                 if key.endswith(("_p_right", "_raw_margin"))}
    return margin, probability, cases


def analyze(output):
    """Read a completed, fully covered run and write declared derived artifacts."""
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed" or summary["status"] != "completed":
        raise ValueError("Analysis requires a completed run")
    cfg = metadata["config"]
    seeds, conditions, histories = cfg["seeds"], cfg["conditions"], cfg["histories"]
    if histories != list(HISTORY_NAMES) or cfg["gammas"] != list(GAMMAS.values()):
        raise ValueError("All five registered paths and four gamma levels required")
    all_conditions = [*conditions, "expected_linear"]
    n, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    times = np.arange(0, n + 1, every)
    with (output / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    index = {(int(r["seed"]), r["condition"], r["history"], r["phase"],
              int(r["trial"]), r["split"]): r for r in curves}
    required = {(s, c, h, phase, int(t), split)
                for s in seeds for c in all_conditions
                for h, phase in [("shared_prefix", "phase1"), *[(h, "phase2") for h in histories]]
                for t in times for split in ("iid", "ood")}
    if len(index) != len(curves) or set(index) != required:
        raise ValueError("Complete unique shared-prefix and branch curves required")
    outcomes = summary["outcomes"]
    keyed = {(int(r["seed"]), r["condition"], r["history"]): r for r in outcomes}
    if len(keyed) != len(outcomes) or set(keyed) != {(s, c, h) for s in seeds for c in all_conditions for h in histories}:
        raise ValueError("All registered seed/family/path outcomes required")
    for row in outcomes:
        if row["condition"] == "expected_linear" and any(row[key] is not None for key in
                ("phase1_acquired", "fresh_acquired", "phase2_target_passed", "qualified_adaptation", "first_observed_75_checkpoint")):
            raise ValueError("Descriptive references must not have success flags")
    resources = json.loads((output / "model_resources.json").read_text(encoding="utf-8"))
    resource_index = {(int(r["seed"]), r["condition"], r["history"]): r for r in resources}
    if len(resource_index) != len(resources) or set(resource_index) != set(keyed):
        raise ValueError("All registered model resource records required")

    def matrix(condition, history, metric="p_target", split="iid", phase="phase2"):
        values = np.asarray([[float(index[s, condition, history, phase, int(t), split][metric])
                              for t in times] for s in seeds])
        if not np.isfinite(values).all():
            raise ValueError("Complete finite checkpoint metrics required")
        return values

    aggregate, contrast_rows, failures, fresh_failures, prefix_failures = [], [], [], [], []
    frozen_rows, gamma_rows = [], []
    criteria = cfg["criteria"]
    for condition in conditions:
        prefix = matrix(condition, "shared_prefix", phase="phase1")
        acquired = ((prefix[:, -1] >= criteria["acquisition_accuracy"])
                    & (prefix[:, -1] - prefix[:, 0] >= criteria["acquisition_gain"]))
        history_aggregate = {}
        for h in histories:
            iid, ood = matrix(condition, h), matrix(condition, h, split="ood")
            gain = iid[:, -1] - iid[:, 0]
            auc = np.trapezoid(iid, dx=every, axis=1) / n
            passed = iid[:, -1] >= criteria["adaptation_accuracy"]
            qualified = ((iid[:, -1] >= criteria["acquisition_accuracy"])
                         & (gain >= criteria["acquisition_gain"])) if h == "fresh_low" else (acquired & passed)
            values = {"phase2_before": iid[:, 0], "phase2_after": iid[:, -1], "phase2_gain": gain,
                      "phase2_auc": auc, "phase2_baseline_adjusted_auc": auc - iid[:, 0],
                      "phase2_ood_after": ood[:, -1],
                      "phase2_greedy_after": matrix(condition, h, "greedy_accuracy")[:, -1],
                      "policy_sensitivity_before": matrix(condition, h, "mean_policy_sensitivity")[:, 0],
                      "policy_sensitivity_after": matrix(condition, h, "mean_policy_sensitivity")[:, -1]}
            history_aggregate[h] = {key: _statistics(value) for key, value in values.items()}
            history_aggregate[h].update(n_seeds=len(seeds), phase2_target_pass_count=int(passed.sum()),
                                        qualified_adaptation_count=int(qualified.sum()),
                                        first_observed_target_checkpoints=[{"seed": s, "trial": next((int(t) for t, value in zip(times, iid[i]) if value >= criteria["adaptation_accuracy"]), None)} for i, s in enumerate(seeds)])
            for i, seed in enumerate(seeds):
                if not qualified[i]:
                    record = {"seed": seed, "condition": condition, "history": h,
                              "phase1_acquired": None if h == "fresh_low" else bool(acquired[i]),
                              "phase2_before": float(iid[i, 0]), "phase2_after": float(iid[i, -1]),
                              "phase2_gain": float(gain[i])}
                    failures.append(record)
                    if h == "fresh_low":
                        fresh_failures.append(record)
        for i, seed in enumerate(seeds):
            if not acquired[i]:
                prefix_failures.append({"seed": seed, "condition": condition,
                                        "phase1_before": float(prefix[i, 0]), "phase1_after": float(prefix[i, -1])})
        partial, unchanged = matrix(condition, "gamma_025"), matrix(condition, "gamma_1")
        partial_auc = np.trapezoid(partial, dx=every, axis=1) / n
        unchanged_auc = np.trapezoid(unchanged, dx=every, axis=1) / n
        per_seed = {"endpoint_benefit": partial[:, -1] - unchanged[:, -1],
                    "initial_advantage": partial[:, 0] - unchanged[:, 0],
                    "gain_difference": (partial[:, -1] - partial[:, 0]) - (unchanged[:, -1] - unchanged[:, 0]),
                    "raw_auc_benefit": partial_auc - unchanged_auc,
                    "baseline_adjusted_auc_benefit": (partial_auc - partial[:, 0]) - (unchanged_auc - unchanged[:, 0]),
                    "ood_endpoint_benefit": matrix(condition, "gamma_025", split="ood")[:, -1] - matrix(condition, "gamma_1", split="ood")[:, -1],
                    "zero_minus_fresh_endpoint": matrix(condition, "gamma_0")[:, -1] - matrix(condition, "fresh_low")[:, -1]}
        for i, seed in enumerate(seeds):
            contrast_rows.append({"seed": seed, "condition": condition,
                                  **{k: float(v[i]) for k, v in per_seed.items()},
                                  "old_rule_acquired": bool(acquired[i])})
        endpoints = np.column_stack([matrix(condition, h)[:, -1] for h in GAMMAS])
        gamma = np.asarray(list(GAMMAS.values()))
        centered = gamma - gamma.mean()
        slopes = endpoints @ centered / np.sum(centered ** 2)
        monotonic = np.all(np.diff(endpoints, axis=1) <= 0, axis=1)
        for i, seed in enumerate(seeds):
            gamma_rows.append({"seed": seed, "condition": condition,
                               "endpoint_slope_per_gamma": float(slopes[i]),
                               "endpoint_nonincreasing_with_gamma": bool(monotonic[i]),
                               **{f"{h}_endpoint": float(endpoints[i, j]) for j, h in enumerate(GAMMAS)}})
        benefits = per_seed["endpoint_benefit"]
        verdict = {"condition": condition, "n_seeds": len(seeds),
                   "mean_endpoint_benefit": float(benefits.mean()),
                   "positive_seed_count": int(np.sum(benefits > 0)),
                   "high_prefix_acquisition_count": int(acquired.sum()),
                   "criterion_met": bool(benefits.mean() >= criteria["endpoint_benefit"]
                                          and np.mean(benefits > 0) >= criteria["seed_pass_fraction"]
                                          and acquired.mean() >= criteria["seed_pass_fraction"])}
        aggregate.append({"condition": condition, "n_seeds": len(seeds), "histories": history_aggregate,
                          "paired_contrasts": {key: _statistics(value) for key, value in per_seed.items()},
                          "resources": {key: _statistics([resource_index[s, condition, "fresh_low"][key] for s in seeds])
                                        for key in ("feature_scale", "initial_feature_gap_rms")},
                          "readout_parameters": sorted({resource_index[s, condition, "fresh_low"]["readout_parameters"] for s in seeds}),
                          "internal_parameters": sorted({resource_index[s, condition, "fresh_low"]["internal_parameters"] for s in seeds}),
                          "shared_high_prefix": {"before": _statistics(prefix[:, 0]), "after": _statistics(prefix[:, -1]),
                                                 "acquisition_count": int(acquired.sum())},
                          "gamma_descriptions": {"slope": _statistics(slopes), "nonincreasing_seed_count": int(monotonic.sum()),
                                                 "scope": "descriptive OLS slope across all four fixed gamma levels; no success gate"},
                          "verdict": verdict})

    raw_rows, boundary_rows = [], []
    for seed in seeds:
        inputs, signs = _inputs(output, seed)
        for condition in all_conditions:
            base = output / f"seed_{seed}" / condition
            before = {split: _raw_margin(base / "shared_prefix" / f"phase1_{n:04d}.npz", x)
                      for split, x in inputs.items()}
            for history in histories:
                for trial in times:
                    for split, x in inputs.items():
                        margin, p, saved_cases = _raw_margin(base / history / f"phase2_{trial:04d}.npz", x)
                        if split + "_p_right" not in saved_cases or not np.allclose(p, saved_cases[split + "_p_right"], atol=2e-12, rtol=0):
                            raise ValueError("Reconstructed raw margin disagrees with primary case probabilities")
                        if split + "_raw_margin" not in saved_cases or not np.allclose(margin, saved_cases[split + "_raw_margin"], atol=2e-12, rtol=0):
                            raise ValueError("Reconstructed raw margin disagrees with primary raw margins")
                        raw_rows.append({"seed": seed, "condition": condition, "history": history,
                                         "phase2_trial": int(trial), "split": split,
                                         "mean_target_raw_margin": float(np.mean(signs[split] * margin)),
                                         "raw_margin_rms": float(np.sqrt(np.mean(margin ** 2))),
                                         "max_abs_raw_margin": float(np.max(np.abs(margin))),
                                         "score_clip_fraction": float(np.mean(np.abs(margin) > 40)),
                                         "mean_confidence": float(np.mean(np.maximum(p, 1 - p)))})
                        actual = float(index[seed, condition, history, "phase2", int(trial), split]["p_target"])
                        baseline = float(index[seed, condition, history, "phase2", 0, split]["p_target"])
                        frozen_rows.append({"seed": seed, "condition": condition, "history": history,
                                            "phase2_trial": int(trial), "split": split,
                                            "frozen_boundary_p_target": baseline, "actual_p_target": actual,
                                            "actual_minus_frozen": actual - baseline, "reward_updates": 0})
                        if trial == 0 and history in GAMMAS:
                            old_margin, old_p, _ = before[split]
                            gamma = GAMMAS[history]
                            old_confidence, confidence = np.maximum(old_p, 1 - old_p), np.maximum(p, 1 - p)
                            if not np.allclose(margin, gamma * old_margin, atol=2e-12, rtol=0):
                                raise ValueError("Boundary raw margin does not equal gamma times prefix")
                            if gamma > 0 and not np.array_equal(np.sign(margin), np.sign(old_margin)):
                                raise ValueError("Positive gamma changed raw decision direction")
                            if np.any(confidence > old_confidence + 2e-12):
                                raise ValueError("Boundary attenuation increased confidence")
                            boundary_rows.append({"seed": seed, "condition": condition, "history": history, "gamma": gamma, "split": split,
                                                  "raw_sign_preserved": bool(np.array_equal(np.sign(margin), np.sign(old_margin))),
                                                  "max_abs_margin_scaling_error": float(np.max(np.abs(margin - gamma * old_margin))),
                                                  "mean_confidence_before": float(old_confidence.mean()), "mean_confidence_after": float(confidence.mean()),
                                                  "strict_confidence_reduction_cases": int(np.sum(confidence < old_confidence)),
                                                  "confidence_case_count": len(p), "confidence_nonincreasing": True})

    reference_rows, reference_aggregate = [], []
    for history in histories:
        item = {"history": history, "n_seeds": len(seeds), "n_checkpoints": len(times)}
        for split in ("iid", "ood"):
            actual = matrix("linear_normalized", history, split=split)
            reference = matrix("expected_linear", history, split=split)
            difference = actual - reference
            item[split] = {"per_seed_checkpoint_mae": _statistics(np.abs(difference).mean(axis=1)),
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
        reference_aggregate.append(item)
    actual_calls = len(seeds) * len(conditions) * 6 * n
    reference_calls = len(seeds) * 6 * n
    if summary["actual_training_trials"] != actual_calls or summary["expected_reference_updates"] != reference_calls:
        raise ValueError("Physical shared-prefix actual/reference call count mismatch")
    derived = {"scope": cfg["scope"], "n_seeds": len(seeds), "all_seeds_included": True,
               "aggregate": aggregate, "primary_hypothesis_from_run": summary["primary_hypothesis"],
               "qualified_adaptation_failures": failures, "fresh_acquisition_failures": fresh_failures,
               "shared_high_prefix_acquisition_failures": prefix_failures,
               "actual_reward_updates": actual_calls, "conditional_reference_updates": reference_calls,
               "counter_notice": "branch lineage counters inherit one high prefix; physical calls count prefix once, not four times",
               "frozen_boundary_baseline": {"rows": len(frozen_rows), "reward_updates": 0, "new_learner": False,
                                            "definition": "each path phase2 trial0 target probability held constant at every checkpoint; endpoint benefit minus initial advantage equals gain difference"},
               "raw_margin_reconstruction": {"rows": len(raw_rows), "max_case_replay_tolerance": 2e-12, "retraining": False},
               "boundary_manipulation_checks": {"rows": len(boundary_rows), "scope": "raw direction and probability confidence are separate; clipping can prevent strict confidence decrease"},
               "conditional_reference": {"scope": "descriptive label-privileged conditional-action drift; not ensemble expectation or reward-learning control; no success flags",
                                          "phase2_trials": times.tolist(), "histories": reference_aggregate,
                                          "pooled_individual_checkpoint_mae": float(np.mean([r["absolute_difference"] for r in reference_rows]))},
               "notice": "all-seed descriptive statistics; no retraining, threshold selection, significance test, confidence-only causal effect, intrinsic learning speed, biological replication or common neural mechanism"}
    for filename, rows in (("paired_attenuation_contrasts.csv", contrast_rows), ("gamma_endpoint_descriptions.csv", gamma_rows),
                           ("raw_margin_summary.csv", raw_rows), ("boundary_confidence.csv", boundary_rows),
                           ("frozen_boundary_baseline.csv", frozen_rows), ("reference_comparison.csv", reference_rows)):
        _write_csv(output / filename, rows)
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
        return (SHORT_NAMES if short else NAMES)[condition] if korean else condition.replace("_", " ")
    def history_name(history):
        return HISTORY_NAMES[history] if korean else history
    def save(fig, filename):
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    def draw(ax, condition, history, metric="p_target", split="iid", factor=100):
        values = matrix(condition, history, metric, split)
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(times))
        ax.plot(times, factor * mean, "o-", color=COLORS[history], label=history_name(history))
        ax.fill_between(times, factor * (mean - sd), factor * (mean + sd), color=COLORS[history], alpha=.08)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, conditions):
        for h in histories:
            draw(ax, condition, h)
        ax.axhline(75, color="gray", linestyle=":", linewidth=1)
        ax.set(title=name(condition), ylim=(-2, 103), xlabel=label("새 규칙 시행", "New-rule trials"),
               ylabel=label("작은 쪽 선택 확률 (%)", "Lower-choice probability (%)"))
    for ax in list(axes.flat)[len(conditions):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(label("같은 prefix에서 감쇠한 뒤 low 연습: 전체 시드 평균±표본 SD", "Shared high prefix, attenuated readout, low practice: all-seed mean ± sample SD"))
    fig.tight_layout(rect=(0, .055, 1, .965))
    save(fig, "attenuation_curves.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, item in enumerate(aggregate):
        values = [r["endpoint_benefit"] for r in contrast_rows if r["condition"] == item["condition"]]
        stats = _statistics(values)
        ax.scatter(i + np.linspace(-.15, .15, len(seeds)), 100 * np.asarray(values), color="#247ba0", alpha=.65)
        ax.errorbar(i, 100 * stats["mean"], yerr=100 * stats["sample_sd"], fmt="ks", capsize=4)
    ax.axhline(0, color="gray", linestyle=":")
    ax.axhline(100 * criteria["endpoint_benefit"], color="#d1495b", linestyle="--")
    ax.set(xticks=np.arange(len(conditions)), xticklabels=[name(c, True) for c in conditions],
           ylabel=label("γ=.25 − γ=1 끝점 (%포인트)", "Gamma .25 minus gamma 1 endpoint (percentage points)"),
           title=label("고정 예산의 복합 조작 효과: 모든 시드와 평균±표본 SD", "Composite intervention endpoint effect: all seeds and mean ± sample SD"))
    fig.tight_layout()
    save(fig, "endpoint_benefit.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    linear = aggregate[0]
    for i, h in enumerate(histories):
        item = linear["histories"][h]
        baseline, gain = item["phase2_before"]["mean"], item["phase2_gain"]["mean"]
        ax.bar(i, 100 * baseline, color=COLORS[h], alpha=.35)
        ax.bar(i, 100 * gain, bottom=100 * baseline, color=COLORS[h], alpha=.8)
        ax.plot(i, 100 * item["phase2_after"]["mean"], "ko")
        ax.plot(i, 100 * baseline, "k_", markersize=20)
    ax.set(xticks=np.arange(len(histories)), xticklabels=[history_name(h) for h in histories], ylim=(0, 103),
           ylabel=label("작은 쪽 선택 확률 (%)", "Lower-choice probability (%)"),
           title=label("단순 선형: 옅은 막대=출발점, 진한 막대=이후 gain, 짧은 선=학습 없는 기준선", "Linear: pale=start, dark=later gain, short line=frozen no-learning baseline"))
    fig.tight_layout()
    save(fig, "initial_gain_endpoint.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for condition in conditions:
        selected = [r for r in boundary_rows if r["condition"] == condition and r["split"] == "iid"]
        means = [np.mean([r["mean_confidence_after"] for r in selected if r["history"] == h]) for h in GAMMAS]
        ax.plot(list(GAMMAS.values()), 100 * np.asarray(means), "o-", label=name(condition))
    ax.set(xlabel=label("head 배수 γ", "Readout multiplier gamma"), ylabel=label("경계 평균 confidence (%)", "Boundary mean confidence (%)"),
           title=label("양수 감쇠는 raw 방향을 보존하며 확신을 줄인다: IID 기술값", "Positive scaling preserves raw direction: IID descriptive confidence"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "boundary_confidence.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split in zip(axes, ("iid", "ood")):
        for h in histories:
            draw(ax, "linear_normalized", h, split=split)
            ax.plot(times, 100 * matrix("expected_linear", h, split=split).mean(axis=0), "--", color=COLORS[h])
        ax.set(title=split.upper(), xlabel=label("새 규칙 시행", "New-rule trials"), ylim=(-2, 103))
    axes[0].set_ylabel(label("작은 쪽 선택 확률 (%)", "Lower-choice probability (%)"))
    axes[1].legend(fontsize=8)
    fig.suptitle(label("선형 실선=실제 보상, 점선=정답쪽을 쓰는 기술 참조", "Linear solid=actual reward; dashed=label-privileged descriptive reference"))
    fig.tight_layout()
    save(fig, "conditional_reference.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(histories))
    ax.bar(positions - .18, [100 * r["iid"]["per_seed_checkpoint_mae"]["mean"] for r in reference_aggregate], .36,
           label=label("개별 경로 MAE 평균", "Mean individual-path MAE"), color="#247ba0")
    ax.bar(positions + .18, [100 * r["iid"]["mean_trajectory_mae"] for r in reference_aggregate], .36,
           label=label("평균 곡선 MAE", "Mean-trajectory MAE"), color="#e09f3e")
    ax.set(xticks=positions, xticklabels=[history_name(h) for h in histories], ylabel=label("실제−참조 절대 차이 (%포인트)", "Actual-reference absolute difference (percentage points)"),
           title=label("평균 곡선에서는 개별 경로 차이가 상쇄될 수 있다", "Individual-path differences can cancel in mean trajectories"))
    ax.legend()
    fig.tight_layout()
    save(fig, "reference_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ("mean_target_raw_margin", "raw_margin_rms"),
                                (label("정답 방향 평균 raw margin", "Mean target-signed raw margin"), label("Raw margin RMS", "Raw margin RMS"))):
        for h in histories:
            values = [np.mean([r[metric] for r in raw_rows if r["condition"] == "linear_normalized" and r["history"] == h and r["phase2_trial"] == int(t) and r["split"] == "iid"]) for t in times]
            ax.plot(times, values, "o-", color=COLORS[h], label=history_name(h))
        ax.set(title=title, xlabel=label("새 규칙 시행", "New-rule trials"))
    axes[1].legend(fontsize=8)
    fig.suptitle(label("보존된 상태와 입력에서 학습 없이 복원한 선형 IID 점수", "Linear IID raw scores reconstructed from archived states without learning"))
    fig.tight_layout()
    save(fig, "raw_margin.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ("readout_change_l2", "internal_change_l2"),
                                (label("출력 계수 변화", "Readout coefficient change"), label("내부 가중치 변화", "Internal weight change"))):
        for condition in conditions:
            for h, linestyle in (("gamma_025", "-"), ("gamma_1", "--")):
                ax.plot(times, matrix(condition, h, metric).mean(axis=0), linestyle, label=name(condition) + " " + history_name(h))
        ax.set(title=title, xlabel=label("새 규칙 시행", "New-rule trials"), ylabel=label("최초 미학습 상태 대비 L2", "L2 from original unlearned state"))
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle(label("출력과 내부의 별도 변화: 실선 γ=.25, 점선 γ=1", "Readout and internal changes separately: solid gamma .25, dashed gamma 1"))
    fig.tight_layout()
    save(fig, "changes.png")

    primary = next(r for r in aggregate if r["condition"] == "linear_normalized")
    verdict = primary["verdict"]
    contrasts = primary["paired_contrasts"]
    fixture_notice = "**소프트웨어 확인용 작은 fixture이며 주요 연구 결과가 아니다.**\n\n" if cfg["scope"] == "integration_test" else ""
    family_table = "\n".join(["| 모델 | γ=.25−γ=1 끝점 평균±SD (%포인트) | 양수 시드 | high 습득 | 판정 |", "|---|---:|---:|---:|---|", *[
        f"| {NAMES[r['condition']]} | {100*r['paired_contrasts']['endpoint_benefit']['mean']:.3f}±{100*r['paired_contrasts']['endpoint_benefit']['sample_sd']:.3f} | {r['verdict']['positive_seed_count']}/{len(seeds)} | {r['verdict']['high_prefix_acquisition_count']}/{len(seeds)} | {'통과' if r['verdict']['criterion_met'] else '미달'} |" for r in aggregate]])
    history_table = "\n".join(["| 선형 경로 | 경계 시작 확률 | 이후 gain | 끝점 | raw AUC | 출발점 조정 AUC |", "|---|---:|---:|---:|---:|---:|", *[
        f"| {HISTORY_NAMES[h]} | {100*primary['histories'][h]['phase2_before']['mean']:.3f}% | {100*primary['histories'][h]['phase2_gain']['mean']:.3f}%p | {100*primary['histories'][h]['phase2_after']['mean']:.3f}% | {100*primary['histories'][h]['phase2_auc']['mean']:.3f}% | {100*primary['histories'][h]['phase2_baseline_adjusted_auc']['mean']:.3f}%p |" for h in histories]])
    failure_table = "\n".join(["| 모델 | 경로 | 시드 | high 습득 | 마지막 확률 |", "|---|---|---:|---|---:|", *[
        f"| {NAMES[r['condition']]} | {HISTORY_NAMES[r['history']]} | {r['seed']} | {'해당없음' if r['phase1_acquired'] is None else r['phase1_acquired']} | {100*r['phase2_after']:.4f}% |" for r in failures]]) if failures else "실제 학습자의 모든 개별 경로가 사전 자격을 확인한 수정/습득 기준을 통과했다."
    report = f"""# 배운 출력 계수의 부분 감쇠 — 결과

{fixture_notice}같은 high 학습 상태에서 출력 계수만 0/.25/.5/1배로 바꾼 네 경로와 fresh low를 같은 새 규칙 {n}시행으로 비교했다. 양수 γ는 경계 raw 선택 방향을 유지한다. 입력이 이미 수량 순서를 제공하므로 실제 그림에서 수량을 알아내거나 인간의 수학 이해를 시험한 결과가 아니다.

## 주 H-A

**단순 선형 H-A: {'통과' if verdict['criterion_met'] else '미달'}.** γ=.25−γ=1의 전체 시드 끝점 차이는 평균 {100*contrasts['endpoint_benefit']['mean']:.3f}±{100*contrasts['endpoint_benefit']['sample_sd']:.3f}%포인트, 양수 {verdict['positive_seed_count']}/{len(seeds)}, 공유 high 습득 {verdict['high_prefix_acquisition_count']}/{len(seeds)}다. 주 과학 기준은 평균15%포인트 이상, 양수8/10 이상, high습득(최종75%와gain15%포인트)8/10 이상을 모두 요구한다. 습득 미달 시드를 평균에서 제외하지 않았다. fixture는 작은 예산의 구현 확인이고10시드 연구 판정을 대신하지 않는다.

{family_table}

다른 여섯 family는 보조 판정이다. γ=.5·0, 단조성·기울기·OOD·fresh 습득·AUC와 참조는 주 결과의 대체가 아니다. 유의성 검정, 실제 배선 우수성이나 같은 전체 학습 자원도 주장하지 않는다.

![모든 경로의 곡선](attenuation_curves.png)

![전체 시드의 고정 예산 끝점 benefit](endpoint_benefit.png)

## 즉시 변화와 이후 연습의 변화

선형 γ=.25−γ=1의 **경계 직후 advantage는 {100*contrasts['initial_advantage']['mean']:.3f}%포인트**, 이후 gain 차이는 {100*contrasts['gain_difference']['mean']:.3f}%포인트다. 두 값의 합이 끝점 benefit이다. 출력 감쇠만으로 잘못된 선택의 확신이 줄고 새 정답 확률이 올라갈 수 있으므로 끝점 차이 모두를 학습 개선으로 해석하지 않는다.

{history_table}

학습 없는 기준선은 각 경로의 phase2 trial0 확률을 전 예산에서 그대로 유지한다. 상태를 바꾸거나 보상 갱신을 실행한 새 학습자가 아니며 실제 보상 호출은0이다. [기준선 CSV](frozen_boundary_baseline.csv)와 [시드별 차이](paired_attenuation_contrasts.csv)에 시작·회복·끝점·조정 AUC를 분리했다. 출발점 조정이나 gain 차이도 intrinsic 학습 속도를 단독으로 식별하지 않는다. 최초 목표 도달은 저장 checkpoint의 관측값으로 정확한 최초 시행이나 지속 성공이 아니다. 성적은 정답 선택 확률의 사례 평균이며 실측 행동 정답률·greedy 성적과 구분한다.

![출발점과 이후 gain과 학습 없는 기준선](initial_gain_endpoint.png)

## 조작과 상태 변화

경계 raw margin은 보존된 전체 IID/OOD 입력·encoder·topology·내부 가중치·head·고정 척도로 학습 없이 복원했으며 primary 사례 확률과 일치를 검사했다. γ>0의 raw 부호/순위 보존과 사례별 confidence 비증가를 확인한다. clipping40에서 양쪽이 포화되면 엄격 감소가 없을 수 있어 strict감소 사례 수와 평균을 [경계 CSV](boundary_confidence.csv)에 남겼다. 확률 동률 허용오차와 raw 부호는 별개다.

![경계 confidence](boundary_confidence.png)

![복원 raw margin](raw_margin.png)

head-only 조작이지만 가소성의 이후 old_head gradient와 internal_weight×head 유효 점수 배선도 달라진다. confidence-only 인과 효과나 p(1−p) 하나가 전체 원인이라는 결론을 내리지 않는다. 출력 계수·내부 가중치·활성 변화는 별도 기록이다. 아래 변화는 최초 미학습 상태와 비교하며 감쇠 뒤 상태만을 기준으로 한 변화가 아니다. 단순 선형에는 KC·MBON이 없고 빈 배열의 활성변화0을 그 회로의 측정값으로 부르지 않는다.

![출력과 내부의 별도 변화](changes.png)

선형·내부 고정 조건의 γ0=fresh는 같은 척도·입력·uniform 아래 상태·선택·보상·확률의 정확한 일치여야 한다. 상속한 prefix 때문에 누적 counter는 다르다. 이는 구현 무결성 검사이며 과학적 성공이 아니다. 가소성 γ0−fresh 차이는 바뀐 내부와 새 head의 상호작용이고 생물학적 기억 저장의 증거가 아니다.

## 별도 행동 평균 참조

선형 참조는 자기 현재 상태에서 (2y−1)p(1−p)를 적용한다. 정답쪽을 직접 받는 기술 계산으로 실제 보상 학습자와 같은 정보의 대조가 아니다. 비선형성 때문에 stochastic 학습의 정확한 전체 기대값도 아니고10시드는 동일 입력의 행동 ensemble도 아니다. 참조에 별도 성공·실패 flags를 붙이지 않는다. 개별 경로 MAE와 평균 곡선 MAE 및 최대 개별 차이는 [분석 JSON](analysis_summary.json), IID/OOD 모든 저장점 차이는 [참조 CSV](reference_comparison.csv)에 따로 보존했다.

![실제 보상과 기술 참조](conditional_reference.png)

![개별 경로와 평균 곡선의 서로 다른 오차](reference_errors.png)

## 미달과 재현

실제 경로의 자격을 확인한 수정/습득 미달은 {len(failures)}건, 공유 high습득 미달은 {len(prefix_failures)}건, fresh습득 미달은 {len(fresh_failures)}건이다. 겹치는 종류를 합쳐 독립 실패 수로 부풀리지 않는다. 참조는 이 집계에 포함하지 않는다.

{failure_table}

물리적 실제 reward-update 호출은 {actual_calls:,}, 별도 참조 갱신은 {reference_calls:,}회다. 공유 high prefix는 seed·family당 실제1회만 실행했으며 네 사본의 상속 counter를 네 번의 호출로 세지 않는다. branch의 이력은 high+low, fresh의 이력은 low뿐이어서 총 경험까지 같지 않다. 평가·복제·감쇠·전처리·학습 없는 기준선은 보상 호출에 더하지 않았다. 모든 errorbar와band는 계산 시드 표본SD이며 신뢰구간이나 생물학적 개체 변동이 아니다.

[사전 프로토콜](../../docs/head-attenuation-protocol.md) · [계산적 배경](../../docs/head-attenuation-background.md) · [모든 원값](summary.json) · [실행 정보](metadata.json) · [동결](registration.json) · [독립 검산](verification.json)

현재 결과 범위는 인공 입력의 경험 의존 변화 Level2다. 새 수량 일반화, 실제 초파리 수량 인지·자연 망각·기억 삭제·수학교육 효과를 시험하지 않았다. 인간의 기초 수량·수학 학습과 연결할 장기 목표는 유지하되 출발 편향·고정 예산 수정 비용·회복량·오류 방향과 대응 인간 시행별 선택/피드백 원자료가 필요하다. 전후 요약 자료만으로 정량적 인간 유사성 Level4나 같은 신경 기작 Level5를 주장하지 않는다.

재현은 study-head-attenuation-v1 별도 checkout과 잠금 환경에서 존재하지 않는 새 결과 경로로 한다. 이전 연구의 실패·등록·source set을 최신 코드에 맞춰 고치지 않는다.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return derived
