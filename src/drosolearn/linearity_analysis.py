"""Descriptive teacher-score surrogate plots; no fitting on transfer inputs."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .linearity import ALPHAS, ANCHORED, FOCAL_INDEX, PAIRS, PRIMARY, REFERENCES, spacing_panel
from .experiment import write_json


NAMES = {"fixed": "실제 배선·출력만 학습", "plastic": "실제 배선·내부도 학습",
         "rewired_fixed": "재배선·출력만 학습", "rewired_plastic": "재배선·내부도 학습",
         "ann_fixed": "일반 신경망·출력만 학습", "ann_plastic": "일반 신경망·내부도 학습"}
SHORT_NAMES = {"fixed": "실제 배선\n출력만", "plastic": "실제 배선\n내부도",
               "rewired_fixed": "재배선\n출력만", "rewired_plastic": "재배선\n내부도",
               "ann_fixed": "일반망\n출력만", "ann_plastic": "일반망\n내부도"}
REFERENCE_NAMES = {"affine": "7입력 affine 참조", "scalar": "수량만 보는 직선 참조"}
METRIC_NAMES = ("unseen_probability_mae", "unseen_margin_mae", "unseen_margin_rmse",
                "unseen_score_rmse", "unseen_greedy_disagreement", "anchored_probability_mae",
                "all_probability_mae", "focal_probability_mae")


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Nonempty finite analysis values required")
    return {"mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def _greedy(probabilities):
    p = np.asarray(probabilities)
    return np.where(np.abs(p - .5) < 1e-12, .5, (p > .5).astype(float))


def _curve_matrix(curves, seeds, condition, split, key):
    selected = {}
    for row in curves:
        if row["condition"] == condition and row["split"] == split:
            pair = (int(row["seed"]), int(row["trial"]))
            if pair in selected:
                raise ValueError("Duplicate curve entry")
            selected[pair] = float(row[key])
    times = sorted({trial for seed, trial in selected if seed in seeds})
    return np.asarray(times), np.asarray([[selected[seed, trial] for trial in times] for seed in seeds])


def analyze(output):
    """Write derived reports without changing raw artifacts or model state."""
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    cfg, outcomes = metadata["config"], summary["outcomes"]
    if summary["status"] != "completed" or metadata["status"] != "completed":
        raise ValueError("Analysis requires a completed run")
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

    def condition_name(condition, short=False):
        return (SHORT_NAMES if short else NAMES)[condition] if korean else condition.replace("_", "\n") if short else condition

    conditions, seeds = cfg["conditions"], cfg["seeds"]
    verdicts = {row["condition"]: row for row in summary["family_verdicts"]}
    aggregate = []
    for condition in conditions:
        rows = [r for r in outcomes if r["condition"] == condition]
        if {r["seed"] for r in rows} != set(seeds) or len(rows) != len(seeds):
            raise ValueError("Analysis requires all registered seed outcomes")
        values = {stage: {reference: {key: _statistics([r[stage][reference][key] for r in rows])
                                      for key in METRIC_NAMES} for reference in REFERENCES}
                  for stage in ("baseline", "final")}
        aggregate.append({"condition": condition, "n_seeds": len(rows), **values,
                          "model_before_accuracy": _statistics([r["before_accuracy"] for r in rows]),
                          "model_after_accuracy": _statistics([r["after_accuracy"] for r in rows]),
                          "model_ood_accuracy": _statistics([r["ood_accuracy"] for r in rows]),
                          "focal_model_before": np.mean([r["focal_model_before"] for r in rows], axis=0).tolist(),
                          "focal_model_after": np.mean([r["focal_model_after"] for r in rows], axis=0).tolist(),
                          "focal_model_greedy_after": np.mean([r["focal_model_greedy_after"] for r in rows], axis=0).tolist(),
                          "focal_references": {stage: {reference: {
                              "p_correct": np.mean([r[stage][reference]["focal_reference_p_correct"] for r in rows], axis=0).tolist(),
                              "greedy_accuracy": np.mean([r[stage][reference]["focal_reference_greedy_accuracy"] for r in rows], axis=0).tolist()}
                              for reference in REFERENCES} for stage in ("baseline", "final")},
                          "verdict": verdicts[condition]})

    panel, primary = spacing_panel(), set(PRIMARY.tolist())
    pair_rows = []
    calibration_rows = []
    for condition in conditions:
        for stage in ("baseline", "final"):
            source = []
            for seed in seeds:
                folder = output / f"seed_{seed}" / condition
                with np.load(folder / f"transfer_{stage}.npz", allow_pickle=False) as saved:
                    source.append({key: saved[key].copy() for key in saved.files})
                with np.load(folder / f"reference_{stage}.npz", allow_pickle=False) as reference:
                    calibration_rows.append({"seed": seed, "condition": condition, "stage": stage,
                                             "affine_rank": int(reference["affine_rank"]),
                                             "affine_score_rmse": float(reference["affine_calibration_score_rmse"]),
                                             "scalar_score_rmse": float(reference["scalar_calibration_score_rmse"]),
                                             "affine_coefficients": reference["affine_coefficients"].tolist(),
                                             "scalar_coefficients": reference["scalar_coefficients"].tolist(),
                                             "affine_singular_values": reference["affine_singular_values"].tolist()})
            for reference in REFERENCES:
                # First calculate errors inside each seed; differences of mean
                # probabilities can cancel and are not a casewise fidelity MAE.
                error = np.asarray([np.abs(d[f"{reference}_map_p_right"] - d["map_p_right"]).mean(axis=2) for d in source])
                margin_error = np.asarray([d[f"{reference}_map_margin"] - d["teacher_map_margin"] for d in source])
                margin_mae = np.abs(margin_error).mean(axis=3)
                margin_rmse = np.sqrt((margin_error ** 2).mean(axis=3))
                disagreement = np.asarray([np.abs(_greedy(d[f"{reference}_map_p_right"])
                                                 - _greedy(d["map_p_right"])).mean(axis=2) for d in source])
                model_p = np.asarray([d["map_p_target"] for d in source])
                reference_p = np.asarray([d[f"{reference}_map_p_target"] for d in source])
                model_g = np.asarray([d["map_greedy_accuracy"] for d in source])
                reference_g = np.asarray([d[f"{reference}_map_greedy_accuracy"] for d in source])
                for i, alpha in enumerate(ALPHAS):
                    for j, (low, high) in enumerate(PAIRS):
                        probability = _statistics(error[:, i, j])
                        raw_mae, raw_rmse = _statistics(margin_mae[:, i, j]), _statistics(margin_rmse[:, i, j])
                        pair_rows.append({"condition": condition, "stage": stage, "reference": reference,
                                          "alpha": float(alpha), "low": low, "high": high,
                                          "ratio": low / high, "difference": high - low, "focal": j == FOCAL_INDEX,
                                          "stratum": "fully_unseen" if j in primary else "one_anchor",
                                          "input_gap": float((panel[i, high - 1] - panel[i, low - 1]) / 8),
                                          "input_midpoint": float((panel[i, high - 1] + panel[i, low - 1]) / 16),
                                          "probability_mae_mean": probability["mean"],
                                          "probability_mae_sample_sd": probability["sample_sd"],
                                          "raw_margin_mae_mean": raw_mae["mean"],
                                          "raw_margin_mae_sample_sd": raw_mae["sample_sd"],
                                          "raw_margin_rmse_mean": raw_rmse["mean"],
                                          "raw_margin_rmse_sample_sd": raw_rmse["sample_sd"],
                                          "greedy_disagreement_mean": float(disagreement[:, i, j].mean()),
                                          "model_p_correct_mean": float(model_p[:, i, j].mean()),
                                          "reference_p_correct_mean": float(reference_p[:, i, j].mean()),
                                          "signed_mean_p_correct_difference": float((reference_p[:, i, j] - model_p[:, i, j]).mean()),
                                          "model_greedy_accuracy_mean": float(model_g[:, i, j].mean()),
                                          "reference_greedy_accuracy_mean": float(reference_g[:, i, j].mean())})
    with (output / "pair_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    write_json(output / "analysis_summary.json", {
        "aggregate": aggregate, "calibration": calibration_rows, "alphas": list(ALPHAS),
        "actual_training_trials": summary["actual_training_trials"], "primary_pairs": "fully_unseen15",
        "scope": cfg["scope"],
        "fidelity_metric": "casewise absolute p_right error, averaged over four maps, fifteen pairs and all cases within each seed",
        "raw_margin_metric": "unclipped right-minus-left score error; reported RMSE is averaged over seed-level RMSE values",
        "reference_scope": "privileged teacher-score surrogate fitted only at trained2/4 quantities; not independent reward learning",
        "uncertainty": "all registered computational seeds; mean and sample SD, no biological independence or significance test"})

    def save(fig, name):
        fig.savefig(output / name, dpi=150)
        plt.close(fig)

    rows_by_condition = {r["condition"]: r for r in aggregate}
    ncols = min(3, len(conditions))
    nrows = (len(conditions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.3 * nrows), squeeze=False)
    for ax, condition in zip(axes.flat, conditions):
        row = rows_by_condition[condition]
        ax.plot(ALPHAS, row["focal_model_after"], "o-", color="tab:blue",
                label=label("학습한 원래 모델", "Trained teacher model"))
        for reference, marker, color in (("affine", "s--", "tab:orange"), ("scalar", "^:", "tab:green")):
            ax.plot(ALPHAS, row["focal_references"]["final"][reference]["p_correct"], marker,
                    color=color, label=REFERENCE_NAMES[reference] if korean else f"{reference} score surrogate")
        ax.set(xticks=ALPHAS, ylim=(0, 1.04), xlabel=label("5·7 입력 간격의 배율", "Focal input gap / original gap"),
               ylabel=label("정답 선택 확률", "Correct-choice probability"), title=condition_name(condition))
        ax.axhline(.5, color="gray", linestyle=":", linewidth=.8)
        ax.legend(fontsize=8, loc="lower right")
    for ax in list(axes.flat)[len(conditions):]:
        ax.set_visible(False)
    fig.suptitle(label("새 5대7: 2·4의 모델 점수만 본 참조의 예측", "New5-vs-7: references see only teacher scores at2/4"), fontsize=14)
    fig.text(.5, .01, label("모든 시드 평균. 참조는 모델 점수를 받은 요약식이며 독립 보상 학습자가 아니다.",
                           "All-seed means; privileged score surrogates are not independent reward learners."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    save(fig, "focal_response.png")

    x, width = np.arange(len(conditions)), .34
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, stage, title in zip(axes, ("baseline", "final"), (label("학습 전", "Before reward learning"), label("학습 후", "After reward learning"))):
        for k, reference in enumerate(REFERENCES):
            data = [r[stage][reference]["unseen_probability_mae"] for r in aggregate]
            ax.bar(x + (k - .5) * width, [100 * d["mean"] for d in data], width=width,
                   yerr=[100 * d["sample_sd"] for d in data], capsize=3,
                   label=REFERENCE_NAMES[reference] if korean else reference)
        ax.axhline(100 * cfg["criteria"]["probability_mae"], color="black", linestyle=":",
                   label=label("주 affine 평균 기준", "Primary affine mean criterion"))
        ax.set(xticks=x, xticklabels=[condition_name(c, short=True) for c in conditions],
               ylabel=label("시행별 선택 확률 MAE (%포인트)", "Casewise probability MAE (percentage points)"), title=title)
        ax.legend(fontsize=8)
    fig.suptitle(label("네 코드·완전 새 15쌍: 모든 시드의 평균±표준편차", "Four maps and fully unseen15 pairs: all-seed mean ± sample SD"), fontsize=14)
    fig.text(.5, .01, label("평균 확률끼리 뺀 값이 아닌 시행별 절대 오차. 선은 affine 주 기준이며 scalar는 보조 지표다.",
                           "Casewise errors, not a difference of mean probabilities; scalar is a secondary reference."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, "probability_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, stage, title in zip(axes, ("baseline", "final"), (label("학습 전", "Before reward learning"), label("학습 후", "After reward learning"))):
        for k, reference in enumerate(REFERENCES):
            data = [r[stage][reference]["unseen_margin_rmse"] for r in aggregate]
            ax.bar(x + (k - .5) * width, [d["mean"] for d in data], width=width,
                   yerr=[d["sample_sd"] for d in data], capsize=3,
                   label=REFERENCE_NAMES[reference] if korean else reference)
        ax.set(xticks=x, xticklabels=[condition_name(c, short=True) for c in conditions],
               ylabel=label("clipping 전 점수 차이의 RMSE", "Unclipped score-margin RMSE"), title=title)
        ax.legend(fontsize=8)
    fig.suptitle(label("확률 변환 전에 남아 있는 점수 오차", "Score error before probability compression"), fontsize=14)
    fig.text(.5, .01, label("낮은 확률 오차만으로 정확한 직선 함수를 주장하지 않는다. 점수 단위와 %포인트는 다르다.",
                           "Low probability error does not prove exact linearity; score units are not percentage points."), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, "raw_margin_errors.png")

    heatmaps = []
    for condition in conditions:
        heatmaps.append(np.asarray([[100 * next(r["probability_mae_mean"] for r in pair_rows
                      if r["condition"] == condition and r["stage"] == "final" and r["reference"] == "affine"
                      and r["alpha"] == alpha and (r["low"], r["high"]) == pair)
                      for pair in PAIRS] for alpha in ALPHAS]))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.5 * nrows), squeeze=False, constrained_layout=True)
    vmax = max(float(h.max()) for h in heatmaps) or 1e-12
    for ax, condition, heatmap in zip(axes.flat, conditions, heatmaps):
        im = ax.imshow(heatmap, aspect="auto", vmin=0, vmax=vmax, cmap="viridis", origin="lower")
        ax.set(xticks=np.arange(len(PAIRS)),
               xticklabels=[f"{a}:{b}{'*' if j in primary else ''}" for j, (a, b) in enumerate(PAIRS)],
               yticks=np.arange(len(ALPHAS)), yticklabels=[str(a) for a in ALPHAS],
               title=condition_name(condition), ylabel=label("입력 간격 배율", "Gap scale"))
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    for ax in list(axes.flat)[len(conditions):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=[a for a in axes.flat if a.get_visible()], shrink=.8,
                 label=label("affine 확률 MAE (%포인트)", "Affine probability MAE (percentage points)"))
    fig.suptitle(label("학습 후 쌍별 오차: *는 완전 새 수량쌍", "After-learning pair errors: * marks fully unseen pairs"), fontsize=14)
    save(fig, "pair_errors.png")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for condition in conditions:
        times, data = _curve_matrix(curves, seeds, condition, "iid", "p_target")
        mean, sd = data.mean(axis=0), data.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(times))
        line, = ax.plot(times, mean, marker="o", label=condition_name(condition))
        ax.fill_between(times, mean - sd, mean + sd, alpha=.08, color=line.get_color())
    ax.axhline(cfg["criteria"]["acquisition_accuracy"], color="gray", linestyle=":",
               label=label("습득의 절대 확률 기준", "Acquisition absolute-probability criterion"))
    ax.set(ylim=(0, 1.05), xlabel=label("보상 학습 시행", "Reward-learning trials"),
           ylabel=label("연습한 2대4의 정답 선택 확률", "Correct-choice probability on trained2-vs-4"),
           title=label("원래 모델의 학습 경로: 참조 적합은 추가 보상 학습이 아니다", "Teacher learning trajectories; reference fitting adds no reward learning"))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "acquisition.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    changes = (("internal_change_l2", label("내부 연결 가중치", "Internal synaptic weights")),
               ("readout_change_l2", label("마지막 선택 가중치", "Readout weights")),
               ("probe_kc_change_l2", label("같은 입력의 KC 활성", "KC activity for fixed input")),
               ("probe_mbon_change_l2", label("같은 입력의 MBON 활성", "MBON activity for fixed input")))
    for ax, (key, title) in zip(axes.flat, changes):
        for condition in conditions:
            times, data = _curve_matrix(curves, seeds, condition, "iid", key)
            ax.plot(times, data.mean(axis=0), label=condition_name(condition))
        ax.set(title=title, xlabel=label("학습 시행", "Training trials"),
               ylabel=label("시작 상태와의 거리 (L2)", "L2 distance from initial state"))
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(label("가중치와 활성 변화의 구분: 크기는 기억의 질을 뜻하지 않는다", "Weights and activity distinguished; distance does not measure memory quality"), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, "changes.png")

    fidelity_table = []
    margin_table = []
    focal_table = []
    for row in aggregate:
        name, verdict = NAMES[row["condition"]], row["verdict"]
        def percent_stats(stage, reference, metric="unseen_probability_mae"):
            data = row[stage][reference][metric]
            return f"{100 * data['mean']:.3f}±{100 * data['sample_sd']:.3f}"
        fidelity_table.append(f"| {name} | {percent_stats('baseline', 'affine')} | {percent_stats('final', 'affine')} | "
                              f"{percent_stats('final', 'scalar')} | {verdict['passing_seed_count']}/{len(seeds)} | "
                              f"{'통과' if verdict['criterion_met'] else '미달'} | {verdict['acquisition_count']}/{len(seeds)} | "
                              f"{verdict['qualified_count']}/{len(seeds)} |")
        margin_table.append("| " + name + " | " + " | ".join(
            f"{row[stage][reference]['unseen_margin_rmse']['mean']:.5f}±{row[stage][reference]['unseen_margin_rmse']['sample_sd']:.5f}"
            for stage, reference in (("baseline", "affine"), ("final", "affine"), ("final", "scalar"))) + " |")
        for source, values in (("원래 학습 모델", row["focal_model_after"]),
                               ("7입력 affine 참조", row["focal_references"]["final"]["affine"]["p_correct"]),
                               ("수량만 보는 직선 참조", row["focal_references"]["final"]["scalar"]["p_correct"])):
            focal_table.append("| " + name + " | " + source + " | " + " | ".join(f"{100*v:.2f}%" for v in values) + " |")
    fidelity = "\n".join(["| 원래 모델 조건 | 학습 전 affine MAE | 학습 후 affine MAE | 학습 후 scalar MAE | 개별 오차 통과 | 주 H-L | 연습 습득 | 습득 및 오차 통과 |",
                            "|---|---:|---:|---:|---:|---|---:|---:|", *fidelity_table])
    raw_margins = "\n".join(["| 원래 모델 조건 | 학습 전 affine RMSE | 학습 후 affine RMSE | 학습 후 scalar RMSE |",
                              "|---|---:|---:|---:|", *margin_table])
    focal = "\n".join(["| 원래 모델 조건 | 반응 함수 | 1/4 간격 | 1/2 간격 | 기존 간격 | 1.5배 간격 |",
                       "|---|---|---:|---:|---:|---:|", *focal_table])
    failed = [r for r in outcomes if not r["acquired"]]
    acquisition_failures = "모든 시드·모델 조건이 습득 기준을 통과했다." if not failed else (
        "습득 미달은 " + "; ".join(f"{NAMES[r['condition']]} 시드 {r['seed']} ({100*r['after_accuracy']:.4f}%, 향상 {100*r['gain']:.4f}%포인트)" for r in failed) + "다. 이 시드도 주 평균에 포함했다.")
    passed = sum(row["verdict"]["criterion_met"] for row in aggregate)
    fixture_notice = ("\n**이 보고서는 소프트웨어 점검용 작은 fixture이며 연구 결과가 아니다.** "
                      "연구용 10시드의 실행·판정·결과를 대신하지 않는다.\n"
                      if cfg["scope"] == "integration_test" else "")
    text = f"""# 단순 참조 함수의 설명 범위
{fixture_notice}

## 먼저 쉬운 설명

복잡한 배선 모델은 2개와 4개 중 큰 쪽을 고르는 연습을 {cfg['trials']}번 했다. 별도로 준비한 2·4 입력에서 모델이 내는 점수만 읽어, 7입력을 더하는 affine 식과 수량만 보는 직선 식을 맞췄다. 이 참조들은 새 수량의 점수나 정답을 보고 다시 맞추지 않았다. 새 문제에서 원래 모델과 얼마나 비슷한 선택 확률을 내는지 검사했다.

**참조는 모델 점수를 제공받은 요약식이며 독립 보상 학습자가 아니다.** 낮은 오차는 이 모델의 반응을 간단한 계산으로 설명할 수 있다는 뜻이다. 참조가 같은 경험으로 더 빨리 배웠다거나 실제 초파리·인간이 같은 식을 사용한다는 뜻은 아니다. 원래 모델의 정답 성적과 참조의 설명 오차도 서로 다른 지표다.

## 미리 정한 오차 기준

주 지표는 네 입력 코드·완전 미학습 15수량쌍에서 원래 모델과 affine 참조의 **시행별 오른쪽 선택 확률 차이의 절댓값**을 평균한 MAE다. 평균 확률끼리 뺀 값은 서로 다른 시행의 오차가 상쇄될 수 있으므로 주 MAE로 사용하지 않았다. 아래 오차는 %포인트 단위이며 ±는 계산 시드의 표본 표준편차다.

연구 설계의 H-L은 전체 10시드 평균 MAE≤{100*cfg['criteria']['probability_mae']:.0f}%포인트 및 같은 개별 기준을 통과한 시드≥8/10을 모두 요구한다. 현재 실행의 시드 수는 {len(seeds)}이며 모든 시드를 포함해 설정된 평균·시드 비율 기준을 적용한 결과 {passed}/{len(conditions)}조건이 통과했다. scalar는 사전 보조 참조이며 주 판정을 대신하지 않는다. 작은 fixture라면 이 판정은 구현 확인으로만 읽는다.

{fidelity}

‘연습 습득’은 최종 2대4 IID 정답 확률≥75%와 시작 대비≥15%포인트 향상을 함께 요구한다. 마지막 열은 습득과 개별 affine 오차 기준을 모두 통과한 시드 수이며, 전체 H-L의 판정을 대체하지 않는다. {acquisition_failures} 실패 시드 삭제·교체나 추가 학습은 하지 않았다.

학습 전 오차도 보존했다. 시작 상태의 거의 무작위인 반응이 이미 직선으로 잘 근사될 수 있으므로, 학습 후의 낮은 오차만으로 새로운 선형 규칙을 발견했다고 말하지 않는다. H-L 미달이면 이번 두 anchor 자료와 사전 오차 범위로는 충분히 설명되지 않았다고 기록한다. 미달을 수량 또는 수학 학습 자체의 불가능성으로 확대하지 않는다.

## 확률 압축 전에 남는 차이

모델은 오른쪽−왼쪽 점수 차이를 clipping한 뒤 sigmoid로 선택 확률로 바꾼다. 큰 점수 차이끼리는 서로 달라도 둘 다 거의 100% 확률이 될 수 있다. 그래서 작은 확률 MAE가 정확한 직선 점수 함수를 증명하지 않는다. 아래는 **clipping 전 원시 점수 차이**의 RMSE, 시드 평균±표본 SD다. 점수 단위이며 %포인트와 다르다. 이 보조 지표에는 사후 성공 기준을 붙이지 않았다.

{raw_margins}

원시 점수 차이 MAE, 후보 점수 RMSE와 greedy 선택 불일치도 [분석 원자료](analysis_summary.json)에 있다. greedy 불일치는 같은 방향이면 0, 반대 방향이면 1, 한쪽만 동률이면 0.5다. 동률은 `|p_right−0.5|<1e−12`다. 정답 방향이 모두 같아도 선택 확률이나 점수 오차는 남을 수 있다.

## 새 5대7의 확률 곡선

아래는 모든 시드의 학습 후 정답 선택 확률 평균이다. 네 코드 모두 수량의 순서와 연습한 2·4 입력을 유지한다. 5·7의 중간 입력값은 0.75로 같고, 사이의 간격만 기존의 1/4·1/2·1·1.5배로 바뀐다. 두 수량은 훈련 입력 구간 밖이다. 이 곡선의 평균값끼리 가까운 것만으로 시행별 주 오차가 작다고 판단하지 않는다.

{focal}

[쌍별 표](pair_summary.csv)는 네 코드·15 완전 미학습쌍·12 anchor 포함쌍을 나눠 원래 모델 성적, 두 참조의 성적과 오차를 남긴다. `probability_mae_mean`은 시행별 절대 오차를 시드마다 먼저 평균한 뒤 시드 평균을 낸 값이다. `signed_mean_p_correct_difference`는 별도의 부호 있는 평균 정답 확률 차이로 주 MAE가 아니다. 5대7 외의 쌍에서는 입력 간격과 중간값이 함께 바뀔 수 있어 간격 하나의 인과 효과로 확대하지 않는다.

## 무엇을 검증했는가

각 시드·모델 조건은 한 번만 보상 학습했고 네 코드 평가가 그 상태를 공유했다. 실제 보상 학습은 총 {summary['actual_training_trials']:,}시행이다. 참조는 학습 전·후 각각 같은 별도 calibration 입력의 모델 점수로 맞췄다. calibration은 2·4만 포함하며 train·IID·OOD·새 수량 평가와 fingerprint가 겹치지 않는다. affine 설계 행렬은 8계수 full rank를 요구하며 rank·특이값·계수·calibration 점수 RMSE를 보존했다. 학습 전후 참조를 따로 맞춘 것은 같은 참조가 독립 학습한 궤적이 아니다.

현재 원래 모델의 점수는 고정 KC ReLU 특징의 가중 합으로 입력 구간마다 affine인 함수다. 전체 구간이 한 직선이라는 보장은 없으며 이번에는 그 근사 범위를 측정했다. 참조와 원래 모델의 정답 성적이 비슷하다고 생물학적 시냅스 법칙·LC11·DAN·물체 세기·수학 의미를 재현했다고 말하지 않는다. 내부 학습 조건은 추가 1,474가중치를 학습하므로 출력 전용 조건과 같은 자원이라는 주장도 하지 않는다.

[사전 프로토콜](../../docs/linearity-protocol.md) · [수학적 설계 확인](../../docs/linearity-background.md) · [실행 정보](metadata.json) · [모든 시드 판정](summary.json) · [독립 검산 기록](verification.json)

독립 검산의 통과 여부는 연결된 검산 기록에서 별도로 확인한다. 모든 시행·선택 난수·보상·배선·checkpoint·calibration 점수·참조 계수·새 수량 점수·확률을 보존해 재계산할 수 있도록 했다. 기존 연구 결과와 실패는 변경하지 않았다. 원래 모델의 경험 의존 변화는 Level 2, 습득을 확인한 뒤 인공 표현에서의 미학습 입력 반응은 제한된 Level 3다. **참조의 높은 설명 정확도 자체가 독립 학습이나 수학 규칙 발견의 증거는 아니다.** 인간과의 정량적 유사성(Level 4)과 관련 신경 기작(Level 5)은 검증하지 않았다. 인간 학습 비교는 장기 목표로 유지하며 입력 정보·선택 지표·학습 전후 변화의 비교 가능성을 먼저 확인한다.

![원래 모델과 두 참조의 새 문제 반응](focal_response.png)

![시행별 선택 확률 오차](probability_errors.png)

![확률 변환 전 점수 차이 오차](raw_margin_errors.png)

![쌍별 affine 오차](pair_errors.png)

![원래 모델의 연습 곡선](acquisition.png)

![가중치와 활성 변화의 구분](changes.png)
"""
    (output / "report.md").write_text(text, encoding="utf-8")
    return aggregate
