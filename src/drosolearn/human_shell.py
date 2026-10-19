"""Prospective shell-task comparison; orchestration and raw-data parsing are separate."""
import numpy as np

from .human_shell_models import CONDITIONS, ADAPTIVE, build_graphs, make_model

MODES = ("conditional", "autonomous")
BEHAVIOR_METRICS = ("acquisition_gain", "reversal_early", "reversal_recovery")


def loss(action, probability):
    raw_p = np.asarray(probability, dtype=float)
    a = np.asarray(action, dtype=float)
    if a.shape != raw_p.shape or not np.isin(a, (0, 1)).all() or not np.isfinite(raw_p).all() or np.any((raw_p < 0) | (raw_p > 1)):
        raise ValueError("Aligned binary actions and finite probabilities required")
    p = np.clip(raw_p, 1e-12, 1 - 1e-12)
    return -(a * np.log(p) + (1 - a) * np.log1p(-p)), (a - raw_p) ** 2


def acquired_reward(action, content):
    if action not in (0, 1) or content not in (0, 1, 2):
        raise ValueError("Invalid action/content")
    return 0 if action == 0 else (0, 1, -1)[content]


def check_records(records, ids, trials):
    if set(records) != set(ids) or len(ids) != len(set(ids)):
        raise ValueError("Registered participant identity mismatch")
    for pid in ids:
        rows = records[pid]
        if len(rows) != trials or [r["trial"] for r in rows] != list(range(1, trials + 1)):
            raise ValueError("Complete original trial order required")
        if any(r["shell"] not in range(1, 7) or r["state"] not in (1, 2)
               or r["content"] not in (0, 1, 2) or r["action"] not in (0, 1) for r in rows):
            raise ValueError("Invalid raw code")


def behavior_metrics(rows, actions=None):
    actions = np.asarray([r["action"] for r in rows] if actions is None else actions)
    if actions.shape != (len(rows),) or not np.isin(actions, (0, 1)).all():
        raise ValueError("Aligned binary actions required")
    acquisition, early, recovery = [], [], []
    for shell in (3, 4):
        indices = [i for i, r in enumerate(rows) if r["shell"] == shell]
        episodes = []
        for i in indices:
            if not episodes or rows[episodes[-1][-1]]["state"] != rows[i]["state"]:
                episodes.append([])
            episodes[-1].append(i)
        if (len(episodes) != 4 or any(len(e) < 6 for e in episodes)
                or [rows[e[0]]["state"] for e in episodes] != [1, 2, 1, 2]):
            raise ValueError("Strict shells require four nonoverlapping six-plus-exposure phases")
        for phase, idx in enumerate(episodes):
            state = rows[idx[0]]["state"]
            optimal = int((shell == 3 and state == 1) or (shell == 4 and state == 2))
            correct = (actions[idx] == optimal).astype(float)
            first, last = float(correct[:3].mean()), float(correct[-3:].mean())
            if phase == 0:
                acquisition.append(last - first)
            else:
                early.append(first)
                recovery.append(last - first)
    return dict(zip(BEHAVIOR_METRICS, (float(np.mean(acquisition)), float(np.mean(early)), float(np.mean(recovery)))))


def _copies(model):
    return model.head.copy(), model.weights.copy(), getattr(model, "values", np.empty(0)).copy()


def _trajectory(model, rows, autonomous=False, uniforms=None):
    n = len(rows)
    p = np.empty(n)
    actions, rewards = np.empty(n, dtype=np.int8), np.empty(n, dtype=np.int8)
    steps = np.empty((3, n))
    for t, row in enumerate(rows):
        shell = int(row["shell"])
        p[t] = model.probability(shell)
        action = int(uniforms[t] < p[t]) if autonomous else int(row["action"])
        reward = acquired_reward(action, int(row["content"]))
        actions[t], rewards[t] = action, reward
        steps[:, t] = model.observe(shell, action, reward)
    if not np.isfinite(p).all() or not np.isfinite(steps).all():
        raise ValueError("Nonfinite trajectory")
    return p, actions, rewards, steps


def fit(calibration_records, cfg, graph):
    ids, seeds, rates = cfg["calibration_ids"], cfg["calibration_seeds"], cfg["rates"]
    n = cfg["trials_per_person"]
    check_records(calibration_records, ids, n)
    if tuple(cfg["conditions"]) != CONDITIONS or tuple(cfg["adaptive_conditions"]) != ADAPTIVE:
        raise ValueError("Registered ordered condition sets required")
    go_count = sum(r["action"] for pid in ids for r in calibration_records[pid])
    trial_count = len(ids) * n
    bias = (go_count + 1) / (trial_count + 2)
    shape = (len(ADAPTIVE), len(rates), len(seeds), len(ids), n)
    probabilities = np.empty(shape)
    scores = np.empty(shape[:-1])
    output_rows = []
    graphs = {seed: build_graphs(graph, seed) for seed in seeds}
    for si, seed in enumerate(seeds):
        for ci, condition in enumerate(ADAPTIVE):
            for ri, rate in enumerate(rates):
                for pi, pid in enumerate(ids):
                    model = make_model(condition, seed, rate, bias, graphs[seed])
                    p, a, _, steps = _trajectory(model, calibration_records[pid])
                    probabilities[ci, ri, si, pi] = p
                    nll, brier = loss(a, p)
                    scores[ci, ri, si, pi] = nll.mean()
                    output_rows.append({"condition": condition, "rate": rate, "seed": seed, "participant_id": pid,
                                        "mean_nll": float(nll.mean()), "brier": float(brier.mean()),
                                        "head_update_calls": model.head_updates, "internal_update_calls": model.internal_updates,
                                        "value_update_calls": model.value_updates,
                                        "head_nonzero_steps": int(np.count_nonzero(steps[0])),
                                        "internal_nonzero_steps": int(np.count_nonzero(steps[1])),
                                        "value_nonzero_steps": int(np.count_nonzero(steps[2]))})
    means = scores.mean(axis=(2, 3))
    selected = {}
    for ci, condition in enumerate(ADAPTIVE):
        best = min(range(len(rates)), key=lambda ri: (float(means[ci, ri]), rates[ri]))
        selected[condition] = float(rates[best])
    selection = {"rates": selected, "calibration_bias": float(bias), "calibration_go_count": int(go_count),
                 "calibration_trial_count": trial_count, "calibration_ids": list(ids),
                 "calibration_seeds": list(seeds), "candidate_rates": list(rates),
                 "candidate_mean_nll": {c: means[i].tolist() for i, c in enumerate(ADAPTIVE)},
                 "selection_rule": "minimum participant/seed mean conditional NLL; exact ties choose smaller rate"}
    return selection, {"probabilities": probabilities}, output_rows


def produce(test_records, cfg, graph, selection):
    ids, seeds = cfg["test_ids"], cfg["test_seeds"]
    n = cfg["trials_per_person"]
    check_records(test_records, ids, n)
    shape = (2, len(CONDITIONS), len(seeds), len(ids), n)
    arrays = {k: np.empty(shape, dtype=np.int8 if k in ("actions", "rewards") else float)
              for k in ("probabilities", "actions", "rewards", "head_steps", "internal_steps", "value_steps")}
    graphs = {seed: build_graphs(graph, seed) for seed in seeds}
    trajectory_rows, witnesses = [], {}
    witness_keys = {}
    for modei, mode in enumerate(MODES):
        for si, seed in enumerate(seeds):
            for pi, pid in enumerate(ids):
                uniforms = np.random.default_rng(seed + 300000 + int(pid[1:])).random(n) if mode == "autonomous" else None
                for ci, condition in enumerate(CONDITIONS):
                    rate = selection["rates"].get(condition, cfg["rates"][0])
                    model = make_model(condition, seed, rate, selection["calibration_bias"], graphs[seed])
                    before = _copies(model)
                    initial = model.witness()
                    p, a, r, steps = _trajectory(model, test_records[pid], mode == "autonomous", uniforms)
                    at = (modei, ci, si, pi)
                    arrays["probabilities"][at], arrays["actions"][at], arrays["rewards"][at] = p, a, r
                    for step_i, key in enumerate(("head_steps", "internal_steps", "value_steps")):
                        arrays[key][at] = steps[step_i]
                    final = model.witness()
                    if pi == 0:
                        for stage, witness in (("initial", initial), ("final", final)):
                            for key, value in witness.items():
                                name = f"{mode}_{condition}_{stage}_{key}"
                                witness_keys.setdefault(name, []).append(np.asarray(value))
                    after = _copies(model)
                    nll, brier = loss(a, p) if mode == "conditional" else (None, None)
                    row = {"mode": mode, "condition": condition, "seed": seed, "participant_id": pid,
                           "mean_nll": None if nll is None else float(nll.mean()),
                           "brier": None if brier is None else float(brier.mean()),
                           **behavior_metrics(test_records[pid], a),
                           "head_update_calls": model.head_updates, "internal_update_calls": model.internal_updates,
                           "value_update_calls": model.value_updates,
                           "head_nonzero_steps": int(np.count_nonzero(steps[0])),
                           "internal_nonzero_steps": int(np.count_nonzero(steps[1])),
                           "value_nonzero_steps": int(np.count_nonzero(steps[2])),
                           "head_change_l2": float(np.linalg.norm(after[0] - before[0])),
                           "internal_change_l2": float(np.linalg.norm(after[1] - before[1])),
                           "value_change_l2": float(np.linalg.norm(after[2] - before[2]))}
                    for key in ("probe_kc", "probe_mbon", "probe_decision"):
                        row[f"{key}_change_l2"] = float(np.linalg.norm(final[key] - initial[key]))
                    trajectory_rows.append(row)
    witnesses.update({key: np.stack(values) for key, values in witness_keys.items()})
    return arrays, trajectory_rows, witnesses


def _interval(values):
    return [float(v) for v in np.quantile(values, [.05, .95], method="linear")]


def summarize(human_test_records, cfg, arrays, trajectory_rows):
    ids, seeds = cfg["test_ids"], cfg["test_seeds"]
    npeople, nseeds = len(ids), len(seeds)
    human = np.asarray([[behavior_metrics(human_test_records[pid])[key] for key in BEHAVIOR_METRICS] for pid in ids])
    human_rows = [{"participant_id": pid, **dict(zip(BEHAVIOR_METRICS, human[i].tolist()))} for i, pid in enumerate(ids)]
    metrics = np.empty((len(CONDITIONS), nseeds, npeople, 3))
    nll = np.empty((len(CONDITIONS), nseeds, npeople))
    brier = np.empty_like(nll)
    lookup = {(r["mode"], r["condition"], r["seed"], r["participant_id"]): r for r in trajectory_rows}
    for ci, condition in enumerate(CONDITIONS):
        for si, seed in enumerate(seeds):
            for pi, pid in enumerate(ids):
                free = lookup[("autonomous", condition, seed, pid)]
                metrics[ci, si, pi] = [free[key] for key in BEHAVIOR_METRICS]
                nll[ci, si, pi] = lookup[("conditional", condition, seed, pid)]["mean_nll"]
                brier[ci, si, pi] = lookup[("conditional", condition, seed, pid)]["brier"]
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    repeats = cfg["bootstrap_repeats"]
    person_indices = rng.integers(0, npeople, (repeats, npeople))
    seed_indices = rng.integers(0, nseeds, (repeats, nseeds))
    human_boot = human[person_indices].mean(axis=1)
    family_reports, boot_models = [], []
    margin = cfg["equivalence_margin"]
    for ci, condition in enumerate(CONDITIONS):
        boot = metrics[ci][seed_indices[:, :, None], person_indices[:, None, :]].mean(axis=(1, 2))
        boot_models.append(boot)
        difference = metrics[ci].mean(axis=(0, 1)) - human.mean(axis=0)
        reports = []
        for mi, key in enumerate(BEHAVIOR_METRICS):
            interval = _interval(boot[:, mi] - human_boot[:, mi])
            equivalent = interval[0] >= -margin and interval[1] <= margin
            distinct = interval[0] > margin or interval[1] < -margin
            reports.append({"metric": key, "human_mean": float(human[:, mi].mean()),
                            "model_mean": float(metrics[ci, :, :, mi].mean()), "difference": float(difference[mi]),
                            "ci90": interval, "equivalent": bool(equivalent),
                            "status": "within_margin" if equivalent else "outside_margin" if distinct else "inconclusive"})
        family_reports.append({"condition": condition, "metrics": reports,
                               "all_three_within_margin": all(r["equivalent"] for r in reports),
                               "inference_role": "primary" if condition == "plastic" else "descriptive_control"})
    primary = CONDITIONS.index("plastic")
    prediction = []
    for baseline in ("uniform", "bias", "persistence"):
        bi = CONDITIONS.index(baseline)
        improvements = nll[bi] - nll[primary]
        boot = improvements[seed_indices[:, :, None], person_indices[:, None, :]].mean(axis=(1, 2))
        interval = _interval(boot)
        prediction.append({"baseline": baseline, "mean_improvement_nats": float(improvements.mean()),
                           "ci90": interval, "passed": interval[0] > cfg["prediction_margin"]})
    human_gain_interval = _interval(human_boot[:, 0])
    model_gain_interval = _interval(boot_models[primary][:, 0])
    human_gain = human_gain_interval[0] > cfg["learning_gain_margin"]
    model_gain = model_gain_interval[0] > cfg["learning_gain_margin"]
    similarity = family_reports[primary]["all_three_within_margin"]
    count = int(np.prod(arrays["actions"].shape))
    return {"status": "completed", "human_test_people": npeople, "model_seeds": nseeds,
            "mode_order": list(MODES), "condition_order": list(CONDITIONS), "behavior_metric_order": list(BEHAVIOR_METRICS),
            "human_metrics": human_rows, "behavior_family_comparisons": family_reports,
            "conditional_family_mean_nll": {c: float(nll[i].mean()) for i, c in enumerate(CONDITIONS)},
            "conditional_family_mean_brier": {c: float(brier[i].mean()) for i, c in enumerate(CONDITIONS)},
            "H_similarity": {"primary_condition": "plastic", "supported": bool(similarity),
                             "claim": "behavioral_equivalence_only", "margin": margin,
                             "notice": "same three summaries within operational margin; not common mechanism or learning proof"},
            "H_prediction": {"primary_condition": "plastic", "supported": all(r["passed"] for r in prediction),
                             "minimum_improvement_nats": cfg["prediction_margin"], "contrasts": prediction},
            "learning_qualification": {"human_initial_gain_ci90": human_gain_interval,
                                       "model_initial_gain_ci90": model_gain_interval,
                                       "human_gain_observed": bool(human_gain), "model_gain_observed": bool(model_gain),
                                       "human_gain_status": "observed" if human_gain else "not_observed",
                                       "gated_learning_pattern_supported": bool(similarity and human_gain and model_gain),
                                       "notice": "human pretraining/experience prevents interpreting absent gain as absent learning"},
            "bootstrap": {"repeats": repeats, "seed": cfg["bootstrap_seed"], "quantiles": [.05, .95],
                          "method": "crossed participant/seed resampling; same people and shared seeds retained across conditions"},
            "test_modeled_trials": count, "conditional_trials": count // 2, "autonomous_trials": count // 2,
            "autonomous_nll_interpreted": False, "mechanism_claim": False, "educational_claim": False}
