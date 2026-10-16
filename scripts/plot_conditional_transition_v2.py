"""등록된 원 기록 번호·순서를 보존하는 별도 표현 수정판. 새 측정은 하지 않는다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


STUDY = "conditional_transition_v1"
PRESENTATION_STUDY = "conditional_transition_figure_correction_v1"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
OLD_SCRIPT = "scripts/plot_conditional_transition.py"
SCRIPT = "scripts/plot_conditional_transition_v2.py"
PRESENTATION_FILES = (SCRIPT, "tests/test_conditional_transition_plot_v2.py", "docs/conditional-transition-figure-correction.md")
INPUTS = ("metadata.json", "registration.json", "metrics.json", "verification.json", "artifact_hashes.json")
CONDITIONS = ("linear", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic", "frozen")
ENVIRONMENTS = ("deterministic", "probabilistic", "uninformative")
SUPPORT = ("both_feedback", "feedback_0_only", "feedback_1_only", "neither")
HUMAN_CELLS = tuple((phase, state, choice) for phase in ("initial", "transfer")
                    for state in ("1", "2") for choice in (0, 1))
MODEL_CELLS = tuple((phase, state, choice) for phase, state in (("phase1", "larger"), ("phase2", "smaller"))
                    for choice in (0, 1))
COLORS = ("#438f84", "#cda05e", "#678cb0", "#e1e3e6")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "그림 입력에 중복 JSON 키가 있습니다")
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("비유한 JSON 값")))


def safe_child(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "안전한 상대 산출물 경로가 필요합니다")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "산출물 경로가 결과 폴더를 벗어납니다")
    return path


def same_rate(actual, expected):
    if expected is None:
        return actual is None
    return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)


def validate_feedback(cells):
    require(set(cells) == {"0", "1"}, "피드백 0/1 셀이 모두 필요합니다")
    for cell in cells.values():
        require(all(type(cell[key]) is int and cell[key] >= 0 for key in ("denominator", "stay", "switch")),
                "분자와 분모는 음수가 아닌 정수여야 합니다")
        d, stay, switch = (cell[key] for key in ("denominator", "stay", "switch"))
        require(stay + switch == d and same_rate(cell["stay_rate"], stay / d if d else None)
                and same_rate(cell["switch_rate"], switch / d if d else None), "기록된 분자·분모·비율이 맞지 않습니다")


def validate_presentation(root, registration_path, main_registration, verification):
    path = Path(registration_path).resolve()
    record = read_json(path)
    require(record["study_id"] == PRESENTATION_STUDY and record["parent_study_id"] == STUDY
            and record["scientific_reanalysis"] is False, "별도 표현 수정 등록이 필요합니다")
    require(type(record["source_git_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", record["source_git_commit"]),
            "표현 수정 소스의 고정 커밋이 필요합니다")
    for key in ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws"):
        require(type(record[key]) is int and record[key] == 0, "표현 수정은 새 측정·학습·생성을 하지 않습니다")
    stamp, verified = (datetime.fromisoformat(value) for value in (record["registered_at_utc"], verification["verified_at_utc"]))
    require(all(value.tzinfo is not None and value.utcoffset().total_seconds() == 0 for value in (stamp, verified))
            and verified <= stamp <= datetime.now(timezone.utc), "원 검산 뒤·그림 전의 UTC 표현 등록이 필요합니다")
    require(set(record["frozen_files"]) == set(PRESENTATION_FILES), "새 표현 파일 세 개만 고정해야 합니다")
    for relative, expected in record["frozen_files"].items():
        require(type(expected) is str and re.fullmatch(r"[a-f0-9]{64}", expected)
                and digest(safe_child(SOURCE_ROOT, relative)) == expected, "새 표현 소스가 등록 바이트와 다릅니다")
    require(record["frozen_files"][SCRIPT] == digest(__file__), "현재 v2 스크립트와 표현 등록이 다릅니다")
    require(set(record["input_hashes"]) == set(INPUTS), "원 결과의 다섯 파일을 정확히 고정해야 합니다")
    require(all(record["input_hashes"][name] == digest(root / name) for name in INPUTS),
            "표현 등록 이후 원 결과 입력이 바뀌었습니다")
    require(record["original_plot_sha256"] == main_registration["frozen_files"][OLD_SCRIPT]
            == digest(safe_child(SOURCE_ROOT, OLD_SCRIPT)), "기존 그림 코드의 부모 바이트를 보존해야 합니다")
    return {"path": str(path), "sha256": digest(path), "source_git_commit": record["source_git_commit"],
            "registered_at_utc": record["registered_at_utc"],
            "frozen_files": record["frozen_files"]}


def registered_human_ids(registration, expected_count):
    entries = [entry for entry in registration["input_files"] if entry["role"] == "human_raw"]
    require(len(entries) == expected_count, "등록된 모든 사람 원 입력이 필요합니다")
    indices = [entry["row_index"] for entry in entries]
    require(all(type(index) is int and index >= 0 for index in indices) and len(set(indices)) == expected_count,
            "원 사람 행 번호가 없거나 중복되었습니다")
    require(len({entry["path"] for entry in entries}) == len({entry["osf_file_id"] for entry in entries}) == expected_count,
            "사람 원 입력의 파일 정체성이 중복되었습니다")
    return [f"human/main/{index}" for index in indices]


def validate_main_inputs(result_dir):
    root = Path(result_dir).resolve()
    meta, reg, metrics, checked, artifacts = (read_json(root / name) for name in INPUTS)
    require(meta["status"] == "completed" and checked["status"] == "passed" and checked["errors"] == [],
            "완료되고 독립 검산이 통과한 결과만 그릴 수 있습니다")
    cfg = reg["config"]
    require(all(record["study_id"] == STUDY for record in (meta, reg, metrics, checked))
            and cfg["study_id"] == STUDY, "다른 연구의 결과입니다")
    require(cfg["scope"] in ("registered_recordwise_descriptive_measurement", "integration_test"), "등록된 기술 분석이 필요합니다")
    require(meta["config"] == cfg, "실행 설정이 등록과 다릅니다")
    require(cfg["model"]["conditions"] == list(CONDITIONS)
            and [env["id"] for env in cfg["model"]["environments"]] == list(ENVIRONMENTS), "모든 조건·환경이 필요합니다")
    seeds = cfg["model"]["seeds"]
    require(type(seeds) is list and seeds and all(type(seed) is int for seed in seeds)
            and len(set(seeds)) == len(seeds), "중복 없는 고정 시드가 필요합니다")
    if cfg["scope"] != "integration_test":
        require(seeds == list(range(11001, 11011)) and cfg["human"]["record_count"] == 25
                and cfg["human"]["trials"] == 1000 and cfg["model"]["trials_per_phase"] == 400,
                "등록한 모든 원 기록이 필요합니다")
    for key in ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws"):
        require(type(cfg[key]) is int and cfg[key] == 0 and type(meta[key]) is int and meta[key] == 0,
                "그림 작업에 새 학습·생성·난수를 허용하지 않습니다")
    require(all(cfg[key] is False for key in ("pooling", "strata_averaging", "causal_inference"))
            and all(metrics[key] is False for key in ("pool_group_or_species_estimates", "averaged_stratum_estimates", "causal_effect_estimated"))
            and all(meta[key] is False for key in ("pooled_group_or_species_estimates", "averaged_stratum_estimates", "causal_effect_estimated"))
            and meta["human_base_metrics_exact_match"] is True, "층 평균·집단 합산·인과 추정이 없는 원 지표 보존 결과가 필요합니다")
    require(checked["metrics_sha256"] == digest(root / "metrics.json")
            and checked["artifact_hashes_sha256"] == digest(root / "artifact_hashes.json")
            and checked["registration_sha256"] == meta["registration_sha256"] == digest(root / "registration.json")
            and checked["config_sha256"] == meta["config_sha256"] == reg["config_sha256"],
            "독립 검산이 현재 결과 바이트와 연결되지 않습니다")
    inventory = reg["input_files"]
    encoded = json.dumps(inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    require(meta["input_files"] == inventory and checked["input_manifest_sha256"] == hashlib.sha256(encoded).hexdigest(),
            "검산한 원 입력 목록과 다릅니다")
    require(meta["frozen_files"] == reg["frozen_files"]
            and set(reg["frozen_files"]) == set(cfg["frozen_files"]), "원 연구의 동결 소스 목록이 바뀌었습니다")
    if cfg["scope"] != "integration_test":
        require(len(reg["frozen_files"]) == 16, "원 연구의 고정 소스 16개가 모두 필요합니다")
    for relative, expected in reg["frozen_files"].items():
        require(type(expected) is str and re.fullmatch(r"[a-f0-9]{64}", expected)
                and digest(safe_child(SOURCE_ROOT, relative)) == expected, "원 연구의 고정 소스 바이트가 바뀌었습니다")
    require(OLD_SCRIPT in reg["frozen_files"], "원 그림 소스의 부모 바이트가 필요합니다")
    timestamp = datetime.fromisoformat(checked["verified_at_utc"])
    require(timestamp.tzinfo is not None and timestamp.utcoffset().total_seconds() == 0
            and timestamp <= datetime.now(timezone.utc), "선행하는 UTC 독립 검산이 필요합니다")
    for relative, expected in artifacts.items():
        require(type(expected) is str and re.fullmatch(r"[a-f0-9]{64}", expected)
                and digest(safe_child(root, relative)) == expected, "검산한 주 산출물이 바뀌었습니다")
    require(artifacts["metrics.json"] == checked["metrics_sha256"]
            and artifacts["registration.json"] == checked["registration_sha256"], "핵심 그림 입력 연결이 없습니다")

    human_ids = registered_human_ids(reg, cfg["human"]["record_count"])
    model_ids = [f"model/{seed}/{condition}/{environment}" for environment in ENVIRONMENTS
                 for condition in CONDITIONS for seed in seeds]
    expected_ids = set(human_ids + model_ids)
    streams = {}
    trial_count = candidate_count = stratum_count = eligible_count = excluded_count = 0
    for stream in metrics["streams"]:
        identifier = stream["stream_id"]
        require(identifier in expected_ids and identifier not in streams, "기록이 중복되거나 등록 범위를 벗어납니다")
        human = identifier.startswith("human/")
        require(stream["kind"] == ("human" if human else "model")
                and stream["coordinate"] == cfg["human" if human else "model"]["coordinate"], "선택 정체성이 다릅니다")
        require(stream["trials"] == (cfg["human"]["trials"] if human else 2 * cfg["model"]["trials_per_phase"])
                and stream["candidates"] == stream["trials"] - 1
                and stream["eligible"] + stream["excluded"] == stream["candidates"], "원 기록 범위와 분모가 다릅니다")
        validate_feedback(stream["by_feedback"])
        require(sum(cell["denominator"] for cell in stream["by_feedback"].values()) == stream["eligible"], "전체 유효 분모가 다릅니다")
        strata = {}
        for row in stream["strata"]:
            key = (row["phase"], row["state"], row["choice"])
            require(key not in strata and type(row["choice"]) is int, "층이 중복되거나 선택 코드가 다릅니다")
            validate_feedback(row["by_feedback"])
            d0, d1 = (row["by_feedback"][str(f)]["denominator"] for f in (0, 1))
            status = "both_feedback" if d0 and d1 else "feedback_0_only" if d0 else "feedback_1_only" if d1 else "neither"
            common = d0 > 0 and d1 > 0
            difference = row["by_feedback"]["1"]["stay_rate"] - row["by_feedback"]["0"]["stay_rate"] if common else None
            require(type(row["common_support"]) is bool and row["common_support"] == common
                    and row["support_status"] == status and same_rate(row["stay_rate_difference_1_minus_0"], difference),
                    "관측 분류 또는 층 안의 차이가 맞지 않습니다")
            strata[key] = row
        require(set(strata) == set(HUMAN_CELLS if human else MODEL_CELLS), "사람 8층·모델 4층을 모두 보존해야 합니다")
        for f in ("0", "1"):
            for key in ("denominator", "stay", "switch"):
                require(sum(row["by_feedback"][f][key] for row in strata.values()) == stream["by_feedback"][f][key],
                        "층의 정수 합이 전체 지표와 다릅니다")
        streams[identifier] = {"source": stream, "strata": strata}
        trial_count += stream["trials"]
        candidate_count += stream["candidates"]
        stratum_count += len(strata)
        eligible_count += stream["eligible"]
        excluded_count += stream["excluded"]
    require(set(streams) == expected_ids and len(streams) == cfg["registered_streams"]
            and trial_count == cfg["registered_input_trials"] and candidate_count == cfg["registered_candidate_pairs"]
            and stratum_count == cfg["registered_strata"], "등록한 모든 기록과 층이 필요합니다")
    require((meta["streams"], meta["input_trials"], meta["candidates"], meta["strata"], meta["eligible"], meta["excluded"])
            == (len(streams), trial_count, candidate_count, stratum_count, eligible_count, excluded_count),
            "메타데이터의 전체 범위가 그림 입력과 다릅니다")
    return {"root": root, "config": cfg, "streams": streams, "human_ids": human_ids,
            "main_registration": reg, "verification": checked,
            "input_hashes": {name: digest(root / name) for name in INPUTS}}


def validate_inputs(result_dir, presentation_registration):
    bundle = validate_main_inputs(result_dir)
    bundle["presentation_registration"] = validate_presentation(bundle["root"], presentation_registration,
                                                               bundle["main_registration"], bundle["verification"])
    return bundle


def register_presentation(result_dir, out):
    bundle = validate_main_inputs(result_dir)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()
    require(re.fullmatch(r"[a-f0-9]{40}", commit), "표현 수정 소스의 정확한 Git 커밋이 필요합니다")
    record = {"study_id": PRESENTATION_STUDY, "parent_study_id": STUDY,
              "registered_at_utc": datetime.now(timezone.utc).isoformat(), "source_git_commit": commit,
              "frozen_files": {relative: digest(safe_child(SOURCE_ROOT, relative)) for relative in PRESENTATION_FILES},
              "input_hashes": bundle["input_hashes"],
              "original_plot_sha256": bundle["main_registration"]["frozen_files"][OLD_SCRIPT],
              "scientific_reanalysis": False, "learning_updates": 0, "learner_constructions": 0,
              "dataset_generations": 0, "rng_draws": 0}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return record


def language():
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    korean = next((name for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic") if name in available), None)
    plt.rcParams.update({"font.family": korean or "DejaVu Sans", "axes.unicode_minus": False, "font.size": 9})
    if korean:
        return {"id": "ko", "title": "같은 선택·상태에서 두 피드백을 모두 관측할 수 있는가?",
                "subtitle": "색은 유효 분모의 존재 여부 · 학습 성과나 사람과 모델의 유사성이 아님",
                "human": "사람: 전체 원본 기록", "model": {"deterministic": "모델: 결정적 보상", "probabilistic": "모델: 확률적 보상", "uninformative": "모델: 무정보 보상"},
                "legend": ["피드백 0·1 모두 1쌍 이상", "피드백 0만 관측", "피드백 1만 관측", "유효 관측 없음"],
                "human_columns": [f"{'처음' if phase == 'initial' else '후반'}\n상태{state}·카드{choice}" for phase, state, choice in HUMAN_CELLS],
                "model_columns": [f"{'1단계' if phase == 'phase1' else '2단계'}\n{'큰' if state == 'larger' else '작은'} 참조·{'작은' if choice == 0 else '큰'} 선택" for phase, state, choice in MODEL_CELLS],
                "record": "기록", "footer": "각 행은 원 사람 기록 또는 조건별 계산 시드 · 모델은 각 조건 안에서 시드 오름차순\n두 분모가 양수여도 정밀성·보상 인과 효과·학습을 보장하지 않음 · 빈 설계층도 보존\n사람 피드백 0은 금전 손실, 모델 0은 무보상 · 새 학습 0회 · Level 4/5 판정 없음"}
    return {"id": "en", "title": "Are both feedback outcomes observed for the same choice and state?",
            "subtitle": "Colors show nonzero eligible denominators, not learning success or human-model similarity",
            "human": "Human: all original records", "model": {"deterministic": "Model: deterministic", "probabilistic": "Model: probabilistic", "uninformative": "Model: uninformative"},
            "legend": ["At least one eligible pair for both", "Feedback 0 only", "Feedback 1 only", "Neither observed"],
            "human_columns": [f"{phase}\nS{state}, card {choice}" for phase, state, choice in HUMAN_CELLS],
            "model_columns": [f"{phase}\n{state} ref; {'small' if choice == 0 else 'large'}" for phase, state, choice in MODEL_CELLS],
            "record": "Record", "footer": "Rows are original human records or condition-specific computational seeds; seed order within each condition\nPositive denominators do not establish precision, causal reward effects, or learning; all designed cells retained\nHuman feedback 0 means monetary loss; model 0 means no reward; no new learning; no Level 4/5 claim"}


def create_figure(result_dir, out, presentation_registration):
    bundle = validate_inputs(result_dir, presentation_registration)
    root, out = bundle["root"], Path(out).resolve()
    require(out.parent == root.parent and out != root, "그림은 주 결과와 다른 새 sibling 폴더에 저장합니다")
    out.mkdir(exist_ok=False)
    cfg, words = bundle["config"], language()
    record = {"study_id": STUDY, "status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_hashes": bundle["input_hashes"], "plot_script_sha256": digest(__file__), "figure_language": words["id"],
              "presentation_study_id": PRESENTATION_STUDY, "presentation_registration": bundle["presentation_registration"],
              "presentation_registration_sha256": bundle["presentation_registration"]["sha256"],
              "source_git_commit": bundle["presentation_registration"]["source_git_commit"],
              "presentation_registered_at_utc": bundle["presentation_registration"]["registered_at_utc"],
              "scientific_reanalysis": False,
              "selection": {"human_records": len(bundle["human_ids"]), "human_stream_order": bundle["human_ids"], "human_strata_per_record": 8,
                            "model_seeds": cfg["model"]["seeds"], "model_conditions": list(CONDITIONS),
                            "model_environments": list(ENVIRONMENTS), "model_strata_per_record": 4, "all_registered_records": True},
              "scope": "분모 존재 분류만 표시; 층 평균·집단/종간 검정·학습 및 인과 판정 없음", "figure_hashes": {}}
    fig = None
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        panels = [(bundle["human_ids"], HUMAN_CELLS, words["human"], None)]
        for env in ENVIRONMENTS:
            ids = [f"model/{seed}/{condition}/{env}" for condition in CONDITIONS for seed in cfg["model"]["seeds"]]
            panels.append((ids, MODEL_CELLS, words["model"][env], env))
        cmap = ListedColormap(COLORS)
        norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
        for ax, (ids, cells, title, env) in zip(axes.flat, panels):
            data = np.array([[SUPPORT.index(bundle["streams"][identifier]["strata"][cell]["support_status"])
                              for cell in cells] for identifier in ids], dtype=int)
            ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto", origin="upper")
            ax.set_title(f"{title} ({len(ids)} × {len(cells)})", pad=12, fontsize=12)
            ax.set_xticks(range(len(cells)), words["human_columns"] if env is None else words["model_columns"], fontsize=9)
            if env is None:
                step = max(1, math.ceil(len(ids) / 25))
                indices = list(range(0, len(ids), step))
                ax.set_yticks(indices, [f"{words['record']} {ids[index].rsplit('/', 1)[1]}" for index in indices], fontsize=8)
            else:
                width = len(cfg["model"]["seeds"])
                ax.set_yticks([index * width + (width - 1) / 2 for index in range(len(CONDITIONS))], CONDITIONS, fontsize=9)
                for edge in range(width, len(ids), width):
                    ax.axhline(edge - .5, color="white", linewidth=1)
            for edge in range(1, len(cells)):
                ax.axvline(edge - .5, color="white", linewidth=.6)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_color("#b1b7bf")
        fig.suptitle(words["title"], fontsize=19, y=.98)
        fig.text(.5, .946, words["subtitle"], ha="center", fontsize=11)
        legend = [Patch(facecolor=color, label=label, edgecolor="#b1b7bf") for color, label in zip(COLORS, words["legend"])]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .088), ncol=4, frameon=False, fontsize=10)
        fig.text(.5, .022, words["footer"], ha="center", va="bottom", fontsize=10, linespacing=1.5)
        fig.subplots_adjust(top=.89, bottom=.17, left=.11, right=.985, hspace=.36, wspace=.31)
        path = out / "feedback_support.png"
        with path.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=160, facecolor="white")
        plt.close(fig)
        fig = None
        record["figure_hashes"][path.name] = digest(path)
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if fig is not None:
            plt.close(fig)
        with (out / "metadata.json").open("x", encoding="utf8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--presentation-registration", type=Path)
    args = parser.parse_args(argv)
    if args.register:
        require(args.presentation_registration is None, "새 표현 등록과 그림 생성을 동시에 실행하지 않습니다")
        register_presentation(args.result_dir, args.out)
        print(json.dumps({"status": "registered", "out": str(args.out)}, ensure_ascii=False))
        return 0
    require(args.presentation_registration is not None, "그림 전에 별도 표현 등록이 필요합니다")
    result = create_figure(args.result_dir, args.out, args.presentation_registration)
    print(json.dumps({"status": result["status"], "figure": str(args.out / "feedback_support.png")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
