# 추가 연구와 Phase 5 재현

각 완료 태그의 새 checkout과 고정 Python 환경을 사용한다. 기존 결과 경로에는 다시 쓰지 않는다. 아래 명령은 저장소 루트에서 실행하며 `reproduced/`는 아직 없는 별도 출력 경로다. 소스·입력·설정이 원 등록과 다르면 실행기가 거부하는 것이 정상이다. 원본 실패와 등록을 새 코드에 맞춰 수정하지 않는다.

## 환경·태그

Windows 기준 실제 Python은 3.12.10이며 각 결과 `metadata.json`과 [이전 환경 영수증](../results/human_shell_runtime_environment.json)에 기록했다. `requirements-lock.txt`가 NumPy·pytest·matplotlib 등을 고정한다. 인간 원자료는 ZIP 안의 텍스트로 읽는다. 캐시 해시는 [인간 비교 재현 문서](human-shell-reproduction.md)를 따른다.

```powershell
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-reproduction
cd DrosoLearn-reproduction
git checkout phase5-synthesis-v1
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python.exe -m drosolearn.phase5_report materialize-parent
```

저장소는 비공개이므로 기존 접근 권한이 필요하다. 완료 태그는 `shell-credit-v1`, `shell-behavior-v1`, `pattern-rule-v1`, `interference-order-v1`, `phase5-synthesis-v1`이다. 최초 smoke는 기존 지침대로 `v0.1.0-smoke`의 별도 checkout에서 재현한다. 이후 소스가 포함된 최종 태그에서 최초 smoke 등록을 억지로 다시 만들지 않는다.

## 실행과 독립 검산

```powershell
.venv/Scripts/python.exe scripts/run_prephase5.py run --config configs/shell_credit_v1.json --registration results/registration_shell_credit_v1.json --out reproduced/shell_credit_v1
.venv/Scripts/python.exe scripts/verify_shell_credit.py --config configs/shell_credit_v1.json --registration results/registration_shell_credit_v1.json --result-dir reproduced/shell_credit_v1 --out reproduced/shell_credit_v1/verification.json

.venv/Scripts/python.exe scripts/run_prephase5.py run --config configs/pattern_rule_v1.json --registration results/registration_pattern_rule_v1.json --out reproduced/pattern_rule_v1
.venv/Scripts/python.exe scripts/verify_pattern_rule.py --config configs/pattern_rule_v1.json --registration results/registration_pattern_rule_v1.json --result-dir reproduced/pattern_rule_v1 --out reproduced/pattern_rule_v1/verification.json

.venv/Scripts/python.exe scripts/run_prephase5.py run --config configs/interference_order_v1.json --registration results/registration_interference_order_v1.json --out reproduced/interference_order_v1
.venv/Scripts/python.exe scripts/verify_interference_order.py --config configs/interference_order_v1.json --registration results/registration_interference_order_v1.json --result-dir reproduced/interference_order_v1 --out reproduced/interference_order_v1/verification.json

.venv/Scripts/python.exe -m drosolearn.phase5_report run --config configs/phase5_synthesis_v1.json --registration results/registration_phase5_synthesis_v1.json --out reproduced/phase5_synthesis_v1
.venv/Scripts/python.exe scripts/verify_phase5_report.py --config configs/phase5_synthesis_v1.json --registration results/registration_phase5_synthesis_v1.json --result-dir reproduced/phase5_synthesis_v1 --out reproduced/phase5_synthesis_v1/verification.json
```

네 번째 연구는 고정된 세 번째 연구의 범용 actor를 재사용하며, 두 연구의 소스 여섯 파일씩을 명시적으로 동결한다. 이후 재현의 선행 gate는 원 저장 결과와 그 검산을 읽는다. `reproduced/`에서 만든 결과로 원 저장 경로를 교체하지 않는다. 새 가설은 이 재현 명령의 시드·예산을 바꾸는 방식이 아니라 별도 연구 ID와 등록으로 만든다.

두 번째 연구에는 원 TNU ZIP·readme PDF·metadata 캐시가 필요하다. 원문 API와 PDF 사이 라이선스 표기의 차이를 보존하며 상업 이용·MIT 재배포 권한으로 확대하지 않는다. 캐시는 Git 밖의 별도 폴더다. `--cache`에 지정한 폴더의 세 원본 이름·크기·SHA256이 [source manifest](../data/human/tnu_shell_2024/source-manifest.json)와 정확히 같아야 한다.

```powershell
.venv/Scripts/python.exe scripts/run_prephase5.py run --config configs/shell_behavior_v1.json --registration results/registration_shell_behavior_v1.json --cache C:/path/to/tnu-cache --out reproduced/shell_behavior_v1
.venv/Scripts/python.exe scripts/verify_shell_behavior.py --config configs/shell_behavior_v1.json --registration results/registration_shell_behavior_v1.json --result-dir reproduced/shell_behavior_v1 --cache C:/path/to/tnu-cache --out reproduced/shell_behavior_v1/verification.json
```

Phase 5는 이미 저장된 결과를 종합하므로 위 원 인간 캐시를 새로 읽지 않는다. 원 관측·설정·보고서·검산 파일의 해시와 JSON 포인터를 읽는다. 공통 등록 컨테이너 이름 `prospective_bounded_followup`은 종합 결과를 미래에 눈가림 확증한 뜻이 아니다. 실제 분석 범위는 설정의 `retrospective_bounded_evidence_synthesis`다.

간섭·순서 원 NPZ는 일반 Git의 단일 파일 제한 때문에 두 연속 바이트 조각으로 보존했다. 위 `materialize-parent`를 먼저 실행하면 원 파일이 없을 때만 정확한 바이트를 복원한다. 원본 SHA·원 크기·각 조각의 Git blob을 검증하고 기존 파일은 덮어쓰지 않는다. [저장 설명](result-storage.md)을 따른다. Phase 5만 새 모듈 진입점에서 이 한 부모의 Git 바이트 검사를 조각으로 수행하며, 이전 공통 코드·과학 결과·나머지 검사는 그대로다. 저장 컨테이너의 차이를 무결성 gate 생략으로 대체하지 않는다.

## 검산과 그림

테스트는 합성 fixture와 변조 탐지를 사용한다. 원 과학 시드를 테스트에서 더 실행해 성공 사례를 추가하지 않는다. 최종 자동검사 명령과 파일별 개수·소스 전후 해시는 `results/prephase5_final_test_validation.json`에 남긴다. 각 연구의 별도 사전 테스트 영수증도 보존한다.

숫자 재현에서는 시행·선택·보상·raw 점수·가중치·활성·주장 gate를 확인한다. NPZ ZIP 컨테이너와 metadata 시간의 바이트가 새 실행과 같아야 한다고 요구하지 않는다. 원 artifact hash는 원 산출물의 변경 탐지용이며, 새 실행의 산술 동등성은 새 독립 검산으로 확인한다.

그림과 읽기용 부록은 고정된 `src/drosolearn/phase5_report.py`의 `render_figures`·`render_evidence_markdown`으로 새 경로에 생성할 수 있다. 실제 생성 파일·코드·원 결과·검산 해시는 `results/phase5_synthesis_figures_v1/provenance.json`에 남긴다. 그림은 추가 학습이나 새 효과 검정이 아니다. 최종 원본 보존 감사는 `results/phase5_synthesis_archive_validation.json`이다.
