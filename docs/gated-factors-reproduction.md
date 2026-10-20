# 후반 상태·수정 크기·방향 배분의 재현

연구 `gated_factors_v1`, 완료 태그 `gated-factors-v1`이다. 부모의 결과를 알고 정한 후향 기술 분석이다. 새 성공 가설 검정이나 새로운 실제 학습이 아니다.

| 항목 | 값 |
|---|---|
| 완료 기준선 | `gated-pathways-v1` / `e74604a67479299825e2ab5572fd3efd8f9970d6` |
| 분석 source commit | `6ee94716af68f22c9feef3b1e3b7085e57b31109` |
| 등록 게시 commit | `a5f3868f638f98ca9256beeb07c2944fca24204d` |
| 등록 SHA-256 | `3b7efbea0a0503a34a30118588a5cb54588d9a3cd3bef4c67689d753346a35a2` |
| 설정 SHA-256 | `f4d6877e76d4bff75439387301d6d4e6fbee38f0634868cc042fe113674da05f` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 게시 bytes 확인 UTC | 2026-10-20T01:08:26.945805+00:00 |
| 실제 집계 UTC | 2026-10-20T01:08:53.785424+00:00 → 2026-10-20T01:08:59.226221+00:00 |
| 독립 검산 UTC | 2026-10-20T01:10:05.028590+00:00 |

과학 소스20개·부모12개를 동결했다. 부모 파일에는 원 `gated_learning_v1/arrays.npz`와 이전 `gated_pathways_v1`의 배열·CSV·요약·독립 검산·등록·게시·archive가 직접 포함된다. 원 학습의 그래프·인코더·출처·원소스·행동/720최적화 검산·사전 동결 경합 증거는 상위 byte 게이트로 재사용하며 새로 학습하지 않는다.

## 저장 결과만 독립 검산

비공개 접근 권한과 전체 이력·태그가 있는 새 clone에서 Python3.12를 사용한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-gated-factors-check
cd DrosoLearn-gated-factors-check
git checkout gated-factors-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_gated_factors.py --config configs/gated_factors_v1.json --registration results/registration_gated_factors_v1.json --result-dir results/gated_factors_v1 --publication-proof results/gated_factors_publication_proof.json --out scratch/gated-factors-verification.json
```

출력 검산 파일은 없어야 한다. 검산기는 새 producer를 import하지 않고 실제 출력 prefix·확률 변화·공통 후반·행동 경우·8셀·세 배분·두 목적지 집합·실제 진단을 독립 재구성한다. 전체14배열·2,880행15열·24비교의 모든240개 시드 기록과 평균, null·0패딩·양수/음수 질량 배분·부모 판정·출처 해시/순서를 확인한다. 기존120,000행동/720최적화 검산을 다시 돌리거나 새로운 정책·보상을 생성하지 않는다.

## 분석을 처음부터 재현

다른 새 clone에서 등록 게시 commit `a5f3868f638f98ca9256beeb07c2944fca24204d`으로 checkout하고 같은 환경을 설치한다. 이 판에는 새 요인 결과 폴더가 없다. 다음을 임시 `copy-proof.py`로 저장한다.

```python
from pathlib import Path
import subprocess
name = 'results/gated_factors_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'gated-factors-v1:' + name]))
```

```text
.venv\Scripts\python.exe copy-proof.py
.venv\Scripts\python.exe scripts/run_gated_factors.py run --config configs/gated_factors_v1.json --registration results/registration_gated_factors_v1.json --publication-proof results/gated_factors_publication_proof.json --out results/gated_factors_v1
.venv\Scripts\python.exe scripts/verify_gated_factors.py --config configs/gated_factors_v1.json --registration results/registration_gated_factors_v1.json --result-dir results/gated_factors_v1 --publication-proof results/gated_factors_publication_proof.json --out results/gated_factors_v1/verification.json
```

결과는 고정 새 폴더에 배타 생성한다. 완료 결과가 있는 checkout에서 덮어쓰지 않는다. metadata UTC·NPZ 컨테이너 bytes는 달라질 수 있지만 같은 환경의 수치·대상·모든 배분·실제 진단·판정은 같아야 한다. 이전 실패의 예산 연장·새 시드·다른 학습률을 재현이라고 부르지 않는다.

## 검사·개발 실패·그림·봉인

```text
.venv\Scripts\python.exe -m pytest tests/test_gated_factors.py tests/test_gated_factors_envelope.py tests/test_gated_factors_verify.py tests/test_balanced_pathways.py tests/test_gated_pathways.py -q
.venv\Scripts\python.exe scripts/render_gated_factors.py --out scratch/gated-factors-figures
```

새158개(요인45·무결성48·독립 검산65)와 기존111개(이전 균형장부67·전환장부44), 총269검사가 통과했다. 최종 JUnit은 `results/gated_factors_prefreeze_tests/gated_factors_prefreeze_final.xml`이며 source20 해시·부모 byte 게이트와 `results/gated_factors_prefreeze_validation.json`으로 연결된다. 실제 집계 전 합성 예시만 사용했다.

초기 정수 합성 셀과 float Shapley 출력의 자료형을 엄격하게 비교하던 검산 도우미 오류1건을 수정했다. `results/gated_factors_development/`에 원 검산기·원 검사·8통과/1실패 JUnit의 bytes를 보존했다. 산술 값의 변경이나 실제 데이터 기준 변경이 아니다. 동결 중 원 실패 XML traceback 끝 공백이 엄격한 공백 형식 검사에 걸렸다. 과학 소스·기준·원자료를 바꾸거나 실패 원문을 다듬지 않고 **그 원문 파일 한 경로만** style 검사에서 제외했다. 나머지 staged 파일은 원래 엄격 검사에 통과했다. `results/gated_factors_prefreeze_evidence_exception.json`은 같은 bytes와 집계 전 처리 순서를 기록하고 소스 커밋에 들어 있다. 이번 완료 변경에는 해당 원문을 다시 stage하지 않아 전체 엄격 검사를 그대로 수행했다.

그림은 독립 검산이 끝난 원 결과를 읽는다. `scratch`는 먼저 만들고 그림 폴더는 없어야 한다. 세 그림과 표시 수치·provenance JSON은 원 배열·요약·독립 검산·renderer 해시를 연결하며 학습을 추가하지 않는다. 진단의 점은 각 계산 시드의 대상 시행 평균이다. 전체 최종 정답률이나 생물학적 개체가 아니다.

기준선18,986파일·2,270,722,506bytes를 현재 원문 bytes의 Git blob 해시로 전수 비교했다. `results/gated_factors_baseline_preservation.json`에 보존했으며 새 literal-byte 전수 비교라고 부르지 않는다. 이후 archive는 기존 경로 diff·source-commit의 사전 기록/개발 실패·등록 source/parent·새 결과·문서·그림·세 현재 색인 해시를 함께 묶는다. 기존 수정 허용 경로는 README·로드맵·상태 JSON뿐이다.

같은 상태의 8개 대입 계산은 새 학습 궤적이 아니며 mixed g를 새 상태에서 재계산하지 않는다. 대칭 장부의 수치적 항등식 검산은 유일한 원인이나 실행 가능한 새 개입의 성공을 증명하지 않는다. [쉬운 결과](gated-factors-findings.md)에서 방향 몫·상태/크기·한쪽 실제 행동을 구분한다.
