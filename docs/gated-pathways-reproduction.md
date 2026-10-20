# 전환 전후 학습 장부의 재현

연구 `gated_pathways_v1`, 완료 태그 `gated-pathways-v1`이다. 이미 완료된 gated 학습의 알려진 결과에서 출발한 후향 기술 분석이며, 새 효과 가설의 확인 검정이 아니다. 새 실제 학습·난수·그래프/모델·최적화는0회다.

| 항목 | 값 |
|---|---|
| 완료 기준선 | `gated-learning-v1` / `b89e9f2baf1aed91b4451538c85ba3e9458df53b` |
| 분석 source commit | `856963ed7b3699f34d392ffc93a842507319a95a` |
| 등록 게시 commit | `3686ab10a28b538434e385e05e834f042b93c2a8` |
| 등록 SHA-256 | `5f73a5e4961e4f531f0027a284d30fac821303f34217d087ede04b468a58918a` |
| 설정 SHA-256 | `ac3d23c7627d5123213c2e563e43d1b5d6c338af71a03fc71e7ade9edbde48c4` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 게시 bytes 확인 UTC | 2026-10-20T00:47:14.982273+00:00 |
| 실제 집계 UTC | 2026-10-20T00:47:33.014884+00:00 → 2026-10-20T00:47:34.727892+00:00 |
| 독립 검산 UTC | 2026-10-20T00:47:54.157186+00:00 |

설정·계획·소스19파일과 부모11파일을 SHA-256으로 동결했다. 부모 실제 학습의 고정 그래프·입력·출처·소스·독립 검산·195배열·CSV·사전 동결 경합 보존 증거는 부모 해시/기록을 따라 대조했다. 부모120,000행동·720최적화의 기존 독립 재구성을 다시 실행하지 않았다.

## 완료 결과만 독립 검산

비공개 접근 권한과 전체 Git 이력·태그가 있는 새 clone에서 Python3.12를 사용한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-gated-pathways-check
cd DrosoLearn-gated-pathways-check
git checkout gated-pathways-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_gated_pathways.py --config configs/gated_pathways_v1.json --registration results/registration_gated_pathways_v1.json --result-dir results/gated_pathways_v1 --publication-proof results/gated_pathways_publication_proof.json --out scratch/gated-pathways-verification.json
```

검산 출력 파일은 없어야 한다. 새 분석 생성자를 import하지 않고 부모 저장 특징·g·방향·보상 기억·계수 prefix에서12배열,1,800행24열,30조건·24대비의 모든 시드/평균 요약을 독립 계산한다. 공통 전환 시계와 각 방법의 자체 관측 집합, 네 번째 발견 시행의 pre 배정, 도움·피해의 부호, 최종 성적 합계와 비용·0분모null을 확인한다. 출력과 원자료의 무결성 검사는 과학적 개선 가설의 성공 판정과 다르다.

## 저장 기록의 집계를 처음부터 재현

다른 새 clone에서 등록 게시 commit `3686ab10a28b538434e385e05e834f042b93c2a8`으로 checkout하고 같은 환경을 설치한다. 이 판에는 새 분석 결과 폴더가 없다. 다음을 임시 `copy-proof.py`로 저장한다.

```python
from pathlib import Path
import subprocess
name = 'results/gated_pathways_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'gated-pathways-v1:' + name]))
```

```text
.venv\Scripts\python.exe copy-proof.py
.venv\Scripts\python.exe scripts/run_gated_pathways.py run --config configs/gated_pathways_v1.json --registration results/registration_gated_pathways_v1.json --publication-proof results/gated_pathways_publication_proof.json --out results/gated_pathways_v1
.venv\Scripts\python.exe scripts/verify_gated_pathways.py --config configs/gated_pathways_v1.json --registration results/registration_gated_pathways_v1.json --result-dir results/gated_pathways_v1 --publication-proof results/gated_pathways_publication_proof.json --out results/gated_pathways_v1/verification.json
```

결과 폴더는 배타 생성한다. 완료 결과가 있는 checkout에서 덮어쓰지 않는다. metadata의 UTC·NPZ 컨테이너 bytes는 달라질 수 있으나 같은 환경의 수치·분류·비율·장부 판정은 같아야 한다. 새 시드·더 긴 예산·새 학습은 재현 절차에 포함되지 않는다.

## 자동검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_gated_pathways.py tests/test_gated_pathways_envelope.py tests/test_gated_pathways_verify.py tests/test_balanced_pathways.py tests/test_gated_learning.py -q
.venv\Scripts\python.exe scripts/render_gated_pathways.py --out scratch/gated-pathways-figures
```

새142개(장부44·무결성41·독립 검산57)와 기존114개(이전장부67·게이트학습47)의 총256검사가 통과했다. 실제 JUnit은 `results/gated_pathways_prefreeze_tests/gated_pathways_prefreeze_final2.xml`, 해시·19소스·부모byte게이트는 `results/gated_pathways_prefreeze_validation.json`에 있다. 개발 검사와 두 최종 검사 모두 합성/기존 고정 구현 검사만 했고, 새 실제 집계를 보고 설계를 조정하지 않았다. 최종2는 runner 설명 문구를 해당 분석으로 바로잡은 뒤의 검사이며 앞선256통과 기록은 작업 폴더에 보존했다. 개발 실패는 없었다.

그림 명령은 완료 태그에서, 독립 검산된 원 결과를 읽어 새 출력 폴더에만 쓴다. `scratch`는 먼저 만들고 그림 폴더는 없어야 한다. 세 그림·`plotted_values.json`·`provenance.json`은 원 배열·요약·독립 검산·renderer 해시를 연결한다. 영어 Pre/Post는 모든 방법의 공통 gated 시간 구간이고 Known/Unknown은 각 방법 자신의 보상 관측 상태다. 그림은 새로운 가설 선택이나 추가 학습이 아니다.

기준선18,953파일·2,264,970,825bytes를 현재 원문 bytes의 Git blob 해시로 전수 비교했다. 기록은 `results/gated_pathways_baseline_preservation.json`이며 새 literal-byte 전수 비교라고 부르지 않는다. 이후 archive는 기존 경로 diff, source-commit의 사전검사 원문, 등록 source/parent, 새 결과·보고서·그림·세 상태 색인의 해시를 묶는다. 기존 수정 허용 경로는 README·로드맵·상태 JSON 세 색인뿐이다.

부모 보호 가설은 설계상 작동 확인으로 통과했고 성적 개선·충분한 습득은 미지지였다는 사실을 유지한다. 새 장부는 유일한 인과 원인이나 일반화·인간 유사성·공통 생물학적 기작의 증거가 아니다. 자세한 수지와 쉬운 해석은 [결과 보고서](gated-pathways-findings.md)에 있다.
