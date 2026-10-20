# 저장된 균형 학습 경로 분석 재현

연구 `balanced_pathways_v1`, 완료 태그 `balanced-pathways-v1`이다. 부모 성적과 미관측 방해 결과를 이미 아는 후향 기술 분석이며 새 학습이나 확증적 효과 검정이 아니다. 분석 방법·코드·부모 원자료를 동결·게시한 뒤 새 집계를 수행했다.

| 항목 | 값 |
|---|---|
| 부모 완료 기준선 | `balanced-learning-v1` / `b836a780b91d81425e534dad7284485f284799ce` |
| 분석 source commit | `6c60a36e45bb4f0fff6c48f4c5df766b92489a69` |
| 등록 게시 commit | `93015c75e3d5b45d561b4c7d3f928fb1369307a2` |
| 등록 SHA-256 | `f37513eb3a382c8676e76089a405a18ef00db58ec57ddcda68b38359794ce6b6` |
| 설정 SHA-256 | `807ba882fa3f72da8363b0b711b26872d446c4b7643814126c520fbc47d916a7` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 분석 UTC | 2026-10-19T23:56:37.387178+00:00 → 2026-10-19T23:56:39.013029+00:00 |
| 독립 검산 UTC | 2026-10-19T23:57:15.545798+00:00 |

정확한17소스·11부모 파일의 SHA-256을 등록했다. 부모190배열·실제96,000선택·510개 방향 최적화의 전체 독립 검산 및 source/graph/input/archive 원문 증거를 변경 없이 재사용한다. 새 분석에서 그래프·학습자·난수·최적화를 만들지 않는다.

## 저장 분석 결과 독립 검산

비공개 접근 권한과 전체 Git 이력·태그가 있는 새 clone에서 실행한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-pathways-check
cd DrosoLearn-pathways-check
git checkout balanced-pathways-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_balanced_pathways.py --config configs/balanced_pathways_v1.json --registration results/registration_balanced_pathways_v1.json --result-dir results/balanced_pathways_v1 --publication-proof results/balanced_pathways_publication_proof.json --out scratch/pathways-verification.json
```

출력은 존재하지 않는 파일을 지정한다. 새 producer를 import하지 않는 검산기가 저장된 g·방향·고정 특징으로 출력 prefix와 실제 보상/관측 기록·checkpoint를 확인하고19배열·1,440행·전체 요약을 재계산한다. 과거 전체 RNG 생성이나 모델 학습을 다시 실행하지 않는다.

## 분석을 처음부터 재현

별도 새 clone에서 등록 게시 commit `93015c75e3d5b45d561b4c7d3f928fb1369307a2`으로 checkout한다. 이 판에는 새 분석 결과 폴더가 없다. 위 환경을 설치하고 다음을 임시 `copy-proof.py`로 저장한다.

```python
from pathlib import Path
import subprocess
name = 'results/balanced_pathways_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'balanced-pathways-v1:' + name]))
```

```text
.venv\Scripts\python.exe copy-proof.py
.venv\Scripts\python.exe scripts/run_balanced_pathways.py run --config configs/balanced_pathways_v1.json --registration results/registration_balanced_pathways_v1.json --publication-proof results/balanced_pathways_publication_proof.json --out results/balanced_pathways_v1
.venv\Scripts\python.exe scripts/verify_balanced_pathways.py --config configs/balanced_pathways_v1.json --registration results/registration_balanced_pathways_v1.json --result-dir results/balanced_pathways_v1 --publication-proof results/balanced_pathways_publication_proof.json --out results/balanced_pathways_v1/verification.json
```

완료 결과를 덮어쓰지 않는다. 위의 별도 새 clone에서 고정 출력 경로를 처음 생성한다. 기존 결과가 있으면 배타 생성으로 거부되는 것이 정상이다.

재현에서는 부모를 다시 학습시키지 않는다. 설정의24부모 조건·기존10계산 시드·400시행을 전부 읽으며 새 시드나 예산을 추가하지 않는다. UTC·NPZ 컨테이너 bytes는 달라질 수 있지만 같은 환경의 수치·행·판정·장부 잔차는 일치해야 한다.

## 장부 정의와 한계

현재 피드백 전의 known_before가4칸이면 full, 그 외는partial이다. 마지막 부호 발견 시행은partial이다. 자기 입력은 own으로 먼저 분류하며 나머지를known_other/unknown_other로 나눈다. 유한 정밀도의 정답 확률 변화는 canonical `y*(sigmoid(z_before+g*(w@D.T))-sigmoid(z_before))`로 계산하고 부모 저장 값과1e-8안에서 대조한다. 아주 작은 부호·0도 버리지 않으며 합계의 부동소수점 잔차를 보고한다.

실제 양쪽 Go에서만 state/scalar/direction8셀을 계산한다. bit4는state,bit2는g,bit1은direction이다. 세 요인 Shapley 가중치는 빈/두요인 집합1/3, 한요인 집합1/6이다. 음수 피해는 먼저 `max(-delta_q,0)`를 적용한 뒤 같은 장부로 나눈다. 미관측 교집합을 고정하며 한쪽만 미관측인 목적지와 한쪽만 Go인 시행은 실제 나머지로 따로 남긴다. 미래 선택을 새로 발생시키거나 대체 방향을 다시 최적화하지 않는다.

완전 관측·양쪽 Go에서만 같은 c와안전제약으로 방향 점수를 비교한다. 실제선택억제상태/g를고정한matched확률변화와 기하목적변화를구분한다. 표의objective통계는각시드내평균을다시평균하며 pooled시행평균과다를수있다. 전체시행수/비율은분자분모를시드별합산한다. 새 인과효과나 모집단 유사성 검정이 아니다.

## 검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_balanced_pathways.py tests/test_pathways_envelope.py tests/test_balanced_pathways_verify.py tests/test_balanced_learning.py -q
.venv\Scripts\python.exe scripts/render_balanced_pathways.py --out scratch/pathways-figures
```

새181개(분석67·무결성40·독립 검산74)와 기존 관련61개, 총242개 고유 검사가 통과했다. 최종 JUnit·동결 소스 해시는 `results/balanced_pathways_prefreeze_validation.json`에 있다. 합성 정수 배열 dtype 오류의 초기 개발 실패 XML도 원문 보존했으며, 그 진단 자체의 끝 공백 한 줄은 수정하지 않는 명시적 예외다. 실제 집계 실패나 데이터·시드 교체가 아니다.

그림은 저장된 독립 검산 통과 결과만 읽는 저장소 CLI로 생성한다. 출력 폴더는 존재하지 않아야 한다. source·표시 수치·PNG·요약·배열·검산 hash를 `balanced_pathways_figures_v1/provenance.json`으로 연결한다. 상태항의 음수는 모델 상태 자체가 더 나쁘다는 뜻이 아니고, 고정 기록의 추가 수정 이득 차이다. 음수 방해질량은 마지막성적차이와단위가다르다.

기존18,884파일·2,176,934,749bytes를 현재 raw bytes의 Git blob 해시와 전수 대조했다. 새 literal-byte 전수 비교로 부르지 않는다. 기존 완료 자료·원소스·원본 실패는 유지하고 README·로드맵·상태JSON 세 색인만 갱신한다. archive는 이후 전체 old-path diff 및 source/parent/result/doc/figure 해시를 결합한다. 원Phase5·인간 비교 음성 결과·닫힌 일반화/간섭 조건과 자동화PAUSED를 유지한다.

[쉬운 설명·전체 조건](balanced-pathways-findings.md)에서 기록상의 이유와 확인되지 않은 설명을 함께 보고한다.
