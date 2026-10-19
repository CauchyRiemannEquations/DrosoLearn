# 균형 방식 실제 학습의 재현

연구 `balanced_learning_v1`, 완료 태그 `balanced-learning-v1`이다. 이전 기하 분석에서 선택한 방향을 새 계산 시드에 전향적으로 적용했다. 이전 결과를 알고 설계했으며, 새 실제 시드의 집계는 등록·게시 뒤 실행했다.

| 항목 | 값 |
|---|---|
| 완료 기준선 | `cone-frontier-v1` / `56e1a31537083825e83d4402cfa9445e7e519b9b` |
| 과학 source commit | `f23a94725f5e76378e5dcc7a2c36552a309947b5` |
| 등록 게시 commit | `223362749d09e6588eb8476ab1abc654e75b9c1c` |
| 등록 SHA-256 | `42de880c7b962e39eed9850281892555477d1d118694e2ec233d692d8c8b9802` |
| 설정 SHA-256 | `257bd4efd46087f6e83abe6e2cc7c35fcdcc5cdbd4ffd17e0be6c815c9f03d8f` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 실행 UTC | 2026-10-19T23:33:00.340401+00:00 → 2026-10-19T23:33:31.879741+00:00 |
| 독립 검산 UTC | 2026-10-19T23:34:28.881350+00:00 |

소스35·선행11·그래프4·입력2파일의 정확한 경로와 SHA-256을 등록했다. v783 KC512/MBON48/1474연결의 고정 부분 투사, 원 입력·인공 이름표, 기존 인코더 규칙을 유지한다. actual seed17501..17510, namespace202610201, 조건당400시행,24조건을 고정했다. 이전 실패의 시드나 예산은 바꾸지 않았다.

## 저장 결과만 독립 검산

비공개 접근 권한과 전체 Git 이력·태그가 있는 새 clone에서 Python3.12로 실행한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-balanced-check
cd DrosoLearn-balanced-check
git checkout balanced-learning-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_balanced_learning.py --config configs/balanced_learning_v1.json --registration results/registration_balanced_learning_v1.json --result-dir results/balanced_learning_v1 --publication-proof results/balanced_learning_publication_proof.json --out scratch/balanced-verification.json
```

출력은 아직 존재하지 않는 파일이어야 한다. 검산은 새 producer와 balanced 학습자를 import하지 않는다. 원 그래프·입력·특징, 난수 순서와 실제 선택·보상·기억·방향·출력 계수,190배열·CSV80열의96,000행, 전체 요약·세 가설을 재구성한다. 실제 관측 상태510개는 원래 특징 공간의 별도 least-squares/비음수 쌍대 상한으로 최적값과 증거를 확인한다. 퇴화점 벡터의 bytes 일치를 강제하지 않는다. 검산이 사용하는 RNG는 저장 경로 재구성용이며 새로운 연구 표본·추가 학습으로 세지 않는다.

## 실제 학습을 처음부터 재현

다른 새 clone에서 등록 게시 commit `223362749d09e6588eb8476ab1abc654e75b9c1c`으로 checkout한다. 이 판에는 새 실제 결과 폴더가 없다. 위 환경을 설치하고 다음을 임시 `copy-proof.py`로 저장한다.

```python
from pathlib import Path
import subprocess
name = 'results/balanced_learning_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'balanced-learning-v1:' + name]))
```

```text
.venv\Scripts\python.exe copy-proof.py
.venv\Scripts\python.exe scripts/run_balanced_learning.py run --config configs/balanced_learning_v1.json --registration results/registration_balanced_learning_v1.json --publication-proof results/balanced_learning_publication_proof.json --out results/balanced_learning_v1
.venv\Scripts\python.exe scripts/verify_balanced_learning.py --config configs/balanced_learning_v1.json --registration results/registration_balanced_learning_v1.json --result-dir results/balanced_learning_v1 --publication-proof results/balanced_learning_publication_proof.json --out results/balanced_learning_v1/verification.json
```

완료 결과가 있는 checkout에서 덮어쓰지 않는다. 고정 출력 경로의 배타 생성은 의도된 보호다. 새 등록·추가 시드·예산 연장으로 재해석하지 않는다. 재현 metadata UTC·NPZ 컨테이너 bytes는 달라질 수 있지만 동일 환경의 실제 수치·선택·보상·캐시 키·판정은 일치해야 한다. 부모 분석이나 옛 학습을 다시 실행할 필요는 없다.

## 검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_balanced_learning.py tests/test_balanced_envelope.py tests/test_balanced_learning_verify.py tests/test_cone_geometry.py tests/test_selective_geometry.py -q
```

새 고유239개(학습61·무결성48·독립 검산130)와 기존84개(cone45·selective39), 총323개가 통과했다. JUnit·동결35소스 해시는 `results/balanced_learning_prefreeze_validation.json`에 있다. 소스 동결 전에 제거한 의미 없는 파일 끝 여분 공백줄은 초기 검사 receipt도 별도로 보존하고 최종 소스 해시를 등록했다. 실제 시드 결과를 사용한 개발 검사는 없다.

검산 뒤 생성한 세 그림과 `plotted_values.json`, `provenance.json`은 실제 결과·요약·독립 검산과 renderer를 해시로 연결한다. 그림은 발표용이며 추가 학습·매개변수 선택이 아니다. 최초 생성에 쓴 source는 `results/balanced_learning_figures_v1/render_source.py`에 bytes 그대로 보존했으며 provenance의 renderer SHA와 같다. 다음 CLI는 같은 그림 계산에서 root·출력 경로만 인자로 받도록 추가한 재현 도구다. 기존 그림을 덮어쓰지 않는다. 전 조건 학습 곡선, 주 조건의 마지막 성적·미관측 방해·실제 경로 길이와 미관측 방해 누계를 표시한다.

```text
.venv\Scripts\python.exe scripts/render_balanced_learning.py --out scratch/balanced-figures
```

독립 검산이 끝난 원자료가 필요하다. `scratch`는 먼저 만들고 그림 출력 폴더는 존재하지 않아야 한다. 최초 그림 renderer와 CLI는 경로 처리만 다르며, 새 출력의 전체 표시 수치를 원 `plotted_values.json`과 정확히 대조했다.

`results/balanced_learning_baseline_preservation.json`은 기준선의18,849파일·2,108,889,643bytes를 현재 raw bytes의 **Git blob 해시로 전수 대조**한 기록이다. 새 literal-byte 전수 비교라고 부르지 않는다. 완료 archive는 이후 기존 경로 diff와 source/parent/data/result/doc/figure 해시를 결합한다. 기존 파일 변경은 README·로드맵·상태 JSON 세 색인뿐이다.

새 원자료·등록·설정·실패 기준·전체 결과를 보존하고, 실제 수정량 상한과 누적 사용량, 출력 계수와 보상 기록 및 내부 고정 가중치, 미관측 연습문제와 새 입력 일반화를 구분한다. [쉬운 결과](balanced-learning-findings.md)에서 한계와 판정을 설명한다.
