# 정규화 약화 분석 재현

연구 `normalization_loss_v1`, 완료 태그 `normalization-loss-v1`이다. 이미 알려진 부모 결과의 후향 기술 분석이며 새 행동 실험이 아니다.

| 항목 | 값 |
|---|---|
| 완료 기준선 | `selective-spillover-v1` / `a19a2abafcdb0db4b1ebdcc4b7629d237d112c55` |
| 분석 source commit | `d9152f2faeb4366ed349a9d23aac8167648def08` |
| 등록 게시 commit | `49a06d9568a8190e3168e5c046666f7e08c79b8b` |
| 등록 SHA-256 | `c7580b1a5fac578eb644df18c3e57abbc604662ea1195f3d9558f10eab3ebecb` |
| 설정 SHA-256 | `852044e63194adaf8d1b550997296a4f5e3752e84815acd37c78e4a5c9720485` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 분석 UTC | 2026-10-19T22:58:25.137153+00:00 → 2026-10-19T22:58:30.581627+00:00 |
| 독립 검산 UTC | 2026-10-19T23:00:46.650878+00:00 |

원 소스13·부모10파일의 정확한 경로와 SHA-256을 등록에 고정했다. 부모175배열·96,000CSV의 원 bytes와
부모 전체 독립 검산·미지지 판정·소스/그래프/입력 Git closure·완료 snapshot70파일을 확인한다.
새 그래프·모델·특징 생성이나 새 학습·난수는 없다. 원자료는 저장소의 부모 결과를 읽으며 새 외부 다운로드는 필요하지 않다.

## 저장 분석을 독립 검산

비공개 저장소 권한과 전체 Git 이력·태그가 있는 새 clone에서 Python3.12 환경으로 실행한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-normalization-check
cd DrosoLearn-normalization-check
git checkout normalization-loss-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_normalization_loss.py --config configs/normalization_loss_v1.json --registration results/registration_normalization_loss_v1.json --result-dir results/normalization_loss_v1 --publication-proof results/normalization_loss_publication_proof.json --out scratch/normalization-verification.json
```

출력은 존재하지 않는 파일로 지정한다. 독립 검산기는 새 분석기를 import하지 않고 저장된 단위 특징으로
pinv·방향·관측 부호·실제 행동/보상/출력 경로를 검산한다. 새 행동을 추출하지 않고 저장된 uniform과 행동의 대응만 확인한다.
같은 g의 B/P/A, 구간·겹치는 미관측 범주·정확한 양/음 질량·0분모·시드/pooled 비율을 재계산하고
**11배열·6,480행·전체 요약**을 대조한다. 정규화 전 참고는 같은 g의 산술이며 별도 자율 정책이 아니다.

## 분석을 처음부터 재현

별도의 새 clone에서 등록 게시 commit `49a06d9568a8190e3168e5c046666f7e08c79b8b`으로 checkout한다. 이 판에는 새 결과 디렉터리가 없다.
위와 같이 환경을 설치한 뒤 완료 태그에 저장된 원 게시 증거를 bytes 그대로 복사한다.
다음을 임시 `copy-proof.py`에 저장해 실행하면 셸의 재인코딩을 피한다.

```python
from pathlib import Path
import subprocess
name = 'results/normalization_loss_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'normalization-loss-v1:' + name]))
```

```text
.venv\Scripts\python.exe scripts/run_normalization_loss.py run --config configs/normalization_loss_v1.json --registration results/registration_normalization_loss_v1.json --publication-proof results/normalization_loss_publication_proof.json --out results/normalization_loss_v1
.venv\Scripts\python.exe scripts/verify_normalization_loss.py --config configs/normalization_loss_v1.json --registration results/registration_normalization_loss_v1.json --result-dir results/normalization_loss_v1 --publication-proof results/normalization_loss_publication_proof.json --out results/normalization_loss_v1/verification.json
```

부모 학습을 다시 실행하지 않는다. 시드·400시행·정답·중립 허용 오차·구간·분모를 바꾸지 않는다.
재현 시 UTC·metadata·NPZ ZIP 컨테이너 bytes는 달라질 수 있지만 같은 환경의 수치·행·판정 일치를 확인한다.
이는 새 과학 표본이나 성공을 찾는 반복이 아니다.

## 검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_normalization_loss.py tests/test_normalization_envelope.py tests/test_normalization_loss_verify.py -q
```

새 검사293개는 합성 값·발명된 소형 저장 경로·가짜 Git 저장소를 쓴다.
관련 기존69개는 `test_selective_spillover.py`30개와 `test_selective_geometry.py`39개다.
최종 JUnit 세 개와 해시는 `results/normalization_loss_prefreeze_validation.json`에 있으며 중복 없이 총362개다.
이전 개발 검사나 최적화 전 반복 결과를 추가로 더하지 않는다.

그림은 동결된 `drosolearn.normalization_loss.render`로 독립 검산 뒤 생성했다.
`results/normalization_loss_figures_v1/provenance.json`이 배열·요약·검산·renderer·그림 해시를 연결한다.
본문은 pooled 비율, CSV/요약은 시드별 비율·평균/표준편차·유효 수까지 함께 제공한다.
그림 N/A는 분모0이며 정확한 tiny mass는 원자료에 남긴다.

이전18,779파일·2,093,773,638bytes의 원문 대조는 `results/normalization_loss_baseline_preservation.json`이다.
`normalization_loss_archive_validation.json`은 이후 기존 경로 Git 차이·새 동결/결과/문서/그림 해시를 결합한 완료 검사다.
두 검사를 같은 시점의 전수 검사라고 부르지 않는다. 기존 과학 소스·결과·실패·Phase5는 보존하고 README·로드맵·상태 JSON만 갱신했다.

[쉬운 결과](normalization-loss-findings.md)에서 점수 약화와 정답률, 실제 관측 기억과 정답표 진단,
출력 변화와 고정 내부 시냅스, 저장 경로 회계와 새 학습 개입을 구분한다.
