# 부분 번짐 절충 실험 재현

연구 ID `spillover_tradeoff_v1`, 완료 태그 `spillover-tradeoff-v1`이다. 완료한 부모 연구와 Phase 5는 변경하지 않는다.

## 고정 실행판과 출처

| 항목 | 값 |
|---|---|
| 완료 기준선 | `update-budget-v1` / `ae3818576f9e3cdc4adcb26164484ff42186c179` |
| 새 source commit | `38a53fbd5096e277548e310d980fbc476ad5a383` |
| 등록 게시 commit | `180faf5405005b021e94914e4cb6c1fb65d5bbac` |
| 등록 SHA-256 | `e1b4b628f0cb162ac664b99d6183e97d4277526494dfc19507e7937613d076a9` |
| 설정 SHA-256 | `05c6332f0abb3939419f567bd03ae984fd0154240cd3da4bcd17609fa6f9a612` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 계산 UTC | 2026-10-19 05:15:38.999899 → 05:15:51.751701 |
| 독립 검산 UTC | 2026-10-19 05:16:59.486332 |

등록은 소스 28·부모 증거 6·그래프 4·입력 2파일을 SHA-256으로 고정한다.
그래프는 저장소의 `data/flywire_783_mb_left_kc512_s701` 네 파일이다. 단일 왼쪽 KC→MBON 투영의
512 KC·48 MBON·1,474연결 중 489 KC·29 MBON이 연결에 참여한다. 출처·라이선스·버전은
원 그래프 manifest와 이전 연구의 기록을 보존한다. 새 인간 자료나 외부 대형 배열은 필요하지 않다.
두 원 입력 catalog의 bytes와 과거 내부 study ID도 보존했다. 새 연구 ID는 새 설정·등록·결과에 있다.

## 저장 결과의 독립 검산

완료 태그의 새 clone에서 Python 3.12 환경을 만든다. 비공개 저장소 접근 권한과 전체 Git 이력·태그가 필요하다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-spillover-check
cd DrosoLearn-spillover-check
git checkout spillover-tradeoff-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_spillover_tradeoff.py --config configs/spillover_tradeoff_v1.json --registration results/registration_spillover_tradeoff_v1.json --result-dir results/spillover_tradeoff_v1 --publication-proof results/spillover_tradeoff_publication_proof.json --out scratch/spillover-verification.json
```

출력은 존재하지 않는 파일 경로를 지정한다. 기존 검산 결과를 덮어쓰지 않는다.
검산기는 새 producer·새 기하 helper를 import하지 않고 입력·고정 특징·SVD·다섯 방향·공통 난수·
120,000행동·보상·전체 출력 상태·6,000정적 평가·698배열·비용·두 가설을 재구성한다.
등록·원격 확인·실행의 SHA와 UTC 순서도 확인한다.

## 자동검사

```text
.venv\Scripts\python.exe -m pytest tests/test_spillover_tradeoff.py tests/test_spillover_geometry.py tests/test_spillover_envelope.py tests/test_spillover_tradeoff_verify.py -q
```

새 검사 194개는 만든 작은 행렬·가짜 모델·가짜 저장소를 사용한다. 관련 기존 82개 검사는
`test_update_budget.py`·`test_update_budget_geometry.py`·`test_update_budget_verify.py`다.
실행 전 두 JUnit 원문과 hash는 `results/spillover_prefreeze_tests` 및
`results/spillover_prefreeze_validation.json`에 있다. 반복 실행한 검사를 중복 집계하지 않는다.

## 같은 설정으로 학습 자체를 재현

원 결과 폴더에서 다시 실행하지 않는다. 별도의 새 clone에서 **등록 게시 commit
`180faf5405005b021e94914e4cb6c1fb65d5bbac`**으로 checkout한다. 그 판에는 소스·입력·등록이 있고
과학 결과 폴더는 없다. 환경을 위와 같이 설치한다.

완료 태그의 원 publication proof를 같은 상대 경로로 bytes 그대로 복사한다. 아래를 임시
`copy-proof.py`로 저장해 실행하면 셸에 의한 줄바꿈 재인코딩을 피할 수 있다.

```python
from pathlib import Path
import subprocess
name = 'results/spillover_tradeoff_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'spillover-tradeoff-v1:' + name]))
```

```text
.venv\Scripts\python.exe scripts/run_spillover_tradeoff.py run --config configs/spillover_tradeoff_v1.json --registration results/registration_spillover_tradeoff_v1.json --publication-proof results/spillover_tradeoff_publication_proof.json --out results/spillover_tradeoff_v1
.venv\Scripts\python.exe scripts/verify_spillover_tradeoff.py --config configs/spillover_tradeoff_v1.json --registration results/registration_spillover_tradeoff_v1.json --result-dir results/spillover_tradeoff_v1 --publication-proof results/spillover_tradeoff_publication_proof.json --out results/spillover_tradeoff_v1/verification.json
```

시드17301..17310·400시행·세 배선×두 표현×다섯 방향·alpha .5 주조건·RNG namespace202610195를 고정한다.
재현의 UTC·metadata·NPZ ZIP 컨테이너 bytes는 달라질 수 있다. 같은 환경에서 배열 수치·행동·보상·판정의
일치를 확인한다. 재현을 새 과학 표본이나 실패를 성공으로 바꾸는 반복으로 세지 않는다.
이번 실제 작업은 과학 실행 한 번과 독립 전체 산술 재계산 한 번이다.

## 저장물과 보존

- `results/spillover_tradeoff_v1`: 등록 사본·metadata·698개 수치 배열·120,000행 CSV·요약·hash·독립 검산.
- `results/spillover_figures_v1`: 두 정적 그림·표시 수치·원자료와 renderer 출처 hash.
- `results/spillover_presentation_validation.json`: 그림 시각 확인·문서 링크·반올림 수치와 해석 구분.
- `results/spillover_baseline_preservation.json`: 기준선의 18,668파일·1,993,639,689bytes 원문 대조.
- `results/spillover_archive_validation.json`: 원문 대조 후 기존 경로 차이와 새 자료 hash를 결합한 완료 검사.
- `results/spillover_prefreeze_whitespace_correction.json`: 등록 전 새 검산 파일의 EOF 빈 줄만 정리한 기록.

기준선 원문 대조와 마지막 색인/hash 검사는 별도 시점의 검사다. README·로드맵·상태 JSON 외에
기존 과학 코드·설정·실패·원자료·결과를 변경하지 않았다. [쉬운 결과 보고서](spillover-tradeoff-findings.md)에
두 미지지 가설과 비용·일반화·인간 비교의 한계를 기록했다.
