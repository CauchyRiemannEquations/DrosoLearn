# 관측 기반 선택적 번짐 억제 실험 재현

연구 ID `selective_spillover_v1`, 완료 태그 `selective-spillover-v1`이다. 이전 부호별 저장 결과 분석과 별도의 새 보상 학습이다.

## 고정 실행판

| 항목 | 값 |
|---|---|
| 완료 기준선 | `signed-spillover-v1` / `4ee00454500b03e2ccae08ba22f2a6625526be83` |
| 과학 source commit | `7f1f67aa1d2355cd00851cf1f9d7196198cf0c9a` |
| 등록 게시 commit | `ae22169faa375e222906dd66cf3a61afd6047af5` |
| 등록 SHA-256 | `22f2ca0ddc9cdb883be1963d4ae88624c6a9adeb4ba76246b7146a457a706553` |
| 설정 SHA-256 | `41977a48b63cde4e96f207636f6071455fb90ff0053750a48881b973d70b7b46` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 계산 UTC | 2026-10-19 06:46:42.016751 → 06:47:06.925208 |
| 독립 검산 UTC | 2026-10-19 06:48:20.852951 |

등록은 정확한 소스28·부모 증거8·그래프4·입력2파일의 SHA-256을 고정한다.
이전 catalog의 bytes와 과거 study ID를 바꾸지 않았다. 부모 archive의 54파일·원자료·소스 Git closure도 확인한다.
이 제한된 부모 스냅샷 검사와 저장소 전체 약2GB의 별도 원문 대조를 구분한다. 실제 그래프는 저장소의
`data/flywire_783_mb_left_kc512_s701`이며, 출처·라이선스·버전은 원 provenance와 이전 분석을 보존한다.
512 KC·48 MBON·1,474연결의 단일 KC→MBON 투영이며 489 KC·29 MBON이 연결에 참여한다.
새 인간 자료나 외부 대형 배열 내려받기는 필요하지 않다.

## 저장 결과의 독립 검산

비공개 저장소 권한과 전체 Git 이력·태그가 있는 새 clone에서 실행한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-selective-check
cd DrosoLearn-selective-check
git checkout selective-spillover-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_selective_spillover.py --config configs/selective_spillover_v1.json --registration results/registration_selective_spillover_v1.json --result-dir results/selective_spillover_v1 --publication-proof results/selective_spillover_publication_proof.json --out scratch/selective-verification.json
```

출력은 존재하지 않는 파일을 지정한다. 검산기는 새 producer·새 기하 helper를 import하지 않고
고정된 이전 독립 검산의 입력·그래프·SVD 산술로 새 방법을 재구성한다. 모든 실제 선택·획득 보상·
관측 부호 전후·routing·동적 방향·마스크·출력 변화·checkpoint·비용·습득·두 가설·정적 조회 참고를 다시 계산한다.
175개 수치 배열과 96,000행 CSV 전체를 비교한다. 원 등록·출처·원격 확인·실행의 해시와 UTC 순서도 검사한다.

## 같은 학습을 처음부터 재현

원 결과를 덮어쓰지 않는다. 별도의 새 clone에서 **등록 게시 commit
`ae22169faa375e222906dd66cf3a61afd6047af5`**으로 checkout한다. 그 판에는 새 결과 폴더가 없다.
위와 같이 환경을 설치하고, 완료 태그의 원 publication proof를 같은 경로에 bytes 그대로 복사한다.
아래 내용을 임시 `copy-proof.py`로 저장해 실행하면 셸의 줄바꿈 재인코딩을 피할 수 있다.

```python
from pathlib import Path
import subprocess
name = 'results/selective_spillover_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'selective-spillover-v1:' + name]))
```

```text
.venv\Scripts\python.exe scripts/run_selective_spillover.py run --config configs/selective_spillover_v1.json --registration results/registration_selective_spillover_v1.json --publication-proof results/selective_spillover_publication_proof.json --out results/selective_spillover_v1
.venv\Scripts\python.exe scripts/verify_selective_spillover.py --config configs/selective_spillover_v1.json --registration results/registration_selective_spillover_v1.json --result-dir results/selective_spillover_v1 --publication-proof results/selective_spillover_publication_proof.json --out results/selective_spillover_v1/verification.json
```

시드17401..17410·400시행·RNG namespace202610196·네 방법 순서·정보 제한·사전 기준을 바꾸지 않는다.
초기 기하와 정답표 없이 가능한81상태의19,440방향이 모두 통과해야 행동 난수가 시작된다.
재현의 UTC·metadata·NPZ ZIP 컨테이너 bytes는 달라질 수 있다. 같은 환경의 배열 수치·행동·보상·판정 일치를 확인한다.
재현을 새 과학 표본이나 실패를 성공으로 바꾸는 반복으로 세지 않는다. 이번 실제 실행은 고정 예산 한 번이다.

## 자동검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_selective_spillover.py tests/test_selective_geometry.py tests/test_selective_envelope.py tests/test_selective_spillover_verify.py -q
```

새 검사250개는 만든 작은 행렬·가짜 신경망·가짜 Git 저장소만 사용한다. 관련 기존82개는
`test_update_budget.py`, `test_update_budget_geometry.py`, `test_update_budget_verify.py`다.
최종 집계 JUnit은 `results/selective_spillover_prefreeze_tests/selective_prefreeze_after_format_tests.xml`의261개와
`selective_envelope_tests.xml`의71개다. 총332개 중 새250·기존82이며 중복 집계하지 않았다.

등록 전 세 새 파일의 EOF 빈 줄만 제거했고 실행 AST는 동일했다. 재검사 전 초안과 이전 JUnit은 보존했다.
`results/selective_spillover_prefreeze_whitespace_correction.json`에 전후 해시·이유가 있고,
`selective_spillover_prefreeze_validation.json`에 최종 소스28개 해시와 사용할 검사 기록을 명시했다.
초기 `selective_prefreeze_final_tests.xml`을 최종 검사 수에 다시 더하지 않는다.

`results/selective_spillover_v1`에 원 등록 사본·metadata·175배열·CSV·요약·hash·독립 검산이 있다.
`results/selective_spillover_figures_v1`에는 고정 renderer로 만든 두 그림·표시 수치·원자료/소스 해시가 있다.
그림의 24조건 끝점과 보상 기억·정적 조회 참고는 요약과 1e-12 허용 오차로 대조했다.

`results/selective_spillover_baseline_preservation.json`은 이전18,742파일·2,039,126,449bytes 원문 대조다.
`results/selective_spillover_archive_validation.json`은 그 이후 기존 경로 차이와 새 연구 해시를 결합한 완료 검사다.
README·로드맵·상태 JSON 세 색인 외에 기존 과학 소스·원자료·설정·성공·실패·결과·Phase5를 바꾸지 않았다.

[쉬운 결과 보고서](selective-spillover-findings.md)에 관측 기록과 출력 학습, oracle과 실제 보상 방법,
정적 조회 참고와 자율 정책 실행, 같은 출력 이동 상한과 전체 자원의 차이를 구분했다.
