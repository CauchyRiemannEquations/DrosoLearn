# 출력 갱신 이동량 제한 연구 재현

별도 연구 ID `update_budget_v1`, 완료 태그 `update-budget-v1`이다.
기존 `representation-update-v1`과 Phase 5를 수정하지 않는다.

## 고정 실행판

| 항목 | 값 |
|---|---|
| 완료 기준선 | `representation-update-v1` / `440f6b57c6176605db37d9a5397e3f82764a5b29` |
| 새 source commit | `fc47813f1a18ec762f737a1e035647ff40509646` |
| 등록 게시 commit | `7ac10804219146127b34364b7dd464ed979239bc` |
| 등록 SHA-256 | `c99a559a22deeb5b6d7c99fadcf377d295c6f7bd503d20bd7a438f668eda2f54` |
| 설정 SHA-256 | `785e9735cf70f6aaa788c3180f64fd895204eac413ef6e84fba8373aca21f2fe` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 계산 UTC | 2026-10-19 04:57:13.422652 → 04:57:22.318805 |
| 독립 검산 UTC | 2026-10-19 04:57:51.987331 |

등록에 명시된 소스 24·부모 증거 6·그래프 4·입력 2파일의 SHA-256을 사용한다.
이전 두 입력 catalog는 bytes와 내부 study ID도 바꾸지 않는다. 새 study ID는 새 config/등록/결과에 있다.
그래프는 저장소의 `data/flywire_783_mb_left_kc512_s701` 네 파일이다.
512 KC·48 MBON·1,474연결의 단일 KC→MBON 투영이며 489 KC·29 MBON이 연결에 참여한다.
새 사람 데이터·대형 맥락 NPZ·외부 그래프 내려받기는 필요하지 않다.

## 저장한 결과의 독립 검산

완료 태그의 새 clone에서 Python 3.12 환경을 만든다.

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
```

새 `scratch` 폴더를 만들고 기존 검산을 덮어쓰지 않는 경로를 지정한다.

```text
.venv\Scripts\python.exe scripts/verify_update_budget.py --config configs/update_budget_v1.json --registration results/registration_update_budget_v1.json --result-dir results/update_budget_v1 --publication-proof results/update_budget_publication_proof.json --out scratch/update-budget-verification.json
```

검산기는 새 producer와 새 기하 helper를 import하지 않는다. 완료한 독립 검산의 그래프·입력·SVD
산술을 재사용해 세 방향과 모든 실제 선택·보상·535개 배열·비용·상한·판정을 다시 구성한다.
원 등록/출처/원격 확인/실행의 hash와 UTC 순서도 확인한다.

새 자동검사 153개는 다음 네 파일이다. 작은 만든 행렬·가짜 신경망·가짜 Git 저장소만 사용한다.

```text
.venv\Scripts\python.exe -m pytest tests/test_update_budget.py tests/test_update_budget_geometry.py tests/test_update_budget_envelope.py tests/test_update_budget_verify.py -q
```

관련 기존 회귀검사 101개는 `test_representation_update.py`, `test_representation_geometry.py`,
`test_representation_update_verify.py`다. 원 JUnit 두 개와 hash는
`results/update_budget_prefreeze_tests` 및 `results/update_budget_prefreeze_validation.json`에 있다.
개발 중 반복 실행한 검사를 더해 세지 않는다.

## 같은 설정으로 학습 자체를 재현

원 결과 폴더에서 다시 실행하지 않는다. 새 clone/별도 checkout에서 **등록 게시 commit
`7ac10804219146127b34364b7dd464ed979239bc`**으로 이동한다. 그 판에는 source·입력·등록이 있고
과학 결과 폴더는 없다. 전체 Git 이력과 기준선 태그를 가져와야 한다.

완료 태그의 원 `results/update_budget_publication_proof.json`을 같은 경로로 bytes 그대로 복사한다.
아래 내용을 임시 `copy-proof.py`로 저장해 새 checkout에서 실행하면 셸의 줄바꿈 재인코딩을 피할 수 있다.

```python
from pathlib import Path
import subprocess
name = 'results/update_budget_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'update-budget-v1:' + name]))
```

환경을 설치한 뒤 고정 예산을 실행하고 독립 검산한다.

```text
.venv\Scripts\python.exe scripts/run_update_budget.py run --config configs/update_budget_v1.json --registration results/registration_update_budget_v1.json --publication-proof results/update_budget_publication_proof.json --out results/update_budget_v1
.venv\Scripts\python.exe scripts/verify_update_budget.py --config configs/update_budget_v1.json --registration results/registration_update_budget_v1.json --result-dir results/update_budget_v1 --publication-proof results/update_budget_publication_proof.json --out results/update_budget_v1/verification.json
```

시드 17201..17210, 400시행, 세 배선×두 표현×세 갱신, RNG namespace 202610194를 바꾸지 않는다.
재현의 UTC·metadata·NPZ ZIP 컨테이너 bytes는 달라질 수 있다. 같은 환경에서 배열 수치·행동·보상·
원 판정이 일치하는지 확인한다. 실패를 성공으로 바꾸기 위한 반복이나 새로운 과학 표본으로 세지 않는다.
이번 작업은 과학 실행 한 번과 별도 전체 산술 재계산 한 번이다.

## 보존·그림·색인

- `results/update_budget_v1`: 원 등록 사본·metadata·535개 수치 배열·72,000행 CSV·요약·5원본 hash·독립 검산.
- `results/update_budget_figures_v1`: 두 정적 그림·표시 수치·원자료 및 renderer 출처 hash.
- `results/update_budget_presentation_validation.json`: 그림 시각 확인·새 문서 링크·수치/해석 구분.
- `results/update_budget_baseline_preservation.json`: 이전 태그의 18,633개 파일·1,967,008,484bytes 원문 대조.
- `results/update_budget_archive_validation.json`: 위 원문 대조 이후 기존 경로 변화가 허용한 세 색인뿐인지,
  새 source/등록/결과/그림/검사 hash가 그대로인지 확인한 완료 기록.
- `results/update_budget_prefreeze_whitespace_correction.json`: 실행 전 테스트 파일 EOF 빈 줄만 정리한 내역.

`scripts/check_update_budget_baseline.py`는 원문 대조만 한다. 이후 `scripts/check_update_budget_archive.py`가
Git 전체 diff와 새 연구의 증거를 결합한다. 두 도구 모두 기존 결과 기록을 덮어쓰지 않는다.
기존 archive와 새 archive를 같은 시점의 원문 대조처럼 설명하지 않는다.
README·로드맵·상태 JSON 세 색인 외에 기존 과학 결과나 Phase 5 보고서를 변경하지 않았다.

H_budget 미지지와 비용의 한계는 [쉬운 결과 보고서](update-budget-findings.md)에 있다.
일반화·인간 유사성·생물학적 비용은 이 재현의 대상이 아니다.
