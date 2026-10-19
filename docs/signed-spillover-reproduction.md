# 저장 결과의 부호별 번짐 분석 재현

연구 ID `signed_spillover_v1`, 완료 태그 `signed-spillover-v1`이다. 재학습 없이 완료 부모의 배열·시행 CSV를 읽는다.

## 고정된 판과 입력

| 항목 | 값 |
|---|---|
| 완료 기준선 | `spillover-tradeoff-v1` / `a9a60a82f73310c983f64589e631b2b8a9e3a052` |
| 분석 source commit | `fd2c1ace641621e58f6d20a7c0f2ff48e6f94631` |
| 등록 게시 commit | `0f1303721777b88bddb5a587b2275c34f56739b8` |
| 등록 SHA-256 | `8e3d7961964183d2949d6f2b4a07ba59531bf8de365b6d98506de736229b25d6` |
| 설정 SHA-256 | `421d1245b247a59a19cb4082b571fcc1debefd25f5ff9f1af6e1adaeeb81c437` |
| Python / NumPy | 3.12.10 / 2.2.6 |
| 실제 분석 UTC | 2026-10-19 06:15:44.539323 → 06:15:47.711317 |
| 독립 검산 UTC | 2026-10-19 06:16:30.498083 |

새 등록은 정확히 소스 15파일·부모 입력 10파일을 SHA-256으로 동결한다. 부모 입력은 부모 등록,
원격 게시 증거, 완료 archive, 결과 폴더의 7파일이다. 부모 archive에 고정된 소스·그래프·입력·그림도
기준선 Git blob과 확인하고, 현재 세 색인은 기준선에 보존한 bytes로 확인한다.
부모 전체 행동의 기존 독립 검산을 증거로 재사용하며 부모 학습을 다시 호출하지 않는다.
출처·라이선스·버전·입력 catalog의 과거 study ID를 포함해 기존 자료는 바꾸지 않았다.

## 저장한 분석 결과 검산

비공개 저장소 접근 권한과 전체 Git 이력·태그가 있는 새 clone에서 실행한다.

```text
git clone https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-signed-check
cd DrosoLearn-signed-check
git checkout signed-spillover-v1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
mkdir scratch
.venv\Scripts\python.exe scripts/verify_signed_spillover.py --config configs/signed_spillover_v1.json --registration results/registration_signed_spillover_v1.json --result-dir results/signed_spillover_v1 --publication-proof results/signed_spillover_publication_proof.json --out scratch/signed-verification.json
```

검산 출력은 존재하지 않는 파일을 지정한다. 새 producer나 새 signed helper를 import하지 않고
부모 배열·CSV로 8개 분석 배열, 480,000개 실제 목적 효과, 24,000개 점검 시점 기대 기여,
1,500개 집계행과 전체 요약을 독립 재구성한다. 행동·난수 추출이나 새 학습은 하지 않는다.

## 같은 분석을 처음부터 재현

원 결과 폴더를 덮어쓰지 않는다. 별도의 새 clone에서 **등록 게시 commit
`0f1303721777b88bddb5a587b2275c34f56739b8`**으로 checkout한다. 그 판에는 부모 자료와 새 분석
소스·설정·등록이 있고 새 분석 결과 폴더는 없다. 위와 같이 환경을 설치한다.

완료 태그의 원 publication proof를 bytes 그대로 복사한다. 다음을 임시 `copy-proof.py`로 저장해 실행한다.

```python
from pathlib import Path
import subprocess
name = 'results/signed_spillover_publication_proof.json'
with Path(name).open('xb') as handle:
    handle.write(subprocess.check_output(['git', 'show', 'signed-spillover-v1:' + name]))
```

```text
.venv\Scripts\python.exe scripts/run_signed_spillover.py run --config configs/signed_spillover_v1.json --registration results/registration_signed_spillover_v1.json --publication-proof results/signed_spillover_publication_proof.json --out results/signed_spillover_v1
.venv\Scripts\python.exe scripts/verify_signed_spillover.py --config configs/signed_spillover_v1.json --registration results/registration_signed_spillover_v1.json --result-dir results/signed_spillover_v1 --publication-proof results/signed_spillover_publication_proof.json --out results/signed_spillover_v1/verification.json
```

30조건·기존 시드17301..17310·저장 400시행·점검 시점0,50,100,200,400·중립 문턱1e-8을 바꾸지 않는다.
새 시드는 추출하지 않는다. 재현 UTC·metadata·NPZ ZIP 컨테이너 bytes는 달라질 수 있지만,
같은 환경에서 배열 수치·분류·요약이 일치하는지 확인한다. 새 과학 표본이나 추가 행동 실험으로 세지 않는다.

## 자동검사·그림·보존

```text
.venv\Scripts\python.exe -m pytest tests/test_signed_spillover.py tests/test_signed_geometry.py tests/test_signed_envelope.py tests/test_signed_spillover_verify.py -q
```

새 검사 228개는 만든 작은 행렬·가짜 저장 경로·가짜 Git 저장소를 사용한다. 관련 기존42개는
`test_spillover_tradeoff.py`, `test_spillover_geometry.py`다. 실행 전 JUnit 두 원문·해시는
`results/signed_spillover_prefreeze_tests`와 `results/signed_spillover_prefreeze_validation.json`에 있다.
반복 실행을 검사 수에 중복 집계하지 않았다.

분석 원자료는 `results/signed_spillover_v1`, 그림과 표시 수치는 `results/signed_spillover_figures_v1`에 있다.
초기 그림의 배치 결함을 확인하고 `scripts/render_signed_figures.py`로 **새 파일명**의 최종 그림을 만들었다.
고정 분석 소스와 초기 PNG·provenance는 보존했고, 바뀐 것은 범례·제목 간격·짧은 색상 막대 이름·0근처 표시뿐이다.
30조건 막대 수치와 3개 평균 영향 행렬의 일치는 1e-12 허용 오차로 확인했으며
`final_provenance.json`·`layout_correction.json`에 초기/최종 해시와 시각 수정 이유를 남겼다.

`results/signed_spillover_baseline_preservation.json`은 이전18,703파일·2,032,301,687bytes의 원문 대조다.
`results/signed_spillover_archive_validation.json`은 그 이후 기존 경로 Git 차이와 새 해시를 결합한 최종 검사다.
두 검사를 같은 시점의 원문 재대조로 설명하지 않는다. README·로드맵·상태 JSON 세 색인 외에 기존
과학 코드·자료·성공·실패·결과·Phase5를 변경하지 않았다.

[쉬운 결과 보고서](signed-spillover-findings.md)에 누적 확률 기여와 최종 성적의 차이, 실제 경로 회계와
갱신 제거의 인과효과의 차이, 같은/반대 정답 범주의 항 개수 차이를 명시했다.
