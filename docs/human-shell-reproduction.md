# 인간–모델 조개 과제 비교 재현

2026-10-19 (Asia/Seoul). 원 실행과 난수 주소 수정 실행은 별도 등록·출력 경로를 사용한다. 같은 경로를 덮어쓰거나 원 설정의 seed·예산·판정 기준을 바꾸지 않는다.

## 고정 버전과 환경

수정 결과까지 포함한 `human-shell-comparison-v2` 태그를 별도 checkout한다. 저장소는 비공개이며 접근 가능한 계정의 기존 Git 인증을 사용한다. Python 3.12 환경에서 저장소 루트가 작업 디렉터리다.

```powershell
git clone --branch human-shell-comparison-v2 https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-shell
Set-Location DrosoLearn-shell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

등록의 runtime은 Python/NumPy 버전을 기록한다. 플랫폼·BLAS 차이는 마지막 부동소수점 자리의 차이를 만들 수 있다. 독립 산술 비교는 동결된 검증기의 허용 오차를 사용하며, 원본 파일의 해시 검증과 산술 오차 검증은 별개다.

## 원자료 캐시

자료는 TNU의 [공개 Zenodo 레코드](https://zenodo.org/records/10066829), 버전 DOI `10.5281/zenodo.10066829`다. [출처·권한 설명](human-shell-source-review.md)과 [파생 시행 저장 설명](human-shell-data-processing-note.md)을 함께 읽는다. 현재 범위는 비상업 연구다. 원 ZIP/PDF는 저장소 밖에 두며 ZIP의 개별 파일을 임의로 교체하지 않는다.

다음은 현재 작업 폴더 밖에 캐시를 만드는 예다.

```powershell
New-Item -ItemType Directory -Path ../DrosoLearn-tnu-cache
Invoke-WebRequest -Uri https://zenodo.org/records/10066829/files/Dataset_TNU.zip -OutFile ../DrosoLearn-tnu-cache/Dataset_TNU.zip
Invoke-WebRequest -Uri https://zenodo.org/records/10066829/files/readme.pdf -OutFile ../DrosoLearn-tnu-cache/readme.pdf
Copy-Item -LiteralPath data/human/tnu_shell_2024/record_snapshot.json -Destination ../DrosoLearn-tnu-cache/metadata.json
```

API 메타데이터에는 변하는 이용 통계가 있으므로 실시간 API 응답 전체의 SHA가 과거와 같다고 가정하지 않는다. `metadata.json`은 저장한 원 API 응답 스냅숏의 정확한 바이트를 사용한다. ZIP/PDF/메타데이터의 크기·SHA는 `data/human/tnu_shell_2024/source-manifest.json`에서 검사한다. 해시가 다르면 실행을 중단하고 원자료를 조사한다.

## 자동검사와 기존 결과의 독립 검산

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_human_shell.py tests/test_human_shell_models.py tests/test_human_shell_source.py tests/test_human_shell_run.py tests/test_human_shell_verify.py tests/test_human_shell_rng.py tests/test_human_shell_rng_run.py tests/test_human_shell_rng_verify.py -q
.\.venv\Scripts\python.exe scripts/verify_human_shell.py --config configs/human_shell_v1.json --registration results/registration_human_shell_v1.json --result-dir results/human_shell_v1 --cache ../DrosoLearn-tnu-cache --out results/MY_SHELL_V1_VERIFICATION.json
.\.venv\Scripts\python.exe scripts/verify_human_shell_rng.py --config configs/human_shell_rng_v2.json --registration results/registration_human_shell_rng_v2.json --result-dir results/human_shell_rng_v2 --cache ../DrosoLearn-tnu-cache --out results/MY_SHELL_V2_VERIFICATION.json
```

원 실행의 독립 검산 통과는 저장 산술을 재현했다는 뜻이다. [원 난수 주소의 설계 오류](human-shell-rng-design-note.md)를 없앴다는 뜻은 아니다. 원 자율 결과의 구간으로 유사성이나 차이를 확정하지 않는다.

## 수정 실행 재현

v2는 저장소에 보존한 v1의 보정·선정값·조건부 예측을 사용한다. 원 20명에서 보정하거나 29명의 조건부 예측을 다시 계산하지 않는다. 새로운 출력 경로에서 같은 자율 선택만 재현한다.

```powershell
.\.venv\Scripts\python.exe scripts/run_human_shell_rng.py run --config configs/human_shell_rng_v2.json --registration results/registration_human_shell_rng_v2.json --cache ../DrosoLearn-tnu-cache --out results/MY_SHELL_V2_REPLAY
.\.venv\Scripts\python.exe scripts/verify_human_shell_rng.py --config configs/human_shell_rng_v2.json --registration results/registration_human_shell_rng_v2.json --result-dir results/MY_SHELL_V2_REPLAY --cache ../DrosoLearn-tnu-cache --out results/MY_SHELL_V2_REPLAY/verification.json
.\.venv\Scripts\python.exe scripts/plot_human_shell.py --result-dir results/MY_SHELL_V2_REPLAY --cache ../DrosoLearn-tnu-cache --out results/MY_SHELL_V2_FIGURES
```

첫 실행 자체의 전체 재현은 `scripts/run_human_shell.py run`에 원 config/registration을 주고 별도 출력 경로로 실행한다. 그 재현은 알려진 난수 주소 문제까지 동일하게 재현하며, 새 과학 표본이나 수정 결과의 대체가 아니다. 원 v1 결과를 새 재현본으로 덮어쓰지 않는다.

새 연구 질문이나 학습 규칙을 바꾸려면 새 프로토콜·설정·소스·결과 경로를 만든다. `register` 명령은 해당 소스가 Git에 그대로 고정되어 있는지 확인한다. 등록을 원격에 먼저 게시하고 게시된 바이트를 확인한 다음 실행한다. 기존 실패를 학습률·시드 교체 또는 추가 훈련으로 성공으로 바꾸지 않는다.
