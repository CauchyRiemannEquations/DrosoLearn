# U/P 실제 학습 재현

완료 태그는 `objective-learning-v1`이며, 과학 코드 버전은 `adeed238058cfb9fb9d2e186846d310bf1f0c296`,
등록 게시 버전은 `6215953da6ed59cbbdabffe63d6c2eebff426c3b`다. 설정·입력·배선 자료·부모 연구·
소스 파일의 해시와 UTC 시각은 등록 및 metadata에 있다. 별도 복제본에서 아래 명령을 사용한다.
완료 원자료·등록·실패를 덮어쓰지 않으며, 검산 출력은 아직 없는 새 경로여야 한다.

```text
git clone --branch objective-learning-v1 https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-objective-review
cd DrosoLearn-objective-review
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pytest tests/test_objective_learning.py tests/test_objective_learning_envelope.py tests/test_objective_learning_verify.py tests/test_cone_geometry.py tests/test_self_blend_learning.py tests/test_probability_alignment.py
.venv/Scripts/python.exe scripts/verify_objective_learning.py --config configs/objective_learning_v1.json --registration results/registration_objective_learning_v1.json --result-dir results/objective_learning_v1 --publication-proof results/objective_learning_publication_proof.json --out results/objective_learning_review_verification.json
```

독립 검산은 새 학습 코드를 호출하지 않고, 192,000번의 실제 선택과 9,600번의 정적 평가,
215개 배열·96개 열을 다시 계산한다. 현재 관측 정보·U/P 목적·기존 B/W 방향·최적성 증거와
일곱 가설도 확인한다. 기존 독립 배선·입력 부호화 보조 코드를 재사용하며, 소스 49개를
모두 고정했다. 방향 최적화의 독립 재검산 횟수도 별도로 기록한다.

원 실행 순서는 사전 검사 385개와 부모 자료 보존 확인 → 소스 49개·부모 증거 14개·배선 자료
4개·입력 자료 2개 고정 커밋 → 새 등록 → 비공개 원격의 원문 확인 → 새 17801..17810 계산 시드의
첫 구성·학습 → 독립 재생이었다. 아래는 당시 실행 명령이다. 기존 결과를 덮어쓰지 않도록
배타 생성하므로, 완료 복제본의 같은 경로에는 다시 실행할 수 없다.

```text
python scripts/run_objective_learning.py register --config configs/objective_learning_v1.json --out results/registration_objective_learning_v1.json
python scripts/run_objective_learning.py run --config configs/objective_learning_v1.json --registration results/registration_objective_learning_v1.json --publication-proof results/objective_learning_publication_proof.json --out results/objective_learning_v1
```

requirements-lock.txt의 Python 패키지와 metadata의 실제 Python/NumPy 버전을 사용한다.
다른 환경에서 검산할 때도 정해 둔 수치 허용치와 Git 원문 비교를 지켜야 한다.
최초의 별도 0.1 smoke 실험을 재현할 때는 `v0.1.0-smoke` 태그를 사용한다.

시행 기록은 `measurements.csv.gz`에 결정적 무손실 gzip으로 저장했다. 압축을 풀면 UTF-8 CSV의
192,000행·96열·CRLF 줄바꿈이 그대로 복원된다. 원본 경로에 덮어쓰지 말고
`gzip.open(path, 'rt', encoding='utf8', newline='')`로 읽는다. `artifact_hashes.json`에는 압축 원문·
배열·metadata·등록·summary의 SHA가 있다. 그림 원값과 해시는
`results/objective_learning_figures_v1/plotted_values.json` 및 `provenance.json`에 있다.

새 전향적 재실험에는 별도 연구 ID·가설·설정·결과 경로와 실행 전 동결·게시가 필요하다.
같은 실패의 시드나 예산을 바꾸지 않는다. 실제 비용·성적·미관측 보호·충분한 습득은 서로 다른
지표다. 새 문제 일반화, 공통 생물학적 기작, 인간 교육 효과는 이번 검증에 포함되지 않는다.
