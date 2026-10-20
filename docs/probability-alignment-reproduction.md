# 고정 상태의 목적 정렬 분석 재현

완료 태그 `probability-alignment-v1`. 과학 소스 `104d4a2038ae33cea0ae774d896380353af8b7dc`,
등록 게시 `82273c65a6ea3102190cbeae24a7da191e68e483`. 부모 결과를 이미 알고 설계한 후향 기술 분석이며
새 효과 가설/학습 성적 검정이 아니다. 모든 해시는 등록과 artifact_hashes.json에 있다.

태그를 별도 checkout하고 requirements-lock.txt대로 환경을 구성한다. 완료 파일을 덮어쓰지 않는다.
검산 출력은 아직 없는 새 경로여야 한다.

```text
git clone --branch probability-alignment-v1 https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-probability-review
cd DrosoLearn-probability-review
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pytest tests/test_probability_alignment.py tests/test_probability_alignment_envelope.py tests/test_probability_alignment_verify.py tests/test_cone_geometry.py tests/test_self_blend_learning.py
.venv/Scripts/python.exe scripts/verify_probability_alignment.py --config configs/probability_alignment_v1.json --registration results/registration_probability_alignment_v1.json --result-dir results/probability_alignment_v1 --publication-proof results/probability_alignment_publication_proof.json --out results/probability_alignment_review_verification.json
```

독립 검산은 원 gated 방법4의 저장 기록에서 상태와 관측 정보를 복원한다. 새 producer/cone solver를
import하지 않고 별도의 ambient 최소제곱/쌍대 상한 계산으로 모든 새 U/P 증거를 확인한다.
검산의 독립 최적화 호출 수는 verification.json에 별도 표시한다. 새 학습/RNG/그래프 호출은0이다.

원 실행의 순서는 사전261검사/부모 Gitblob 확인 → 소스24/부모17고정 커밋 → 등록 → 비공개 원격
등록 bytes 확인 → 11,949새 U/P최적화와 고정상태 집계 → 독립 검산이었다.
아래는 원래 배타 생성 명령이며 완료 checkout의 같은 경로에는 다시 실행할 수 없다.

```text
python scripts/run_probability_alignment.py register --config configs/probability_alignment_v1.json --out results/registration_probability_alignment_v1.json
python scripts/run_probability_alignment.py run --config configs/probability_alignment_v1.json --registration results/registration_probability_alignment_v1.json --publication-proof results/probability_alignment_publication_proof.json --out results/probability_alignment_v1
```

그림 원값과 해시는 results/probability_alignment_figures_v1/plotted_values.json 및 provenance.json에 있다.
27배열·900행17열,30조건300시드기록의 독립 산술과 유한 확률/일차 목적은 별도로 검산한다.
합성 검사 개발의2실패는 results/probability_alignment_development에 원문 ZIP/JSON으로 보존했다.
같은 기준의 최종 가상 검사를 통과한 뒤 등록했으며 실제 계산 실패를 시드/예산 변경으로 바꾸지 않았다.

P의 선형 지지 최대값은 비선형 유한 증분의 전역 최적값이 아니다. 후보들은 같은 실제 상태/g에서
한 번 평가됐고, 합한 값은 새 정책의 학습 경로/최종 성적이 아니다. 실제 다음 개입에는 별도
관측정보/계산자원/습득/보호 기준과 새로운 실행 전 동결·게시·결과 경로가 필요하다.
