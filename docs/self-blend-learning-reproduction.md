# 자기 우선 혼합 학습 재현

완료 태그는 `self-blend-learning-v1`, 과학 소스는 `7507ba8321b556652d3f2fd33b0b4dc7d3798e6a`,
실행 전 원격 등록은 `7eb5e4d35216ed3cf6245196f01cc27e3760f771`다. Python/numpy 버전은
`results/self_blend_learning_v1/metadata.json`, 설정은 `configs/self_blend_learning_v1.json`에 있다.
기존 결과 경로를 덮어쓰지 않는다. 과학 실패를 같은 등록의 새 시드나 긴 예산으로 바꾸지 않는다.

## 보관 결과 확인

태그의 별도 checkout에서 requirements-lock.txt대로 환경을 구성한다.
현재 작업 위치의 완료 결과·등록·증거는 그대로 둔다.

```text
git clone --branch self-blend-learning-v1 https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-self-blend-review
cd DrosoLearn-self-blend-review
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pytest tests/test_self_blend_learning.py tests/test_self_blend_envelope.py tests/test_self_blend_learning_verify.py tests/test_gated_learning.py tests/test_balanced_learning.py
.venv/Scripts/python.exe scripts/verify_self_blend_learning.py --config configs/self_blend_learning_v1.json --registration results/registration_self_blend_learning_v1.json --result-dir results/self_blend_learning_v1 --publication-proof results/self_blend_learning_publication_proof.json --out results/self_blend_recheck_review.json
```

마지막 경로는 아직 없는 새 파일이어야 한다. 독립 검산은 새 producer/helper를 import하지
않고144,000실제 행동/기억/출력prefix/원B/혼합/201배열/87CSV열/세 가설을 계산한다.
코호트를 새로 학습하는 명령이 아니다. 계산 순서는 재현해도 zip container 메타데이터나
재실행 시간의 bytes까지 같다는 보장은 없으므로 배열·CSV 값과 기준 판정을 비교한다.

## 원래 전향적 순서

당시 부모/기준선 검산 → 가상 검사342개 → 소스44/부모11/그래프4/입력2 해시와 코드
커밋 → 신규 등록 → 비공개 원격 등록 bytes 검증 → 첫 새 코호트 실행 → 독립검산 순이었다.
처음 사용한 명령은 아래와 같다. 완료 checkout에서는 배타 생성 보호 때문에 다시 쓸 수 없다.

```text
python scripts/run_self_blend_learning.py register --config configs/self_blend_learning_v1.json --out results/registration_self_blend_learning_v1.json
python scripts/run_self_blend_learning.py run --config configs/self_blend_learning_v1.json --registration results/registration_self_blend_learning_v1.json --publication-proof results/self_blend_learning_publication_proof.json --out results/self_blend_learning_v1
```

완전한 새 전향적 재실험에는 별도 연구ID/프로토콜/경로·실행 전 등록·원격 증거가 필요하다.
이전 결과를 지우거나 원 등록의 날짜·해시를 바꾸지 않는다. 고정된 가상 검증만 원하면
자동검사의900xx 허구 시드와 가상 그래프 fixture를 사용한다.

원본 결과의 SHA-256은 artifact_hashes.json, 원본 독립 검산은 verification.json,
사전342개 검사와 두 개발 실패 ZIP은 results/self_blend_prefreeze_validation.json의
source-commit 결합 영수증으로 확인한다. 각 그림의 숫자와 renderer/원자료 해시는
results/self_blend_figures_v1/provenance.json과 plotted_values.json에 있다.

3:1은 단위 방향 혼합이다. 자기75%·도움25%의 실제 성적 보장 또는 같은 실제 비용을
뜻하지 않는다. 새 입력 일반화나 생물학적 기작은 이 재현의 판정 대상이 아니다.
