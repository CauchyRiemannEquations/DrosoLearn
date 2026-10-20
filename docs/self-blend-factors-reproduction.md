# 자기 우선 혼합의 세 요소 분석 재현

완료 태그 `self-blend-factors-v1`. 과학 소스 `a050fda72e31683961872c8d6c2c7d197d4face3`,
등록 게시 `64f5bca3a889da56c2244c958cc842c5b90c55a4`. 원래 부모 결과는 이미 알려진 후향 기술 분석이며
새 효과 가설을 검정하지 않았다. 설정/프로토콜/원자료 해시와UTC시각은 등록 및metadata에 있다.

태그의 별도 checkout에서 requirements-lock.txt대로 환경을 구성한다. 완료 원자료·등록·
실패를 덮어쓰지 않는다. 아래 검산 출력은 아직 없는 새 파일이어야 한다.

```text
git clone --branch self-blend-factors-v1 https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-self-blend-factors-review
cd DrosoLearn-self-blend-factors-review
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pytest tests/test_self_blend_factors.py tests/test_self_blend_factors_envelope.py tests/test_self_blend_factors_verify.py tests/test_gated_factors.py tests/test_self_blend_learning.py
.venv/Scripts/python.exe scripts/verify_self_blend_factors.py --config configs/self_blend_factors_v1.json --registration results/registration_self_blend_factors_v1.json --result-dir results/self_blend_factors_v1 --publication-proof results/self_blend_factors_publication_proof.json --out results/self_blend_factors_review_verification.json
```

독립 검산은 새 producer/helper를 import하지 않고 저장된D/g/w/prefix에서144,000시행·
576,000효과·14배열·3,600행15열과8셀/세 배분/혼합항등식/이전 그림 원장을 다시 계산한다.
새 학습·난수·모델·그래프·방향 최적화는 실행하지 않는다. 부모의 원래 독립 실제 재생
영수증과 입력/source Git bytes도 함께 확인한다.

원래 집계 순서는 사전295검사·부모 byte gate → source22/parent13 고정커밋 → 신규등록 →
비공개 원격 등록 bytes 검증 → 첫 저장자료 집계 → 독립검산이었다. 당시 명령은 아래이며,
완료 checkout에서는 배타 생성 규칙 때문에 같은 경로를 다시 쓸 수 없다.

```text
python scripts/run_self_blend_factors.py register --config configs/self_blend_factors_v1.json --out results/registration_self_blend_factors_v1.json
python scripts/run_self_blend_factors.py run --config configs/self_blend_factors_v1.json --registration results/registration_self_blend_factors_v1.json --publication-proof results/self_blend_factors_publication_proof.json --out results/self_blend_factors_v1
```

새 분석을 추가하려면 별도ID/계획/설정/경로/실행 전 동결·게시가 필요하다. 원본arrays.npz와
measurements.csv 및summary/metadata/등록 해시는artifact_hashes.json, 독립 산술은verification.json,
그림 원값/해시는results/self_blend_factors_figures_v1의plotted_values.json/provenance.json에 있다.

사전 형식검사 최초 마감은 신규 검사 파일의 끝 빈 줄 때문에 중단됐다. 원본295통과JUnit·
검사·당시 영수증은results/self_blend_factors_prefreeze_style_attempt01.zip에 보존했다.
실제 변경은 후행LF2바이트 제거/최종LF1개뿐이며 과학 내용·다른21소스/기준/부모입력은
그대로다. 최초 note의 개수 표현은 최종 archive의 원문 byte 대조가 정정한다. 새 295개 검사 통과와
새 source hash를 등록 전에 결합했으며 원래 ZIP·실패·봉인 원문을 바꾸지 않았다.

혼합 8셀은 g를 새 상태로 재계산하지 않는 단일 수정 대입이다. 새 정책이나 고유 인과효과·최적 비율을
증명하지 않는다. 공통 상한은 같은 실제 누적 비용이 아니며 새 일반화·인간·생물 기작 주장은 없다.
