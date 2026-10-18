# Phase 4 제한된 증거 종합의 사전 계획

2026-10-19 (Asia/Seoul). `phase4_synthesis_v1`은 새 인간 효과를 추정하는 실험이 아니라, 앞서 검증한 자료와 지표를 연결하는 **증거 종합과 계수 정합성 점검**이다. 사람이 읽는 [종합 보고서](phase4-synthesis.md)와 기계가 읽는 [비교 설계](../configs/phase4_synthesis_v1.json)를 함께 동결한다. 원자료·기존 등록·완료 결과·실패는 바꾸지 않는다.

핵심 질문은 ‘여섯 학습 현상마다 지금 어떤 측정을 비교할 수 있으며, 어디부터 추가 근거가 필요한가’다. 수량비, 반복, 일반화·전이, 보상 반전, 간섭, 학습 순서를 모두 포함한다. Honoré S1/S6, Marković의 검증된 원본 25기록, 의미 게이트가 닫힌 OpenNeuro를 같은 관측 단위라고 합치지 않는다. 해당되지 않는 자료도 표에서 삭제하지 않는다.

실행 전에 설정·두 스크립트·두 시험 파일·본 계획·종합 보고서와 명시한 기존 증거의 SHA-256, 소스 Git 버전을 등록한다. 비공개 원격 저장소에 등록을 보존하고 바이트 일치를 확인한 뒤 새 결과 경로에서 한 번 실행한다. 사전에 본 결과를 종합하므로 새 맹검 확증 연구라고 부르지 않는다. `time_mixture_v1`은 별도의 사전 등록·검산이 끝난 결과만 입력으로 받는다.

주 점검은 다음과 같다.

1. 여섯 현상 각각에 세 인간 자료의 상태, 측정·단위·모델 근거·차이·추가 필요 근거가 빠짐없이 있는가.
2. 인간 과제의 source-coded 기술 측정 허용과 정확한 역사적 실행판 미확인을 함께 보존하는가. OpenNeuro 의미 미해결과 원본 미검증 49행도 남기는가.
3. 기존 `conditional_transition_v1`의 모든 기록·층에서 피드백별 정수 분모와 유지·전환 수가 원 기록의 분모 및 유효쌍을 정확히 분할하는가. 사람/환경별로 세는 것은 **양쪽 피드백이 관측되는 층의 개수**다. 비율·학습률·종별 성적은 평균하지 않는다.
4. 사람이 읽는 논문 근거와 앞으로 검증할 가설을 구분하고, 실패한 모델 기준을 사람과의 차이 검정으로 재분류하지 않는가.

주 실행은 `synthesis.json`, 등록 사본, 실행 메타데이터와 결과 해시를 새 폴더에 저장한다. 독립 스크립트는 주 함수를 가져오지 않고 원 증거 JSON에서 정수 개수와 모든 닫힌 게이트를 다시 읽고, 동결 파일·소스 Git·등록과 실행 시각·결과 해시를 대조한다. 산술과 무결성이 모두 맞아야 점검 통과다. 실패하면 그 결과를 보존하고 같은 등록을 고쳐 통과시키지 않는다.

`bounded_phase4_deliverables_completed=true`는 **현재 확보 자료에 대한 비교 체계·기술 측정·차이·미검증 가설의 종합이 완성됨**만 뜻한다. `direct_level4_test_completed=false`와 `direct_level4_verdict=untested_not_testable_with_current_pairing`을 함께 남긴다. ‘유사성 없음’이라는 통계 결론이나 모든 Phase 2·3 과제의 완료로 읽지 않는다. 동일 신경 기작과 교육 효과는 검정하지 않는다. 사람 56명의 집계 표와 원본 25기록, 계산 시드를 초파리 개체 수로 바꾸지 않는다.

재현은 최종 완료 태그의 별도 checkout에서, 기존 출력과 다른 경로를 사용한다.

```powershell
python scripts/audit_phase4_synthesis.py register --config configs/phase4_synthesis_v1.json --out NEW_REGISTRATION.json
python scripts/audit_phase4_synthesis.py run --config configs/phase4_synthesis_v1.json --registration NEW_REGISTRATION.json --out NEW_RESULTS
python scripts/verify_phase4_synthesis.py --config configs/phase4_synthesis_v1.json --registration NEW_REGISTRATION.json --result-dir NEW_RESULTS --out NEW_VERIFICATION.json
```

재현 시각은 원 실행 시각과 다르다. 별도 등록은 원 분석을 대체하지 않으며 원 완료 결과를 덮어쓰지 않는다. 외부 인간 MAT 파일의 재분석은 이번 종합에 포함하지 않는다. 기존 검산이 보증하는 범위를 넘어 원자료 자체를 다시 검증했다고 표현하지 않는다.
