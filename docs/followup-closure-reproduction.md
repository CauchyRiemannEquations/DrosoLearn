# 최종 후속 종합과 재현 확인

원래의 첫 Phase 5와 17개 후속 결과는 원 경로/원 태그에 그대로 있다. 이번 마감은 `followup-closure-v1`, `drosolearn-bounded-complete-v1`에 별도 기록했다. 기존 원본을 지우거나 같은 출력 폴더를 덮어쓰지 않는다.

## 무엇을 다시 실행했고 무엇은 원 영수증을 확인했는가

| 확인 | 실제 범위 |
|---|---|
| 기존 전체 보존 | 19,206개 기존 Git blob의 현재 바이트 해시 비교. 세 상태 인덱스만 이후 갱신 허용. |
| 17개 연구의 재현 연결 | 완료 태그·원 소스/입력·등록·비공개 게시·원 다섯 산출물·독립 검산·실행 시각·과거 실패의 해시와 리터럴 재검증. |
| 최신 저장 분석의 실제 재실행 | 원 버전 독립 verifier를 한 번 실행해 192,000선택·16배열·2,160행·480경로·34,560선택/보상 칸 재산술. |
| 새 종합 | 17연구·24주장·80원 JSON 참조·66원 판정 관측·6보존 앵커의 연결을 생산/독립 코드로 대조. |
| 하지 않은 실행 | 옛 17개 학습/최적화 전부 재실행, 새 학습·시드·그래프·방향 최적화·효과 검정·인간 실험. |

최초 고유 자동검사는 267개(새 종합146개 + 원 저장 분석 관련121개)다. 실제 독립 소비자에서 원 인간 요약의 `study_id` 부재를 처리하지 못한 실패가 발생했다. 실패 당시 소스와 출력은 `results/followup_closure_verification_attempt01.json` 및 ZIP에 보존했다. 원 결과·기준을 바꾸지 않고 별도 소비자 v2를 네 소스·54입력으로 동결/게시한 뒤 추가 84개 자동검사와 전체 독립 비교를 통과했다. 원 267개가 이 실제 스키마 문제를 놓쳤다는 한계와 추가 검사 수를 함께 남긴다. 원 연구별 시험 수를 합치지 않았다. 환경의 Python/NumPy 버전은 `results/followup_closure_v1/metadata.json`, 잠금 의존성은 `requirements-lock.txt`에 있다.

## 소스와 입력의 고정

- 소스 commit: `2b9cef9fb475846b71cc63ea0629ce4e36b790ac`.
- 등록 게시 commit: `16a0643017141d521df1e4548180ccb72c87ad2a`.
- 등록 SHA-256: `be38caae22634f5b096b50d0ceb4d5f0f5a9a09d476aa484128fce282569f188`.
- 직접 동결: 소스35파일, 부모253파일. 원 연구의 소스·입력 묶음은 그 원 등록의 해시와 Git 소스에 이어진다. 재현에 재사용한 과거 소스가 직접 입력에도 나타날 수 있으며 중복 독립 증거로 세지 않는다.
- 새 저장 재검산 영수증: `results/followup_closure_reproduction_v1/objective_pathways_reverification.json`.
- 실행 명령·시작/종료·원 코드 해시: 같은 폴더의 `invocation.json`.

Git의 원 태그·소스·부모 입력 확인과 현재 원자료의 SHA-256 비교는 바이트 보존 검사다. 그 검사가 새 생물학적 독립 재현을 뜻하지 않는다. 사람 자료의 원 권한·가공 범위·출처와 FlyWire/DrosoMem 버전·라이선스는 원 provenance/감사 문서를 그대로 따른다. 새 자료를 다운로드하거나 공개 배포하지 않았다.

## 실제 최초 실행 순서

아래는 고정된 첫 실행 기록이다. 종합 코드를 커밋하고 등록을 비공개 원격에 게시·bytes 확인한 다음 저장 재검산을 실행했다. 새 종합 전에 재검산이 통과해야 한다. 고정 경로에 결과가 이미 있으면 fail closed 하며 원본을 덮어쓰지 않는다.

```powershell
python -m pytest tests/test_followup_closure.py tests/test_followup_closure_envelope.py tests/test_followup_closure_verify.py tests/test_objective_pathways.py tests/test_objective_pathways_verify.py -q
python scripts/run_followup_closure.py register --config configs/followup_closure_v1.json --out results/registration_followup_closure_v1.json
# 등록 bytes의 비공개 원격 게시 확인 후 다음 명령을 실행한다.
python scripts/verify_objective_pathways.py --config configs/objective_pathways_v1.json --registration results/registration_objective_pathways_v1.json --result-dir results/objective_pathways_v1 --publication-proof results/objective_pathways_publication_proof.json --out results/followup_closure_reproduction_v1/objective_pathways_reverification.json
python scripts/run_followup_closure.py run --config configs/followup_closure_v1.json --registration results/registration_followup_closure_v1.json --publication-proof results/followup_closure_publication_proof.json --out results/followup_closure_v1
# 최초 원 소비자는 인간 앵커의 원 스키마에서 실패했다. 그 실패와 원 소스를 보존한다.
python scripts/verify_followup_closure.py --config configs/followup_closure_v1.json --registration results/registration_followup_closure_v1.json --result-dir results/followup_closure_v1 --publication-proof results/followup_closure_publication_proof.json --out results/followup_closure_v1/verification.json
# 추가 가상 자료 시험을 먼저 통과한 다음 소비자 v2의 소스/입력을 동결·등록·비공개 게시했다.
python -m pytest tests/test_followup_closure_consumer_v2.py -q
# 게시 증명을 받은 뒤 아래 전체 독립 검산이 통과했다.
python scripts/verify_followup_closure_consumer_v2.py --config configs/followup_closure_v1.json --registration results/registration_followup_closure_v1.json --result-dir results/followup_closure_v1 --publication-proof results/followup_closure_publication_proof.json --consumer-registration results/registration_followup_closure_consumer_v2.json --consumer-publication-proof results/followup_closure_consumer_v2_publication_proof.json --out results/followup_closure_consumer_v2/verification.json
python scripts/check_followup_closure_archive.py
```

## 완료 후 읽기 전용 확인

완료 태그의 별도 checkout에서 의존성을 잠금 파일대로 준비하고 원 자료의 해시와 두 통과 영수증을 읽어 확인할 수 있다. 최신 저장 분석 verifier와 종합 소비자 v2는 `--out`만 아직 없는 저장소 안의 안전한 새 경로로 바꾸어 다시 읽기 전용 검산할 수 있다. 원 태그·입력·결과와 통과 영수증은 그대로 두고, 기존 파일과 원 실패 위치의 덮어쓰기는 거부한다. 원 실패 소비자의 명령을 양성 재현 명령으로 취급하지 않는다. archive 검사기도 완료 영수증을 덮어쓰지 않으므로 이미 존재하면 종료한다.

전체 원 실험 재현이 필요한 경우에는 각 원 연구의 재현 문서와 등록/게시 시점의 별도 checkout 절차를 따른다. 초기 최초 실험은 AGENTS.md 지침대로 `v0.1.0-smoke`의 별도 checkout을 사용한다. 완료된 태그에서 고정 출력 경로만 임의로 바꾸거나 기존 결과를 삭제해 재실행하지 않는다. 새 연구·새 조건에는 별도 사전 계획과 등록이 필요하다.

소비자 v2의 정확한 네 소스·54입력·설정·커밋·UTC는 [별도 등록](../results/registration_followup_closure_consumer_v2.json), 실제 비공개 원격 바이트 확인은 [게시 증명](../results/followup_closure_consumer_v2_publication_proof.json), 추가 시험의 고유 수·XML·전후 소스 해시는 [별도 시험 영수증](../results/followup_closure_consumer_v2_prefreeze_validation.json)에 있다. 완료 태그의 독립 읽기 검산은 새 등록을 만들지 않고 이 원 등록과 게시 증명을 사용한다.

실행 전 등록, 결과 읽기, 후향 설명, 실제 학습, 시각·문서 형식 수정은 서로 다른 작업이다. 이전 오류와 수정 원본을 새 종합에 맞춰 다시 쓰지 않았다. 이번 종합은 알려진 결과를 묶었으며 새 눈가림 가설 검정이나 메타분석이 아니다.

[최종 보고서](followup-closure-final-report.md) · [증거 부록](followup-closure-evidence-appendix.md) · [해석 한계](followup-closure-limitations.md)
