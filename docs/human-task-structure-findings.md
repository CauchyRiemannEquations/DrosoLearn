# 사람 과제 구조 검증 결과

작성일: 2026-10-15 (Asia/Seoul). 연구 ID `human_task_structure_v1`.

## 먼저 쉽게 설명하면

**25개 원본의 기록 관계가 모두 맞았다.** 카드를 골랐다는 기록과 과제가 유리하다고 정한 카드, 정오 표시가 서로 일관됐다. 정보를 요청하거나 응답하지 않은 칸은 카드 오답으로 채우지 않았다. 같은 조건에서 사용한 상태·잠재 결과 일정도 완전히 같았다.

이 결과는 사람이 잘 배웠다는 뜻이 아니다. 다음 단계에서 ‘같은 선택을 유지했는지’를 셀 때 사용할 기록의 뜻과 구획을 확인한 결과다. 카드 배치를 나타내는 `random_lr`를 정보 요청 가용성으로 쓰지 않기로 했다.

## 고정 범위와 판정

| 확인 항목 | 결과 |
|---|---|
| 기존 원본·가공본 대조가 통과한 원본 | Main 25개 전부 |
| 1,000시행 배열 모양·허용 코드·결측·C/A/S 관계 | 25개 모두 통과 |
| formatter regular 조건 안의 S/R 일정 | 15개에서 한 가지 일정으로 동일 |
| formatter irregular 조건 안의 S/R 일정 | 10개에서 한 가지 일정으로 동일 |
| 미대응 자료 | 기존 49행 미검증 상태 보존 |
| 독립 원본 검산 | 원본25개·일정2종·동결파일47개 확인, 통과 |
| 합성 자동 시험 | 46개 통과 |

S가 바뀌는 원 시행을 새 상태 첫 시행으로 보존했고 각 기록의 구간 시작·길이·S/R 해시는 [원 결과](../results/human_task_structure_v1/audit.json)에 있다. 실제 받은 결과의 변화나 사람이 선택을 바꾼 위치로 반전을 추정하지 않았다. 처음 800시행과 마지막 200시행의 간격 규칙 구획은 S 자체의 반전과 별개다.

## 실행 전 동결과 재현

- 소스 커밋: `b9b206399521fb5055be2cb1832c3085c13959f5`.
- 등록 UTC: `2026-10-15T04:01:03.996548+00:00`.
- GitHub 동일 등록 바이트 확인 UTC: `2026-10-15T04:01:23.899018+00:00`.
- 실제 검사 UTC: `2026-10-15T04:01:28.190037+00:00` → `2026-10-15T04:01:28.286630+00:00`.
- 독립 검산 UTC: `2026-10-15T04:01:34.015365+00:00`.
- 원 결과 SHA-256: `f4e777748518a1490bc6ec19b4c3c0caf6e560e6cfa1ca85f5e2e8c227b01547`.

[사전 계획](human-task-structure-protocol.md), [문헌·원 코드 근거](human-task-structure-sources.md), [등록](../results/registration_human_task_structure_v1.json), [원격 동결](../results/human_task_structure_prefreeze_evidence.json), [독립 검산](../results/human_task_structure_v1/verification.json), [시험 기록](../results/test-validation-human-task-structure.json)을 함께 보존했다. 이미 구조와 변환을 탐색한 후 고정한 후속 감사이며 눈가림 확증 연구가 아니다.

저장소 루트에서 고정 의존성을 설치한 Python으로 실행한다. 캐시는 기존 공식 OSF 원본 캐시이며 저장소 밖에 둔다. 기존 출력을 덮어쓰지 않는다.

```powershell
python scripts/audit_human_task_structure.py register --cache ../DrosoLearn-osf-origin-cache --out results/registration_TASK_NEW.json
python scripts/audit_human_task_structure.py run --cache ../DrosoLearn-osf-origin-cache --registration results/registration_TASK_NEW.json --out results/TASK_NEW.json
python scripts/verify_human_task_structure.py --result results/TASK_NEW.json --registration results/registration_TASK_NEW.json --cache ../DrosoLearn-osf-origin-cache --out results/TASK_VERIFICATION_NEW.json
python -m pytest tests/test_human_task_structure.py tests/test_human_task_structure_verify.py -q
```

캐시가 없다면 [앞선 원본 감사](human-osf-origin-protocol.md)의 `register --fetch` 절차로 고정 URL·버전·해시의 원본을 복구한다. 원 MATLAB 프로그램을 실행하지 않는다. 최초 결과는 완료 태그 `human-task-structure-v1`의 별도 checkout과 동일 입력에서 재현한다.

## 해석과 다음 단계

공개 과제 코드는 2019 버전이고 검증된 원본 명단은 2018 시각을 포함한다. 논문의 피드백 노출 시간과 공개 코드도 일치하지 않는다. 정확한 과거 실행판까지 확인됐다고 주장하지 않는다. 후속 허용 범위는 **코드 근거와 원본 구조가 함께 지지하는 source-coded 카드 선택의 기술 집계**다.

사람의 정답률·학습 곡선·반전 비용·집단 차이·사람과 모델의 유사성을 이번 검사에서 계산하지 않았다. 구조 통과는 Level 4/5 결과가 아니다. 다음 지표는 별도 코드·시험·프로토콜·입력 목록을 실행 전에 다시 고정한다. 기존 DrosoMem과 완료한 자료·코드·설정·등록·실패·결과는 변경하지 않았다.
