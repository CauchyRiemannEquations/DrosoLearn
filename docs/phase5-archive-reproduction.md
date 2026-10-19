# 최종 보존 감사 재실행

종합의 과학 실행·독립 산술 검산은 [원 재현 안내](phase5-reproduction.md)를 따른다. 이 추가 문서는 종합 후의 파일 보존 감사를 재실행하는 방법이다. 원 등록에 동결된 문서는 수정하지 않았다.

`phase5-synthesis-v1` checkout에서 환경을 준비하고, 원 인간 ZIP/PDF/API 스냅숏의 정확한 바이트를 외부 캐시에 둔다. 큰 원 NPZ를 조각에서 먼저 복원한다. 원 감사 파일을 덮어쓰지 않도록 새 출력 경로를 사용한다.

```powershell
.venv/Scripts/python.exe -m drosolearn.phase5_report materialize-parent
.venv/Scripts/python.exe scripts/verify_phase5_archive.py --cache C:/path/to/tnu-cache --out results/MY_PHASE5_ARCHIVE.json
```

이 코드는 원 `human-shell-comparison-v2`의 모든 Git blob을 읽어 세 문서 인덱스를 제외한 파일의 바이트와 대조한다. 추가 5개 등록의 소스·입력·부모·그래프·원자료·산출물 해시, 실행 전 원격 게시 증거와 시간 순서, 28개 완료 태그의 원 결과, 최종 시험 영수증·그림·문서 링크·주장 범위를 검사한다. 새 학습·난수·효과 검정은 없다. 원 결과 경로가 현재 source commit의 Git blob 대신 두 조각에 들어 있는 경우는 정확히 지정한 O 배열 한 파일뿐이며 원 SHA·길이가 같아야 한다.

첫 감사 helper는 기존 `shell_behavior_v1` 요약에도 `study_id`가 있다고 가정해 KeyError로 중단했다. 그 요약은 원래 이 필드가 없고, 등록·metadata·검산·원 요약 해시로 연구가 식별된다. 원 요약을 고치지 않았다. 감사 코드에서 이 한 알려진 누락만 허용하며 명시적 불일치와 다른 연구의 누락은 거부한다. [개발 기록](../results/phase5_archive_engineering_evidence.json)을 보존한다. 이것은 학습 결과의 실패나 원 독립 산술 검산 오류가 아니다.

최종 감사 자체의 소스는 `scripts/verify_phase5_archive.py`와 `scripts/check_prephase5_baseline.py`이며 최종 감사에 해시를 남긴다. 이들은 과학 실행이 끝난 후의 보존 점검 도구다. 현재 HEAD·완료 태그·원격 main이 같은지의 게시 후 확인은 감사가 미래 사실로 미리 선언하지 않고 별도 수행한다.
