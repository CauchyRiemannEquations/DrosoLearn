# 공식 OSF 원본과 가공 CSV의 부분 재현 감사 계획

작성일: 2026-10-15 (Asia/Seoul). 연구 ID: `human_osf_origin_v1`. 기존 인간 코드 의미 감사와 별도로 진행하는 원본 변환 감사다. 인간 학습 성과나 사람/모델 유사성을 분석하지 않는다.

## 먼저 쉽게 설명하면

공개된 원본 파일과 다른 저장소의 가공 표가 같은 기록에서 나왔는지 확인한다. 원본을 보고 가공 프로그램의 정해진 계산을 다시 수행한 뒤, **값 하나와 빈칸 하나까지 같은지** 검사한다.

전체 가공 자료는 74행이지만 정상 공개 목록에서 이름이 정확히 대응하는 원본은 Main의 25행이다. 이 25행을 모두 검사한다. 나머지 49행은 원본 미대응 상태로 남긴다. 25행이 전부 맞아도 전체 74행이나 인간 학습 비교의 준비가 끝났다고 부르지 않는다.

## 범위와 선정 기준

고정 설정은 [human_osf_origin_manifest.json](../configs/human_osf_origin_manifest.json)이다. 공식 OSF 데이터 component `75jrf`의 공개 목록과 고정 pybefit 커밋 `813aeb5ad2c7bbed90ecd7d947bce9fe981c9702`의 formatter·가공 CSV를 대상으로 한다.

선정은 성과를 보기 전 **원 파일명의 정확한 일치와 공개 가용성**만으로 고정한다. formatter에 있는 전체 Main 50행과 Pilot 24행을 먼저 목록에 남기고, 완전한 공식 목록과 대조한다. 공식 `Main` 목록은 비어 있었고 `Pilot`이라는 이름의 폴더 목록은 154개 항목이었다. 폴더 이름을 참가자 집단이나 가공본 집단 코드로 해석하지 않는다.

| 가공 자료의 원래 행 구분 | 전체 행 | 고정 목록에서 원본이 유일하게 대응한 행 | 미대응 행 |
|---|---:|---:|---:|
| Main | 50 | 25 | 25 |
| Pilot | 24 | 0 | 24 |
| 합계 | 74 | 25 | 49 |

Pilot 24행의 파일명은 22개 고유 항목이며 기존 중복 두 쌍은 보존한다. 중복을 자동 삭제하거나 74행을 74명의 독립 참가자로 부르지 않는다. 파일명이 중복 대응하면 임의로 하나를 선택하지 않는다. 현재 고정된 유일 대응 25개는 전부 감사하며, 잘 맞는 원본만 나중에 고르지 않는다.

사전 탐색에서는 파일명·가용성·해시·원 변수와 모양을 확인했다. **등록 전 원본→가공 CSV의 전 값·NaN 비교는 실행하지 않았다.** 파일명 대응 조사와 값의 동일성 검증을 구분한다. 이미 자료 구조를 본 탐색적 감사이므로 눈가림 확증 사전등록으로 표시하지 않는다.

## 원자료·권한·개인정보 처리

원본 MAT, 원 formatter와 라이선스, 고정 가공 CSV는 저장소 밖 별도 캐시에 보존한다. 공식 목록·권한 JSON 5개는 재현을 위해 `data/human/osf_origin_evidence_20261015/`에 원바이트 스냅샷과 별도 manifest·README로 보존하고, 캐시에는 그 고정 바이트를 복사할 수 있다. 각 입력의 고정 URL·파일 ID·버전·크기·SHA-256은 manifest에 둔다. OSF가 알려 준 해시는 받은 바이트에서 직접 계산한 SHA-256과 구분하여 확인한다. 캐시 경로는 저장소 내부 또는 캐시 밖으로 빠지는 경로를 허용하지 않는다.

공식 데이터 component의 공개 상태 및 명시적 CC BY 4.0 연결을 원본 사용의 근거로 검사한다. formatter의 MIT 선언과 원 데이터의 권한을 구분한다. **OSF 원본의 허가를 GitHub 가공 CSV 전체의 재배포 허가로 자동 확장하지 않는다.** 개별 파일의 모든 예외를 검토했다고 주장하지 않으며, 이번 저장소에는 원본 응답과 가공 CSV를 새로 게시하지 않는다.

행별 대조 결과에는 원 파일명 대신 SHA-256 토큰, OSF 파일 ID, 가공본의 행 인덱스를 사용한다. 참가자의 실제 이름을 추정하거나 대응시키지 않는다. 보존한 공식 목록 스냅샷에는 원래 공개된 익명 코드 파일명과 시각이 포함되어 있다. 따라서 결과의 `participant_names_or_original_filenames_reported=False`는 행별 대조 보고서의 범위이며, 메타데이터 원문에도 파일명이 없다는 뜻이 아니다. 공개 식별자를 다른 개인정보에 연결하지 않는다.

## 원 formatter에서 고정한 변환

원 formatter를 실행하거나 import하지 않는다. Python AST를 정적으로 읽어 문자 그대로 선언된 파일명 목록과 연결 순서만 확인한다. 실제 변환은 새 검사 코드가 다음 규칙을 직접 구현한다. 결과에서 더 잘 맞는 다른 변환을 찾지 않는다.

각 원본은 1,000시행이며 필요한 변수와 모양은 `A: 1×1000`, `C: 1×1000`, `S: 1000×1`, `R: 1000×2`다. 원본 변수의 원 순서를 보존한다.

| 가공 열 | 고정 재현 규칙 | 해석상 주의 |
|---|---|---|
| `responses` | 원 `A`에서 1을 뺌. 원 1/2는 가공 0/1, 원 3은 가공 2, NaN은 유지 | 가공 2는 정보 요청 선택이며 이진 정답 선택으로 합치지 않음 |
| `outcomes`의 이진 선택 | 선택한 `R`의 `-1/+1`을 `(R+1)/2`로 변환 | 실제 결과이며 정확도나 예정 보상 확률이 아님 |
| `outcomes`의 정보 요청 | 원 상태 `S`에 1을 더해 2/3으로 기록 | 2/3은 상태 정보 코드이며 돈 보상의 크기가 아님 |
| `outcomes`의 무응답 | NaN 유지 | 보상 0이나 특정 선택으로 채우지 않음 |
| `correct_responses` | 원 `C`를 위치 그대로 복사 | 변환 동일성을 확인하며 사람의 정답률을 집계하지 않음 |

Main/Pilot의 formatter 파일명 목록 순서와 가공 CSV의 `subject` 행 번호를 고정 대응으로 사용한다. CSV의 열은 시행 `0`부터 `999`까지이며 `responses`에는 `condition`도 있다. 원 formatter의 집단 순서에 따른 `regular/irregular` 표시를 검사한다. 파일 위치나 행동 결과로 집단을 다시 정하지 않는다.

자료형·차원·범위를 벗어난 원본, 중복 키, 미지 코드, 잘못된 헤더 또는 불완전한 목록은 문제로 남긴다. 원본의 행·열을 이동하거나 일부 시행·참가자를 삭제해 가공본에 맞추지 않는다. 원본 값과 가공 값이 다르면 원인을 확정하지 않은 상태로 보고한다.

## 등록과 성공·실패 기준

등록은 원본 비교를 실행하기 전에 모든 캐시 입력의 바이트 크기·SHA-256, 전체 74행의 가용성 표, 유일 대응 25개, manifest, 코드·테스트·이 문서의 해시와 Git 버전을 기록한다. 동결 파일은 기존 코드·설정·테스트·문서·환경 명세 8개와 공식 스냅샷 5개·그 manifest·README 7개를 합한 **15개**다. 환경 명세는 `requirements-lock.txt`와 `requirements-human-candidate-audit.txt`다. 등록은 캐시의 무결성과 파일명 대응을 검사하지만 원본→CSV 전 값 비교는 수행하지 않는다. Python·NumPy·SciPy 버전과 실행 시작·완료 UTC를 별도로 기록한다.

고정 25개 원본에 대해 `25 × 1,000 × 3 = 75,000`개 값을 전부 대조한다. 정수 코드의 값은 정확히 같아야 하며 허용오차를 사용하지 않는다. NaN은 같은 위치의 NaN과만 같다고 처리하고 CSV 빈칸과 원본 NaN의 위치를 함께 검사한다. 유한한 0을 결측으로 바꾸지 않는다. `1`과 `1.0`의 직렬화 차이는 파싱한 수치값의 동일성과 원 파일 바이트 차이를 구분한다.

부분 원본 게이트의 통과에는 다음 조건이 모두 필요하다.

1. 고정 입력·코드·테스트·프로토콜·manifest 해시가 등록과 일치한다.
2. 전체 74행의 가용성 및 중복 기록이 동일하며, 가용한 25개를 모두 검사했다.
3. 각 원본의 3개 배열·1,000시행 전체에서 차원·값·NaN 위치의 불일치가 0개다.
4. 비교한 값은 총 75,000개이며 조건 표시도 고정 formatter 순서와 같다.
5. 독립 검증기가 원본에서 변환을 다시 계산해 동일한 판정을 확인한다.

어느 원본의 읽기·스키마·코드 검사에 실패하면 해당 행의 실패와 이유를 보존하고 전체 25개 게이트는 실패로 둔다. 비교하지 못한 값이 있는데 `불일치 0`만 보고 성공이라고 하지 않는다. 실행 예외도 새 JSON에 기록한다. 입력이나 변환을 고쳐 같은 등록으로 재실행하지 않는다.

**미대응 49행은 검증되지 않은 자료다.** 0불일치나 실패 성적이라고 집계하지 않는다. 25개가 통과해도 전체 자료 계보, 사람의 독립성, 원 연구의 모든 분석 재현, 가공본의 포괄적 재배포 권한 또는 인간 학습 비교 준비를 통과한 것은 아니다.

## 구현·독립 검증·재현 명령

주 도구는 `scripts/audit_human_osf_origin.py`, 독립 검증기는 `scripts/verify_human_osf_origin.py`다. 합성 자료 자동 테스트는 원본과 가공 값의 일치·단일 값 차이·NaN 차이·축/순서 오류·정보 요청·결측 처리·중복과 미대응·동결 및 덮어쓰기 방어를 검사한다. 실제 사람 자료를 테스트에 사용하지 않는다.

독립 검증기는 주 재구성·비교 함수를 불러오지 않고 캐시·공식 목록·고정 formatter와 가공 CSV에서 전 값을 다시 확인한다. 원본과 가공본을 한꺼번에 수정하고 해시만 갱신하는 손상이 통과하지 않도록 등록·manifest·코드 동결도 함께 확인한다.

재현에는 Python 3.12 기본 환경과 `requirements-human-candidate-audit.txt`의 고정 의존성을 사용한다. 저장소 루트에서 실행하며, 고정 외부 캐시를 재사용하거나 등록의 `--fetch`로 누락 입력을 복구한다. 원본 쪽 캐시는 33개 파일이며 원본 MAT 25개, formatter·라이선스 3개, 공식 목록 3개와 권한 JSON 2개로 구성된다. 가공 캐시는 CSV 6개다. 예시 캐시 이름을 자신의 실제 외부 캐시 경로로 바꾼다.

`register --fetch`는 manifest의 `bundled_path`가 있는 공식 JSON 5개를 저장소의 고정 스냅샷에서 로컬 복사한다. 원본 MAT·formatter·라이선스·가공 CSV는 고정 OSF revision 또는 Git 커밋의 정상 공개 HTTPS URL에서 복구한다. 받은 바이트의 크기·SHA-256을 검사하고, 기존 캐시가 다른 값이면 덮어쓰지 않고 중단한다. 원 formatter나 원 연구 코드를 실행하는 과정이 아니다. `--fetch`는 등록 모드에서만 허용하며 값 대조는 수행하지 않는다.

캐시를 복구할 때에는 다음 명령으로 고정한 정상 공개 URL과 해시를 조회할 수 있다. 원본 코드나 formatter를 실행하는 명령이 아니다. 각 URL에서 받은 바이트는 manifest의 `path`에 대응하는 외부 캐시 위치에 저장하고 크기·SHA-256을 검사한다. 기존 파일은 덮어쓰지 않는다. 접근 실패나 해시 변경이 있으면 같은 자료로 취급하지 않는다.

```powershell
.\.venv\Scripts\python.exe -c "import json; m=json.load(open('configs/human_osf_origin_manifest.json', encoding='utf-8')); [print(x['path'], x['url'], x['sha256']) for x in m['source_files']+m['derived_sources']]"
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-human-candidate-audit.txt
.\.venv\Scripts\python.exe scripts/audit_human_osf_origin.py register --fetch --cache ../DrosoLearn-osf-origin-cache --derived-cache ../DrosoLearn-human-cache --out results/registration_OSF_ORIGIN_NEW.json
.\.venv\Scripts\python.exe scripts/audit_human_osf_origin.py run --cache ../DrosoLearn-osf-origin-cache --derived-cache ../DrosoLearn-human-cache --registration results/registration_OSF_ORIGIN_NEW.json --out results/MY_OSF_ORIGIN_REPLAY.json
.\.venv\Scripts\python.exe scripts/verify_human_osf_origin.py --result results/MY_OSF_ORIGIN_REPLAY.json --registration results/registration_OSF_ORIGIN_NEW.json --cache ../DrosoLearn-osf-origin-cache --derived-cache ../DrosoLearn-human-cache --out results/MY_OSF_ORIGIN_VERIFICATION.json
```

`run`에는 `--registration`이 필요하다. `--out`은 새 JSON **파일** 경로이며 디렉터리가 아니다. 등록·결과·독립 검증은 각각 새 파일에 저장한다. 기존 출력이 있으면 덮어쓰기를 거부한다. 최초 실행의 source freeze 이후에는 완료 태그의 별도 checkout과 같은 고정 입력을 사용해 재현한다.

## 결과 보존과 다음 단계

결과에는 검사한 원본 25개와 대응 행, 전체 가용성 표, 49개 미검증 상태, 비교 값 수와 불일치 위치, 조건 표시·무결성·권한 관계, 등록·원격 동결 근거·실행 시각·환경·독립 검증을 남긴다. 성공 여부와 관계없이 보존한다. 캐시를 공개 저장소에 복사하거나 원본을 수정하지 않는다.

OpenNeuro의 선택·기대 코드 의미가 미해결이라는 기존 결과는 그대로 유지한다. 이 독립 부분 감사의 통과로 그 문제를 해결했다고 주장하지 않는다. 사람의 학습 곡선·보상률·정답률·반전 비용·RL 적합·사람/모델 비교는 실행하지 않는다. 필요한 의미·표본·과제 구조를 확인한 뒤 별도 행동 연구 프로토콜로 진행한다. 기존 DrosoMem과 완료된 원본·코드·설정·등록·실패·결과는 변경하지 않는다. 이번 결과는 Level 4 또는 Level 5의 증거가 아니다.
