# 인간 시행 자료의 공개 권한 재확인

검토일: 2026-10-15 (Asia/Seoul). 실제 공개 API 접근은 2026-10-15 01:08:49–01:11:32 UTC에 수행했다. 이전 조사에서 기록한 timeout은 당시의 관찰로 보존한다.

먼저 쉽게 말하면, 이번에는 자료가 있는지에 더해 **그 자료를 어떤 조건으로 이용할 수 있다고 원 출처가 표시하는지** 확인했다. 두 OSF 연구 공간에서 CC BY 4.0 표시를 직접 찾았다. 아직 사람의 성적이 모델과 비슷한지를 계산한 것은 아니다.

## 이번에 직접 확인한 표시

| 공식 출처 | 공개 여부와 라이선스 표시 | 확인 범위 |
|---|---|---|
| [Marković 프로젝트 h526v](https://api.osf.io/v2/nodes/h526v/) | `public: true`, CC BY 4.0 | 프로젝트 노드 메타데이터. copyright holder Dimitrije Markovic, 2019 |
| [그 하위 Behavioural data 75jrf](https://api.osf.io/v2/nodes/75jrf/) | 하위 데이터 component 자체가 `public: true`, CC BY 4.0 관계를 명시 | 저자 저장소 MIT에서 추론하지 않고 실제 데이터 component의 표시를 확인 |
| [Cheng 프로젝트 x2rau](https://api.osf.io/v2/nodes/x2rau/) | `public: true`, CC BY 4.0 | copyright year 2022. holder 문자열은 빈 값이므로 임의로 채우지 않음 |

공식 노드 세 개는 라이선스 ID `563c1cf88c5e4a3877f9e96a`를 직접 연결한다. [공식 라이선스 API](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/)는 `CC-By Attribution 4.0 International`과 [CC BY 4.0 legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)를 반환했다. 실제 원 응답, UTC 시각, HTTP 상태, 크기와 SHA-256은 [새 자료 manifest](../data/human/osf_permissions_20261015/source-manifest.json)에 보존했다.

관련 연구는 Marković, Reiter & Kiebel (2022), [Revealing human sensitivity to a latent temporal structure of changes](https://doi.org/10.3389/fnbeh.2022.962494), Cheng et al. (2023), [The database for extracting numerical and visual properties of numerosity processing in the Chinese population](https://doi.org/10.1038/s41597-023-01933-6)이다. 논문의 라이선스와 데이터 노드의 라이선스를 구분해 확인했다.

## 파일 목록이 알려주는 것

[Marković의 children API](https://api.osf.io/v2/nodes/h526v/children/)는 `Behavioural data`와 실험 코드 component를 별도로 반환한다. 데이터 component의 폴더 목록에는 `Main`, `Pilot`이 있다. 빈 최상위 OSFStorage 목록만 보고 전체 자료가 없다고 결론 내리면 안 된다. 이번 저장소에는 개별 참가자 파일명이 포함된 Pilot 목록을 넣지 않았다.

Cheng의 최상위 두 페이지에는 17개 항목이 있었으며 `Dataset1`부터 `Dataset5`까지 다섯 폴더를 확인했다. 각 폴더에는 `StimList_DatasetN.xlsx`, `pics`, `alltrial_DatasetN.mat`가 실제 목록에 있다. 각 XLSX/MAT 목록은 현재 버전 1, 크기와 서버가 선언한 SHA-256을 제공한다. 이는 공식 목록에서 파일의 존재와 버전 표시를 확인한 결과다. 파일 본문을 내려받거나 내부 열을 이번에 재검증한 결과는 아니다.

## 적용 범위와 남은 확인

명시 라이선스 관계를 확보한 범위는 위 OSF 프로젝트·데이터 component다. 개별 파일 내부의 별도 권한 예외는 읽지 않았다. OSF 원본과 GitHub `pybefit` 가공 CSV의 바이트·변환 대응은 확인하지 않았으므로, 기존 CSV에 CC BY 4.0을 자동으로 적용하지 않는다. 이전 GitHub 자료 권한 유보와 원본 cache 정책도 그대로 둔다.

이번 보존물은 공식 공개 메타데이터 원 응답 14개다. 참가자 응답 자료, 자극 이미지, 성과 곡선은 포함하지 않는다. API 응답 자체의 해시는 직접 계산했으며, 목록에 등장하는 자료 파일 해시는 서버의 선언이다. 두 종류의 해시를 같은 검증으로 설명하지 않는다. 자료 이용 시 원 저자·논문·자료 DOI·라이선스와 변형 여부를 함께 남기는 범위를 준비했다.

웹 도구에서는 공식 페이지가 403 또는 접근 불가로 보고됐고, Python 공개 요청에서는 페이지 HTML이 공통 OSF application shell만 반환했다. 일부 추가 API 요청은 HTTP 502였지만 정상 순차 재요청에서 위 메타데이터를 확보했다. 성공과 실패를 모두 새 manifest에 기록했다. 접근 실패를 비공개 또는 권한 없음의 증거로 해석하지 않는다.

## 재현

다음 명령은 저장한 공식 JSON의 해시, 공개 노드 ID, 명시 라이선스 관계, legalcode URL과 파일 목록 구조만 검증한다. 네트워크를 요청하거나 사람의 학습 성과를 분석하지 않는다.

```powershell
python scripts/check_human_source_permissions.py
```

새 검증 결과를 보존하려면 사용하지 않은 경로를 지정한다. 기존 출력은 덮어쓰지 않는다.

```powershell
python scripts/check_human_source_permissions.py --out results/human_trial_permissions_v1/metadata-validation.json
```

합성 메타데이터 테스트는 다음과 같다. 실제 참가자 자료는 테스트 fixture로 사용하지 않았다.

```powershell
python -m pytest tests/test_human_source_permissions.py -q
```

이 단계는 비교 자료의 출처·권한과 목록을 확인하는 작업이다. 사람과 모델의 행동 유사성 또는 공통 신경 기작을 검증하는 Level 4·5 판정은 수행하지 않았다.
