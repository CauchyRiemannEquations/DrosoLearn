# 인간 시행별 공개 자료 — 출처와 선정 기록

검토일: 2026-10-15 (Asia/Seoul). 시행별 학습 자료의 가용성을 확인하는 제한된 탐색이다. 체계적 문헌고찰 전체나 인간·모델 유사성 검정은 아니다. 기존 Honoré & Noël의 사전·사후 요약 자료와 모든 완료 연구를 수정하지 않는다.

## 먼저 쉽게 설명하면

사람의 ‘연습 전후 성적표’에서 한 걸음 더 들어가 ‘매번 무엇을 골랐고 어떤 피드백을 받았는지’의 기록을 찾았다. 파일을 찾았다고 바로 모델과 비교할 수 있는 것은 아니다. 시행 순서가 있어도 피드백이 없거나, 같은 사람이 중복으로 들어 있거나, 화면 왼쪽 선택을 어떤 도형 선택으로 바꿀 단서가 없을 수 있다.

## 실제 파일을 확인한 세 후보

| 자료 | 직접 확인한 단위 | 활용 후보와 제약 |
|---|---|---|
| Stolz / OpenNeuro ds004295 v1.0.0 | 26개 참가자 레이블의 사건 TSV, 별도 행동 MAT 23개 행 | 선택·기대·피드백 순서 점검 후보. 수량 과제가 아니며 TSV의 도형 ID와 실제 시행 번호가 없음. MAT 행과 참가자 ID의 대응을 가정하지 않음 |
| Cochrane et al. / TEfits `anstrain` | 4개 ID × 250시행, 6열의 예제 발췌 자료 | 수량 과제 시행 형식 후보. 전체 연구 자료가 아니고 피드백·반전·세션·정확한 두 수량이 없음. `anstrain_s1`는 포함된 s1의 완전 중복 |
| Marković et al. / pybefit temporal reversal | main 50행 + pilot 24행, 행마다 1,000시행 | 선택·피드백·정오·잠재 상태 구조를 검증할 후보. pilot의 중복 두 쌍, 정보 요청 제3선택, 별도 데이터 재배포 권한 미확인 때문에 독립 참가자 수와 분석 표본을 확정하지 않음 |

### 1. CC0 원본을 보존한 주 자료

Stolz, Pickering & Mueller (2022)의 [OpenNeuro 공식 버전](https://openneuro.org/datasets/ds004295/versions/1.0.0/download), DOI [10.18112/openneuro.ds004295.v1.0.0](https://doi.org/10.18112/openneuro.ds004295.v1.0.0)을 실제 확인했다. Git 태그 `1.0.0`의 commit은 `c7c5d3a164425af3e65cc908ef0d41f01725984c`다. 라이선스는 공식 페이지와 원본 `dataset_description.json`의 CC0 선언을 근거로 한다. 26개 사건 파일·메타데이터 3개·행동 MAT 2개, 총31파일/1,934,671바이트만 바이트 그대로 보존했다. 출처·버전·해시는 [manifest](../data/human/stolz_2022_openneuro/source-manifest.json), 권한과 범위는 [저자 표시](../data/human/stolz_2022_openneuro/ATTRIBUTION.md)에 있다.

TSV의 열은 `onset`, `duration`, `sample`, `value`다. 한 행은 선택·기대 평정·피드백·과제 경계 등의 **사건**이며 한 시행이 아니다. [고정 원본 코드북](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004295/c7c5d3a164425af3e65cc908ef0d41f01725984c/task-task_events.json)으로 각 사건을 해석한다. `response_time`은 코드북에 가능한 항목으로 설명돼 있지만 실제 TSV 열에는 없다. 선택부터 피드백까지의 간격을 반응 시간으로 부르지 않는다.

[원 논문 Methods 2.1·2.5](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14235)에 모집26명·분석23명, 두 과제 각280시행, 과제 시작의 별도 연습4회가 기재돼 있다. 도형 좌우 배치는 매 시행 무작위다. 피드백 확률은70/30과30/70이며 첫 과제의 반전은82·150·225시행 뒤, 두 번째는86·160·223시행 뒤다. 이는 문헌의 예정 시점이며, 원파일에 시행별 경계·유리한 도형·도형 위치가 직접 기록됐다는 뜻은 아니다. 연습·누락·원 시행번호와의 정렬을 별도로 검증해야 한다. 원 논문의 신경 결과는 이번 행동 자료 감사에서 재분석하지 않았다.

REW 양성 피드백은10센트, 음성은0센트다. PUN 양성은소음 회피, 음성은소음 발생이다. 따라서 양성 피드백을 모든 맥락의 돈 보상이나 정답으로 통일하지 않는다. 0–3의 기대 평정도 모델의 실제 선택 확률과 같다고 가정하지 않는다. 별도 MAT에는 조건별 선택·피드백·시간 등이23개 cell×280값으로 있으나 참가자 ID 변수가 없어 TSV26개 레이블과 임의로 연결하지 않는다. 두 MAT의 공통 변수는 같은 자료이며 독립 표본으로 합치지 않는다.

### 2. 수량 과제의 실제 예제 시행 자료

저자 저장소 [TEfits](https://github.com/akcochrane/TEfits)의 commit `0ef49fcdb7037f98028e6264608602ddd0ca24a6`에서 `data/anstrain.RData`와 `data/anstrain_s1.RData`를 실제 확보하고 정적 RData 파서로 읽었다. 출처 논문은 [Cochrane, Cui, Hubbard & Green (2019)](https://doi.org/10.3758/s13414-018-01636-w)이며 [데이터 문서](https://github.com/akcochrane/TEfits/blob/0ef49fcdb7037f98028e6264608602ddd0ca24a6/man/anstrain.Rd)는 일부 발췌임을 명시한다. upstream 문서의 DOI에는 끝의 `-w`가 빠져 있어 원파일을 고치지 않고 정식 논문 DOI와 구분했다.

`anstrain`은1,000행×6열(`ratio`, `resp`, `trialNum`, `sizeRat`, `acc`, `subID`)이다. 각 ID의 `trialNum`은1–250이며 결측은 없다. `acc`와 `resp==(ratio>0)`의 일치는 정오 코드의 정합성 검사다. 이 값을 평균해 성과를 비교하거나 학습 곡선을 적합하지 않았다. `anstrain_s1`의250행은 이미 들어 있는 s1과 정확히 같으므로 합쳐1,250 독립 시행으로 세지 않는다. signed `ratio`를 실제 두 수량의 비율이나 Weber 지표로 자동 변환하지 않는다.

패키지 `LICENSE`·`DESCRIPTION`은 MIT이며 별도의 원 연구 전체 자료 라이선스는 확인하지 못했다. 이를 CC 라이선스의 전체 연구 공개 자료라고 표시하지 않는다. 이번 저장소에는 원 RData를 재배포하지 않고 고정 URL·해시와 구조 감사, 외부 캐시를 읽는 재현 도구를 보존한다. 원 연구 전체의 참가자 표본·세션·제외 규칙을 이4개 예제 ID로 대신하지 않는다.

### 3. 반전 구조는 풍부하지만 중복을 확인한 후보

[Marković, Reiter & Kiebel (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9621332/), DOI `10.3389/fnbeh.2022.962494`의 코드 저장소 [pybefit](https://github.com/dimarkov/pybefit), commit `813aeb5ad2c7bbed90ecd7d947bce9fe981c9702`를 확인했다. `examples/temporal_rev_learn/behavioral_data/{main,pilot}`의 세 CSV와 `states_and_rewards.mat`를 실제 읽었다. 본문이 연결하는 OSF의 라이선스 정보는 이번 정상 공개 접근에서 timeout/접근 실패로 확인하지 못했다. 저장소의 MIT 소프트웨어 선언과 원 데이터의 별도 권한은 구분하며 원 CSV/MAT는 이 저장소에 재배포하지 않는다.

CSV의 `subject`는 익명 행 번호다. `responses`의0/1은 카드 선택, main의2는 정보 요청이며 빈칸은 무응답이다. `correct_responses.csv`는 이름과 달리 정답 카드 번호가 아니라 정오 값이다. MAT의 `S`가 상태, `R`가 각 선택의 예정된 결과다. 시행별 정오·힌트·결과 코드는 원 상태와 대조했으며 성과 평균은 계산하지 않았다.

pilot의 행8/9와11/12는 세 CSV에서 각각 완전히 같다. [고정 변환 코드](https://github.com/dimarkov/pybefit/blob/813aeb5ad2c7bbed90ecd7d947bce9fe981c9702/examples/temporal_rev_learn/behavioral_data/pilot/format_behavioral_data.py#L23)에도 같은 원 MAT 파일 두 쌍이 중복 기재돼 있다. 원행은 삭제하지 않았으며74개 행을74명의 독립 사람이라고 단정하지 않는다. 자동 중복 제거로 표본을 확정하는 일도 이번에는 하지 않는다. main의 제3선택을 실패나 무응답으로 바꿔 이진 모델에 넣지 않는다.

## 실제 확보에 도달하지 못한 후보도 보존

[Hofmann, Kinder & Pekár (2024)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1287429/full)는 수량 학습 자료 후보이나 확인한 Data availability는 저자 제공 요청 방식이다. 이번에는 저자에게 연락하거나 새로운 인간 실험을 하지 않았다. [Cheng et al. (2023)](https://www.nature.com/articles/s41597-023-01933-6)은 수량·시각 단서 지각검사의 시행 자료와 [OSF](https://osf.io/x2rau/)를 명시하나 파일 API가 반복 timeout해 실제 파일을 검증하지 못했다. 논문상 `STIMID`는 자극 ID이며 시행 순서로 해석할 수 없다. 확보 실패는 자료가 세상에 존재하지 않는다는 결론이 아니다.

## 다음 비교에 남은 조건

수량 예제 자료는 순서가 있지만 피드백·반전 조작이 없다. 반전 사건 자료는 선택·피드백이 있지만 수량 과제가 아니고 도형·원 시행번호 대응이 부족하다. 보다 풍부한 반전 후보는 중복·권한·추가 선택을 먼저 해결해야 한다. 그러므로 현재 자료로 인간과 모델의 반전 성공률이나 학습 속도가 같다고 검정하지 않는다. 실제 비교를 시작할 때에는 관측 선택과 모델 확률의 차이, 표본 단위, 학습/평가 구분, 공통 지표와 반증 조건을 별도 프로토콜로 먼저 고정한다.
