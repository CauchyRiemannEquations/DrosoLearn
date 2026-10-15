# 저자 코드와 기록 연결 근거 검토

고정 버전: OpenNeuro ds004295 v1.0.0, commit `c7c5d3a164425af3e65cc908ef0d41f01725984c`.

자료·코드 9개, 총 49,249바이트를 정상 공개 URL에서 확인했고, 그중 새로 보존한 MATLAB 5개는 총 43,141바이트다. 정확한 원 URL, 고정 Git URL, 포인터 해시, payload 크기·SHA256, UTC 접근 시각은 [새 원본 manifest](../data/human/stolz_2022_linkage_evidence/source-manifest.json)에 있다. MATLAB 5개는 Git-annex 포인터이며, 공개 S3 payload의 크기·해시가 고정 포인터의 SHA256E 키와 일치했다. 아래 줄 번호는 **실제 payload의 LF 기준 줄 번호**다. GitHub blob은 포인터만 보여주므로 그 페이지에 실제 코드 줄 번호를 붙이면 잘못된 링크가 된다.

## 새 근거 폴더에 넣을 원 코드 5개

1. `code/Q_learning.m` — 4,879바이트, `c34cd9ae66e933100579ee7d3a9b234a444963533d7e7467d3801fce6e984451`
2. `code/RevEx1_modelfit.m` — 7,581바이트, `1f0db0b2a5320c9dbf80237e02788b90edb7004dac1aa2a4df7cc59340ef17e2`
3. `code/modeling_calcRSQ_AIC_BIC.m` — 3,591바이트, `6a997a3d414accc93a46751b40cf587bf69011642584d7d69ddb9a721a2afcb7`
4. `code/modeling_calculatePR.m` — 3,603바이트, `3b2c7df7d7ff67aec8bab2badceab5b8e34f7098d499cd447105d66e216973cf`
5. `code/singleTrial_RegressionAnalyses.m` — 23,487바이트, `8a7ff74ad8c9dfba1717accf5892a52f37282157c77a9dac70ced83474a349a4`

나머지 확인 파일은 원 `README`, `dataset_description.json`, `participants.json`, `task-task_events.json`이다. `participants.json`은 열 설명뿐이며 실제 인구통계 행이 아니고 이번 새 보존 폴더에는 포함하지 않았다. 나머지 세 파일은 기존 고정 원본을 참조한다. `dataset_description.json`의 CC0 및 기존 보존 README/사건 코드북을 권한·해석 근거로 참조한다. `participants.tsv`, 추가 MAT, EEG, 자극 파일은 이번 코드 조사에서 내려받거나 분석하지 않았다.

## 명시된 것과 없는 것

| 질문 | 원 코드 근거 | 판정 |
|---|---|---|
| 행동 MAT cell과 BIDS ID 대응 | `RevEx1_modelfit.m` 34행에서 완성된 MAT를 읽고 78행의 숫자 subj로 순회. `modeling_calculatePR.m` 31,55행도 같은 구조 | 명시 ID 목록·대응표·정렬 규칙 없음 |
| 제외 목록 | `singleTrial_RegressionAnalyses.m` 55–56행에서 행동/BadEEGTrials MAT를 읽고 83–84행에서 EEG 배열의 `kickers`를 삭제. 298–299,353–354행에도 반복 | `kickers`의 값·대입·BIDS ID 정의가 공개 코드에 없음. 추가 MAT는 읽지 않음 |
| 행동 원자료 추출 | 5개 MATLAB 모두 완성된 분석 입력을 읽어서 모형 적합·예측오차·EEG 회귀에 사용 | 원 응답 로그/사건→행동 MAT의 변환 코드 없음 |
| 피드백 척도 | `Q_learning.m` 31–38행은 context1에서 0/1, context2에서 r−1로 -1/0을 사용 | 두 맥락의 유리한 피드백을 동일 금전 보상으로 부르지 않음 |
| 기대 코드 | `singleTrial_RegressionAnalyses.m` 316–326행은 MAT의 3/4/5/6을 0,1/3,2/3,1로 변환 | MAT의 3/4/5/6은 BIDS 사건의 기대 0/1/2/3과 다른 표현. 새 작업에서는 ordinal 대응만 확인하고 주관적 확률 검증으로 부르지 않음 |
| action 1/2의 실제 뜻 | `Q_learning.m` 42–45행은 a를 행동값 배열의 인덱스로 사용. 원 논문 Methods2.6.1은 모형 선택지를 shape1/shape2로 정의 | MAT가 도형 선택 모형 입력이라는 근거는 있으나 숫자1/2→특정 도형 및 매 시행 화면 좌우 변환 규칙은 공개 코드에 없음 |

원 논문 [Methods 2.5–2.6](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14235)은 도형 좌우를 시행마다 무작위로 제시하고, 모형은 도형별 행동값을 갱신한다고 설명한다. 따라서 BIDS의 좌우를 그대로 MAT action과 같다고 가정할 수 없다. 논문은 첫 과제/둘째 과제별 예정 반전 및 별도 연습 4회도 설명하지만, 본 코드에는 원 시행 번호와 도형–좌우 대응을 복원할 과제 실행 코드가 없다. 예정 반전표를 관측 선택 번호에 바로 붙이면 안 된다.

## 재사용하면 안 되는 결측 처리

`RevEx1_modelfit.m` 83–88행은 too-slow action을 0으로 설명하며 action에서 0을 먼저 삭제한 뒤, 삭제된 action의 `==0` 마스크로 feedback을 삭제한다. 원래 0이 있었다면 두 번째 마스크는 더 이상 원 결측 위치를 가리키지 않는다. 이 코드는 대응 행을 동시에 삭제하는 올바른 원자료 추출 근거로 사용하면 안 된다. 현재 보존 MAT에서 이 문제가 실제 발생했다고 주장하지 않는다. 이번에는 실제 MAT/TSV를 새로 읽거나 매칭하지 않았다.

## 다음 파이프라인에 주는 제한

REW/PUN의 피드백 두 전체 열을 원 순서 그대로 모두 만족하는 유일 대응을 찾는 검사는 파일 간 기록 연결 후보를 평가할 수 있다. 대응이 생겨도 공식 저자 ID 표의 발견, 자극/좌우 변환의 검증, 원 시행 누락 해결, 정답 상태 복원은 별도 문제다. 일부 일치·가장 가까운 순열·사후 제외·임의 결측 삭제·유리한 열 선택으로 강제로 연결하지 않는다. 일치가 없는 경우와 여러 후보가 있는 경우를 보존하고, 동일 참가자의 독립 측정이나 확률적 독립 비트라는 가정을 추가하지 않는다.
