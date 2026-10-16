# 조건부 관측 가능성 그림의 표현 수정 기록

작성일: 2026-10-16 (Asia/Seoul). 표현 수정 ID `conditional_transition_figure_correction_v1`. 원 측정 ID `conditional_transition_v1`.

## 무엇을 고쳤나

원 그림 코드는 사람 원 기록의 행 번호가 0부터 연속한다고 가정했다. 실제 등록된 25개 원 행 번호는 `0–14, 27–36`이다. 이 차이 때문에 원 그림은 입력 검사에서 멈췄고 PNG를 만들지 못했다. 실패를 [원 실패 기록](../results/conditional_transition_figure_failure_v1.json)으로 보존했다.

사람 기록 25개가 있다는 사실은 그 원 행 번호가 0–24라는 뜻이 아니다. 새 `plot_conditional_transition_v2.py`는 원 등록의 `input_files`에서 `role=human_raw`인 항목들의 **row_index와 원 입력 순서**를 그대로 읽는다. 번호가 0·27처럼 떨어져 있어도 허용한다. 원 입력의 누락·중복 행 번호·중복 파일 정체성이나 측정 결과의 누락·중복 경로는 거부한다. 그림의 행 표시와 별도 metadata도 실제 원 행 번호와 순서를 남긴다.

## 유지한 자료와 해석

이 수정은 그림의 ID 대응과 출판 절차를 고친 것이다. 이미 등록한 원 그림·프로토콜·설정·실행 코드·시험·측정 결과·독립 검산을 수정하지 않는다. 과학적 측정·검산을 재실행하지 않고 기존에 통과한 분자·분모·지원 분류만 표시한다. 모델 ID·층 정의·그림의 네 영역·색의 뜻·비교 범위는 바꾸지 않는다.

사람 8층·모델 4층과 설계층 안의 관측 0을 그대로 보존한다. 두 피드백의 유효 분모가 모두 1 이상이라는 표시는 정밀성·보상 인과 효과·학습 성과·사람/모델 유사성을 뜻하지 않는다. 추가 학습·학습자 구성·데이터 생성·난수 추출은 모두 0회이며 Level 4/5 판정을 추가하지 않는다. 자동 이어가기는 PAUSED를 유지한다.

## 원 등록과 별도 표현 등록의 연결

원 측정 등록의 소스 16개는 그 등록의 해시로 현재 보존 바이트를 검사한다. 기존 그림 코드의 SHA는 새 코드에 맞춰 바꾸지 않고, 원 부모 소스 바이트로 확인한다.

별도 표현 등록에는 새 그림 스크립트·이 시험·이 문서 **3개만** 동결 소스로 넣는다. 다음 입력은 완료된 원 측정 폴더의 `metadata.json`, `registration.json`, `metrics.json`, `verification.json`, `artifact_hashes.json` **5개**다. 이들의 정확한 바이트 해시와 원 그림 SHA, 새 소스의 Git 커밋·UTC 등록 시각·새 학습 자원 0을 함께 보존한다.

그림 생성은 독립 검산 통과·원 주 산출물의 해시·원 입력 목록의 canonical SHA·원 소스 16개·새 표현 소스 3개·별도 등록의 5개 입력 해시를 모두 확인한다. 표현 등록은 원 검산 이후이며 실제 그림 생성보다 앞서야 한다. 최초 v2 그림은 새 표현 소스와 등록을 원격에 고정하고 동일 바이트를 확인한 뒤 새 sibling 결과 폴더에 생성한다.

새 환경에서 측정을 재현하면 실행 UTC 등 때문에 5개 파일의 해시가 달라진다. 이 경우 과거 표현 등록을 재사용하지 않고, 아래 등록 모드로 **그 재현 결과에 대한 새 표현 등록**을 먼저 만든다. 등록 모드는 PNG를 만들지 않는다.

## 재현 명령

완료한 원 측정과 검산이 `results/CONDITIONAL_NEW`에 있고 새 소스 3개를 Git에 고정한 상태에서 저장소 루트의 Python 3.12 고정 환경으로 실행한다.

```powershell
python scripts/plot_conditional_transition_v2.py --register --result-dir results/CONDITIONAL_NEW --out results/registration_CONDITIONAL_FIGURE_NEW.json
python scripts/plot_conditional_transition_v2.py --result-dir results/CONDITIONAL_NEW --presentation-registration results/registration_CONDITIONAL_FIGURE_NEW.json --out results/CONDITIONAL_FIGURES_NEW
```

두 명령 사이에 최초 배포와 같은 원격 동결 증거를 별도로 보존한다. 원 측정 등록과 새 표현 등록은 역할이 다르며 어느 쪽도 소급 변경하지 않는다. 기존 경로는 덮어쓰지 않는다. 실제 최초 v2 결과 경로는 `results/conditional_transition_figures_v2`다.

## 수정판의 검증 범위

새 시험은 수작업 집계만 사용한다. 비연속 행 번호 0·27과 그 반대 입력 순서, 누락·중복 거부, 원 그림의 연속 번호 가정 재현, 새 표현 등록 생성·배타 출력·해시 결합을 확인한다. 수작업 자료에서만 PNG를 만들어 실제 행 번호 표시와 배치를 눈으로 확인한다. 실제 사람·모델 원 시행 자료를 개발용으로 다시 집계하거나 새로운 학습을 실행하지 않는다.

시험 전후 새 소스 3개의 SHA-256, 실행 명령·UTC·종료 상태·출력을 별도 작업 기록 `work/conditional-transition-figure-v2-tests.json`에 남긴다. 원 실패와 수정판의 성공을 구분하며 원 그림이 성공했다고 고쳐 보고하지 않는다.
