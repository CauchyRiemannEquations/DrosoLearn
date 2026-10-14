# DrosoLearn

**Exploring the Foundations of Mathematical Learning Through a Drosophila Connectome**

초파리 커넥톰 제약을 가진 계산 모델에서 수량·기초 규칙을 배우는 과정, 일반화, 보상 수정과 학습 간섭을 조사한다. 인간의 기초 수량 인지·수학 학습에 대해 검증 가능한 가설을 만드는 것이 장기 목표다. 커넥톰의 우수성이나 인간과 동일한 신경 메커니즘을 전제하지 않는다.

쉽게 말하면, 초파리 뇌의 **연결 지도**를 참고해 작은 계산 모델을 만들고, 경험이 선택을 어떻게 바꾸는지 관찰하는 연구다. 지금은 “2와4 중 어떤 쪽을 선택하면 보상을 받는가”를 배우게 한다. 물체 그림에서 개수를 알아내는 단계나 사람의 수학 이해를 재현한 단계는 아직 아니다.

| 용어 | 쉬운 풀이 |
|---|---|
| 커넥톰 | 어떤 뉴런이 어떤 뉴런과 연결돼 있는지 나타낸 지도 |
| 가중치 | 연결을 통해 신호가 전달되는 계산상의 강도. 실제 시냅스 효능과 동일하지 않음 |
| 내부 학습 | 모델 안쪽 연결의 계산 가중치를 바꾸는 것 |
| 출력 학습 | 마지막 선택을 정하는 가중치를 바꾸는 것 |
| 보상 반전 | 처음에는 큰 쪽이 정답이었다가 작은 쪽이 정답으로 바뀌는 실험 |
| 시드 | 같은 알고리즘을 다른 무작위 초기 조건으로 시작하는 번호. 초파리 개체 수가 아님 |
| 사전 기준 | 결과를 보기 전에 정한 성공·실패 판정 규칙 |

[DrosoMem](https://github.com/CauchyRiemannEquations/DrosoMem)은 완료된 독립 연구다. 원본 저장소·실험을 변경하지 않고, 특정 커밋의 소규모 데이터를 출처와 함께 재사용한다. 기존 M11 실패 및 이후 제한적 내부 가소성 연구의 부정적 기능 결과를 연구 출발점에 포함했다.

## 현재 구현

- 실제 FlyWire FAFB v783 부분집합의 KC→MBON 1,474쌍을 이용한 one-hop projection. 기존 recurrent reservoir의 복제는 아니다.
- **추상적 수량 보상 연합**: `n/8` 입력이 수량·순서를 이미 제공한다. 시각 수량 추출·LC11 회로·수학 기호 이해는 검증하지 않는다.
- 2대4 습득→동일 모델 보상 반전. 인공 nuisance OOD와 미학습 수량쌍 probe는 분리 평가한다.
- 원래/재배선 × 고정/가소성, 일반 random sparse ANN, frozen·random reward·internal-only: 9조건.
- 전향적 로컬 동결, 다중 seed, 원자료·가중치·활동 checkpoint, 자동 그림 및 별도 검증.

초기 결과와 테스트 상태는 [초기 실행 보고서](results/smoke_v1/report.md), [독립 검증](results/smoke_v1/verification.json), [실패를 포함한 해석](docs/initial-findings.md)에서 확인한다. 3시드 smoke에서 original fixed/plastic의 습득은 각각2/3·3/3시드가 기준을 통과했지만 **보상 반전은 모든 조건에서 실패**했다. 일반 ANN에서도 습득이 나타나 원래 배선 우위를 지지하지 않았다. 이 실행은 확증·생물학적 검증이 아니다.

## 설치와 실행

최초 smoke는 `v0.1.0-smoke`에 보존했다. 아래 최초 실험 명령은 그 태그의 **별도 checkout**에서 사용한다. 후속 모듈을 추가한 현재 main에서는 옛 등록의 source file set이 달라져 실행이 거부되는 것이 정상이다. 옛 등록 파일을 수정하지 않는다.

```powershell
git clone --branch v0.1.0-smoke https://github.com/CauchyRiemannEquations/DrosoLearn.git DrosoLearn-smoke
Set-Location DrosoLearn-smoke
```

Python **3.12**에서 저장소 root를 작업 디렉터리로 사용한다. Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q

# 기존 등록의 source/config 해시가 동일하면 기존 동결로 새 디렉터리에 재현
.\.venv\Scripts\python.exe -m drosolearn.cli run --config configs/smoke.json --registration results/registration_smoke_v1.json --out results/my_smoke_replay
.\.venv\Scripts\python.exe -m drosolearn.cli verify --out results/my_smoke_replay
```

Linux/macOS는 `python3.12 -m venv .venv`와 `.venv/bin/python`으로 실행한다. BLAS/platform에 따른 부동소수점 차이가 있을 수 있으며 버전·platform과 tolerance를 보존한다. 각 run은 약 21,600개의 학습 시행과 27 seed-condition을 포함한다. 결과 경로가 존재하면 덮어쓰기를 거부한다.

코드·데이터·프로토콜·설정을 변경할 때는 이유를 amendment로 남기고 **새 등록**을 먼저 만든다:

```powershell
.\.venv\Scripts\python.exe -m drosolearn.cli register --config configs/smoke.json --out results/registration_NEW.json
.\.venv\Scripts\python.exe -m drosolearn.cli run --config configs/smoke.json --registration results/registration_NEW.json --out results/NEW
.\.venv\Scripts\python.exe -m drosolearn.cli verify --out results/NEW
```

`configs/confirmatory.json`은 별도 10시드 계획이며 초기 결과를 본 후 수정한 설계의 무조건적 확증으로 사용하지 않는다. 최초 실행 전에 새로운 source/protocol 동결과 수정 내역을 확인한다.

## 후속: 이전 학습과 보상 반전의 비용

쉽게 말하면, 이번에는 **처음부터 작은 쪽을 배우는 모델**과 **큰 쪽을 배운 뒤 작은 쪽으로 바꾸는 모델**을 비교한다. 마지막 선택 담당 가중치만 초기화하는 대조도 두어, 이전 학습이 어디에 영향을 남겼는지 계산적으로 살핀다. 초기화는 컴퓨터 안에서 하는 조작이며 실제 뇌의 기억 삭제가 아니다.

[새 프로토콜](docs/reversal-protocol.md)은 별도10시드·6모델·5학습 이력, 기존과 같은 단계당400시행과 학습률을 사용한다. **옛 실패를 더 오래 훈련해 성공으로 바꾸는 실험이 아니다.** 결과는 [후속 보고서](results/reversal_v1/report.md)에 보존한다. 이 후속 명령은 현재 main의 새 소스 동결 버전에서 실행한다:

실행 결과의 쉬운 해석은 [reversal-findings.md](docs/reversal-findings.md)에 있다. 처음부터 새 규칙을 배우는 것은 대부분 성립했지만 반전 기준은 전 조건에서 미달했다. **반전의 낮은 출발점과 학습 속도 자체를 구분**해야 하며, 과정 성적의 비용을 곧바로 느린 학습률이라고 해석하지 않는다. 내부도 학습한 모델에서 선택 부분 초기화 이후 수치상 이득이 나타나, 내부 변화가 무조건 방해한다는 해석도 지지되지 않았다.

후속 실행 버전은 `study-reversal-v1` 태그로 보존한다. 이후 main에 새 코드가 추가되면 그 태그를 별도 checkout하고 Python3.12 환경을 설치해 재현한다. 아래는 Windows PowerShell에서 해당 checkout의 `.venv`를 사용하는 명령이다.

```powershell
.\.venv\Scripts\python.exe -m drosolearn.reversal run --config configs/reversal_diagnostic.json --registration results/registration_reversal_v1.json --out results/MY_REVERSAL_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.reversal verify --out results/MY_REVERSAL_REPLAY
```

코드·설정·새 프로토콜을 변경했다면 `python -m drosolearn.reversal register --config configs/reversal_diagnostic.json --out results/registration_NEW.json`으로 실행 전에 새 기록을 만든다. 경로가 존재하면 덮어쓰기를 거부한다.

인간 공개 자료의 실제 확보·원본 성적표 구조 검증은 [human-data-readiness.md](docs/human-data-readiness.md)에 있다. 아이56명의 연습 전후 점수 자료를 확보했으나, 연습 한 번씩의 기록이나 보상 반전 기록은 없어 현재 반전 곡선과 직접 비교할 수 없다.

## 후속: 입력이 미리 알려 준 수량 순서

이번에는 연습에 쓰는2와4의 입력을 그대로 두고, **처음 보는 수량의 입력 코드만 바꿔** 같은 모델을 평가한다. 숫자가 큰 코드에 더 많은 물체 수를 대응시키는 기존 입력과, 고유한 임의 이름표를 쓰는 입력을 비교한다. [사전 계획](docs/encoding-protocol.md), [설계의 수학적 성질](docs/encoding-background.md), [실행 보고서](results/encoding_v1/report.md)를 보존한다.

실행 결과의 [쉬운 설명](docs/encoding-findings.md)도 보존했다. 실제 출력 전용 모델의 새 수량 성적은50.1%→86.1%, 내부 학습 모델은50.1%→94.0%였고 재배선·일반 신경망도 비슷했다. 이는 순서가 제공된 인공 입력에서 보상 경험으로 반응이 변한 결과이며 수량 의미를 스스로 발견한 증거가 아니다. 임의 이름표의 전수 평균50%는 미리 증명된 성질이다. 이름표 조작은 수량쌍의 입력 간격도 바꾸므로 순서만 독립 조절했다고 주장하지 않는다.

720개 이름표 배치를 모두 평가하지만 새로720번 학습하지 않는다. 실제로는 수량 정보가 있는 모델과 수량 정보를 뺀 모델 두 개씩만 학습한다. 두 수량 모두 연습하지 않은15쌍의 임의 이름표 평균50%는 **실행 전에 증명된 대칭**이며 새로운 생물학적 실패가 아니다. 일부 수량을 연습에서 본12쌍은 별도로 보고한다. 순서 코드를 받은 모델이 높은 성적을 내더라도, 수량의 순서를 스스로 발견했다는 주장으로 바꾸지 않는다.

재현 버전은 `study-encoding-v1` 태그의 별도 checkout을 사용한다. Python3.12 `.venv` 설치 후 새 경로로 실행한다:

```powershell
.\.venv\Scripts\python.exe -m drosolearn.encoding run --config configs/encoding_diagnostic.json --registration results/registration_encoding_v1.json --out results/MY_ENCODING_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.encoding verify --out results/MY_ENCODING_REPLAY
```

소스·설정·프로토콜을 변경할 때에는 `python -m drosolearn.encoding register --config configs/encoding_diagnostic.json --out results/registration_NEW.json`으로 새 실행 전 기록을 만든다. 이전 등록·결과를 최신 코드에 맞춰 수정하지 않는다.

## 후속: 순서를 유지한 입력 간격

이번에는 수량의 순서를 유지하면서 **같은5대7 문제의 입력값 사이 거리만** 바꾼다. 두 입력의 중간값과 연습한2·4, 전체 범위는 고정한다. 한 번 학습한 모델을 기존 간격의1/4·1/2·1·1.5배에서 검사한다. 정답 선택 확률과, 확률이 큰 쪽만 고르는 성적을 따로 보고한다. [사전 계획](docs/spacing-protocol.md), [수학적 확인](docs/spacing-background.md), [실행 보고서](results/spacing_v1/report.md)를 보존한다.

실행 결과의 [쉬운 설명](docs/spacing-findings.md)에서 같은5대7의 확률과 선택을 구분했다. 실제 출력 전용 모델은 기존 간격81.7%→1/4간격59.3%, 내부 학습 모델은94.2%→66.8%로 확률이 낮아졌다. 확률이 높은 쪽만 고르면 모든 조건·시드·간격에서 주 쌍 정답률100%였다. 그래도 정답 확률75%라는 사전 전이 기준의1/4간격 통과 수0/10은 그대로 보존했다. 재배선·일반 신경망도 비슷해 원래 배선의 고유성을 입증하지 않는다.

이 진단은 입력 표현과 선택 함수가 성적에 주는 영향을 살핀다. 단순 선형 점수와sigmoid만으로도 확률 변화와 선택 방향 보존이 나타날 수 있으므로 새 수량 원리·인간의Weber법칙·생물학적 기작의 발견으로 해석하지 않는다. 실제 학습 예산은6모델×10새시드×400시행=24,000이며 네 코드는 추가 학습이 아니다.

재현 버전은 `study-spacing-v1` 태그의 별도 checkout을 사용한다. Python3.12 `.venv` 설치 뒤 새 경로에 실행한다:

```powershell
.\.venv\Scripts\python.exe -m drosolearn.spacing run --config configs/spacing_diagnostic.json --registration results/registration_spacing_v1.json --out results/MY_SPACING_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.spacing verify --out results/MY_SPACING_REPLAY
```

소스·자료·프로토콜을 바꾸면 같은 연구를 덮어쓰지 않는다. 새 계획을 먼저 작성하고 `python -m drosolearn.spacing register --config configs/spacing_diagnostic.json --out results/registration_NEW.json`으로 별도 동결한다. 현재 연구 설정은 사전 시드·예산·학습률·판정 기준을 변경하면 등록을 거부한다. 새로운 과학적 설정은 프로토콜과 검증기에서도 명시적으로 구분한다.

## 후속: 모델 반응을 단순한 계산으로 설명하기

연습한2·4의 별도 입력에서 원래 모델의 후보 점수만 읽어, **7입력의 선형 합산식**과 **수량만 보는 직선**을 맞춘다. 새 수량의 점수·정답으로 다시 맞추지 않고 네 입력 간격·완전 미학습15쌍의 시행별 예측 오차를 검사한다. [사전 계획](docs/linearity-protocol.md), [수학적 의미](docs/linearity-background.md), [실행 보고서](results/linearity_v1/report.md)를 보존한다.

이 참조는 모델 점수를 제공받은 요약식이며 독립 보상 학습자나 동등한 정보·자원의 대조가 아니다. 낮은 확률 오차가 정확한 직선 점수 또는 같은 신경 기작을 뜻하지 않으므로 원시 점수 차이와 선택 불일치도 함께 남긴다. 주 기준은 학습 후 affine 확률MAE≤2%포인트의 전체시드 평균과개별통과≥8/10이며, 습득 실패를 주 평균에서 제외하지 않는다. 새5001–5010시드·6모델·400시행, 실제 보상 학습24,000번을 사용한다.

실행과 독립 검산을 완료했고 [쉬운 설명](docs/linearity-findings.md)을 보존했다. 실제 출력 전용·내부 학습 모델의 확률 예측 오차는 각각0.952·0.386%포인트였으며 주 기준6/6조건, 개별오차각10/10통과였다. 연습 습득 미달3개도 포함했다. 학습 전부터 참조 오차가 작았고 내부 학습 모델의 원시 점수 차이 오차는 더 컸으므로 선형 규칙 발견이나 더 선형적인 내부 기작을 주장하지 않는다. 자동 테스트525개와 모든 학습·평가·참조 계산의 독립 검산이 통과했다.

재현은 `study-linearity-v1` 태그의 별도 checkout과 Python3.12 환경에서 새 결과 경로를 사용한다:

```powershell
.\.venv\Scripts\python.exe -m drosolearn.linearity run --config configs/linearity_diagnostic.json --registration results/registration_linearity_v1.json --out results/MY_LINEARITY_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.linearity verify --out results/MY_LINEARITY_REPLAY
```

새 과학적 설정은 별도 프로토콜·검증기·등록으로 구분한다. 등록 명령은 `python -m drosolearn.linearity register --config configs/linearity_diagnostic.json --out results/registration_NEW.json`이며 기존 경로를 덮어쓰지 않는다. 현재 연구의 시드·예산·학습률·오차 기준은 변경하면 등록을 거부한다.

## 후속: 같은 보상 경험으로 배우는 단순 모델

이번에는 원래 모델의 점수를 받는 설명식 대신 **7개 계수를 선택·보상으로 직접 학습하는 선형 정책**을 검사한다. 초기 출력은 모두 50%다. 연습 입력만으로 초기 특징 차이의 크기를 맞추는 선형·네트워크 조건, 원래 척도의 선형 조건, 선택과 무관한 무작위 보상 조건을 함께 보존한다. [사전 계획](docs/reward-linear-protocol.md), [계산적 배경](docs/reward-linear-background.md), [실행 보고서](results/reward_linear_v1/report.md)를 보존한다.

새6001–6010시드·9조건·400시행으로 실제 학습은36,000번이다. 두 주 질문은 보상 의존 습득과 습득 후 완전 미학습15쌍의 일반화다. 입력의 수량 순서는 이미 제공되며 결정적 두 선택의0/1보상에는 정답 쪽을 복원할 정보가 있다. 척도를 맞춰도7/48출력 계수와 추가1,474내부 가중치의 자원이 모두 같지는 않다. 이전 연구의 초기 출력·척도와 달라진 별도 진단으로 해석한다.

실행 결과의 [쉬운 해설](docs/reward-linear-findings.md)을 보존했다. 척도를 맞춘 독립 선형은 연습96.85%·완전 새15쌍96.01%의 정답 선택 확률, 습득 및습득후전이각10/10으로 두주기준을통과했다. 원래 척도의 선형64.71%·71.16%는사전gate0/10이지만 결정적새입력선택은100%였다. 무작위 보상은연습평균47.98%였고 두보조조건의습득미달20건을포함해보존했다. 실제·재배선·일반망의정규화조건도학습했으므로원래배선의고유성을입증하지않는다. 자동테스트724개와모든학습·평가의독립검산이통과했다.

재현은 `study-reward-linear-v1` 태그의 별도 checkout과 Python3.12 환경에서 새 결과 경로를 사용한다:

```powershell
.\.venv\Scripts\python.exe -m drosolearn.reward_linear run --config configs/reward_linear_diagnostic.json --registration results/registration_reward_linear_v1.json --out results/MY_REWARD_LINEAR_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.reward_linear verify --out results/MY_REWARD_LINEAR_REPLAY
```

새 과학적 설정은 별도 프로토콜·검증기·등록으로 구분한다. 등록 명령은 `python -m drosolearn.reward_linear register --config configs/reward_linear_diagnostic.json --out results/registration_NEW.json`이며 기존 경로를 덮어쓰지 않는다. 현재 연구의 시드·조건·척도 규칙·초기화·예산·학습률·판정 기준은 변경하면 등록을 거부한다.

## 후속: 단순 학습자의 보상 반전 비용

이번에는 **처음부터 작은 쪽을 배우는 경우**와 **큰 쪽을 배운 뒤 작은 쪽으로 바꾸는 경우**를 비교한다. 출력 계수만0으로 되돌리는 경계 대조와 작은 쪽을 계속 연습하는 대조도 포함한다. [사전 계획](docs/bandit-reversal-protocol.md), [계산적 배경](docs/bandit-reversal-background.md), [실행 보고서](results/bandit_reversal_v1/report.md)를 보존한다.

단순 선형과 기존 여섯 정규화 네트워크, 네 학습 이력, 새7001–7010시드를 사용한다. 단계당400시행이며 실제 선택·보상 학습196,000회와 별도 기대 갱신 참조28,000회를 구분한다. 참조는 정답 정보를 사용해 현재 상태에서 행동을 평균한 갱신을 반복하는 계산 경로로, 확률적 학습 궤적의 모집단 평균이나 독립 보상 학습자가 아니다. 주 H-B는 단순 모델의 fresh/high 습득과 고정400회 뒤의 끝점 비용을 함께 판정한다. 출발점·향상·원시 및 시작점을 보정한 곡선 면적을 별도 기록하며 낮은 끝점이나 면적을 학습 속도 자체의 증거로 읽지 않는다.

실행과 독립 검산을 완료했고 [쉬운 해설](docs/bandit-reversal-findings.md)을 보존했다. 단순 모델은 처음부터96.96%·반전44.51%·출력 초기화96.96%였다. 주 끝점 비용 H-B는 통과했고 반전 성공 기준은2/10이었다. 반전도3.03%→44.51%로 회복했으며 전체 실제 조건의 반전 미달40건을 보존했다. 기대 참조는 평균 곡선과 개별 경로의 차이를 따로 보고 완전한 설명으로 읽지 않는다. 관련 테스트401개와 모든 갱신·상태·평가의 독립 검산이 통과했다. 자동 보고서의 표 구분선만 보정한 기록도 함께 남겼고 소스·기준·수치는 바꾸지 않았다.

재현은 `study-bandit-reversal-v1` 태그의 별도 checkout과 Python3.12 환경에서 새 결과 경로를 사용한다:

```powershell
.\.venv\Scripts\python.exe -m drosolearn.bandit_reversal run --config configs/bandit_reversal_diagnostic.json --registration results/registration_bandit_reversal_v1.json --out results/MY_BANDIT_REVERSAL_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.bandit_reversal verify --out results/MY_BANDIT_REVERSAL_REPLAY
```

새 설정은 별도 프로토콜·검증기·등록으로 구분한다. 등록 명령은 `python -m drosolearn.bandit_reversal register --config configs/bandit_reversal_diagnostic.json --out results/registration_NEW.json`이며 기존 경로를 덮어쓰지 않는다. 현재 연구의 시드·조건·학습 이력·초기화·척도·예산·학습률·판정 기준은 고정한다.

## 연구 문서

| 문서 | 내용 |
|---|---|
| [핵심 질문](docs/research-question.md) | 다섯 연구 질문, 반증과 Level 1–5 |
| [DrosoMem 감사](docs/drosomem-audit.md) | 현재 구현·실험 결과·실제 데이터 범위·재사용 판단 |
| [문헌 검토](docs/literature-review.md) | 지정 논문 3편과 추가 일차 연구의 실제 근거 |
| [기술 설계](docs/technical-design.md) | 입력·회로·가소성·로그 인터페이스와 선택 이유 |
| [사전 프로토콜](docs/experimental-protocol.md) | 입력 분할·보상·대조군·고정 판정 기준 |
| [로드맵](docs/research-roadmap.md) | Phase 0–5와 다음 단계의 gate |
| [인간 비교](docs/human-learning-comparison.md) | 공통 측정 지표·공개 데이터·차이와 한계 |
| [LIMITATIONS](LIMITATIONS.md) | 추상 입력·모델·생물학·교육적 해석 경계 |

## 데이터·재현·GitHub

`data/`에 고정된 작은 원본 subset을 동봉한다. 큰 전뇌 자료나 원시 synapse point cloud는 다운로드하지 않았다. [파일 출처 manifest](data/source-manifest.json), [데이터 조건](data/LICENSE.md), 원본 `provenance.json`과 해시를 함께 보존한다. 새 Python 코드는 작성한 구현이며 DrosoMem runtime 코드를 복사하지 않았다.

`results/`에는 실패를 포함한 결과와 메타데이터만 기록한다. 학습 평가가 좋지 않더라도 예산을 늘려 같은 가설의 성공을 찾지 않는다. 다음 실험은 실패 원인과 입력·가소성·자원의 가설을 분리해 등록한다.

새 GitHub 저장소 생성·배포 절차는 [repository-creation.md](docs/repository-creation.md)에 있다. DrosoMem에 push하거나 원본 결과를 교체하지 않는다. 소프트웨어는 [MIT](LICENSE), FlyWire 파생 데이터는 별도의 CC BY-NC 4.0 조건을 따른다.
