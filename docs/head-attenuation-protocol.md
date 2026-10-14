# 배운 선택 계수의 부분 감쇠 — 사전 프로토콜 v1

작성일: 2026-10-15 (Asia/Seoul). [기존 보상 반전 결과](bandit-reversal-findings.md)를 확인한 뒤, 새 시드 **8001–8010을 실행하기 전에** 정한 후속 계획이다. 기존 완료 연구의 소스·프로토콜·설정·등록·원자료·결과와 실패는 변경하지 않는다. 새 코드·문서·설정·전향적 해시 동결·배타 생성 결과 경로를 사용한다. 비공개 Git 원격 보존은 외부 공인 공개 사전등록이나 완전한 눈가림 확증 연구를 뜻하지 않는다.

## 쉬운 설명과 주 질문

지난 실험에서 단순 선형 모델은 처음부터 작은 쪽을 배우면 최종 선택 확률 96.96%였지만, 큰 쪽을 배운 뒤 작은 쪽으로 바꾸면 44.51%였고 반전 기준은 2/10시드만 통과했다. 선택 담당 계수를 0으로 만들면 96.96%였다. 이번에는 **옛 선택의 방향을 유지한 채 계수의 크기만 줄였을 때, 같은 새 규칙 연습 뒤 작은 쪽 선택 확률이 높아지는지** 묻는다. 이전 연구의 소스 commit `fceca94dc6e8137085112ad3a5b75a3cf3894e50`, 재현 태그 `study-bandit-reversal-v1`, 결과와 미달 기록을 유지한다.

예를 들어 2개와 4개 중 4개를 고르면 보상을 주는 연습을 400번 한다. 그 상태를 네 사본으로 복제하고 선택 계수에 0, 0.25, 0.5, 1을 곱한다. 양수인 0.25와 0.5에서는 raw 점수의 선택 방향을 유지한다. 이후 모든 사본이 같은 입력과 선택 난수를 사용해 작은 쪽 보상 규칙을 400번 연습한다. 처음부터 작은 쪽만 배우는 별도 대조도 둔다.

계수를 줄이면 학습 없이도 잘못된 선택의 확신이 낮아져, 반전 직후 작은 쪽 확률이 50%에 가까워진다. 따라서 마지막 성적 차이를 모두 학습의 개선으로 부르지 않는다. **직후 차이, 이후 회복량 차이, 고정 예산 끝점 차이**를 따로 기록한다. 이 실험의 주 질문은 부분 감쇠라는 복합 조작의 끝점 효과이며, 확신만의 인과 효과나 학습 능력 자체의 속도를 분리하는 실험이 아니다.

입력은 물체 그림이 아닌 이미 순서를 제공하는 숫자다. 실제 초파리의 세기 능력, 인간 수학 이해, 자연 망각이나 실제 기억 삭제를 시험하지 않는다.

## 주 H-A와 사전 판정

주 대상은 `linear_normalized`다. 각 시드의 2단계 400시행 종료 IID 낮은 수량 선택 확률에서 다음을 계산한다.

`endpoint_benefit = p_small(gamma_025, phase2 trial400) − p_small(gamma_1, phase2 trial400)`

**H-A**는 다음 세 조건을 모두 요구한다.

1. 공유 high prefix가 최종 높은 수량 IID 확률 `≥0.75`, 시작 대비 gain `≥0.15`를 함께 충족한 시드가 **8/10 이상**이다.
2. `endpoint_benefit`의 **전체 10시드 평균이 ≥0.15**다.
3. `endpoint_benefit`가 엄격히 `>0`인 시드가 **8/10 이상**이다. 정확히 0은 양수로 세지 않는다.

습득 미달 시드를 평균·양수 개수에서 제외하지 않는다. 0.15는 15%포인트 차이이고, 75%·15%포인트 습득 gain·8/10과 함께 실행 전에 정한 연구용 실용 기준이다. 유의성 검정이나 생물학적 임계값이 아니다. 하나라도 미달하면 H-A 미지지로 보존한다. high 습득 미달은 ‘이미 배운 선택을 수정한다’는 해석을 제한한다.

`fresh_low`의 최종 IID 확률 `≥.75`와 시작 대비 gain `≥.15`는 별도 습득 기록이다. 이를 H-A의 추가 gate로 넣지 않는다. 다른 여섯 네트워크에 동일한 끝점 판정을 적용하되 보조 판정으로 표시하고 선형 주 결과를 대신하지 않는다. `.5`, `0`, 감쇠 수준의 끝점 기울기·단조성, OOD, 회복량, AUC와 행동 평균 참조는 모두 기술적 보조다. 새 결과를 본 뒤 감쇠 수준·시드·주 지표·기준을 골라 주 결과로 승격하지 않는다.

## 모델 조건과 자원

| 조건 | 표현과 연결 | 학습 부분 |
|---|---|---|
| `linear_normalized` | 7차원 인공 입력 | 출력 계수 7개 |
| `fixed` / `plastic` | 실제 KC→MBON 부분집합, MBON 슬롯 48개 | 출력 48개 / 출력과 내부 가중치 1,474개 |
| `rewired_fixed` / `rewired_plastic` | 차수 보존 재배선 | 같은 출력 / 출력과 내부 가중치 |
| `ann_fixed` / `ann_plastic` | 무작위 희소 일반망 | 같은 출력 / 출력과 내부 가중치 |

초기 head는 전 조건 정확히 0, 출력 학습률 .1, 내부 학습률 .01이다. 초기 특징 RMS `g`는 **미학습 상태의 1단계 train 입력만**으로 계산하고 이후 고정한다. 과거의 초기화·척도·시드가 다른 연구를 동일 실험으로 합치지 않는다.

실제 연결은 FlyWire v783의 보존된 nominal 512 KC·48 MBON 부분집합이며 1,474쌍에 489 KC·29 MBON이 참여한다. 연결 없는 19 MBON 슬롯도 남긴다. 재배선은 KC outdegree·MBON indegree·MBON별 incoming weight multiset을 보존하고 ANN은 뒤 두 특성을 보존한다. encoder와 KC ReLU는 고정이고 LC11·DAN·APL·재귀 회로나 생리 시간을 구현하지 않는다. 계수 수·특징 방향·상관·내부 추가 학습 자원은 같지 않으므로 전체 자원 동등성이나 실제 커넥톰의 우수성을 주장하지 않는다.

## 공유 prefix와 다섯 경로

| 경로 | 1단계 | 경계 조작 | 2단계 |
|---|---|---|---|
| `fresh_low` | 보상 학습 생략, 같은 무보상 척도 전처리 | 초기 상태 유지 | low 400 |
| `gamma_0` | 공유 high 400 상태 사본 | head × 0 | low 400 |
| `gamma_025` | 같은 상태 사본 | head × .25 | low 400 |
| `gamma_05` | 같은 상태 사본 | head × .5 | low 400 |
| `gamma_1` | 같은 상태 사본 | head × 1 | low 400 |

각 seed·family에서 high prefix를 **실제로 한 번만** 학습하고 그 종료 상태를 네 개의 독립 사본으로 복제한다. `seed_{seed}/{condition}/shared_prefix/phase1_{trial:04d}.npz`에 공유 궤적을 저장하며 branch는 해당 prefix를 가리킨다. 사본의 state·메모리를 공유하여 다른 branch가 영향을 받으면 안 된다. 각 branch의 경계 직전 상태는 공유 종료 상태와 동일해야 한다.

감쇠는 head만 바꾼다. 내부 가중치·encoder·bias·topology·initial upper bound·고정 `g`·학습률·갱신 counter·RNG 상태와 2단계 stream은 보존한다. 경계 전후 상태와 조작량을 저장하고 감쇠를 보상 갱신으로 세지 않는다. γ>0에서 모든 IID/OOD 사례의 raw 점수 차이 `d_after=γ d_before`, raw 부호와 후보 순위 보존을 검사한다. confidence `max(p,1−p)`는 사례별 비증가여야 하고 평균·엄격감소 사례 수를 기록한다. 확률 clipping 때문에 모든 사례의 엄격 감소를 일반 법칙으로 요구하지 않는다. raw 부호와 `|p−.5|<1e−12`를 동률로 처리하는 greedy 지표는 구분한다. γ=0은 점수 0·확률 .5의 별도 완전 초기화 대조다.

가소성 모델에서는 내부 가중치를 바꾸지 않아도 `internal_weight×head`로 표현되는 유효 점수 배선이 감쇠되며, 다음 내부 갱신의 `old_head`도 줄어든다. 즉 초기 확신, 실제 선택/보상, 출력 갱신, 내부 eligibility가 함께 달라질 수 있다. confidence-only 인과 결론을 내리지 않는다.

선형과 내부 고정 네트워크에서는 γ0 branch의 2단계 상태·선택·보상·확률이 같은 `g`와 입력·uniform을 가진 `fresh_low`와 정확히 같아야 한다. 누적 counter는 이전 400회 때문에 branch가 더 크며 별도 기록한다. 이 관계는 구현 무결성 검사이고 새로운 과학적 성공이 아니다. 가소성 모델의 γ0−fresh 차이는 바뀐 내부와 새 head의 상호작용이며 생물학적 시냅스 기억의 증거가 아니다.

실제 reward-update 호출은 `10×7×(400+4×400+400)=168,000`회다. branch의 학습 이력 counter는 각 800, fresh는 400이지만, 사본의 상속된 prefix counter를 네 번의 실제 호출로 세지 않는다. branch 네 개의 총 이력 수와 물리적 호출 수를 분리한다. 모든 경로의 **새 규칙 연습은 400회**이며 총 보상 경험까지 같다고 말하지 않는다. 평가·전처리·감쇠·복제·학습 없는 경계 기준선은 이 수에 더하지 않는다. 조기 종료·시드 교체·사후 예산 연장·학습률 변경은 하지 않는다.

## 입력, 고정 척도와 갱신

`x=(n/8, .1×(여섯 nuisance 좌표−.5))`를 입력한다. 두 후보의 수량과 위치·nuisance를 네 행 교차 설계로 균형화한다. 초기 raw 특징 `z`에 대해 `g=sqrt(mean_i sum_j (z_right−z_left)^2)`를 1단계 train 400개에서 계산한다. 선형은 `z=x`, 네트워크는 초기 raw MBON이다. `g`는 양의 유한값이어야 하며 같은 시드·family의 prefix·모든 branch·fresh에서 같다. 2단계·평가 입력·정답·보상이나 학습 뒤 내부 상태로 재계산하지 않는다. 같은 초기 고정/가소성 배선 쌍의 `g`도 같아야 한다.

이는 전체 초기 train 입력의 무보상 batch 관찰을 허용한 전처리다. 미래 입력을 전혀 보지 못하는 순수 온라인 실험이라고 부르지 않는다. 초기 평균 제곱 특징 차이 길이만 맞추며 방향·상관이나 내부 변화 후 척도까지 같게 하지 않는다.

`φ=z/g`, `d=(φ_right−φ_left)·head`, `p=sigmoid(clip(d,−40,40))`, `a=1[u<p]`, `r=1[선택이현재규칙정답]`이다. 학습 API는 입력·선택·scalar 보상만 받는다.

`f=(a−p)(r−.5)`

`head ← head + .1 f (φ_right−φ_left)`

가소성은 갱신 전 head로 `eligibility=old_head[post]×(KC_right−KC_left)[pre]/g`를 계산하고 `weights←clip(weights+.01 f eligibility,0,2×initial_weights)`를 사용한다. head·내부·활성 변화를 따로 보존한다. 결정적 두 선택 0/1 보상은 정답쪽을 복원할 수 있는 정보이므로 어려운 무라벨 수학 규칙 발견으로 표현하지 않는다. clipping 미분을 적용하지 않는 이 갱신을 모든 상태에서 clipped 확률의 정확한 기울기라고 부르지 않는다.

## 새 시드와 난수 stream

주 과학 시드는 **8001–8010**이고 개발 fixture는 **101/102**만 사용한다. 새 과학 시드로 설계·척도·예산·기준을 튜닝하지 않는다.

| stream | 난수 시드 | 행 수 |
|---|---|---:|
| phase1 train | `seed*100+31` | 400 |
| phase2 train | `seed*100+34` | 400 |
| IID 평가 | `seed*100+32` | 128 |
| nuisance OOD 평가 | `seed*100+33` | 128 |
| phase1 선택 uniform | `seed+280000` | 400 |
| phase2 선택 uniform | `seed+300000` | 400 |
| encoder / 재배선 / ANN | `seed+70000` / `seed+50000` / `seed+60000` | — |

네 split의 방향 독립 자극 fingerprint를 실제 비교해 중복을 검사한다. 수량 oracle은 정답 확률 1, nuisance-only는 .5여야 한다. 다른 난수 시드만으로 누출 부재를 가정하지 않는다. 2단계 입력·uniform은 모든 branch와 fresh에서 같고, 정책이 다르면 실제 선택과 보상은 달라질 수 있다. prefix 복제·fresh 생략·γ0 조작은 2단계 RNG 위치를 바꾸면 안 된다. 평가 중 상태·counter·척도·RNG를 바꾸지 않는다.

## 평가와 보조 기술 지표

공유 prefix와 각 2단계의 0·100·200·300·400시행에 IID/OOD를 평가한다. phase2 trial0은 경계 조작 뒤 상태다. fresh에는 phase1 학습 궤적이 없다. `p_target`은 사례별 정답 선택 **확률의 평균**이고, 뽑힌 행동의 실측 정답률이나 greedy 정확도와 다르다. `p_larger`, `p_smaller`, 전체 사례 `p_right`, raw 점수 복원과 `p(1−p)`를 보존한다.

| 지표 | 정의와 용도 |
|---|---|
| 주 끝점 benefit | γ.25−γ1의 phase2 최종 IID 확률, 전체 시드 |
| 즉시 경계 advantage | γ.25−γ1의 phase2 trial0 IID 확률; 학습 없이 생긴 차이 |
| branch 회복량 | `final−trial0`; 낮은 출발점의 회복과 구분 |
| 회복량 차이 | `gain(.25)−gain(1)=endpoint_benefit−initial_advantage` |
| 학습 없는 기준선 | 각 branch의 trial0 확률을 고정 예산 전체에서 유지하는 기준선; 상태 변화·reward update 없음 |
| raw / 출발점 조정 AUC | 다섯 checkpoint 사다리꼴 적분/400; 조정값은 raw AUC−trial0 |
| 목표 기준과 수정 자격 | branch final IID≥.75, 공유 high 습득과 둘 다 만족한 개별 시드 수; fresh는 .75와 gain .15 |
| 최초 관측 도달 | IID≥.75인 첫 저장 checkpoint; 미도달 null; 정확한 최초 시행·지속 성공 아님 |
| γ 수준 기술 | γ0/.25/.5/1 전체 끝점·개별 시드 선형 기울기와 단조 순서 count; 새 성공 gate 없음 |
| γ0−fresh | 끝점·AUC·시작점 및 내부 상태 차이; 고정 모델 일치는 무결성 검사 |
| OOD | 인공 nuisance 분포 이동의 확률과 차이, IID와 별도 보고 |

branch 2단계 성공에 추가 gain .15는 요구하지 않는다. 출발점·회복량·끝점·AUC 하나만으로 학습 능력 자체가 느려졌다거나 빨라졌다고 결론 내리지 않는다. 특히 γ.25는 학습 없이도 시작점이 높아진다. 학습 없는 기준선과 gain 차이를 남기되, baseline 조정 역시 같은 intrinsic speed를 분리해 추정하는 인과 분석은 아니다. 모든 평균·표본 SD·개별 시드·미달을 보존하며 errorbar는 신뢰구간이 아니다.

## 행동 평균 참조와 해석 한계

선형만 별도 `expected_linear` 참조를 계산한다. 현재 참조 상태에서 `(2y−1)p(1−p)`를 사용해 head를 `.1×factor×Δφ`만큼 갱신한다. 공유 high prefix 400을 한 번 만들고 같은 γ 네 branch를 복제한 뒤 low 400, 별도 fresh low 400을 계산한다. 총 `10×(400+4×400+400)=24,000`회의 결정적 참조 갱신을 실제 reward calls 168,000과 분리한다.

참조는 정답쪽 `y`를 직접 사용하는 기술 계산이며 실제 보상 학습자와 같은 정보의 대조가 아니다. 참조에 습득·성공·실패 flags를 붙이지 않는다. 자신의 현재 상태에서 한 단계 행동 평균식을 반복한 경로는 비선형성 때문에 stochastic 학습의 정확한 전체 기대값이나 동일 입력 ensemble 평균도 아니다. 같은 seed의 실제/참조 IID·OOD checkpoint별 차이, 개별 경로 MAE, 평균 곡선 MAE와 최대 개별 차이를 따로 보고한다. 평균에서 상쇄된 오차를 숨기지 않는다. 참조 성적이나 곡선 일치로 예산·threshold를 바꾸지 않는다.

`p(1−p)`의 경계 변화는 계산적 설명의 한 부분이다. 특징의 방향·크기, 실제 선택과 보상, 출력·내부 갱신과 clipping도 영향을 주므로 이 계수 하나를 전체 원인이라고 선언하지 않는다. [계산적 배경](head-attenuation-background.md)에 자체 도출과 기존 문헌의 확인 범위를 구분한다.

## 실행 전 동결, 원자료와 독립 검산

새 설정·프로토콜·배경·전체 Python 소스·투영 원자료·manifest·license·환경 잠금 파일을 새 등록의 해시로 동결한다. source commit과 등록 commit을 비공개 원격에 보존하고 원격 byte 해시·UTC 시간을 확인한 뒤 과학 실행을 시작한다. 존재하지 않는 새 결과 경로를 배타 생성하고 config·코드 버전·환경·dirty 상태·시작/종료 시간·모든 시드·경로·모델 자원·lineage counters·물리적 호출 수를 보존한다. 이전 등록을 최신 source file set에 맞춰 고치지 않는다.

개발 테스트는 TRAIN-only 고정 척도, prefix 실제 한 번, 분기 독립성, 허용된 head-only 감쇠, boundary raw sign/rank·confidence, 고정 γ0=fresh, stream 동일성, 평가 상태 비변경, 실제/참조 budget 분리, H-A 경계·all-seed 평균·실패 보존과 참조의 무성공 flags를 검사한다. 작은 fixture 수치로 과학 가설을 판정하지 않는다.

독립 검산은 producer 학습 클래스·척도·요약·기대참조 함수를 호출하지 않고 별도 산술로 초기 척도, 실제 선택·보상·head/내부 갱신, 모든 checkpoint와 사례 확률·raw 점수·경계 허용 변화·공유 prefix, 참조 경로와 판정을 복원한다. hashes·split fingerprints·coverage·counter lineage·physical calls·graph 불변량을 확인한다. NaN·누출·해시 불일치·허용하지 않은 상태변화는 과학적 미지지와 다른 무결성 실패이며 기록을 보존한다.

```powershell
.\.venv\Scripts\python.exe -m drosolearn.head_attenuation register --config configs/head_attenuation_diagnostic.json --out results/registration_head_attenuation_v1.json
.\.venv\Scripts\python.exe -m drosolearn.head_attenuation run --config configs/head_attenuation_diagnostic.json --registration results/registration_head_attenuation_v1.json --out results/MY_HEAD_ATTENUATION_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.head_attenuation verify --out results/MY_HEAD_ATTENUATION_REPLAY
.\.venv\Scripts\python.exe -m drosolearn.head_attenuation analyze --out results/MY_HEAD_ATTENUATION_REPLAY
```

완료 보존용 태그 `study-head-attenuation-v1`의 별도 checkout과 잠금 환경으로 재현한다. 기존 최초 smoke는 `v0.1.0-smoke`, 기존 bandit 연구는 `study-bandit-reversal-v1`의 각 source set을 사용한다.

## 반증과 인간 비교

H-A 지지는 이 입력·고정 척도·학습률·400시행 조건에서 γ.25 부분 감쇠의 전체 끝점 benefit이 사전 기준을 충족한다는 뜻이다. 미지지는 그 기준을 충족하지 못한 결과이며 모든 조건의 반전이 쉽다거나 감쇠 효과가 없다는 일반 결론이 아니다. γ0나 다른 family의 좋은 성적으로 주 미달을 바꾸지 않는다. 즉시 경계 advantage와 이후 gain 차이를 함께 해석한다.

현재 범위는 인공 입력에서 경험에 따라 선택이 바뀌는 **Level 2**다. 새 수량 일반화는 실행하지 않는다. 초파리·인간 자료의 정량 유사성 Level 4나 관련 신경 기작 Level 5를 주장하지 않는다. 계산 seed는 생물학적 개체나 인간 참가자가 아니다. 인간 기초 수량·수학 학습과 비교하는 장기 목표는 유지하되, 출발 편향·고정 예산 수정 비용·회복량·오류 방향의 관측 대응과 시행별 선택·피드백 원자료를 먼저 확보하고 별도 지표·가설로 등록한다. 현재 전후 요약 인간 자료로 이 반전 궤적을 직접 비교하거나 교육 효과를 주장하지 않는다. 비슷한 행동 형태가 같은 신경 원인을 뜻하지 않는다.
