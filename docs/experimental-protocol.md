# 사전 실험 프로토콜 v1

작성일: 2026-10-14 (Asia/Seoul). 첫 실행 전에 코드·데이터·이 문서·설정의 SHA-256을 `register` 명령으로 동결한다. 이는 로컬의 전향적 기록이며 공인 저장소의 외부 사전등록이 아니다. 결과를 보고 판정 기준을 바꾸지 않는다. 변경이 필요하면 기존 결과와 실패를 보존하고 새 등록·새 결과 디렉터리를 만든다.

## 질문과 검증 범위

이미 수량과 순서를 제공한 인공 입력에서, 보상 경험이 선택 확률을 바꾸는가? 제한된 KC→MBON 내부 가중치 갱신이 출력층 학습과 어떤 차이를 보이는가? 이 실험은 수량 추출 또는 수학 규칙 발견을 검증하지 않는다. 초파리 시각 수량 인지와 LC11→KC 경로는 미구현이다.

주 가설 H1: 보상 선택 정책이 학습 전보다 향상하여 습득 기준을 충족한다. H1 실패는 입력에 정보가 없다는 뜻과 다르다. 입력이 이미 해결 정보를 제공하므로 분석적 oracle가 1.0이어도 정책이 학습하지 못할 수 있다.

H2: H1을 통과한 조건이 훈련 범위 밖 인공 nuisance 입력에서도 선택 기준을 유지한다. H3: H1 통과 조건이 보상 반전에 적응한다. 내부 가소성·배선의 성능 차이는 초기에는 방향을 전제하지 않는 기술 분석이다. 커넥톰 우수성을 성공 조건으로 삼지 않는다.

## 실제 사용 회로와 모델

원본 DrosoMem `7fb156c193c7a499fc2266a164b705729abbab21`의 `flywire_783_mb_left_kc512_s701`를 그대로 보존한다. 686뉴런·3,309연결 중 KC→MBON만 선택한다. 512 KC와 48 MBON의 nominal 차원을 유지하나 실제 연결은 489 KC→29 MBON의 1,474쌍·10,468접촉이다. 19 MBON은 입력이 없다. 125 DAN과 1 APL 및 재귀 회로는 계산하지 않는다. 기존 reservoir 코드를 그대로 사용하는 연구가 아니다.

KC 활성은 `max(E x + b, 0)`이다. E와 b는 seed별 고정 무작위 인공 encoder이며 모든 조건에 동일하다. 초기 KC→MBON 가중치는 연결별 시냅스 개수를 각 MBON의 보존된 KC 입력 접촉 수 합으로 나눈다. 이러한 정규화는 생리학 측정값이 아니다. MBON은 weighted sum, 출력은 두 후보의 MBON 활성 차이와 48차원 head의 내적이다. `p_right = sigmoid(head · (z_right − z_left))`로 행동을 샘플링한다. 모델에는 지속 상태, 실제 생리 시간, 자동 망각이 없다.

## 데이터와 단서 통제

주 학습 수량쌍은 2 대 4. 첫 400시행은 큰 수량, 다음 400시행은 작은 수량 선택에 1, 나머지는 0 보상이다. 반전에 같은 모델을 이어 쓰며 초기화하지 않는다. training phase별 서로 다른 입력을 사용한다. 고정 예산이고 조기 종료하지 않는다.

입력은 `[n/8, 0.1*(u_1−0.5), …, 0.1*(u_6−0.5)]`다. 여섯 nuisance 이름은 크기·총면적·밀도·간격·배치·범위를 상기시키는 **인공 descriptor**이며 실제 이미지의 기하학이 아니다. 실제 면적·밀도의 종속 관계를 이 synthetic vector가 구현했다고 주장하지 않는다. 시각 이미지 확장은 별도의 교차/반전 통제와 cue regression이 필요하다.

각 family의 두 nuisance vector U,V를 낮은/높은 수량과 교차하여 U-low/V-high, V-low/U-high를 만들고 양쪽 위치를 교환한다. 4행씩 정확히 균형화한 뒤 순서를 섞는다. 따라서 수량을 제거하면 동일한 인공 입력이 두 정답과 동수 대응한다. 단일 nuisance 기준 및 nuisance만 쓰는 고정 정책의 기대 정확도는 0.5다. 각 descriptor를 하나씩 조사하고 분석적 수량 oracle 정확도 1.0을 확인한다.

훈련·IID 평가 nuisance는 [0.25,0.75], OOD는 [0,0.25]∪[0.75,1] 범위다. eval은 split당 128행(32 family, 64개의 좌우 교환을 제외한 고유 자극)이며 독립 개체 128개가 아니다. 수량쌍·좌우 교환을 정규화한 실제 입력 fingerprint 중복을 전체 split 사이 검사한다. 학습에 쓰인 2와 4는 IID/OOD에서 의도적으로 반복된다. 새 수량쌍 1대3·3대4·4대8은 이차 probe로 두며 습득 gate를 분리한다. 3대4에는 기존 수량 4가 포함되고 1대3·4대8이 모두 완전히 미관측 수량은 아님을 보고한다.

seed별 stream은 train acquisition `seed*100+1`, reversal `+2`, IID `+3`, OOD `+4`, transfer `+10+i`이다. 모델 encoder/head는 `seed+70000`, 행동 RNG `+80000`, 무정보 보상 `+90000`, 재배선 `+50000`, ANN `+60000`이다. 조건별 학습·평가 입력, encoder, head 초기값과 행동 uniform stream을 맞춘다. 행동 확률이 달라지면 행동이 달라지는 것은 의도한 정책 비교다.

## 조건과 자원

| 조건 | 배선 | 내부 갱신 | 출력 갱신 | 목적 |
|---|---|---|---|---|
| fixed | 원래 KC→MBON | 0 | 48파라미터 | 고정 구조+출력 학습 |
| plastic | 원래 | 1,474파라미터 | 48 | 내부 추가 학습 |
| rewired_fixed | 차수 보존 재배선 | 0 | 48 | 배선 대조 |
| rewired_plastic | 동일 재배선 | 1,474 | 48 | 배선×가소성 2×2 |
| ann_fixed | 일반 무작위 sparse feedforward | 0 | 48 | 일반 신경망 대조 |
| ann_plastic | 동일 일반망 | 1,474 | 48 | 동등 내부 파라미터 대조 |
| frozen | 원래 | 0 | 0 | 무학습 대조 |
| random_reward | 원래 | 1,474 | 48 | 행동 정답과 독립 Bernoulli(0.5) 보상 |
| internal_only | 원래 | 1,474 | 0 | 출력 고정 시 내부 변화의 계산적 기여 진단 |

48개 출력 중 실제로 29개만 활성 입력을 받는다. ANN도 postsynaptic degree·입력 가중치 multiset 및 19개 inactive MBON을 유지한다. ANN은 KC outdegree를 보존하지 않는다. 재배선은 중복 없이 14,740개의 accepted double-edge swap을 수행하며 KC outdegree, MBON indegree, 각 MBON 입력 가중치 multiset/합을 실제 검사한다. KC outgoing strength, 유형·구획·거리·전체 회로는 보존하지 않는다. 재배선 차이를 파일에 기록한다.

고정 대 가소성은 총 trainable resource가 같지 않다. 가소성 효과와 자원 효과가 혼합될 수 있으므로 파라미터·시행·업데이트 횟수와 internal-only 결과를 보고한다. 초기 코드만으로 생물학적으로 허용된 최적 가소성을 선정했다고 주장하지 않는다.

## 갱신 규칙

선택 행동 a∈{0,1}, 선택 보상 r∈{0,1}, 선택 전 p와 고정 baseline 0.5로 `f=(a−p)*(r−0.5)`를 정의한다. head는 `head += 0.1*f*(z_right−z_left)`다. 내부는 **갱신 전 head**로 `w_e += 0.01*f*head_old[post_e]*(kc_right[pre_e]−kc_left[pre_e])`를 적용하고 [0,2*w_initial_e]에 clipping한다. 이는 단순화된 전역 reward-gated policy gradient다. 해당 eligibility는 현재 두 후보의 활성 차이이며 시간 trace나 실제 DAN 구획 신호가 아니다.

optimizer는 입력·선택 행동·보상만 받는다. 평가 정답과 양 후보 정답 label은 optimizer API에 없다. 그러나 이 완전한 이진 보상 설정에서는 보상과 행동으로 정답을 추론할 수 있다. ‘어려운 생물학적 강화학습’을 재현했다고 주장하지 않는다. finite-difference 검사는 clipping되지 않은 head/internal gradient를 확인한다. random_reward는 보상 permutation이 아닌 독립 Bernoulli stream이다.

## 평가와 실패 기준

주 지표는 **확률적 정책의 정확한 선택 정확도** `mean(p(target))`이며 CSV 필드명은 `expected_reward`다. 이는 같은 정책으로 무한히 행동을 샘플링했을 때의 기대 정답률이다. 이 평가에서 행동을 추가 샘플링하거나 학습하지 않는다. greedy `accuracy`는 진단이며 baseline의 미세한 random logit가 우연히 1.0 greedy 성능을 보일 수 있으므로 습득 판정에 쓰지 않는다. 이 구분은 첫 실행 전에 고정한다.

- 습득: 마지막 acquisition IID 지표 ≥0.75 **그리고** baseline 대비 ≥0.15 증가.
- 일반화: 습득 통과 **그리고** 마지막 acquisition OOD ≥0.70.
- 반전: 최초 습득 통과 **그리고** 마지막 reversal IID ≥0.75. 반전 직후 지표도 보존.
- 첫 기준 도달 시행: 100시행 단위 IID probe에서 처음 ≥0.75인 checkpoint. 지속 도달/정밀 latency가 아닌 보조 지표이며 미도달은 null/right-censored.
- 새 수량쌍: 원자료와 확률·greedy 성능을 모두 남기되 습득 여부를 함께 기록. n/8이 순서를 주므로 성공해도 규칙 발견으로 해석하지 않음.
- 구조적 해결 가능성: oracle=1.0, 모든 nuisance 기준=0.5가 아니면 분석을 진행하지 않음.
- 누출·해시 불일치·NaN·중복 edge·재배선 불변량 위반·평가중 변경은 실행/검증 실패이며 해당 기록을 보존.

smoke seeds는 [101,202,303], 3시드로 구현과 기록을 점검한다. 전체 시드 결과를 보고하고 유의성 검정·확증·구조 우수성 결론을 내리지 않는다. 확장 feasibility config는 [1001,…,1010]의 별도 10시드로 위 기준 통과 비율 ≥0.8을 설정하나 현재 실행하지 않는다. 초기 결과를 본 후 수정한 코드/학습률에는 amendment와 새로운 평가 조건을 사용한다. 10시드 통과도 인간/실제 초파리 일반화를 뜻하지 않는다. 배선 advantage 가설은 향후 별도의 효과 크기·자원 통제·통계 규약을 등록한다.

## 기록·검증·해석

모든 시행의 행동·보상·선택 확률·활성 norm·내부/출력 step 변화를 CSV로 보존한다. baseline과 100시행마다 같은 IID 16행 고정 probe의 **모든 KC/MBON 활동, 내부/출력 가중치, 입력 encoder와 행동 확률**을 NPZ로 저장한다. 실제 모든 시행의 전체 neuron vector를 저장하지는 않으며 저장 입력·시드·규칙으로 재현 가능하다. split당 평가 개별 사례는 phase 시작/끝을 저장한다. seed별 원본 입력·family ID·재배선도 저장한다.

별도 검증 코드는 학습 클래스와 평가 함수를 호출하지 않고 checkpoint와 입력으로 활성·출력·지표·판정을 다시 계산한다. SHA-256, cue baseline, split 누출, 조건별 갱신 범위, seed/조건/시행 coverage와 행동/보상 stream을 확인한다. 전체 optimizer trajectory의 독립 생리학 재현을 의미하지 않는다.

내부 가중치, 출력층, 동일 입력의 활성 변화는 별도 plot과 필드로 구분한다. 해석은 추상 입력의 Level 1/2 및 습득을 통과한 제한된 Level 3까지다. 인간 데이터 재분석이 없는 Level 4, 생물학적 검증이 없는 Level 5는 주장하지 않는다. 인간 학습의 비교 계획은 [human-learning-comparison.md](human-learning-comparison.md)에 유지한다.
