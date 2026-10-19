# 인공 기호 패턴 식별과 제한적 전이: 실행 전 계획

2026-10-19 (Asia/Seoul). 연구 ID `pattern_rule_v1`. 앞선 두 후속 연구의 실행·독립 검산과
별도 소스·설정·입력 명세·그래프 해시 동결 및 비공개 원격 등록 확인 뒤 한 번 실행한다.
현재 문서는 긍정적인 학습 결과를 보고 기준을 정한 문서가 아니다. 이전 소스와 실패는 수정하지 않는다.

## 쉽게 설명하면

ABAB처럼 번갈아 나오는 짧은 문자열과 ABBA처럼 두 개씩 묶이는 주기 문자열을 구별하게 한다.
네 예제를 익히는 것과 새 기호·새 배치·새 길이에서도 구별하는 것을 따로 시험한다.
문자열을 모델 앞에 직접 놓아 주는 인공 과제이며, 초파리가 문자를 읽거나 수학을 이해했다는 시험은 아니다.

## 입력 명세와 지름길

| set 순서 | 목표 1 | 목표 0 | 평가 역할 |
|---|---|---|---|
| acquisition | ABAB / BABA | ABBA / BAAB | 네 훈련 예제 습득 |
| phase_diagnostic | ABAB / BABA | AABB / BBAA | 익숙한 두 anchor와 새 단일 class 진단 |
| new_alphabet | CDCD / DCDC | CDDC / DCCD | 두 class 모두 새 기호·입력 |
| new_alphabet_phase | CDCD / DCDC | CCDD / DDCC | 새 기호와 주기 4 phase 변화 |
| new_length | ABABAB / BABABA | ABBAAB / BAABBA | 길이 6의 두 새 class |

A=0, B=1, C=2, D=3이다. 20평가 슬롯은 16개 고유 문자열이다. phase의 두 anchor는 훈련과
같고, 새 alphabet/combined의 교대 두 문자열도 서로 같다. 이 중복을 독립 증거로 세지 않는다.
phase에서 새 문자열 두 개는 목표 0뿐이므로 항상 NoGo인 기계도 그 둘을 맞춘다.
**phase만 바꾼 결과에는 균형 일반화 성공 gate를 적용하지 않는다.** old anchor와 새 단일 class를 분리한다.

이 유한 명세의 실제 입력은 `data/synthetic/pattern_rule_v1/input_catalog.json`에 보존해
원 byte hash를 등록한다. producer는 그 파일을 직접 읽고 명세의 배열·형식·값과 일치하는지
확인한 뒤 사용한다. 파일이 없으면 런타임에서 대신 생성하지 않는다. 입력 파일 준비에는 모델이나 행동 RNG를 쓰지 않는다.

각 네 사례 set의 class별 전체 기호 빈도·첫 기호·마지막 기호 분포가 같다. 길이 4는 각 기호 두 번,
길이 6은 각 세 번이다. 분석적 인접 교대 oracle, 네 문자열 exact lookup(미관측 pGo=.5),
first-last 다름 지름길, 마지막 기호 복사와 first/last/frequency-only 최적 가능 성적을 exact count로 보고한다.
oracle은 학습자가 아니며 lookup은 훈련 라벨을 받은 분석 대조이다. first-last 지름길은 훈련과
새 alphabet을 풀어도 phase·새 길이에서 실패한다. 새 기호 한 set의 성공만으로 규칙 이해를 주장하지 않는다.

## 인공 이력과 모델

입력은 `[s0/3,...,s(L-1)/3,0 padding to six slots,L/6]`의 7차원이다.
모델은 짧은 전체 문자열을 인공 이력 버퍼로 받는다. 내부 기억이나 recurrent state를 학습시킨 것이 아니다.
길이 좌표는 0기호/padding을 구분하고, 길이 6에서는 새 좌표가 실제로 활성화된다.
기호값의 간격과 순서를 인공 입력이 제공한다는 한계를 남긴다.

두 행동 중 NoGo의 입력은 영벡터, Go의 입력은 위 벡터다. actor에는 현재 벡터·자기 선택·획득 보상만
제공한다. Go이면 종류 1에서 +1, 종류 0에서 -1이고 NoGo는 0이다. NoGo의 잠재 Go 결과와 목표
라벨은 학습자에게 주지 않는다. 숨은 표적은 환경과 평가에서만 사용한다.

조건은 fixed, plastic, rewired_fixed, rewired_plastic, ann_fixed, ann_plastic, linear, uniform, lookup_q의 9개다.
일차 조건은 plastic이다. 실제 부분 배선/재배선/ANN은 기존 `Learner` 초기화와 `build_graphs`를
새 `VectorActor`에서 사용한다. encoder seed는 model_seed+100000, head는 0으로 시작한다.
출력 학습률 .1, 내부 학습률 .01, `(a-p)*reward` 정책경사와 갱신 전 head 기반 eligibility,
원래 마스크·비음수 clipping을 유지한다. shell 과제의 기존 소스는 수정하지 않는다.

특징 차이 RMS는 **네 훈련 입력만** 사용한 초기 상태에서 고정한다. OOD 입력·정답·성과로 척도를 맞추지 않는다.
linear도 네 raw 훈련 벡터의 RMS로 정규화하며 head 7개만 학습한다. uniform의 pGo=.5, head/내부 가중치는 빈 배열이다.
모델별 자원과 내부 가소성의 추가 갱신 수를 기록하며 완전히 같은 학습 자원이라고 주장하지 않는다.

lookup_q는 원 입력 벡터와 길이에 대응하는 네 훈련 정체성×두 행동의 Q table을 0에서 시작해
자기 선택의 획득 보상만으로 학습한다. alpha=.1, beta=4이다. 새 입력의 pGo는 .5이며 평가 때
table 행을 추가하지 않는다. head/내부/활성은 빈 배열로, value table·그 변경·호출 수와 분리한다.
라벨을 바로 받은 분석 exact lookup과 다른 **보상으로 학습한 암기 대조**이다.

등록 전 개발 초안의 8조건에서 lookup_q를 추가한 이유는, 앞선 Study 1의 가치 표현 진단을
근거로 신경망 실패와 유한 과제/예산의 해결 가능성을 분리할 학습 대조가 필요했기 때문이다.
패턴 연구의 실제 학습을 실행하기 전에 정한 추가이며, 일차 plastic과 기존 성공 기준·시드·학습 길이는 그대로다.
등록된 8조건 실행이 있었던 것으로 표현하거나 추가 대조를 일차 모델로 바꾸지 않는다.

## 예산·난수·평가

model seed는 15101..15110의 10개다. seed마다 네 예제를 한 번씩 섞은 100 block, 총 400회
선택·피드백을 모든 조건에서 수행한다. 순서 주소는 `SeedSequence([202610191,seed,0])`,
행동 uniform 주소는 `[202610191,seed,1]`이다. 같은 seed에서 모든 조건이 순서와 uniform을 공유한다.
주소는 합하지 않는다. 구성 시드와 행동 추출을 인간/동물 개체 수로 바꾸지 않는다.

checkpoint는 0,50,100,200,400이다. 매번 5 set×4입력을 보상·업데이트·행동 추출 없이 평가한다.
총 실제 피드백은 9×10×400=36,000회, 정적 probe는 9×10×5×5×4=9,000사례다.
greedy 동률 pGo=.5는 Go를 택하며 일차 지표와 분리한다.

모든 시행에 현재 예제·pGo·행동·보상·출력/내부/value 호출 수·실제 new-minus-old 가중치 변화량과
비영 여부를 남긴다. 의도한 delta가 부동소수점 반올림으로 실제 변화를 만들지 못하면 변화량은 0이다.
모든 조건·seed·checkpoint에서 head·내부 가중치·encoder·mask·초기/상한·정규화 척도 및
20 probe의 KC/MBON/결정 특징, clipping 전 raw 점수 차이와 확률, lookup value table을 저장한다.
고정 probe 활성은 내부의 지속 기억 상태가 아니다.

## 습득 gate와 별도 일반화 판정

각 seed의 획득은 마지막 훈련 기대 정답 선택확률 ≥.80, 초기 대비 gain≥.25,
종류별 기대 정답률 각각 ≥.70을 모두 요구한다. 전체 10seed 중 8개 이상 통과할 때
그 조건의 습득을 지지하며, 일차 H_acquisition은 plastic이다.

new_alphabet/new_alphabet_phase/new_length 각각에서, 같은 seed가 획득한 뒤 마지막
기대 정답률 ≥.75, train 대비 손실 ≤.10, 종류별 성적 각각 ≥.70이면 해당 OOD를 통과한다.
전체 10seed 중 습득과 해당 OOD를 함께 통과한 수가 8개 이상이고 연구 단위 습득 gate도
통과할 때만 제한적 전이를 지지한다. 습득 미달이면 OOD 수치를 남기되
`not_assessable_without_acquisition`으로 표시한다. seed를 버리거나 더 잘 맞는 조건을 일차 조건으로 바꾸지 않는다.

phase-only에는 일반화 pass를 만들지 않는다. 각 class와 익숙한 anchor·새 단일 class를 기술한다.
분류 가능한 입력 정보, 학습으로 인한 변화, 습득 후 제한적 새 입력 전이를 구분하며, 새로운 규칙 family
전이·다음 항 예측·일반적 수학 이해를 시험했다고 쓰지 않는다. 유한 16문자열과 10계산 seed는
모든 수열/사람/초파리 집단을 대표하지 않는다.

## 검증·중단·다음 작업

개발 검사는 실제 seed/예산을 쓰지 않는 `integration_test` fixture만 실행한다. 공통 등록기는
그 scope를 실제 연구로 등록하지 못하게 한다. graph·원 소스·고정 catalog 생성 코드·설정·선행 결과를
새 등록으로 동결하고 선행 Study 1·2의 완료·검산 및 원격 등록 확인 뒤 실제 학습을 한 번 실행한다.

독립 검산기는 producer/actor/metric 함수를 import하지 않고 유한 catalog·중복·oracle/단서 균형,
원 그래프 변환·encoder·정규화·순서·공유 uniform·선택·보상·실제 가중치 변화·전체 checkpoint와
gate 판정을 재구성한다. 실패를 성공으로 바꾸기 위한 추가 seed·횟수·표현 검색은 하지 않는다.
무결성 오류와 과학적 습득/전이 미지지를 구분해 원본을 보존한다.

결과의 최대 범위는 이 표현·모델·유한 family의 정보 식별(Level 1), 경험에 따른 변화(Level 2),
gate를 통과한 제한적 새 입력 전이(Level 3)다. 인간 학습 유사성·실제 초파리의 기호 이해·공통 신경 기작·
교육 효과는 새로 주장하지 않는다. 간섭·순서 연구는 별도 등록과 자체 습득/공동학습 조건으로 뒤따른다.
