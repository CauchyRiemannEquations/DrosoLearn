# DrosoLearn 기술 설계 v0.1

## 확정된 초기 범위

Python 3.12와 NumPy를 기본으로 사용한다. 686개 원본 node 및3,309 edge subset을 그대로 보존하되 모델에서는 512 KC→48 MBON 사이의1,474 edge만 읽는다. 원래 해시를 검사한 다음 원본 decimal-string ID를 float로 변환하지 않고 역할 표식과 연결을 교차 검증한다. 실제 계산에 참여하는489 KC/29 MBON과 연결이 없는 차원을 metadata로 분리한다.

기존 DrosoMem recurrent reservoir 전체를 재사용하면 학습되지 않은 재귀 dynamics·DAN ordinary signed edges와 내부 갱신을 함께 바꿀 위험이 있다. 첫 구현은 허용 연결의 변화와 output 변화를 관찰하는 feedforward 최소 모델로 좁힌다. 따라서 단순함은 메커니즘 분리의 선택이며 기존 reservoir의 결과를 그대로 이어받은 것으로 해석하지 않는다.

## 모듈 인터페이스

| 파일 | 책임 |
|---|---|
| `data.py` | 인공 수량/nuisance cross, split fingerprints, oracle/cue 분석적 기준 |
| `graph.py` | 해시·역할 검증, KC→MBON 추출, degree-preserving rewiring, matched ANN |
| `model.py` | frozen encoder/KC 활성, sparse MBON sum, stochastic output, reward-only update |
| `experiment.py` | 로컬 등록, drift 차단, stream, 모든 시드/조건/phase 실행, raw logs/checkpoint |
| `analysis.py` | 학습·반전·내부/출력/활동·transfer 그림, 모든 시드 보고서 |
| `audit.py` | 학습 구현과 독립된 checkpoint 재계산·무결성·판정 검증 |
| `cli.py` | register/run/analyze/verify 실행 명령 |

numpy sparse edge arrays `pre,post,weights`를 사용한다. DAG 한 단계를 계산하므로 SciPy/PyTorch/GPU는 필수가 아니다. API `Learner.update(inputs, action, reward)`는 정답 label을 받지 않는다. 평가 및 보상 환경에만 정답이 있다. 완전 이진 보상의 정보량은 생물학 실험과 다르다.

## 가소성과 제한

mask는 실제 KC→MBON에만 허용한다. edge가 새로 생성되지 않고 각 edge의 초기 정규화 강도의 두 배를 upper bound로 삼는다. 전역 reward-gated gradient는 생물학적 규칙의 탐색을 위한 계산적 출발점이며 KC→MBON anti-Hebbian/DAN compartment 모델의 검증된 복제가 아니다. 단계별 rule와 gradient 검증은 [프로토콜](experimental-protocol.md)에 있다.

original/rewired fixed/plastic 2×2가 구조와 내부 갱신의 요소를 분리한다. 같은 신경 수·활성 output 수·입력·행동 stream을 맞추고 총 trainable parameter 차이를 기록한다. `internal_only`는 readout을 고정한 추가 진단이나 실제 생리학의 검증은 아니다. 최종 정확도가 천장에 도달하면 학습 속도·반전·간섭 차이를 사전에 별도 연구로 정하며 현재 결과에서 우수성을 만들지 않는다.

## 재현성과 실패 보존

config, 실행 전 source/data/protocol hash, git commit·작업트리 상태, package/platform, source graph provenance, seed mapping, reproduction command를 저장한다. 기존 출력 디렉터리와 등록은 exclusive 생성한다. 실패 시 partial records와 failure metadata를 남긴다. seed별 입력·family ID·topology·checkpoint는 pickle 없이 NPZ로 보존하며 CSV/JSON에는 난수와 모델 조건을 명시한다.

내부 weights, head weights와 동일한16행 probe의 KC/MBON activity는 서로 다른 필드다. checkpoint는100시행 간격이며 전체 신경 vector를 매시행 저장하지 않는다. 매시행 입력과 선택·보상·norm·가중치 step을 보존해 replay의 기반을 만든다. 별도 검증은 독립식으로 전체 시행 갱신과 저장 checkpoint를 재계산한다. 생물학적 회로·원래 초파리 실험의 재현을 검증한 것은 아니다.

## 확장 인터페이스

다음 변경은 버전별로 명시적으로 추가한다: 다른 KC seed subset, recurrent fixed reservoir adapter, 구획별 KC→MBON/DAN modulation, temporal eligibility/decay, 실제 LC11 visual input, 학습 순서와 충돌 과제, 기호 규칙 과제의 acquisition/transfer 분리. 이들 모두 현재 실행 완료 상태로 취급하지 않는다. 사람 공개자료는 별도 provenance·schema adapter와 공통 지표 계층으로 연결하고 모델 시행을 사람 참가자 자료와 합치지 않는다.
