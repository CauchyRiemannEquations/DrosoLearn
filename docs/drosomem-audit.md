# DrosoMem 원본 감사와 DrosoLearn 재사용 범위

확인일: 2026-10-14 (Asia/Seoul). 대상은 공개 저장소 [CauchyRiemannEquations/DrosoMem](https://github.com/CauchyRiemannEquations/DrosoMem)의 `main`, 커밋 **`7fb156c193c7a499fc2266a164b705729abbab21`**이다. 마지막 커밋 시간은 `2026-10-13T12:42:57+09:00`, 제목은 “Seal scientific manuscript review audit and preserve original draft evidence”다. 이후 upstream 변경은 이 감사에 포함되지 않는다.

원본은 작업용 `work/DrosoMem_snapshot/`에 얕은 sparse clone으로 읽었다. 처음 전체 이력 clone은 자료 규모 때문에 중단했다. 원본 소스·데이터·연구 결과를 수정하거나 upstream에 push하지 않았다. sparse clone에 소스, 설정, tests, docs, 번들 data는 있고 대규모 results 전체는 없다. M11의 `results/rule_transfer_v1/main/summary.json`은 해당 커밋의 Git blob으로 별도 읽었다. 역사적 실험의 수치 재실행을 수행한 감사는 아니다. 본 감사가 새로 수행한 검증은 번들 데이터의 SHA256, CSV 구조·뉴런 역할·연결 수, LC11 표식의 전체 원자료 교차 확인이다. [기계 판독 감사 기록](drosomem-audit.json)에 확인 범위와 해시를 남긴다.

## 1. 코드·데이터 라이선스의 경계

- DrosoMem 코드·문서는 MIT, `Copyright (c) 2026 Flying contributors`다. substantial code를 복사하면 원래 저작권 고지와 MIT 전문을 함께 보존한다.
- Shiu/Spiller 소프트웨어 고지는 `data/UPSTREAM_LICENSE.txt`, MIT, `Copyright (c) 2023 Philip Shiu and Nico Spiller`다.
- FlyWire public v783 유래 데이터는 별도 **CC BY-NC 4.0**다. 위 MIT 고지가 데이터의 비상업적 이용·출처 표시 조건을 바꾸지 않는다. DrosoLearn의 코드 라이선스와 파생 data 라이선스를 구분한다.
- 공개 출처는 [FlyWire v783 archive](https://zenodo.org/records/10676866), [FlyWire 지침](https://home.flywire.ai/guidelines), [DrosoMem DATA_SOURCES](https://github.com/CauchyRiemannEquations/DrosoMem/blob/7fb156c193c7a499fc2266a164b705729abbab21/DATA_SOURCES.md)다. 배포 시 현재 원자료 인용 지침도 확인한다. DrosoMem을 FlyWire 연결·표식의 원저자로 인용하지 않는다.

## 2. 원자료와 실제 번들 범위

연결 원자료는 Shiu/Spiller가 배포한 `Connectivity_783.parquet`이며 source commit은 `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`이다. 알려진 SHA256은 `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347`, source rows는 15,091,983이다. 이 약 101 MB parquet는 원본 Git에 번들되지 않는다. 이번 감사에서는 parquet 전체를 다운로드하거나 모든 원자료 edge를 재추출하지 않았다.

표식은 `flyconnectome/flywire_annotations` commit `8587524c1748ce5ef2080822a2fc890fc03bf597`의 `Supplemental_file1_neuron_annotations.tsv`다. 이번에 공개 원본을 직접 다운로드했고 SHA256 `9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be`가 기대값과 일치했다. 표식 139,248행이다. 연결 node universe의 별도 `Completeness_783.csv`도 원본을 받아 해시 `bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311`와 138,639개 ID를 확인했다. 업데이트된 v783 표식을 사용하는 것이며, 2024 최초 출판 당시의 annotation snapshot이라고 주장하지 않는다.

| 번들 폴더 (`data/` 아래) | 뉴런 | 유향 연결 | 선택·해석 |
| --- | ---: | ---: | --- |
| `flywire_783_subset` | 300 | 3,303 | threshold 후 incident synapse strength 상위 hub; 해부학적 회로 아님 |
| `flywire_783_connected_300_s101` | 300 | 5,477 | seeded weakly connected 성장; 생물학적 표본 아님 |
| `flywire_783_connected_300_s202` | 300 | 2,438 | 같은 source의 overlapping computational subset |
| `flywire_783_connected_300_s303` | 300 | 1,682 | weak connectivity가 strong connectivity를 뜻하지 않음 |
| `flywire_783_hubs_1000` | 1,000 | 24,604 | hub 순위 유도 부분망 |
| `flywire_783_mb_left_kc512_s701` | 686 | 3,309 | 이번 초기 구현의 파생 연결 원본 |
| `flywire_783_mb_left_kc512_s702` | 686 | 3,241 | 향후 KC sampling 민감도 확인 후보 |

모든 번들 graph는 root ID·edge를 제공한다. named-cell annotations는 두 MB subset에 별도로 들어 있다. ID는 큰 정수이므로 float 경유를 금지하고 JSON/CSV에서 decimal string으로 다룬다. `edges.csv`는 `pre,post,count,sign`, `neurons.json`은 순서가 정해진 모든 ID, `provenance.json`은 원본·파생 해시 및 추출 조건이다. matrix convention은 **`W[post, pre]`**이며 값은 `count × upstream sign`이다. 접촉 수와 upstream 부호는 실측 생리학적 시냅스 효능이 아니다.

## 3. KC·MBON·DAN·APL의 가용 범위

MB s701/s702 각각은 좌측 **KC 512, MBON 48, DAN 125, APL 1**을 포함한다. selection은 `cell_class == Kenyon_Cell/MBON/DAN`과 `cell_type == APL`의 exact annotation으로 역할을 정한다. 좌측 전체 pool은 KC 2,580, MBON 48, DAN 166, APL 1이다. KC는 seed 701/702로 512개를 uniform without replacement 샘플링한다. 모든 좌측 MBON/APL을 남기고, **KC 샘플링 전 전체 KC/MBON pool**과 어느 방향으로든 ≥5 접촉 edge가 있는 DAN 125개를 남긴다. 나머지 DAN 41개 제외 목록도 provenance에 기록되어 있다. 이후 선택 ID에 induced graph, ≥5 접촉 threshold, autapse 제거를 적용한다. 특정 DAN은 KC sampling 뒤 실제 edge를 잃을 수 있다.

**s701의 직접 KC→MBON 경로**를 실제 CSV에서 다시 세었다. 1,474개 directed pair, 합 10,468개 contact, 연결된 presynaptic KC 489개, postsynaptic MBON 29개다. 따라서 48 MBON 모두를 관측해도 19개는 이 one-hop 파생 경로에서 KC 입력이 없다. s702의 KC→MBON pair는 1,431개다. 512→48 투사 행렬을 쓸 때 node universe와 zero columns를 보존하고, nominal dimensions와 실제 active dimensions를 따로 보고한다.

초기 DrosoLearn은 s701에서 이 경로만 추출하는 **단순화된 계산 모델**을 사용한다. 원래 686-node 재귀 reservoir의 전체 dynamics, DAN/APL activity, compartment별 신호, 시각 입력 회로를 재사용한 모델이 아니다. reward scalar로 KC→MBON의 선택된 가중치를 변경해도 이를 실제 DAN 재현으로 부르지 않는다. 접촉 topology는 실제 데이터에서 왔지만 입력·활성·학습 규칙·행동 출력은 모델 설계다.

## 4. LC11: 원자료는 있지만 번들된 시각 회로는 없음

전체 pinned annotation에서 **exact `cell_type == LC11`은 127개(좌 66/우 61)**이고 `super_class == visual_projection`이다. 모든 127 root ID가 같은 pinned `Completeness_783.csv`의 138,639 source-node universe에 존재했다. 따라서 LC11 원자료를 원천적으로 이용할 수 없다고 쓰면 부정확하다.

반면 위 **7개 번들 subset에는 LC11 ID가 모두 0개**다. 원본 모듈에 retina/dot image→LC11 encoding, 검증된 LC11 receptive-field dynamics, LC11→MB 경로의 실측 자극 모델은 구현되어 있지 않다. 이번 감사는 source 전체 parquet에서 LC11의 모든 in/out edge 또는 LC11→KC/MBON 도달 경로를 재구성하지 않았다. LC11 label과 source-node membership은 확인했고, 기능적 회로 구현은 확인하지 않았다. 이 구분을 유지한다.

후속 시각 연구는 full graph를 체크섬으로 확보한 뒤 LC11/상류/하류 coverage, optic-lobe 범위, stimulus mapping, trial dynamics를 별도 등록해야 한다. 현재 사전 계산된 수량 또는 KC 인공 코드로 점 집합을 주는 실험은 **추상적 수량 학습 벤치마크**이며 살아 있는 초파리의 시각적 수량 인지를 재현했다고 주장할 수 없다.

## 5. 기존 시뮬레이션의 인터페이스와 재사용 판단

| 원본 모듈 | 검증한 구현·입출력 | DrosoLearn 판단 |
| --- | --- | --- |
| `flying.data.connectome.load_connectome(directory)` | CSV/ID SHA256 확인; CSR, IDs, provenance 반환; duplicate pair/unknown ID/sign/count 검증 | 로더 설계·schema를 참고하되 이번 모델은 별도 구현 |
| `flying.data.mushroom_body.load_roles(directory, ids)` | annotations SHA256, ID 집합·exact role label 검증; roles 및 DataFrame 반환 | 파생 KC→MBON 경로 선택의 출처 |
| `extract_subset`, `build_circuits` | raw·annotation 체크섬 확인; threshold/seed/induced extraction | 미래 재구축 절차의 근거; 초기에는 실행 불필요 |
| `brain.reservoir.Reservoir` | `x←(1−leak)x+leak·tanh(Wx+encoder(symbol))`; fixed weights | 보존할 고정 재귀 baseline 후보; 현재 one-hop 모델과 혼동 금지 |
| `brain.mushroom_body.KCEncoder/CircuitReservoir` | fixed sparse digit patterns KC에만 주입; symbol당 여러 update | 인공 입력이라는 한계 보존; 수량 visual encoder로 전용 주장 금지 |
| `models.readout.Readout`, `SelectedReadout` | train-only mean/SD와 affine10-class softmax, full-batch Adam | 상태·출력층 변화 분리 원칙 참고; digit-specific 코드를 그대로 쓰지 않음 |
| `brain.mushroom_body.role_shuffled` | 역할 블록 내 directed double-edge swaps; per-node in/out degree 보존 | 성공 swap수·original overlap 기록 필수; 독립 균일 null이라고 가정 금지 |
| `brain.role_random.role_random` | 역할 블록 N/E·weight multiset 대응, degree 미보존 | 약한 대조군이므로 강한 degree control과 구별 |
| `visualization.activity`, experiment checkpoint/manifest/verify scripts | 상태/학습 곡선·가중치/출력 hash·원자료/설정/코드 기록 | 저장 단위와 독립 검증 구조를 참고; 역사적 manifests는 수정 금지 |
| `data.whole_brain.prepare/load_graph` | raw node crosswalk, full sparse caches, retained-input coverage; 별도 raw 다운로드 필요 | 큰 모델 성공 탐색의 근거 아님; 현재 scope 밖 |

기본 `pyproject.toml`은 Python ≥3.10, numpy `>=1.26,<3`, scipy `>=1.11,<2`, pandas `>=2,<4`, matplotlib `>=3.8,<4`, mpmath `>=1.3,<2`, threadpoolctl `>=3,<4`다. parquet 추출만 pyarrow `>=14,<26`가 추가 필요하다. Brian2 기반 LIF는 optional `brian2==2.10.1`이며 별도 lock이 있다. 기존 실험의 재현은 그 실험 producer revision과 lock을 써야 하며 최신 source로 과거 replay 해시를 덮어쓰면 안 된다. 이번 초기 파이프라인은 새 Python 코드로 최소 dependencies를 정한다.

원본 최소 그래프 로드는 다음과 같다(이 코드는 DrosoMem snapshot을 읽는 예시다).

```python
from pathlib import Path
from flying.data.connectome import load_connectome
from flying.data.mushroom_body import load_roles

path = Path('data/flywire_783_mb_left_kc512_s701')
weights, root_ids, provenance = load_connectome(path)
roles, annotations = load_roles(path, root_ids)
# weights.shape == (686, 686), weights.nnz == 3309
# weights[post, pre]; KC-to-MBON edge mask uses pre-role KC / post-role MBON.
```

## 6. 보존해야 할 연구 결과와 최신 원본의 추가 범위

사용자 요약에 있는 핵심 결론을 현재 원본에서 확인했다. 고정된 source-based computational reservoir에서 **과거 입력의 operational decoding** 근거가 있다. 실제 배선이 모든 matched null보다 더 좋다는 주장은 성립하지 않았고, periodic next-symbol 예측이나 쉬운 tracking의 ceiling은 우위 증거가 아니다. 계산 seed/overlapping subset/replay 횟수는 독립 개체 수가 아니다.

**M10은 task preflight READY**다. `x[t+1]=(a·x[t]+b) mod11`, 3-symbol prefix와8-symbol continuation, train275/seen-rule-new-start275/held-out-coefficient440, 총990 affine episode다. 같은 조건의 iid/shuffled family까지2,970 episode와63 control evaluations를 설계 검증했다. source neural model은 fit하지 않았다. history-3 table은 seen-rule 시작값을 전부 완주하지만 held-out coefficient를 하나도 완주하지 못했다. known-form oracle은440개 coefficient test를 완주하며 task solvability를 보인다. oracle은 공식을 이미 아는 privileged reference다. a=10 period2는 lag2-copy로 풀리므로 broad rule learning과 분리한다.

**M11은 acquisition-limited primary/secondary FAIL**다. 실제 s701/10 matched partial null/3-slot shift를6 K11 maps와3 families에서216 affine heads로 비교했다. 각 head는48 observations,539 nominal affine scalars와96 train-only scaler moments,2,200 train rows다. recurrence/encoder/template는 학습하지 않았다. primary는 period2를 제외한 a=7/8/9의330 held-out episodes다.

| endpoint | discovery | confirmation | 사전 gate |
| --- | ---: | ---: | ---: |
| training teacher accuracy | 1,277/6,600 =19.35% | 1,245/6,600 =18.86% | ≥50% |
| held-out first next | 131/990 =13.23% | 138/990 =13.94% | ≥25% |
| target-free8-symbol completion | 5/990 =0.51% | 10/990 =1.01% | ≥10% |
| real−shift completion | +0.505pp | +1.010pp | ≥5pp |
| strictly above all10 nulls | false | false | true와 별도 material gap |

학습 습득 자체가 기준 미달이므로 결과를 “규칙은 학습했으나 일반화만 실패”로 표현할 수 없다. `summary.json`의 FAIL 및 all primary gates false를 실제 blob으로 읽었다. 역사적216 independent refits/trajectory replay는 원본 archive의 검증 보고이며 이번 감사에서 재수행한 것은 아니다.

**새 감사에서 확인한 추가 사항:** 현재 DrosoMem은 출력층 학습만 존재하는 코드로 끝나지 않는다. 이후/병행 내부 학습 연구를 구현했지만 기능적 학습 성공은 별도다.

- Phase4: supervised KC→MBON 가소성432 runs; sign·edge support·post-row absolute budget 보존. real200-digit paired 비교24건 중 개선5/동률3/악화16. 가중치 변화의 구현이 일반적 학습 성공이나 dopamine physiology를 증명하지 않는다.
- Phase5: scalar reward와 eligibility, frozen supervised warm-start policy. real trace는 frozen 대비24 pair 모두 악화. refit readout은 일부 상태 정보를 회복했지만 reward-learning 우위를 확립하지 않았다. 업데이트에는 global classifier feedback이 있고 DAN 자체가 reward를 전송하지 않는다.
- Phase5B: γ1/pedc의 PPL101/MBON11 매핑, actual KC/DAN spike-gated LTD 구현과 causal controls 통과. s701179/s702134 candidate KC→MBON11 edges 중 paired conditioning14/11 변경. 그러나 isolated baseline MBON11 probe가0이므로 기능 반응 gate FAIL, pi recall 평균3→2. 시간 상수·학습률은 모델 선택이며 생리적 재현이 아니다.
- 후속5: 새 KC maps3개×부분 회로2개×5 arms=30 cases,6 paired cases baseline CS+가 모두0;24 controls 연결 변화0. fixed decoder gain0pp로≥5pp gate FAIL. causal/local routing 성공과 behavioral/functional learning 성공을 구분했다.
- whole-brain 규모 feasibility와 일부 source-pathway sensitivity는 실행 근거가 있지만 whole-brain/real-wiring 우위를 확립하지 않았다. partial→full 규모 증가는 이전 실패를 취소하지 않는다.

따라서 DrosoLearn의 새 기여는 “처음 내부 학습을 만들었다”가 아니다. **수량/보상 과제를 대상으로 출력층 학습·내부 변화·일반화를 별도 측정하고, 고정/가소성/재배선/일반 ANN 대조를 사전에 정의한다**는 다른 연구 질문이다. 기존 실패가 있음을 출발점으로 삼고 동일 과제의 해결 가능성, nuisance cues, acquisition gate부터 확인한다.

## 7. provenance 및 재현 계획

1. 복사·파생 전에 source repository/commit/path/원본 파일 SHA256를 기록한다. 새 `data/`에는 사용한 최소 파생 자료와 원본 provenance·license notice를 같이 둔다.
2. 파생 edge의 역할 선택 조건, 원본 threshold≥5/autapse 제거, 전체 node ordering, KC→MBON count1,474/contact10,468을 manifest로 확인한다. 학습 중 count와 plastic efficacy를 같은 필드에 덮어쓰지 않는다.
3. stimulus seed, nuisance strata, train/validation/test split hashes, config, protocol hash와 구현 commit을 run 시작 전에 봉인한다. runtime environment/seed/learned parameter budget/reproduce command를 결과별 남긴다.
4. baseline projections와 readout hashes, plastic synapse 변화, trial activations를 별도 저장한다. training reward·greedy choice·probe accuracy를 서로 바꾸어 보고하지 않는다.
5. 모든 condition/seed를 보존하고 최적 seed만 고르지 않는다. smoke 실행과 본실험을 구분하고 failed acquisition이 generalization pass로 대체되지 않게 한다.
6. 역사적 DrosoMem records는 immutable 참고 자료다. 본 연구의 results는 신규 DrosoLearn에 생성하고 과거 M11/Phase4/5 결과·해시·판정은 변경하지 않는다.

## 8. 감사의 해석 한계

이 문서는 원본 코드·문서·선택된 canonical 결과·파생 데이터의 정적/구조 검증이다. 전체15,091,983 edge source 재추출, 모든 원본 실험 replay, physiological cell parameters 검증, LC11 stimulus path 구현, 원본 생물학적 사실의 독립 재현은 수행하지 않았다. 문헌에서 얻는 생물학적 근거는 `literature-review.md`에서 별도로 다룬다. 초기에 사용할 one-hop projection의 차이를 솔직히 기록하고, 도달한 claim level을 모델 discriminability/learning/generalization/behavioral similarity로 분리한다.
