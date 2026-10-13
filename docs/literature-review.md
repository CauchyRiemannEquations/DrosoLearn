# DrosoLearn 선행연구 검토

검토일: 2026-10-14 (Asia/Seoul). 이 문서는 실제 논문의 관찰, 논문의 계산 모델, DrosoLearn의 새 가설을 구분한다. 문헌의 보고 결과는 DrosoLearn에서 재현한 결과가 아니다. 사람 대상 신규 실험과 살아 있는 초파리 실험은 초기 연구 범위에 포함하지 않는다.

## 지정 논문 확인

| 문헌과 확인 범위 | 직접 확인된 근거 | DrosoLearn에 적용할 경계 |
|---|---|---|
| **Bengochea et al. (2023)**, *Numerical discrimination in Drosophila melanogaster*, Cell Reports 42(7), 112772. DOI: [10.1016/j.celrep.2023.112772](https://doi.org/10.1016/j.celrep.2023.112772). [공개 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10442639/)의 Results·STAR Methods·data statement 확인. | 실제 초파리의 더 큰 수량에 대한 자발적 선호, 수량 비율 의존성, 시각적 단서 통제 조건, 당 용액 연합 후 선호 변화, LC11 기능 억제 후 수행 저하를 보고했다. | LC11 관여는 전체 수량 회로, LC11→KC 입력 경로, 초파리의 수학 기호 이해를 입증하지 않는다. KC에 수량을 직접 주는 과제는 추상적 벤치마크다. |
| **Huang et al. (2024)**, *Dopamine-mediated interactions between short- and long-term memory dynamics*, Nature 634, 1141–1149. DOI: [10.1038/s41586-024-07819-w](https://www.nature.com/articles/s41586-024-07819-w). 본문·Methods·data/code statement 확인. | 후각 조건화 중 DAN·MBON 전압 활동을 측정하고, 단기 기억 구획의 MBON 피드백이 다른 구획의 장기 기억 가소성을 조절하는 근거와 계산 모델을 제시했다. | 시각 수량 학습의 직접 증거가 아니다. 보상 scalar로 선택된 내부 가중치를 갱신하는 구현은 별도의 단순화된 계산 규칙이다. |
| **Chen & Li (2014)**, *Association between individual differences in non-symbolic number acuity and math performance: A meta-analysis*, Acta Psychologica 148, 163–172. DOI: [10.1016/j.actpsy.2014.01.016](https://doi.org/10.1016/j.actpsy.2014.01.016), [PMID 24583622](https://pubmed.ncbi.nlm.nih.gov/24583622/). 서지와 PubMed 초록 확인; 유료 본문 전체는 확인하지 않았다. | 비기호 수량 정확성과 수학 수행의 횡단 연구 36표본, N=4,705에서 r=0.20, 95% CI [0.14, 0.26]을 보고했다. 종단 관련성도 검토했다. | 관련성은 인과, 훈련의 교육 효과, 종간 동일 메커니즘을 증명하지 않는다. 표본별 원자료·분석 코드의 공개 저장소는 이번 검토에서 확인하지 못했다. |

### Bengochea 연구에서 첫 실험으로 가져올 구조

1 대 3 조건의 보상 학습은 적은 수량과 당 용액을 연결하는 단회 3분 Pavlovian 조건화, 2시간 뒤 선호 평가였다. 해당 학습군은 큰 수량 선호가 감소하여 유의한 선호가 없는 상태가 되었다. 이를 모든 수량쌍에서 적은 수량으로 완전히 반전했다고 요약하지 않는다. 단서 통제는 물체 크기, 총 어두운 면적, 표시 범위, 밀도, 배열 등을 여러 조건으로 조작했다. 원 논문의 행동 지표는 영역별 체류/통과 분포에 기반한 선호 지수이므로 계산 모델의 이진 선택 정확도와 바로 같은 값으로 취급하지 않는다. [원 논문 Results, Figure 5 및 STAR Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10442639/)

**DrosoLearn의 제안:** 처음에는 입력 부호화를 명시한 수량 연합/구별 과제로 학습 전후 변화를 확인한다. 시각 입력을 사용하는 확장에서는 여러 단서 통제군, 단서와 수량의 일치/불일치 평가, 표면 단서 기준 모델을 함께 둔다. 추상 입력 실험으로 LC11 회로를 재현했다고 주장하지 않는다.

### Huang 연구와 내부 학습 규칙의 차이

논문의 계산 모델은 Janelia **hemibrain v1.2.1**의 연결 존재를 제약으로 사용한다. 5개 미만 시냅스 연결은 제외하고, 나머지 강도는 시냅스 수를 그대로 사용하지 않고 측정 활동에 적합했다. KC→MBON의 양방향 anti-Hebbian 규칙은 CS–US 순서에 의존한다. 빠른 구획과 느린 구획의 피드백을 포함하며, 연속 미분방정식과 가정을 둔 재귀 근사식을 구분한다. FlyWire v783 연결 수에 임의 보상 갱신을 적용하면 이 모델의 충실한 복제가 아니다. [원 논문 Methods: Connectome- and physiology-constrained computational model](https://www.nature.com/articles/s41586-024-07819-w)

**DrosoLearn의 제안:** Phase 2에서 KC→MBON mask, 허용 가중치 범위, 보상/처벌 부호, 갱신 시점, 감쇠와 eligibility trace를 명시한다. DAN 회로가 없거나 구획별 역할을 검증하지 못했다면 조절 신호를 ‘인공적인 보상 신호’로 기록한다. 내부 가중치 변화, 출력층 갱신, 일시적 활동 변화를 별도 로그로 보존한다.

## 추가 일차 연구와 설계상 함의

| 문헌 | 확인된 연구 내용 | 새 연구에 필요한 검증 |
|---|---|---|
| Hige et al. (2015), *Heterosynaptic Plasticity Underlies Aversive Olfactory Learning in Drosophila*, Neuron 88(5), 985–998. [DOI 10.1016/j.neuron.2015.11.003](https://doi.org/10.1016/j.neuron.2015.11.003), [공개 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/) | 냄새와 특정 DAN 활성 연합 후 냄새 특이적인 MBON 입력 시냅스 억제, 시간 순서·구획 의존성, sparse odor code 중첩과 기억 일반화의 관계를 보고했다. | 가소성을 전체 시냅스에 일괄 허용하지 않는다. 입력 코드 중첩을 독립 변수로 조작하여 일반화와 간섭의 관계를 시험하는 것은 DrosoLearn의 새 가설이다. |
| Aso & Rubin (2016), *Dopaminergic neurons write and update memories with cell-type-specific rules*, eLife 5:e16135. [DOI 10.7554/eLife.16135](https://doi.org/10.7554/eLife.16135), [PMID 27441388](https://pubmed.ncbi.nlm.nih.gov/27441388/) | DAN 유형과 냄새 대비 활성 시점에 따라 학습 요구량, 기억 감쇠와 수정 특성이 달랐다. 개별 DAN의 optogenetic 자극을 사용했다. | 모든 DAN을 동일한 signed reward-prediction-error 뉴런으로 간주하지 않는다. 학습률·감쇠·시간 순서의 민감도 분석을 별도로 계획한다. |
| DeWind et al. (2015), *Modeling the approximate number system to quantify the contribution of visual stimulus features*, Cognition 142, 247–265. [DOI 10.1016/j.cognition.2015.05.016](https://doi.org/10.1016/j.cognition.2015.05.016), [공개 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4831213/) | 수량, 크기와 간격 차원이 수량 선택에 미치는 영향을 회귀로 구분하고, 비수량 단서를 무시한 Weber fraction 추정의 문제를 다뤘다. | 수량·개체면적·총면적·밀도·표시 범위를 모두 동시에 독립 고정할 수 있다고 선언하지 않는다. 공변량 및 일치/불일치 평가가 필요하다. |
| Halberda, Mazzocco & Feigenson (2008), *Individual differences in non-verbal number acuity correlate with maths achievement*, Nature 455, 665–668. [DOI 10.1038/nature07246](https://doi.org/10.1038/nature07246), [PMID 18776888](https://pubmed.ncbi.nlm.nih.gov/18776888/) | 14세 참가자의 짧은 점 배열 비교 수행과 과거 수학 성취 간 관련성을 연구했다. 비율별 정확도와 psychophysical Weber fraction이 비교 후보다. | 수량 민감도가 이후 수학 학습을 원인적으로 결정한다고 해석하지 않는다. 참가자 원자료 공개는 확인하지 못했다. |
| Gilmore et al. (2013), *Individual Differences in Inhibitory Control, Not Non-Verbal Number Acuity, Correlate with Mathematics Achievement*, PLOS ONE 8(6):e67374. [원 논문 및 DOI 10.1371/journal.pone.0067374](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0067374) | 시각 단서 불일치 시행의 억제 요구가 수량 비교와 수학 성취의 관련성에 기여할 수 있음을 보여줬다. | cue congruency별 오류를 분리한다. ‘비율 효과가 있으면 ANS’라는 단일 판정은 사용하지 않는다. |
| Honoré & Noël (2016), *Improving Preschoolers’ Arithmetic through Number Magnitude Training: The Impact of Non-Symbolic and Symbolic Training*, PLOS ONE 11(11):e0166685. [원 논문 및 DOI 10.1371/journal.pone.0166685](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0166685) | 56명 유아의 기호·비기호·통제 훈련을 비교했다. 수량 처리 향상과 산술 전이를 구분했으며 기호 훈련의 산술 향상이 더 컸다. | 비기호 수량 향상이 기호 산술 향상으로 자동 전이된다는 가설에 반례를 포함한다. 모델에서 기호를 주는 것은 인공 입력이다. |
| Kattner, Cox & Green (2016), *Transfer in Rule-Based Category Learning Depends on the Training Task*, PLOS ONE 11(10):e0165260. [원 논문 및 DOI 10.1371/journal.pone.0165260](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165260) | 범주 구별과 범주 식별 훈련이 새로운 범주 구별로 전이되는 양상이 달랐다. | 수학/수량 연구 자체는 아니다. 습득과 전이의 별도 평가 및 과제 구조 비교의 방법론적 사례로 사용한다. |

## 공개 자료 접근성 및 재사용 결정

이번 최소 실행은 아래 외부 행동 자료나 MATLAB 코드를 가져오지 않는다. 공개 링크 확인과 실제 파일의 해시 검증·스키마 검증·재분석은 구분한다. 사후 인간 비교를 실행할 때에는 파일별 출처, DOI, 내려받은 날짜, SHA-256, 라이선스, 포함·제외 기준을 새 manifest에 고정한다.

| 자료 | 확인한 접근성 / 조건 | 현재 상태 |
|---|---|---|
| Bengochea 행동 자료·분석 코드 | [Zenodo 8052276](https://zenodo.org/records/8052276), DOI 10.5281/zenodo.8052276. BENGOCHEA_etal_2023.zip, 931,633,676 bytes, 공개 metadata의 MD5 b4bda9565cb8ad0377e2f3db44418d40. REST metadata에서 CC BY 4.0 확인. | metadata 확인; 압축파일 미다운로드·미재분석. 내부 코드별 추가 라이선스는 사용 전 확인. |
| Huang 신경 전압 trace | [Zenodo 10998457](https://zenodo.org/records/10998457), DOI 10.5281/zenodo.10998457. Figure1–5.zip 공개 metadata 및 CC BY 4.0 확인. 논문의 원시 voltage imaging은 reasonable request 조건이며 공개 trace와 구분된다. | metadata 확인; raw image 공개를 가정하지 않음. |
| Huang 계산 모델 | [저자 GitHub](https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model). README에 MATLAB v2020b/v2022a, recurrent equation fitting, GPL-3.0-or-later 명시. | 별도 코드 미복사. 재사용 시 commit·원문 라이선스·파생물 조건 보존 필요. |
| DeWind 자극/모형 | 공개 본문과 supplementary code 참조 확인. 인간 시행별 원자료의 명시적 공개 archive는 이번 확인에서 찾지 못함. | 방법을 인용하며 원자료가 있다고 가정하지 않음. |
| Honoré & Noël 인간 자료 | 논문 Supporting Information의 S1–S6 Data XLSX 목록 공개: [수량 비교 S1 Data](https://doi.org/10.1371/journal.pone.0166685.s002), [덧셈 S6 Data](https://doi.org/10.1371/journal.pone.0166685.s007). 논문은 CC BY. | 자료 목록 확인; 파일 내부 granularity·결측·정확한 라이선스 재확인은 Phase 4 작업. |
| Kattner et al. 인간 범주 전이 자료 | [S1 Data ZIP](https://doi.org/10.1371/journal.pone.0165260.s001), CategoryTransfer_RawData.Rdata와 ReadMe.txt가 있다고 논문이 명시. 논문은 CC BY. | 목록 확인; 수량 자료로 오분류하지 않음. |

## 첫 연구의 선택 근거와 미해결점

첫 실험은 **인공적으로 부호화한 수량에 대한 보상 선택/구별 학습**으로 제한하는 것이 현실적이다. 고정 구조+출력 학습, 선택된 내부 가소성, 재배선+동일 학습, 일반 인공망을 비교할 인터페이스를 만든다. LC11부터 KC까지 검증된 입력 경로를 만들기 전에는 생물학적 시각 수량 인지 복제라고 부르지 않는다.

정확한 수량 scalar, one-hot 또는 tuning code를 입력하면 수량의 추출은 이미 인코더가 제공한다. multiplicative noise나 log tuning을 넣고 비율 의존성이 나타나면 그것이 입력 가정에서 생겼는지 인코더 대조군으로 확인해야 한다. 사람이 물체를 보고 수량을 추정하는 현상과 추상 수량 값에 대한 모델의 연합 학습을 분리한다.

시각 입력 확장, DAN compartment mapping, 생리학적 시간 단위, 다중 기억 시간척도, 인간 원자료의 비교 가능성은 후속 과제다. 이 문헌 검토만으로 같은 학습 메커니즘이나 고등학교 수학 교육 효과를 주장할 근거는 없다. 정량 비교 체계는 [human-learning-comparison.md](human-learning-comparison.md)에 정의한다.
