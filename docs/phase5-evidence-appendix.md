# Phase 5 기계 판독 증거의 읽기용 부록

기존 결과를 종합한 문서다. 새 학습·인간 실험·효과 검정은 0회다.

## 1. 수량과 단순 규칙을 경험으로 학습하는가?

- **model_scope** (bounded, Level 0): 공개 v783 부분집합의 KC→MBON 한 단계 계산 경로를 사용한다.
  해석 범위: 시각 LC11·DAN 동역학·전체 재귀 뇌 회로는 구현하지 않았다.
  근거: `docs/drosomem-audit.json/reuse_plan/first_model` = `"New independently implemented one-hop KC to MBON projection"`
  근거: `docs/drosomem-audit.json/reuse_plan/data_subset` = `"flywire_783_mb_left_kc512_s701"`
  근거: `docs/drosomem-audit.json/reuse_plan/ignore_for_initial_model` = `["DAN/APL dynamics","Recurrent whole-graph dynamics","Visual LC11 circuit"]`
- **quantity_learning** (bounded, Level 2): 순서가 제공된 인공 수량 입력에서 학습과 표현 의존성이 확인됐다.
  해석 범위: 수량을 시각적으로 추출하거나 초파리의 수학 이해를 재현한 결과가 아니다.
  근거: `results/encoding_v1/summary.json/family_verdicts/1/qualified_seed_count` = `10`
  근거: `results/encoding_v1/summary.json/family_verdicts/1/criterion_met` = `true`
- **simple_reward_learning** (supported, Level 2): 별도 단순 선형 대조도 보상으로 연습 과제를 습득했다.
  해석 범위: 단순한 표현과 학습 규칙으로 설명할 대안이며 실제 배선의 고유 우위를 뜻하지 않는다.
  근거: `results/reward_linear_v1/summary.json/hypotheses/H_R` = `{"acquisition_count":10,"criterion_met":true,"mean_iid_reward_gap":0.48865709781276045,"positive_seed_count":10}`
  근거: `results/stochastic_feedback_linear_v1/summary.json/primary_hypothesis/probabilistic_acquisition_count` = `10`
- **pattern_acquisition** (not_supported, Level 2): 인공 패턴의 주 가소성 모델은 연습 습득 0/10으로 사전 기준을 충족하지 못했다.
  해석 범위: 이 표현·400시행·모델의 결과이며 생물학적 불가능성이나 모든 학습법의 실패가 아니다.
  근거: `results/pattern_rule_v1/summary.json/H_acquisition` = `{"condition":"plastic","supported":false}`
  근거: `results/pattern_rule_v1/summary.json/families/1/acquisition_pass_seeds` = `0`
  근거: `results/pattern_rule_v1/summary.json/families/8/acquisition_pass_seeds` = `10`

## 2. 학습 중 활성·내부·출력은 어떻게 바뀌는가?

- **internal_intervention** (supported, Level 2): 같은 첫 학습 상태의 반전 개입에서 내부 갱신 허용이 결과를 바꿨다.
  해석 범위: 전체 정책 경로 효과이며 추가 갱신 자원·후속 보상 경로까지 같거나 생물학적 규칙이라는 주장은 아니다.
  근거: `results/learning_ablation_v1/summary.json/primary_hypotheses/H_I/criterion_met` = `true`
  근거: `results/learning_ablation_v1/summary.json/primary_hypotheses/H_I/mean_effect` = `0.2611788773506773`
  근거: `results/learning_ablation_v1/summary.json/primary_hypotheses/H_H/mean_effect` = `0.3693634218935594`
- **state_decomposition** (supported, Level 0): 저장된 출력·내부 특징·교차항을 별도 산술로 분해했다.
  해석 범위: 정적 재조합을 실제 학습 분기나 생물학적 원인의 기여율로 읽지 않는다.
  근거: `results/state_decomposition_v1/summary.json/arithmetic_identity_passed` = `true`
  근거: `results/state_decomposition_v1/summary.json/cases` = `204800`
  근거: `results/state_decomposition_v1/summary.json/learning_updates` = `0`
- **cross_cue_credit** (supported, Level 2): 보상 규칙이 고정된 별도 과제에서 다른 문제로 번지는 출력 갱신을 분리하자 최종 수행이 높아졌다.
  해석 범위: 추가 고정 계수와 행렬 풀이가 필요한 계산적 대조이며 원 인간 과제 실패의 전체 원인이나 신경 기작은 아니다.
  근거: `results/shell_credit_v1/summary.json/H_credit` = `{"effect_criterion_met":true,"interpretation":"stationary computational cross-cue update intervention; not the cause of the historical human-task failure","isolated_acquisition_gate":true,"mean_endpoint_effect":0.2238343002341348,"n_seeds":5,"positive_seeds":5,"supported":true}`
  근거: `results/shell_credit_v1/summary.json/acquisition/0/mean_endpoint_accuracy` = `0.6190182634950381`
  근거: `results/shell_credit_v1/summary.json/acquisition/1/mean_endpoint_accuracy` = `0.8428525637291729`
- **head_scaling** (bounded, Level 2): 같은 이력에서 출력 계수 축소가 반전 결과를 바꾼 계산 효과를 보존한다.
  해석 범위: 즉시 경계 변화와 이후 학습 효과를 구분하며 자연 망각·본질적 학습 속도·교육 효과로 확대하지 않는다.
  근거: `results/head_attenuation_v1/summary.json/primary_hypothesis/criterion_met` = `true`
  근거: `results/head_attenuation_v1/summary.json/primary_hypothesis/mean_endpoint_benefit` = `0.37025106468720415`

## 3. 미학습 사례·규칙에 일반화하는가?

- **scalar_transfer** (bounded, Level 3): 순서를 가진 인공 수량 표현에서는 별도 선형 학습자도 습득 후 새 수량쌍 기준을 통과했다.
  해석 범위: 입력이 순서를 제공한 제한적 전이이며 추상적 수학 규칙 발견이나 커넥톰 고유 성능의 증거가 아니다.
  근거: `results/reward_linear_v1/summary.json/hypotheses/H_G` = `{"criterion_met":true,"joint_identity_transfer_count":10}`
- **linear_explanation** (bounded, Level 0): 기존 신경망 선택 확률은 단순 선형 참조로 가까이 설명되는 범위가 있었다.
  해석 범위: 참조는 원 모델 점수를 받았다. 독립 보상 학습이나 같은 내부 기작의 증거가 아니다.
  근거: `results/linearity_v1/summary.json/family_verdicts/1/mean_affine_probability_mae` = `0.0038632548605362907`
  근거: `results/linearity_v1/summary.json/family_verdicts/1/criterion_met` = `true`
- **pattern_generalization** (not_assessable, Level 3): 주 패턴 모델은 습득 gate 미달로 세 새 입력 집합의 일반화 판정을 보류한다.
  해석 범위: 습득을 실패한 수치를 일반화 성공 또는 학습 후 일반화 실패로 대체하지 않는다.
  근거: `results/pattern_rule_v1/summary.json/H_generalization` = `[{"joint_acquisition_and_ood_pass_seeds":0,"set":"new_alphabet","status":"not_assessable_without_acquisition","supported":false},{"joint_acquisition_and_ood_pass_seeds":0,"set":"new_alphabet_phase","status":"not_assessable_without_acquisition","supported":false},{"joint_acquisition_and_ood_pass_seeds":0,"set":"new_length","status":"not_assessable_without_acquisition","supported":false}]`
- **memorization_control** (not_supported, Level 3): 네 입력을 보상으로 외운 Q는 습득했지만 미학습 입력 전이를 통과하지 못했다.
  해석 범위: 새 입력에서 p=.5인 설계상의 암기 대조다. 다른 일반화 모델의 불가능성은 아니다.
  근거: `results/pattern_rule_v1/summary.json/families/8/acquisition_supported` = `true`
  근거: `results/pattern_rule_v1/summary.json/families/8/ood` = `[{"joint_acquisition_and_ood_pass_seeds":0,"set":"new_alphabet","status":"not_supported","supported":false},{"joint_acquisition_and_ood_pass_seeds":0,"set":"new_alphabet_phase","status":"not_supported","supported":false},{"joint_acquisition_and_ood_pass_seeds":0,"set":"new_length","status":"not_supported","supported":false}]`
  근거: `results/pattern_rule_v1/summary.json/shortcuts/unique_strings` = `16`
  근거: `results/pattern_rule_v1/summary.json/shortcuts/phase_is_balanced_generalization_test` = `false`
- **drosomem_failure_retained** (not_supported, Level 0): 기존 독립 DrosoMem M11의 습득·전이 기준 미달을 그대로 계승한다.
  해석 범위: 이번 종합에서 과거 실험 전체를 재실행한 것이 아니라 고정 원 요약을 감사했다.
  근거: `docs/drosomem-audit.json/repository/commit` = `"7fb156c193c7a499fc2266a164b705729abbab21"`
  근거: `docs/drosomem-audit.json/repository/upstream_modified` = `false`
  근거: `docs/drosomem-audit.json/M11_archived_summary/git_blob_sha256` = `"661bc66ee2a655f8d336747f64c139de2a38b4a149bd6d1fb2cef6fba3e82720"`
  근거: `docs/drosomem-audit.json/M11_archived_summary/content/primary_pass` = `false`

## 4. 반복·반전·간섭·순서·유지는 어떤 영향을 주는가?

- **reversal_history** (supported, Level 2): 단순 모델도 이전 반응의 영향을 받아 반전과 처음부터 학습의 결과가 달랐다.
  해석 범위: 생물학적 기억·모델 고유 학습 속도·인간과 동일 원리의 증거가 아니다.
  근거: `results/bandit_reversal_v1/summary.json/primary_hypothesis` = `{"condition":"linear_normalized","criterion_met":true,"fresh_acquisition_count":10,"high_prefix_acquisition_count":10,"mean_endpoint_cost":0.5245449722473937,"n_seeds":10,"name":"H_B","positive_seed_count":10,"sd_endpoint_cost":0.24080895916800138}`
- **contextual_interference** (not_assessable, Level 2): 주 가소성 모델의 상충 과제 뒤 유지 손실의 사전 판정은 not_assessable이다.
  해석 범위: 각 과제 습득과 공동학습·충돌 입력 기준을 먼저 요구한다. gate가 닫히면 기술적 수치만 남기며 효과를 확증하지 않는다.
  근거: `results/interference_order_v1/summary.json/qualification_supported` = `false`
  근거: `results/interference_order_v1/summary.json/H_interference` = `{"ci90":[-0.02514287093724811,-0.012845094325649651],"mean_retention_loss":-0.019196562148705503,"status":"not_assessable","supported":false}`
  근거: `results/interference_order_v1/summary.json/families/1/qualification_pass_seeds` = `0`
- **contextual_order** (not_assessable, Level 2): 주 가소성 모델의 A→B와 B→A의 학습 순서의 사전 판정은 not_assessable이다.
  해석 범위: 각 과제 습득과 공동학습·충돌 입력 기준을 먼저 요구한다. gate가 닫히면 기술적 수치만 남기며 효과를 확증하지 않는다.
  근거: `results/interference_order_v1/summary.json/qualification_supported` = `false`
  근거: `results/interference_order_v1/summary.json/H_order` = `{"ci90":[-0.006545670363546001,0.005929160612490878],"mean_order_delta":-0.0003185509155236188,"status":"not_assessable","supported":false}`
  근거: `results/interference_order_v1/summary.json/families/1/qualification_pass_seeds` = `0`
- **context_joint_control** (bounded, Level 2): 맥락별 Q와 신경망의 개별·공동학습 성적을 모두 보존한다.
  해석 범위: 네 일관 문제만 맞혀 전체 평균이 높아지는 지름길을 충돌 문제별 기준으로 막았다.
  근거: `results/interference_order_v1/summary.json/analytic_context_blind_witness` = `{"conflicting_id_accuracy":0.5,"consistent_id_accuracy":1.0,"each_class_accuracy":0.8333333333333334,"overall_accuracy":0.8333333333333334,"passes_conflict_qualification":false,"passes_old_mean_and_class_gate":true}`
  근거: `results/interference_order_v1/summary.json/families/8/qualification_pass_seeds` = `10`
- **idle_scope** (untested, Level 0): 갱신 없는 대기 복제는 상태 불변 점검이며 자연 시간 망각은 미검증이다.
  해석 범위: 시간 상수·자연 감쇠를 구현하지 않은 모형에서 망각의 생물학적 부재를 주장하지 않는다.
  근거: `results/interference_order_v1/summary.json/interpretation/natural_time_forgetting_tested` = `false`
  근거: `results/interference_order_v1/summary.json/interpretation/idle_is_state_copy_invariant` = `true`
  근거: `results/interference_order_v1/summary.json/counts/idle_feedback_trials` = `0`

## 5. 인간 기초 학습과 정량 비교할 공통점과 차이는 무엇인가?

- **direct_human_similarity** (not_supported, Level 4): 실제 공개 조개 과제의 직접 비교에서 주 모델의 등록된 인간 행동 유사성은 지지되지 않았다.
  해석 범위: 한 과제·모델·혼합 코호트의 결과다. 모든 인간/초파리 공통 원리의 부재나 인간 학습 실패를 뜻하지 않는다.
  근거: `results/human_shell_rng_v2/summary.json/H_similarity` = `{"claim":"behavioral_equivalence_only","margin":0.1,"notice":"same three summaries within operational margin; not common mechanism or learning proof","primary_condition":"plastic","supported":false}`
  근거: `results/human_shell_rng_v2/summary.json/learning_qualification` = `{"gated_learning_pattern_supported":false,"human_gain_observed":true,"human_gain_status":"observed","human_initial_gain_ci90":[0.11494252873563221,0.25287356321839083],"model_gain_observed":false,"model_initial_gain_ci90":[-0.05982758620689659,0.07241379310344824],"notice":"human pretraining/experience prevents interpreting absent gain as absent learning"}`
  근거: `results/human_shell_rng_v2/summary.json/behavior_family_comparisons/1/metrics/2` = `{"ci90":[-0.4042337164750958,-0.22871647509578547],"difference":-0.31532567049808424,"equivalent":false,"human_mean":0.3122605363984674,"metric":"reversal_recovery","model_mean":-0.0030651340996168588,"status":"outside_margin"}`
- **human_prediction** (not_supported, Level 4): 다음 인간 선택 예측은 균등보다 나았지만 편향·반복 대조를 모두 넘는 기준은 미달했다.
  해석 범위: 세 예측 대조와 자율 행동 유사성을 구분한다. 사람의 사전 연습·시각·운동·실제 시간은 재현하지 않았다.
  근거: `results/human_shell_rng_v2/summary.json/H_prediction` = `{"contrasts":[{"baseline":"uniform","ci90":[0.0138171543138575,0.03997941299582021],"mean_improvement_nats":0.027100757068630524,"passed":true},{"baseline":"bias","ci90":[-0.04015869927073437,-0.006684220796720543],"mean_improvement_nats":-0.022574043787609247,"passed":false},{"baseline":"persistence","ci90":[-0.08659626474177753,0.006525056187373817],"mean_improvement_nats":-0.04275710430798365,"passed":false}],"minimum_improvement_nats":0.01,"primary_condition":"plastic","supported":false}`
- **rng_design_history** (design_limited, Level 0): 첫 인간 비교의 자율 난수 주소 중복과 별도 수정 이력을 보존한다.
  해석 범위: 산술 검산 통과와 추론 설계 타당성은 별개다. 첫 자율 결과를 새 눈가림 증거로 합치지 않는다.
  근거: `results/human_shell_rng_v2/verification.json/original_autonomous_design_limitation_preserved` = `true`
  근거: `results/human_shell_rng_v2/verification.json/new_calibration_observations` = `0`
  근거: `results/human_shell_rng_v2/verification.json/conditional_arrays_byte_identical` = `true`
- **behavior_metric_limits** (supported, Level 0): 같은 세 점수의 재생 반례와 보상 부호 개입은 점수만으로 학습 기작을 식별할 수 없음을 보인다.
  해석 범위: 계획된 구성적 반례이며 공정한 학습 경쟁자·새 인간 전략·인간 유사성 확증이 아니다.
  근거: `results/shell_behavior_v1/summary.json/metric_identification` = `{"notice":"Exact equality was planned by construction; it limits mechanism identification and is not a new human learning effect.","planned_counterexample_verified":true,"scripted_policy_is_eligible_fair_learning_control":false,"scripted_policy_online_updates":0,"scripted_policy_uses_precomputed_action_tape":true,"three_summaries_uniquely_identify_reward_learning":false}`
  근거: `results/shell_behavior_v1/summary.json/reward_intervention` = `{"disagreement_fraction":0.36320641762452105,"evaluated_choices":50112,"fixed_tape_choice_change":0,"notice":"Reward signs reversed in a synthetic intervention; original human rewards and original comparison are unchanged.","q_vs_fixed_tape_disagreeing_choices":18201}`
- **human_source_gate** (bounded, Level 0): 다른 공개 카드 자료의 25개 원본 구조 검증과 역사적 실행판의 미확인을 구분한다.
  해석 범위: 닫힌 과거 자료 조건은 조개 자료 비교가 완료돼도 열리지 않는다.
  근거: `results/human_task_structure_v1/audit.json/gate` = `{"direct_human_model_similarity_allowed":false,"exact_historical_task_execution_version_verified":false,"source_coded_descriptive_metric_allowed":true}`
  근거: `results/phase4_synthesis_v1/synthesis.json/verified_original_human_records` = `25`
  근거: `results/phase4_synthesis_v1/synthesis.json/unverified_original_human_records_retained` = `49`
- **time_mixture_counterexample** (supported, Level 0): 시간별 선택 성향과 보상 확률이 함께 변하는 무학습 반례를 보존한다.
  해석 범위: 피드백별 조건부 차이는 곧 보상의 인과 효과가 아니다. 기존 인간/모델 기록의 실제 원인을 특정한 것은 아니다.
  근거: `results/time_mixture_v1/metrics.json/hypotheses` = `{"constant_p_control_zero":true,"constant_q_control_zero":true,"exact_time_conditional_zero":true,"joint_constant_control_zero":true,"time_mixture_counterexample":true}`
- **biological_education_limits** (untested, Level 0): 공통 신경 기작과 고등 수학교육 효과는 검증하지 않았다.
  해석 범위: 행동 유사성이 나와도 Level 5를 자동 주장할 수 없다. 이번 직접 유사성도 미지지다.
  근거: `results/human_shell_rng_v2/summary.json/mechanism_claim` = `false`
  근거: `results/human_shell_rng_v2/summary.json/educational_claim` = `false`

## 전체 완료 연구

| 연구 | 역할 | 완료 태그 | 검산 |
|---|---|---|---|
| smoke_v1 | initial_artificial_quantity_learning | v0.1.0-smoke | passed |
| reversal_v1 | quantity_reward_reversal_diagnostic | study-reversal-v1 | passed |
| encoding_v1 | input_encoding_diagnostic | study-encoding-v1 | passed |
| spacing_v1 | input_spacing_diagnostic | study-spacing-v1 | passed |
| linearity_v1 | linear_solvability_diagnostic | study-linearity-v1 | passed |
| reward_linear_v1 | reward_rule_linear_control | study-reward-linear-v1 | passed |
| bandit_reversal_v1 | bandit_reward_reversal | study-bandit-reversal-v1 | passed |
| head_attenuation_v1 | output_weight_scale_diagnostic | study-head-attenuation-v1 | passed |
| human_trial_data_audit_v1 | historical_human_source_structure_audit | human-trial-data-audit-v1 | passed |
| human_trial_linkage_v1 | historical_source_row_linkage_audit | human-trial-linkage-v1 | passed |
| human_semantics_v1 | unresolved_historical_field_semantics | human-semantics-v1 | passed |
| human_osf_origin_v1 | historical_osf_provenance | human-osf-origin-v1 | passed |
| human_task_structure_v1 | historical_task_source_gate | human-task-structure-v1 | passed |
| feedback_transition_v1 | descriptive_feedback_transition_not_similarity | feedback-transition-v1 | passed |
| metric_calibration_v1 | metric_identifiability_counterexamples | metric-calibration-v1 | passed |
| stochastic_feedback_linear_v1 | stochastic_linear_learning_control | stochastic-feedback-linear-v1 | passed |
| stochastic_feedback_network_v1 | stochastic_network_learning_comparison | stochastic-feedback-network-v1 | passed |
| learning_ablation_v1 | quantity_learning_internal_output_ablation | learning-ablation-v1 | passed |
| state_decomposition_v1 | quantity_frozen_state_decomposition | state-decomposition-v1 | passed |
| conditional_transition_v1 | conditional_metric_descriptive_identifiability | conditional-transition-v1 | passed |
| time_mixture_v1 | exact_time_mixture_counterexample | time-mixture-v1 | passed |
| phase4_synthesis_v1 | bounded_historical_measurement_synthesis | phase4-synthesis-v1 | passed |
| human_shell_v1 | preserved_rng_design_limited_comparison | human-shell-v1-design-limited | passed |
| human_shell_rng_v2 | repaired_direct_human_model_comparison | human-shell-comparison-v2 | passed |
| shell_credit_v1 | stationary_cross_cue_update_intervention | shell-credit-v1 | passed |
| shell_behavior_v1 | saved_trace_description_and_planned_metric_counterexample | shell-behavior-v1 | passed |
| pattern_rule_v1 | artificial_pattern_acquisition_then_ood | pattern-rule-v1 | passed |
| interference_order_v1 | contextual_curricula_with_acquisition_and_joint_gates | interference-order-v1 | passed |
