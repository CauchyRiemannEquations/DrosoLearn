# 후속 종합의 주장·원값·버전 부록

원 JSON 값은 리터럴 그대로 보존했다. 새로운 평균·효과·성능 순위가 아니다. 한 주장이 여러 지점을 참조하거나 같은 지점이 재사용될 수 있어 독립 증거 80개라는 뜻은 아니다. Level은 주장 수준이며 음성/보류/미검증 수준을 양성 성과로 세지 않는다.

`drosomem_independent_history`의 ‘미지지’는 원 M11의 습득·전이 판정을 가리키며, 원 저장소와 실패 기록의 보존에 실패했다는 뜻이 아니다. 이 부록의 값과 원 판정은 그대로 보존했다.

## 24개 주장

| ID | 질문 | Level | 상태 | 주장 |
|---|---:|---:|---|---|
| [update_separation_with_extra_movement](#update_separation_with_extra_movement) | 2 | 2 | 지지 | 원 입력에서 다른 연습 입력으로의 출력 갱신을 계산적으로 분리한 조건은 원 공유 조건보다 높았고 사전 습득을 통과했다. |
| [identity_label_alone](#identity_label_alone) | 1 | 2 | 미지지 | 입력 이름표를 공급하는 것만으로 사전 학습 개선·습득 기준을 충족하지 못했다. |
| [same_movement_budget](#same_movement_budget) | 2 | 2 | 미지지 | 같은 시행당 출력 이동량 상한의 제한 투영은 충분한 습득 기준에 도달하지 못했다. |
| [partial_spillover_tradeoff](#partial_spillover_tradeoff) | 2 | 2 | 미지지 | 사전 가운데 절충은 두 끝점보다 실용적으로 우수하다는 기준과 충분한 습득 기준을 통과하지 못했다. |
| [signed_spillover_accounting](#signed_spillover_accounting) | 2 | 0 | 기술·산술 | 저장된 변화에서 자기 수정·같은 정답 입력 도움·반대 정답 입력 방해를 따로 기록하고 최종 변화와 대조했다. |
| [observed_selective_suppression](#observed_selective_suppression) | 2 | 2 | 미지지 | 실제 획득 보상으로 관측된 방해를 선택 억제한 주 조건은 고정 개선·충분한 습득 기준을 통과하지 못했다. |
| [normalization_weakens_remaining_effects](#normalization_weakens_remaining_effects) | 2 | 0 | 기술·산술 | 방해를 줄인 뒤 이동 길이를 제한하는 정규화가 자기 수정과 남겨 둔 도움의 실제 점수 효과도 약화시킨 장부를 보존한다. |
| [safe_direction_geometry](#safe_direction_geometry) | 2 | 0 | 기술·산술 | 관측 안전 조건과 같은 단위 이동 상한에서 다른 수정 방향의 점수 영향을 비교할 수 있었다. |
| [balanced_actual_learning](#balanced_actual_learning) | 2 | 2 | 미지지 | 균형 방향의 실제 학습은 실용 개선·충분한 습득·미관측 비악화의 세 기준을 충족하지 못했다. |
| [balanced_actual_pathways](#balanced_actual_pathways) | 2 | 0 | 기술·산술 | 균형 방식의 작은 최종 차이를 부분 관측·완전 관측 구간과 실제 선택·수정 효과로 나눈 장부를 보존한다. |
| [early_isolation_gate](#early_isolation_gate) | 4 | 2 | 미지지 | 초반에 번짐을 막고 네 보상 부호 확인 뒤 균형 방향을 여는 조건은 충분한 습득·실용 개선을 통과하지 못했다. |
| [gated_actual_pathways](#gated_actual_pathways) | 2 | 0 | 기술·산술 | 전환 전 보호와 전환 뒤 자기 수정·도움 변화 및 실제 누적 비용을 따로 기록했다. |
| [gated_state_size_direction](#gated_state_size_direction) | 2 | 0 | 기술·산술 | 양쪽 실제 수정에서 상태·실제 계수·방향의 대칭 배분과 한쪽만 고친 나머지로 후반 차이를 복원했다. |
| [self_priority_actual_learning](#self_priority_actual_learning) | 2 | 2 | 미지지 | 자기 수정을 우선하는 고정 혼합은 기존 전환 균형보다 주 평균이 낮았고 개선·충분한 습득은 미지지였다. |
| [self_priority_help_loss](#self_priority_help_loss) | 2 | 0 | 기술·산술 | 자기 수정 회복과 다른 문제 도움 감소를 실제 크기·상태·방향 및 정규화된 혼합 재료로 나누어 보존했다. |
| [fixed_state_probability_alignment](#fixed_state_probability_alignment) | 2 | 0 | 기술·산술 | 전체 목적 U와 현재 확률 목적 P는 동일한 저장 상태의 단일 수정에서 작은 차이를 보였다. |
| [uniform_probability_actual_learning](#uniform_probability_actual_learning) | 2 | 2 | 미지지 | U/P를 각자의 실제 현재 상태·선택·보상 경로에 적용했지만 실용 개선·충분한 습득·P의 실용 추가효과는 미지지였다. |
| [unknown_training_input_operational_protection](#unknown_training_input_operational_protection) | 4 | 2 | 지지 | U/P의 아직 보상을 모르는 다른 연습 입력의 운영상 보호 기준은 통과했다. |
| [objective_actual_path_decomposition](#objective_actual_path_decomposition) | 2 | 0 | 기술·산술 | U/P의 최종 차이를 상태·실제 크기·방향과 한쪽만 실제 선택·보상을 받은 경로로 정확히 나누었다. |
| [human_similarity_retained](#human_similarity_retained) | 5 | 4 | 미지지 | 기존 공개 인간 조개 과제의 직접 비교에서 등록한 주 행동 유사성과 종합 예측 기준은 미지지였다. |
| [pattern_generalization_gate_retained](#pattern_generalization_gate_retained) | 3 | 3 | 판정 보류 | 원 패턴 주 모델은 습득 자격 미달로 새 입력 일반화 판정이 보류된 상태를 유지한다. |
| [interference_order_gate_retained](#interference_order_gate_retained) | 4 | 2 | 판정 보류 | 기존 맥락별 과제의 개별·공동 습득 자격 미달로 간섭·순서의 확증 판정은 보류다. |
| [drosomem_independent_history](#drosomem_independent_history) | 3 | 0 | 미지지 | 독립 완료 프로젝트 DrosoMem의 M11 습득·전이 미달과 원 저장소 불변을 보존한다. |
| [shared_mechanism_and_education_untested](#shared_mechanism_and_education_untested) | 5 | 5 | 미검증 | 공통 신경 기작과 인간 고등 수학교육 효과는 검증하지 않았으며 이번 종합도 그 주장을 추가하지 않는다. |

## update_separation_with_extra_movement

**원 입력에서 다른 연습 입력으로의 출력 갱신을 계산적으로 분리한 조건은 원 공유 조건보다 높았고 사전 습득을 통과했다.**

상태: 지지. 질문 2, Level 2. 해석 한계: 자기 점수 수정 폭을 맞춘 투영이며 더 큰 전체 가중치 이동과 추가 고정 계산을 썼다. 네 연습 문자열의 결과이고 생물학적 내부 가소성·일반화의 증거가 아니다.

원 파일: [results/representation_update_v1/summary.json](../results/representation_update_v1/summary.json). JSON Pointer: `/H_update`. SHA-256: `c7dbeaa3e1ec358babfa0e3e8bcce90e463da839c585960c2f72aad979301208`.

```json
{
  "supported": true,
  "mean_effect": 0.36246854976211856,
  "positive_seeds": 10,
  "treated_acquisition_pass_seeds": 10,
  "numeric_update_qualified_all": true,
  "mean_spillover_squared_reduction": 2.693120842211743
}
```

원 파일: [results/representation_update_v1/summary.json](../results/representation_update_v1/summary.json). JSON Pointer: `/families/0/mean_final_expected_accuracy`. SHA-256: `c7dbeaa3e1ec358babfa0e3e8bcce90e463da839c585960c2f72aad979301208`.

```json
0.5028849410756661
```

원 파일: [results/representation_update_v1/summary.json](../results/representation_update_v1/summary.json). JSON Pointer: `/families/1/mean_final_expected_accuracy`. SHA-256: `c7dbeaa3e1ec358babfa0e3e8bcce90e463da839c585960c2f72aad979301208`.

```json
0.8653534908377847
```

원 파일: [results/representation_update_v1/summary.json](../results/representation_update_v1/summary.json). JSON Pointer: `/families/1/acquisition_pass_seeds`. SHA-256: `c7dbeaa3e1ec358babfa0e3e8bcce90e463da839c585960c2f72aad979301208`.

```json
10
```


## identity_label_alone

**입력 이름표를 공급하는 것만으로 사전 학습 개선·습득 기준을 충족하지 못했다.**

상태: 미지지. 질문 1, Level 2. 해석 한계: 이름표는 공급된 암기 표현이며 모든 표현의 효과 부재를 입증하지 않는다.

원 파일: [results/representation_update_v1/summary.json](../results/representation_update_v1/summary.json). JSON Pointer: `/H_representation`. SHA-256: `c7dbeaa3e1ec358babfa0e3e8bcce90e463da839c585960c2f72aad979301208`.

```json
{
  "supported": false,
  "mean_effect": 0.04012593982900708,
  "positive_seeds": 10,
  "treated_acquisition_pass_seeds": 0,
  "numeric_update_qualified_all": true,
  "mean_spillover_squared_reduction": null
}
```

원 파일: [results/representation_update_v1/summary.json](../results/representation_update_v1/summary.json). JSON Pointer: `/families/2/mean_final_expected_accuracy`. SHA-256: `c7dbeaa3e1ec358babfa0e3e8bcce90e463da839c585960c2f72aad979301208`.

```json
0.5430108809046732
```

원 파일: [results/representation_update_v1/summary.json](../results/representation_update_v1/summary.json). JSON Pointer: `/families/2/acquisition_pass_seeds`. SHA-256: `c7dbeaa3e1ec358babfa0e3e8bcce90e463da839c585960c2f72aad979301208`.

```json
0
```


## same_movement_budget

**같은 시행당 출력 이동량 상한의 제한 투영은 충분한 습득 기준에 도달하지 못했다.**

상태: 미지지. 질문 2, Level 2. 해석 한계: 더 큰 이동을 허용한 성공과 같은 자원이라고 읽지 않는다. 같은 상한과 실제 총 이동량 동일성은 다르다.

원 파일: [results/update_budget_v1/summary.json](../results/update_budget_v1/summary.json). JSON Pointer: `/H_budget`. SHA-256: `428ca57a337b87209121e320817c39cfca019c99564835d2e5d27a202d7267f9`.

```json
{
  "supported": false,
  "mean_effect": 0.07157582697619291,
  "positive_seeds": 10,
  "treated_acquisition_pass_seeds": 0,
  "numeric_budget_qualified_all": true
}
```

원 파일: [results/update_budget_v1/summary.json](../results/update_budget_v1/summary.json). JSON Pointer: `/families/1/mean_final_expected_accuracy`. SHA-256: `428ca57a337b87209121e320817c39cfca019c99564835d2e5d27a202d7267f9`.

```json
0.8646438246707712
```

원 파일: [results/update_budget_v1/summary.json](../results/update_budget_v1/summary.json). JSON Pointer: `/families/1/acquisition_pass_seeds`. SHA-256: `428ca57a337b87209121e320817c39cfca019c99564835d2e5d27a202d7267f9`.

```json
10
```

원 파일: [results/update_budget_v1/summary.json](../results/update_budget_v1/summary.json). JSON Pointer: `/families/2/mean_final_expected_accuracy`. SHA-256: `428ca57a337b87209121e320817c39cfca019c99564835d2e5d27a202d7267f9`.

```json
0.5746739616923948
```

원 파일: [results/update_budget_v1/summary.json](../results/update_budget_v1/summary.json). JSON Pointer: `/families/2/acquisition_pass_seeds`. SHA-256: `428ca57a337b87209121e320817c39cfca019c99564835d2e5d27a202d7267f9`.

```json
0
```


## partial_spillover_tradeoff

**사전 가운데 절충은 두 끝점보다 실용적으로 우수하다는 기준과 충분한 습득 기준을 통과하지 못했다.**

상태: 미지지. 질문 2, Level 2. 해석 한계: 고정 비율·새 코호트의 결과다. 성공 비율이나 시드를 사후 탐색하지 않았다.

원 파일: [results/spillover_tradeoff_v1/summary.json](../results/spillover_tradeoff_v1/summary.json). JSON Pointer: `/H_tradeoff`. SHA-256: `7de0bf4d5f8e70d1a630c7bd4382674f30b3db3d0860cb9fa45c10efd290e5f8`.

```json
{
  "supported": false,
  "primary_alpha": 0.5,
  "endpoint_comparisons": {
    "shared": {
      "mean_effect": 0.04538539950930502,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": -0.020735901125033485,
      "positive_seeds": 0,
      "practical_advantage_pass": false
    }
  },
  "numeric_spillover_qualified_all": true,
  "requires_acquisition": false
}
```

원 파일: [results/spillover_tradeoff_v1/summary.json](../results/spillover_tradeoff_v1/summary.json). JSON Pointer: `/H_acquisition`. SHA-256: `7de0bf4d5f8e70d1a630c7bd4382674f30b3db3d0860cb9fa45c10efd290e5f8`.

```json
{
  "supported": false,
  "primary_alpha": 0.5,
  "acquisition_pass_seeds": 0,
  "numeric_spillover_qualified_all": true,
  "requires_tradeoff_advantage": false
}
```

원 파일: [results/spillover_tradeoff_v1/summary.json](../results/spillover_tradeoff_v1/summary.json). JSON Pointer: `/families/2/mean_final_expected_accuracy`. SHA-256: `7de0bf4d5f8e70d1a630c7bd4382674f30b3db3d0860cb9fa45c10efd290e5f8`.

```json
0.5484518720705218
```


## signed_spillover_accounting

**저장된 변화에서 자기 수정·같은 정답 입력 도움·반대 정답 입력 방해를 따로 기록하고 최종 변화와 대조했다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 정확한 누적 장부이며 어느 갱신을 없앤 새 학습 성적이나 유일한 인과 원인이 아니다.

원 파일: [results/signed_spillover_v1/summary.json](../results/signed_spillover_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `fb3d02498abf37538e5a3ee16ab568db05cc10862d6f981c579d0fd802fd4745`.

```json
{
  "residual_axis": [
    "static_M",
    "trial_g",
    "trial_p",
    "trial_own_score",
    "trial_other_score_l2",
    "trial_head_step_l2",
    "checkpoint_head",
    "checkpoint_logits",
    "checkpoint_probabilities",
    "margin_telescoping",
    "probability_telescoping"
  ],
  "max_residual_by_axis": [
    0.0,
    0.0,
    7.771561172376096e-16,
    2.92300905702092e-16,
    4.13048032649571e-16,
    4.579669976578771e-16,
    0.0,
    2.7755575615628914e-15,
    6.661338147750939e-16,
    2.7755575615628914e-15,
    6.661338147750939e-16
  ],
  "all_reconstruction_and_telescoping_checks_passed": true,
  "all_mass_balances_passed": true,
  "all_classification_counts_complete": true
}
```

원 파일: [results/signed_spillover_v1/summary.json](../results/signed_spillover_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `fb3d02498abf37538e5a3ee16ab568db05cc10862d6f981c579d0fd802fd4745`.

```json
{
  "prior_outcomes_known": true,
  "confirmatory_behavioral_hypothesis": false,
  "source_labels_used_in_analysis_only": true,
  "pathwise_accounting_not_isolated_causal_effect": true,
  "uniform_source_checkpoint_expectation_not_actual_next_schedule": true,
  "probability_expectation_not_sigmoid_of_expected_margin": true,
  "neutral_counts_do_not_zero_exact_masses": true,
  "internal_synapses_fixed_in_parent": true,
  "generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false,
  "output_cost_optimization_performed": false
}
```


## observed_selective_suppression

**실제 획득 보상으로 관측된 방해를 선택 억제한 주 조건은 고정 개선·충분한 습득 기준을 통과하지 못했다.**

상태: 미지지. 질문 2, Level 2. 해석 한계: 평가 정답을 사용하는 oracle와 보상 부호 기억의 정적 조회는 주 출력 선택 성능을 대신하지 않는다.

원 파일: [results/selective_spillover_v1/summary.json](../results/selective_spillover_v1/summary.json). JSON Pointer: `/H_selective`. SHA-256: `814a9d0be4a4146068b567c912b74021cc3bec8a4d43c45f7e28c27a6f7bc2f1`.

```json
{
  "supported": false,
  "endpoint_comparisons": {
    "shared": {
      "mean_effect": 0.06547113927155898,
      "positive_seeds": 10,
      "practical_advantage_pass": true
    },
    "isolated": {
      "mean_effect": 0.0032132092878687348,
      "positive_seeds": 9,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false,
  "oracle_can_replace_primary": false
}
```

원 파일: [results/selective_spillover_v1/summary.json](../results/selective_spillover_v1/summary.json). JSON Pointer: `/H_acquisition`. SHA-256: `814a9d0be4a4146068b567c912b74021cc3bec8a4d43c45f7e28c27a6f7bc2f1`.

```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_selective_advantage": false,
  "oracle_can_replace_primary": false
}
```

원 파일: [results/selective_spillover_v1/summary.json](../results/selective_spillover_v1/summary.json). JSON Pointer: `/families/3/mean_final_expected_accuracy`. SHA-256: `814a9d0be4a4146068b567c912b74021cc3bec8a4d43c45f7e28c27a6f7bc2f1`.

```json
0.5677692589235284
```

원 파일: [results/selective_spillover_v1/summary.json](../results/selective_spillover_v1/summary.json). JSON Pointer: `/families/3/acquisition_pass_seeds`. SHA-256: `814a9d0be4a4146068b567c912b74021cc3bec8a4d43c45f7e28c27a6f7bc2f1`.

```json
0
```


## normalization_weakens_remaining_effects

**방해를 줄인 뒤 이동 길이를 제한하는 정규화가 자기 수정과 남겨 둔 도움의 실제 점수 효과도 약화시킨 장부를 보존한다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 같은 실제 계수로 계산한 영향 질량이며 정답률 손실이나 정규화 없는 새 학습의 성적이 아니다.

원 파일: [results/normalization_loss_v1/summary.json](../results/normalization_loss_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `7c8e1cc06372e74088290837a8ce1ec1bfe614e44cbb2affd2a4fae845a088ab`.

```json
{
  "residual_axis": [
    "gram",
    "actual_influence",
    "desired_influence",
    "scaled_desired",
    "reciprocal_scale",
    "own_score",
    "other_score",
    "head_step",
    "scalar",
    "unit_direction",
    "checkpoint_logits",
    "stage_accounting"
  ],
  "max_residual_by_axis": [
    0.0,
    4.440892098500626e-16,
    0.0,
    2.3245294578089215e-15,
    2.220446049250313e-16,
    4.0245584642661925e-16,
    4.387264367034004e-16,
    5.204170427930421e-16,
    0.0,
    2.220446049250313e-16,
    2.886579864025407e-15,
    6.938893903907228e-18
  ],
  "all_reconstruction_checks_passed": true,
  "all_stage_and_partition_balances_passed": true
}
```

원 파일: [results/normalization_loss_v1/summary.json](../results/normalization_loss_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `7c8e1cc06372e74088290837a8ce1ec1bfe614e44cbb2affd2a4fae845a088ab`.

```json
{
  "prior_outcomes_known": true,
  "confirmatory_behavioral_hypothesis": false,
  "saved_scalar_reference_not_counterfactual_policy": true,
  "score_loss_not_accuracy_loss": true,
  "phase_is_descriptive_not_causal": true,
  "true_labels_evaluator_only": true,
  "unknown_category_overlaps": true,
  "neutral_counts_do_not_zero_exact_masses": true,
  "internal_synapses_fixed_in_parent": true,
  "generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## safe_direction_geometry

**관측 안전 조건과 같은 단위 이동 상한에서 다른 수정 방향의 점수 영향을 비교할 수 있었다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 저장 상태 방향 최적화는 실제 학습 성적이나 미래 최종 성적의 전역 최적성을 보장하지 않는다.

원 파일: [results/cone_frontier_v1/summary.json](../results/cone_frontier_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `476853adfed1e745f50d7bed36e3a9213e27ed4f0df0f4894483739653527300`.

```json
{
  "all_unit_caps_passed": true,
  "all_observed_and_oracle_constraints_passed": true,
  "all_case_certificates_passed": true,
  "all_phase_balances_passed": true,
  "max_certificate_residual": 1.7055801215803967e-14,
  "max_step_bound_violation": 1.3877787807814457e-17
}
```

원 파일: [results/cone_frontier_v1/summary.json](../results/cone_frontier_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `476853adfed1e745f50d7bed36e3a9213e27ed4f0df0f4894483739653527300`.

```json
{
  "prior_outcomes_known": true,
  "confirmatory_behavioral_hypothesis": false,
  "same_saved_scalar_not_autonomous_policy": true,
  "no_unknown_signs_in_observed_constraints": true,
  "oracle_is_full_information_safety_reference": true,
  "oracle_not_universal_observed_upper_bound": true,
  "own_optimum_not_joint_improvement_guarantee": true,
  "balanced_not_complete_Pareto_frontier": true,
  "score_not_accuracy": true,
  "unknown_harm_evaluator_only": true,
  "unit_cap_not_equal_processing_cost": true,
  "generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## balanced_actual_learning

**균형 방향의 실제 학습은 실용 개선·충분한 습득·미관측 비악화의 세 기준을 충족하지 못했다.**

상태: 미지지. 질문 2, Level 2. 해석 한계: 주 조건의 작은 양수 평균을 성공으로 바꾸지 않으며 미관측은 보상 미관측 연습 입력이다.

원 파일: [results/balanced_learning_v1/summary.json](../results/balanced_learning_v1/summary.json). JSON Pointer: `/H_gain`. SHA-256: `38d25b109819dcce836096ce696fa6d52be8a2bb63e2a62065b947732c10084f`.

```json
{
  "supported": false,
  "endpoint_comparisons": {
    "feedback_selective": {
      "mean_effect": 0.0018710252240668468,
      "positive_seeds": 5,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": 0.006715572422411775,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```

원 파일: [results/balanced_learning_v1/summary.json](../results/balanced_learning_v1/summary.json). JSON Pointer: `/H_acquisition`. SHA-256: `38d25b109819dcce836096ce696fa6d52be8a2bb63e2a62065b947732c10084f`.

```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_gain": false
}
```

원 파일: [results/balanced_learning_v1/summary.json](../results/balanced_learning_v1/summary.json). JSON Pointer: `/H_unknown`. SHA-256: `38d25b109819dcce836096ce696fa6d52be8a2bb63e2a62065b947732c10084f`.

```json
{
  "supported": false,
  "mean_balanced_minus_feedback_damage": 0.011364622409407766,
  "nonworsening_seeds": 0,
  "empirical_observed_sample_only": true,
  "population_equivalence_claimed": false,
  "fixed_400_trial_total_is_primary": true,
  "all_numeric_and_information_gates_passed": true
}
```

원 파일: [results/balanced_learning_v1/summary.json](../results/balanced_learning_v1/summary.json). JSON Pointer: `/families/3/mean_final_expected_accuracy`. SHA-256: `38d25b109819dcce836096ce696fa6d52be8a2bb63e2a62065b947732c10084f`.

```json
0.5927987701051496
```


## balanced_actual_pathways

**균형 방식의 작은 최종 차이를 부분 관측·완전 관측 구간과 실제 선택·수정 효과로 나눈 장부를 보존한다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 원 상태 대입과 실제 경로를 구분한다. 새 학습이나 원인의 고유 비율을 주장하지 않는다.

원 파일: [results/balanced_pathways_v1/summary.json](../results/balanced_pathways_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `20ad09156f0473951df6c872a748fb819fd649c4168bc3c6527f7fdc037b4b4b`.

```json
{
  "all_exact_accounting_gates_passed": true,
  "residuals": {
    "endpoint_telescoping": 7.771561172376096e-16,
    "per_destination_probability_Shapley": 8.673617379884035e-19,
    "per_destination_harm_Shapley": 8.673617379884035e-19,
    "endpoint_phase_category_difference": 8.881784197001252e-16,
    "endpoint_action_case_difference": 8.81239525796218e-16,
    "unknown_membership_difference": 0.0,
    "paired_Go_probability_bridge": 2.7755575615628914e-17,
    "paired_Go_shared_unknown_harm_bridge": 6.938893903907228e-18
  },
  "last_discovery_trial_assigned_to_partial": true,
  "zero_effects_and_roundoff_signs_preserved": true,
  "both_Go_only_bridge_no_NoGo_zero_direction_ambiguity": true,
  "full_known_objectives_compare_same_information_constraints": true
}
```

원 파일: [results/balanced_pathways_v1/summary.json](../results/balanced_pathways_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `20ad09156f0473951df6c872a748fb819fd649c4168bc3c6527f7fdc037b4b4b`.

```json
{
  "retrospective_descriptive_only": true,
  "parent_H_gain": false,
  "parent_H_acquisition": false,
  "parent_H_unknown": false,
  "new_confirmatory_hypothesis": false,
  "exact_path_accounting_not_unique_causal_explanation": true,
  "crossed_one_step_cells_are_not_relearned_policy": true,
  "crossed_directions_can_have_different_original_information": true,
  "unknown_harm_bridge_intersection_mask_fixed_across_cells": true,
  "same_realized_movement_budget_claimed": false,
  "internal_synapses_and_probe_activity_fixed": true,
  "observed_memory_is_separate_from_output_weights": true,
  "novel_input_generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## early_isolation_gate

**초반에 번짐을 막고 네 보상 부호 확인 뒤 균형 방향을 여는 조건은 충분한 습득·실용 개선을 통과하지 못했다.**

상태: 미지지. 질문 4, Level 2. 해석 한계: 보호 통과는 부분 관측 동안 다른 입력을 움직이지 않고 완전 관측 뒤 미관측 집합이 사라지는 설계상의 운영 검사다.

원 파일: [results/gated_learning_v1/summary.json](../results/gated_learning_v1/summary.json). JSON Pointer: `/H_gain`. SHA-256: `a076d7d049f2cbb8c63faf093e32820b6bb0071b5cb7aebcc88dd2f0475e952b`.

```json
{
  "supported": false,
  "endpoint_comparisons": {
    "balanced": {
      "mean_effect": 0.0018647163414324819,
      "positive_seeds": 9,
      "practical_advantage_pass": false
    },
    "feedback_selective": {
      "mean_effect": 0.004603776042436947,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": 0.006808307524882362,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```

원 파일: [results/gated_learning_v1/summary.json](../results/gated_learning_v1/summary.json). JSON Pointer: `/H_acquisition`. SHA-256: `a076d7d049f2cbb8c63faf093e32820b6bb0071b5cb7aebcc88dd2f0475e952b`.

```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_gain": false
}
```

원 파일: [results/gated_learning_v1/summary.json](../results/gated_learning_v1/summary.json). JSON Pointer: `/H_protection`. SHA-256: `a076d7d049f2cbb8c63faf093e32820b6bb0071b5cb7aebcc88dd2f0475e952b`.

```json
{
  "supported": true,
  "maximum_primary_seed_unknown_damage": 0.0,
  "absolute_tolerance": 1e-08,
  "comparisons": {
    "balanced": {
      "mean_effect": -0.05108806781749028,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    },
    "feedback_selective": {
      "mean_effect": -0.038461854820264735,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    }
  },
  "operational_protection_expected_by_design": true,
  "novel_learning_mechanism_demonstrated": false,
  "all_numeric_and_information_gates_passed": true
}
```

원 파일: [results/gated_learning_v1/summary.json](../results/gated_learning_v1/summary.json). JSON Pointer: `/families/4/mean_final_expected_accuracy`. SHA-256: `a076d7d049f2cbb8c63faf093e32820b6bb0071b5cb7aebcc88dd2f0475e952b`.

```json
0.5759520190768139
```


## gated_actual_pathways

**전환 전 보호와 전환 뒤 자기 수정·도움 변화 및 실제 누적 비용을 따로 기록했다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 전환 시점의 공통 구간과 각 방법 자신의 관측·상태·선택을 같다고 가정하지 않는다.

원 파일: [results/gated_pathways_v1/summary.json](../results/gated_pathways_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `ad08104cf638954f9cb94b427b380a65b7ba5e220912ea2adf76d23eee7e967e`.

```json
{
  "all_exact_accounting_gates_passed": true,
  "residuals": {
    "endpoint_telescoping": 8.881784197001252e-16,
    "positive_negative_identity": 3.3306690738754696e-16,
    "phase_category_partition": 1.5265566588595902e-16,
    "contrast_endpoint": 9.992007221626409e-16,
    "phase_cost_path": 1.2434497875801753e-14,
    "phase_cost_energy": 1.1102230246251565e-15
  },
  "common_reference_phase_uses_prefeedback_memory": true,
  "last_discovery_trial_assigned_to_pre": true,
  "each_method_categories_use_own_observations": true,
  "zero_effects_and_roundoff_signs_preserved": true,
  "no_counterfactual_policy_or_direction_evaluated": true
}
```

원 파일: [results/gated_pathways_v1/summary.json](../results/gated_pathways_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `ad08104cf638954f9cb94b427b380a65b7ba5e220912ea2adf76d23eee7e967e`.

```json
{
  "retrospective_descriptive_only": true,
  "parent_H_gain": false,
  "parent_H_acquisition": false,
  "parent_H_protection": true,
  "new_confirmatory_hypothesis": false,
  "exact_path_accounting_not_unique_causal_explanation": true,
  "reference_gate_is_common_time_split_not_all_methods_knowledge": true,
  "same_realized_movement_budget_claimed": false,
  "internal_synapses_and_probe_activity_fixed": true,
  "observed_memory_is_separate_from_output_weights": true,
  "novel_input_generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## gated_state_size_direction

**양쪽 실제 수정에서 상태·실제 계수·방향의 대칭 배분과 한쪽만 고친 나머지로 후반 차이를 복원했다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 혼합 단일 수정 셀은 재학습한 정책이나 유일한 인과 개입이 아니다.

원 파일: [results/gated_factors_v1/summary.json](../results/gated_factors_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `fd9e7e1e219c15f810b32ed06492c70aa5cf9a398ccc400e9c6499a2a85bf02a`.

```json
{
  "all_exact_accounting_gates_passed": true,
  "residuals": {
    "factual_parent_effects": 1.7468665403086447e-15,
    "post_group_endpoint_reconstruction": 0.0,
    "shapley_signed_efficiency": 3.469446951953614e-18,
    "shapley_positive_efficiency": 3.469446951953614e-18,
    "shapley_negative_efficiency": 3.469446951953614e-18,
    "positive_negative_allocation_identity": 3.469446951953614e-18,
    "post_component_reconstruction": 4.440892098500626e-16,
    "zero_padded_ineligible": 0.0
  },
  "common_post_uses_prefeedback_memory": true,
  "only_actual_both_Go_crossed": true,
  "own_other_partition": true,
  "all_action_cases_retained": true,
  "tiny_signs_preserved": true,
  "same_recorded_fact_corners_match": true,
  "no_future_policy_or_new_training": true
}
```

원 파일: [results/gated_factors_v1/summary.json](../results/gated_factors_v1/summary.json). JSON Pointer: `/families/0/means/component_endpoint_contributions`. SHA-256: `fd9e7e1e219c15f810b32ed06492c70aa5cf9a398ccc400e9c6499a2a85bf02a`.

```json
[
  [
    -0.0005710834171478485,
    -0.00019985568059563793,
    -0.028731463208552847,
    -0.0019296237687163528,
    0.000821282206873035,
    0.0
  ],
  [
    -0.0003373077970431497,
    7.332511909304471e-06,
    0.03695754715147366,
    0.0,
    0.0007914795266821719,
    0.0
  ]
]
```

원 파일: [results/gated_factors_v1/summary.json](../results/gated_factors_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `fd9e7e1e219c15f810b32ed06492c70aa5cf9a398ccc400e9c6499a2a85bf02a`.

```json
{
  "retrospective_descriptive_only": true,
  "parent_H_gain": false,
  "parent_H_acquisition": false,
  "parent_H_protection": true,
  "new_confirmatory_hypothesis": false,
  "accounting_allocation_not_unique_causal_explanation": true,
  "mixed_one_step_cells_not_relearned_policy": true,
  "unpaired_actions_not_assigned_to_zero_direction": true,
  "same_realized_movement_budget_claimed": false,
  "internal_synapses_and_probe_activity_fixed": true,
  "observed_memory_is_separate_from_output_weights": true,
  "novel_input_generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## self_priority_actual_learning

**자기 수정을 우선하는 고정 혼합은 기존 전환 균형보다 주 평균이 낮았고 개선·충분한 습득은 미지지였다.**

상태: 미지지. 질문 2, Level 2. 해석 한계: 같은 새 코호트 내 비교이며 다른 실험 코호트와 성적 순위를 합치지 않는다. 보호는 설계상 운영 검사다.

원 파일: [results/self_blend_learning_v1/summary.json](../results/self_blend_learning_v1/summary.json). JSON Pointer: `/H_gain`. SHA-256: `6358ea2d9ac5b34980e02d39338c40e4c50fb5983632f3b044731cd3ec35fb9c`.

```json
{
  "supported": false,
  "endpoint_comparisons": {
    "gated_balanced": {
      "mean_effect": -0.0043857329099483144,
      "positive_seeds": 1,
      "practical_advantage_pass": false
    },
    "balanced": {
      "mean_effect": -0.0026480081884590057,
      "positive_seeds": 3,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": 0.0029544817620496765,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```

원 파일: [results/self_blend_learning_v1/summary.json](../results/self_blend_learning_v1/summary.json). JSON Pointer: `/H_acquisition`. SHA-256: `6358ea2d9ac5b34980e02d39338c40e4c50fb5983632f3b044731cd3ec35fb9c`.

```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_gain": false
}
```

원 파일: [results/self_blend_learning_v1/summary.json](../results/self_blend_learning_v1/summary.json). JSON Pointer: `/H_protection`. SHA-256: `6358ea2d9ac5b34980e02d39338c40e4c50fb5983632f3b044731cd3ec35fb9c`.

```json
{
  "supported": true,
  "maximum_primary_seed_unknown_damage": 0.0,
  "absolute_tolerance": 1e-08,
  "comparisons": {
    "gated_balanced": {
      "mean_effect": 0.0,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    },
    "isolated": {
      "mean_effect": 0.0,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    }
  },
  "operational_protection_expected_by_design": true,
  "novel_learning_mechanism_demonstrated": false,
  "all_numeric_and_information_gates_passed": true
}
```

원 파일: [results/self_blend_learning_v1/summary.json](../results/self_blend_learning_v1/summary.json). JSON Pointer: `/families/4/mean_final_expected_accuracy`. SHA-256: `6358ea2d9ac5b34980e02d39338c40e4c50fb5983632f3b044731cd3ec35fb9c`.

```json
0.5768724084106928
```

원 파일: [results/self_blend_learning_v1/summary.json](../results/self_blend_learning_v1/summary.json). JSON Pointer: `/families/5/mean_final_expected_accuracy`. SHA-256: `6358ea2d9ac5b34980e02d39338c40e4c50fb5983632f3b044731cd3ec35fb9c`.

```json
0.5724866755007445
```


## self_priority_help_loss

**자기 수정 회복과 다른 문제 도움 감소를 실제 크기·상태·방향 및 정규화된 혼합 재료로 나누어 보존했다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 정답률 보존 비율·새 학습 성적·고유한 원인 비율이 아니다.

원 파일: [results/self_blend_factors_v1/summary.json](../results/self_blend_factors_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `0407fb71c6f22bbe3c21bcbbb7726450589d6f47420b1c2c9c13312b644034a9`.

```json
{
  "all_exact_accounting_gates_passed": true,
  "residuals": {
    "prior_primary_post_bookkeeping": 9.298117831235686e-16,
    "saved_unit_mixture_geometry": 1.9392994997399807e-15,
    "factual_parent_effects": 1.7925814179770492e-15,
    "post_group_endpoint_reconstruction": 1.3183898417423734e-15,
    "shapley_signed_efficiency": 3.469446951953614e-18,
    "shapley_positive_efficiency": 3.469446951953614e-18,
    "shapley_negative_efficiency": 3.469446951953614e-18,
    "positive_negative_allocation_identity": 3.469446951953614e-18,
    "post_component_reconstruction": 4.440892098500626e-16,
    "zero_padded_ineligible": 0.0
  },
  "common_post_uses_prefeedback_memory": true,
  "only_actual_both_Go_crossed": true,
  "own_other_partition": true,
  "all_action_cases_retained": true,
  "tiny_signs_preserved": true,
  "same_recorded_fact_corners_match": true,
  "no_future_policy_or_new_training": true
}
```

원 파일: [results/self_blend_factors_v1/summary.json](../results/self_blend_factors_v1/summary.json). JSON Pointer: `/families/0/means/component_endpoint_contributions`. SHA-256: `0407fb71c6f22bbe3c21bcbbb7726450589d6f47420b1c2c9c13312b644034a9`.

```json
[
  [
    0.00040985025249831025,
    0.0001661279572841186,
    0.02261946445381665,
    -0.0006950449131485709,
    0.0012407292196135635,
    0.0
  ],
  [
    0.00029761352156798375,
    1.606739864701059e-06,
    -0.027631946366678317,
    -0.0009255335065022033,
    0.0001313997317352114,
    0.0
  ]
]
```

원 파일: [results/self_blend_factors_v1/summary.json](../results/self_blend_factors_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `0407fb71c6f22bbe3c21bcbbb7726450589d6f47420b1c2c9c13312b644034a9`.

```json
{
  "retrospective_descriptive_only": true,
  "parent_H_gain": false,
  "parent_H_acquisition": false,
  "parent_H_protection": true,
  "new_confirmatory_hypothesis": false,
  "accounting_allocation_not_unique_causal_explanation": true,
  "mixed_one_step_cells_not_relearned_policy": true,
  "unpaired_actions_not_assigned_to_zero_direction": true,
  "same_realized_movement_budget_claimed": false,
  "internal_synapses_and_probe_activity_fixed": true,
  "observed_memory_is_separate_from_output_weights": true,
  "novel_input_generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## fixed_state_probability_alignment

**전체 목적 U와 현재 확률 목적 P는 동일한 저장 상태의 단일 수정에서 작은 차이를 보였다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 당시 새 방향 최적화가 있었지만 새로운 학습 경로는 0회다. 고정 상태의 이득을 미래 최종 성적에 더하지 않는다.

원 파일: [results/probability_alignment_v1/summary.json](../results/probability_alignment_v1/summary.json). JSON Pointer: `/finite_alignment_summary`. SHA-256: `757ed2f4014ad804467df55fa3ace89ef3ea5e9d1384e2d4c4a5db636a40f361`.

```json
{
  "mean_exact_P_minus_U_per_update": 3.8933151170545127e-07,
  "positive_seed_means": 9,
  "pooled_positive_trials": 1548,
  "pooled_negative_trials": 72,
  "pooled_zero_trials": 345,
  "pooled_negative_beyond_tolerance_trials": 0,
  "worst_Q_difference": -1.4814359461379212e-10,
  "new_learned_performance_claim": false
}
```

원 파일: [results/probability_alignment_v1/summary.json](../results/probability_alignment_v1/summary.json). JSON Pointer: `/counts/new_optimization_calls`. SHA-256: `757ed2f4014ad804467df55fa3ace89ef3ea5e9d1384e2d4c4a5db636a40f361`.

```json
11949
```

원 파일: [results/probability_alignment_v1/summary.json](../results/probability_alignment_v1/summary.json). JSON Pointer: `/counts/new_training_trials`. SHA-256: `757ed2f4014ad804467df55fa3ace89ef3ea5e9d1384e2d4c4a5db636a40f361`.

```json
0
```

원 파일: [results/probability_alignment_v1/summary.json](../results/probability_alignment_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `757ed2f4014ad804467df55fa3ace89ef3ea5e9d1384e2d4c4a5db636a40f361`.

```json
{
  "retrospective_descriptive_only": true,
  "new_confirmatory_hypothesis": false,
  "tangent_global_maximum_is_expected_integrity_property": true,
  "finite_probability_global_optimum_claimed": false,
  "summed_fixed_state_gains_predict_new_trajectory": false,
  "actual_future_policy_relearned": false,
  "state_weighting_effect_separated_from_uniform_objective_control": true,
  "processing_cost_equalized": false,
  "parent_H_gain": false,
  "parent_H_acquisition": false,
  "parent_H_protection": true,
  "novel_input_generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## uniform_probability_actual_learning

**U/P를 각자의 실제 현재 상태·선택·보상 경로에 적용했지만 실용 개선·충분한 습득·P의 실용 추가효과는 미지지였다.**

상태: 미지지. 질문 2, Level 2. 해석 한계: 작은 수치 차이와 사전 성공 기준을 구분한다. 같은 수정 상한이며 실제 총 이동과 계산 비용은 다르다.

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/H_gain`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
{
  "gated_uniform": {
    "supported": false,
    "endpoint_comparisons": {
      "gated_balanced": {
        "mean_effect": -0.0018705134196065677,
        "positive_seeds": 2,
        "practical_advantage_pass": false
      },
      "isolated": {
        "mean_effect": 0.006338476769223856,
        "positive_seeds": 10,
        "practical_advantage_pass": false
      },
      "gated_self_blend": {
        "mean_effect": 0.0026083562276982275,
        "positive_seeds": 9,
        "practical_advantage_pass": false
      }
    },
    "all_numeric_and_information_gates_passed": true,
    "requires_acquisition": false
  },
  "gated_probability": {
    "supported": false,
    "endpoint_comparisons": {
      "gated_balanced": {
        "mean_effect": 0.0006817172395877757,
        "positive_seeds": 8,
        "practical_advantage_pass": false
      },
      "isolated": {
        "mean_effect": 0.008890707428418198,
        "positive_seeds": 10,
        "practical_advantage_pass": false
      },
      "gated_self_blend": {
        "mean_effect": 0.005160586886892571,
        "positive_seeds": 10,
        "practical_advantage_pass": false
      }
    },
    "all_numeric_and_information_gates_passed": true,
    "requires_acquisition": false
  }
}
```

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/H_acquisition`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
{
  "gated_uniform": {
    "supported": false,
    "acquisition_pass_seeds": 0,
    "all_numeric_and_information_gates_passed": true,
    "requires_gain": false
  },
  "gated_probability": {
    "supported": false,
    "acquisition_pass_seeds": 0,
    "all_numeric_and_information_gates_passed": true,
    "requires_gain": false
  }
}
```

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/H_state`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
{
  "supported": false,
  "comparison": "gated_probability_minus_gated_uniform",
  "mean_effect": 0.002552230659194343,
  "positive_seeds": 10,
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/families/4/mean_final_expected_accuracy`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
0.5845420564847277
```

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/families/6/mean_final_expected_accuracy`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
0.5826715430651213
```

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/families/7/mean_final_expected_accuracy`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
0.5852237737243156
```


## unknown_training_input_operational_protection

**U/P의 아직 보상을 모르는 다른 연습 입력의 운영상 보호 기준은 통과했다.**

상태: 지지. 질문 4, Level 2. 해석 한계: 새 입력 일반화나 새 보호 기작이 아니다. 초반 분리와 완전 관측 뒤 미관측 집합 부재의 설계상 기대를 확인했다.

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/H_protection`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
{
  "gated_uniform": {
    "supported": true,
    "maximum_primary_seed_unknown_damage": 0.0,
    "absolute_tolerance": 1e-08,
    "comparisons": {
      "gated_balanced": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      },
      "isolated": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      }
    },
    "operational_protection_expected_by_design": true,
    "novel_learning_mechanism_demonstrated": false,
    "all_numeric_and_information_gates_passed": true
  },
  "gated_probability": {
    "supported": true,
    "maximum_primary_seed_unknown_damage": 0.0,
    "absolute_tolerance": 1e-08,
    "comparisons": {
      "gated_balanced": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      },
      "isolated": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      }
    },
    "operational_protection_expected_by_design": true,
    "novel_learning_mechanism_demonstrated": false,
    "all_numeric_and_information_gates_passed": true
  }
}
```

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/interpretation/unknown_means_reward_unobserved_not_novel_input`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
true
```

원 파일: [results/objective_learning_v1/summary.json](../results/objective_learning_v1/summary.json). JSON Pointer: `/interpretation/generalization_tested`. SHA-256: `63ef1d5f4d6513f2dea536771db412b42dca921e1399fe7d26852a419dcac131`.

```json
false
```


## objective_actual_path_decomposition

**U/P의 최종 차이를 상태·실제 크기·방향과 한쪽만 실제 선택·보상을 받은 경로로 정확히 나누었다.**

상태: 기술·산술. 질문 2, Level 0. 해석 한계: 방향 전체가 손해였다고 단순화하지 않으며 음수 상태 몫을 더 나쁜 정답 확률로 읽지 않는다. 대칭 장부는 유일한 인과 원인이 아니다.

원 파일: [results/objective_pathways_v1/summary.json](../results/objective_pathways_v1/summary.json). JSON Pointer: `/validation`. SHA-256: `86a1b02af51060d495f2ecc9e8b4c07ce91ad01ec8b557baa79ed34629b2ad00`.

```json
{
  "all_exact_accounting_gates_passed": true,
  "residuals": {
    "factual_parent_reconstruction": 2.1316282072803006e-14,
    "complete_endpoint_accounting": 1.0547118733938987e-15,
    "all_path_telescoping": 2.7755575615628914e-15,
    "shapley_efficiency": 1.3010426069826053e-18,
    "signed_positive_negative_identity": 8.673617379884035e-19,
    "ineligible_zero_padding": 0.0
  },
  "same_prefeedback_full_gate_and_prefix_checked": true,
  "only_actual_both_Go_crossed": true,
  "one_sided_actual_choices_retained": true,
  "recorded_common_uniform_and_own_probabilities_checked": true,
  "both_Go_rewards_identical_and_scalar_difference_from_probability": true
}
```

원 파일: [results/objective_pathways_v1/summary.json](../results/objective_pathways_v1/summary.json). JSON Pointer: `/families/0/means/component_endpoint_contributions`. SHA-256: `86a1b02af51060d495f2ecc9e8b4c07ce91ad01ec8b557baa79ed34629b2ad00`.

```json
[
  [
    -0.0007864040508569701,
    -0.000392946493730736,
    -0.01895876307140617,
    -0.002004548328618086,
    0.0003936782072424774,
    0.0
  ],
  [
    -0.002044308406843389,
    7.592041066591018e-05,
    0.02140576573463236,
    -0.0006547340501382532,
    0.0010958266294462352,
    0.0
  ]
]
```

원 파일: [results/objective_pathways_v1/summary.json](../results/objective_pathways_v1/summary.json). JSON Pointer: `/families/1/means/component_endpoint_contributions`. SHA-256: `86a1b02af51060d495f2ecc9e8b4c07ce91ad01ec8b557baa79ed34629b2ad00`.

```json
[
  [
    -0.0004244425631897894,
    -0.00020571802805653222,
    -0.018537747157984303,
    -0.0014950361952192168,
    0.0003758465472107564,
    0.0
  ],
  [
    -0.000879848076681495,
    1.4204888646846414e-05,
    0.021495236232894475,
    -0.0006297593921323633,
    0.0009689809840995114,
    0.0
  ]
]
```

원 파일: [results/objective_pathways_v1/summary.json](../results/objective_pathways_v1/summary.json). JSON Pointer: `/families/2/means/component_endpoint_contributions`. SHA-256: `86a1b02af51060d495f2ecc9e8b4c07ce91ad01ec8b557baa79ed34629b2ad00`.

```json
[
  [
    0.00018207786658622514,
    4.9382233253797654e-05,
    0.001167263732031292,
    -0.00014956372911027887,
    0.00021272623736936502,
    0.0
  ],
  [
    0.0007898266157014497,
    4.755431131486451e-05,
    -6.073193743546491e-05,
    -0.0001933198242940068,
    0.0005070151537772654,
    0.0
  ]
]
```

원 파일: [results/objective_pathways_v1/summary.json](../results/objective_pathways_v1/summary.json). JSON Pointer: `/interpretation`. SHA-256: `86a1b02af51060d495f2ecc9e8b4c07ce91ad01ec8b557baa79ed34629b2ad00`.

```json
{
  "retrospective_descriptive_only": true,
  "new_confirmatory_hypothesis": false,
  "accounting_allocation_not_unique_causal_explanation": true,
  "mixed_cells_not_relearned_policy": true,
  "NoGo_potential_rewards_or_directions_not_generated": true,
  "same_realized_movement_budget_claimed": false,
  "parent_U_gain": false,
  "parent_P_gain": false,
  "parent_U_acquisition": false,
  "parent_P_acquisition": false,
  "parent_U_protection": true,
  "parent_P_protection": true,
  "parent_state_gain": false,
  "novel_input_generalization_tested": false,
  "new_human_experiment": false,
  "shared_neural_mechanism_claim": false,
  "education_effect_claim": false
}
```


## human_similarity_retained

**기존 공개 인간 조개 과제의 직접 비교에서 등록한 주 행동 유사성과 종합 예측 기준은 미지지였다.**

상태: 미지지. 질문 5, Level 4. 해석 한계: 한 공개 과제·모델의 결과이며 모든 인간/초파리 공통 원리의 부재가 아니다. 이번 후속에 새 인간 비교는 없다.

원 파일: [results/human_shell_rng_v2/summary.json](../results/human_shell_rng_v2/summary.json). JSON Pointer: `/H_similarity/supported`. SHA-256: `042518425a54e558b1fc99298f9f7f1ca98a734d85f6efdc3931994f5985b838`.

```json
false
```

원 파일: [results/human_shell_rng_v2/summary.json](../results/human_shell_rng_v2/summary.json). JSON Pointer: `/H_prediction/supported`. SHA-256: `042518425a54e558b1fc99298f9f7f1ca98a734d85f6efdc3931994f5985b838`.

```json
false
```

원 파일: [results/human_shell_rng_v2/summary.json](../results/human_shell_rng_v2/summary.json). JSON Pointer: `/mechanism_claim`. SHA-256: `042518425a54e558b1fc99298f9f7f1ca98a734d85f6efdc3931994f5985b838`.

```json
false
```

원 파일: [results/human_shell_rng_v2/summary.json](../results/human_shell_rng_v2/summary.json). JSON Pointer: `/educational_claim`. SHA-256: `042518425a54e558b1fc99298f9f7f1ca98a734d85f6efdc3931994f5985b838`.

```json
false
```


## pattern_generalization_gate_retained

**원 패턴 주 모델은 습득 자격 미달로 새 입력 일반화 판정이 보류된 상태를 유지한다.**

상태: 판정 보류. 질문 3, Level 3. 해석 한계: 네 연습 문자열의 후속 출력 학습을 새 기호·배치·길이의 일반화로 대체하지 않는다.

원 파일: [results/pattern_rule_v1/summary.json](../results/pattern_rule_v1/summary.json). JSON Pointer: `/H_acquisition/supported`. SHA-256: `815c3f3d2ea37e65fd7f5adb453b5a637b264d159817752041ee728e94cdd961`.

```json
false
```

원 파일: [results/pattern_rule_v1/summary.json](../results/pattern_rule_v1/summary.json). JSON Pointer: `/H_generalization`. SHA-256: `815c3f3d2ea37e65fd7f5adb453b5a637b264d159817752041ee728e94cdd961`.

```json
[
  {
    "set": "new_alphabet",
    "joint_acquisition_and_ood_pass_seeds": 0,
    "supported": false,
    "status": "not_assessable_without_acquisition"
  },
  {
    "set": "new_alphabet_phase",
    "joint_acquisition_and_ood_pass_seeds": 0,
    "supported": false,
    "status": "not_assessable_without_acquisition"
  },
  {
    "set": "new_length",
    "joint_acquisition_and_ood_pass_seeds": 0,
    "supported": false,
    "status": "not_assessable_without_acquisition"
  }
]
```


## interference_order_gate_retained

**기존 맥락별 과제의 개별·공동 습득 자격 미달로 간섭·순서의 확증 판정은 보류다.**

상태: 판정 보류. 질문 4, Level 2. 해석 한계: 학습하지 못한 과제의 차이를 학습 뒤 간섭이나 순서 효과로 다시 이름 붙이지 않는다.

원 파일: [results/interference_order_v1/summary.json](../results/interference_order_v1/summary.json). JSON Pointer: `/qualification_supported`. SHA-256: `e83f9cb51e6c85f5a50b049091009f9a1ff289598162dd508fbb98c8aee3e5ef`.

```json
false
```

원 파일: [results/interference_order_v1/summary.json](../results/interference_order_v1/summary.json). JSON Pointer: `/H_interference/status`. SHA-256: `e83f9cb51e6c85f5a50b049091009f9a1ff289598162dd508fbb98c8aee3e5ef`.

```json
"not_assessable"
```

원 파일: [results/interference_order_v1/summary.json](../results/interference_order_v1/summary.json). JSON Pointer: `/H_order/status`. SHA-256: `e83f9cb51e6c85f5a50b049091009f9a1ff289598162dd508fbb98c8aee3e5ef`.

```json
"not_assessable"
```


## drosomem_independent_history

**독립 완료 프로젝트 DrosoMem의 M11 습득·전이 미달과 원 저장소 불변을 보존한다.**

상태: 미지지. 질문 3, Level 0. 해석 한계: 고정 원 요약의 출처 감사를 계승하며 새 재실행이나 그 프로젝트의 결과 수정이 아니다.

원 파일: [docs/drosomem-audit.json](../docs/drosomem-audit.json). JSON Pointer: `/repository/commit`. SHA-256: `879ed87cbaadc52481c6e93206f7aa86c8345f574c5e142b9f510a936e988ea8`.

```json
"7fb156c193c7a499fc2266a164b705729abbab21"
```

원 파일: [docs/drosomem-audit.json](../docs/drosomem-audit.json). JSON Pointer: `/repository/upstream_modified`. SHA-256: `879ed87cbaadc52481c6e93206f7aa86c8345f574c5e142b9f510a936e988ea8`.

```json
false
```

원 파일: [docs/drosomem-audit.json](../docs/drosomem-audit.json). JSON Pointer: `/M11_archived_summary/content/primary_pass`. SHA-256: `879ed87cbaadc52481c6e93206f7aa86c8345f574c5e142b9f510a936e988ea8`.

```json
false
```


## shared_mechanism_and_education_untested

**공통 신경 기작과 인간 고등 수학교육 효과는 검증하지 않았으며 이번 종합도 그 주장을 추가하지 않는다.**

상태: 미검증. 질문 5, Level 5. 해석 한계: 행동 유사성이 양성이어도 같은 기작을 자동 결론 낼 수 없다. 이번 공개 과제의 직접 유사성도 미지지다.

원 파일: [results/human_shell_rng_v2/summary.json](../results/human_shell_rng_v2/summary.json). JSON Pointer: `/mechanism_claim`. SHA-256: `042518425a54e558b1fc99298f9f7f1ca98a734d85f6efdc3931994f5985b838`.

```json
false
```

원 파일: [results/human_shell_rng_v2/summary.json](../results/human_shell_rng_v2/summary.json). JSON Pointer: `/educational_claim`. SHA-256: `042518425a54e558b1fc99298f9f7f1ca98a734d85f6efdc3931994f5985b838`.

```json
false
```


## 연구별 원 버전과 판정

모든 값은 원 등록/요약에 묶인다. 다음 66개 관측 객체는 과학 가설과 산술 무결성 검사를 섞어 성공 수로 더하지 않는다.


### representation_update_v1

유형: actual_learning. 완료 태그 `representation-update-v1` → `440f6b57c6176605db37d9a5397e3f82764a5b29`. 최초 소스 `6f112178879425f8c91c30daee261e81198e1c73`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17101,
    17102,
    17103,
    17104,
    17105,
    17106,
    17107,
    17108,
    17109,
    17110
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_representation_update_v1.json) · [요약](../results/representation_update_v1/summary.json) · [독립 검산](../results/representation_update_v1/verification.json) · [보존 감사](../results/representation_update_archive_validation.json) · [재현](../docs/representation-update-reproduction.md)

원 요약 위치 `/H_update`:
```json
{
  "supported": true,
  "mean_effect": 0.36246854976211856,
  "positive_seeds": 10,
  "treated_acquisition_pass_seeds": 10,
  "numeric_update_qualified_all": true,
  "mean_spillover_squared_reduction": 2.693120842211743
}
```

원 요약 위치 `/H_representation`:
```json
{
  "supported": false,
  "mean_effect": 0.04012593982900708,
  "positive_seeds": 10,
  "treated_acquisition_pass_seeds": 0,
  "numeric_update_qualified_all": true,
  "mean_spillover_squared_reduction": null
}
```


### update_budget_v1

유형: actual_learning. 완료 태그 `update-budget-v1` → `ae3818576f9e3cdc4adcb26164484ff42186c179`. 최초 소스 `fc47813f1a18ec762f737a1e035647ff40509646`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17201,
    17202,
    17203,
    17204,
    17205,
    17206,
    17207,
    17208,
    17209,
    17210
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_update_budget_v1.json) · [요약](../results/update_budget_v1/summary.json) · [독립 검산](../results/update_budget_v1/verification.json) · [보존 감사](../results/update_budget_archive_validation.json) · [재현](../docs/update-budget-reproduction.md)

원 요약 위치 `/H_budget`:
```json
{
  "supported": false,
  "mean_effect": 0.07157582697619291,
  "positive_seeds": 10,
  "treated_acquisition_pass_seeds": 0,
  "numeric_budget_qualified_all": true
}
```


### spillover_tradeoff_v1

유형: actual_learning. 완료 태그 `spillover-tradeoff-v1` → `a9a60a82f73310c983f64589e631b2b8a9e3a052`. 최초 소스 `38a53fbd5096e277548e310d980fbc476ad5a383`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17301,
    17302,
    17303,
    17304,
    17305,
    17306,
    17307,
    17308,
    17309,
    17310
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_spillover_tradeoff_v1.json) · [요약](../results/spillover_tradeoff_v1/summary.json) · [독립 검산](../results/spillover_tradeoff_v1/verification.json) · [보존 감사](../results/spillover_archive_validation.json) · [재현](../docs/spillover-tradeoff-reproduction.md)

원 요약 위치 `/H_tradeoff`:
```json
{
  "supported": false,
  "primary_alpha": 0.5,
  "endpoint_comparisons": {
    "shared": {
      "mean_effect": 0.04538539950930502,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": -0.020735901125033485,
      "positive_seeds": 0,
      "practical_advantage_pass": false
    }
  },
  "numeric_spillover_qualified_all": true,
  "requires_acquisition": false
}
```

원 요약 위치 `/H_acquisition`:
```json
{
  "supported": false,
  "primary_alpha": 0.5,
  "acquisition_pass_seeds": 0,
  "numeric_spillover_qualified_all": true,
  "requires_tradeoff_advantage": false
}
```


### signed_spillover_v1

유형: saved_record_analysis. 완료 태그 `signed-spillover-v1` → `4ee00454500b03e2ccae08ba22f2a6625526be83`. 최초 소스 `fd2c1ace641621e58f6d20a7c0f2ff48e6f94631`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17301,
    17302,
    17303,
    17304,
    17305,
    17306,
    17307,
    17308,
    17309,
    17310
  ],
  "training_trials": 400,
  "parent_learning_result": "results/spillover_tradeoff_v1"
}
```

[등록](../results/registration_signed_spillover_v1.json) · [요약](../results/signed_spillover_v1/summary.json) · [독립 검산](../results/signed_spillover_v1/verification.json) · [보존 감사](../results/signed_spillover_archive_validation.json) · [재현](../docs/signed-spillover-reproduction.md)

원 요약 위치 `/validation/all_reconstruction_and_telescoping_checks_passed`:
```json
true
```

원 요약 위치 `/validation/all_mass_balances_passed`:
```json
true
```

원 요약 위치 `/validation/all_classification_counts_complete`:
```json
true
```


### selective_spillover_v1

유형: actual_learning. 완료 태그 `selective-spillover-v1` → `a19a2abafcdb0db4b1ebdcc4b7629d237d112c55`. 최초 소스 `7f1f67aa1d2355cd00851cf1f9d7196198cf0c9a`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17401,
    17402,
    17403,
    17404,
    17405,
    17406,
    17407,
    17408,
    17409,
    17410
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_selective_spillover_v1.json) · [요약](../results/selective_spillover_v1/summary.json) · [독립 검산](../results/selective_spillover_v1/verification.json) · [보존 감사](../results/selective_spillover_archive_validation.json) · [재현](../docs/selective-spillover-reproduction.md)

원 요약 위치 `/H_selective`:
```json
{
  "supported": false,
  "endpoint_comparisons": {
    "shared": {
      "mean_effect": 0.06547113927155898,
      "positive_seeds": 10,
      "practical_advantage_pass": true
    },
    "isolated": {
      "mean_effect": 0.0032132092878687348,
      "positive_seeds": 9,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false,
  "oracle_can_replace_primary": false
}
```

원 요약 위치 `/H_acquisition`:
```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_selective_advantage": false,
  "oracle_can_replace_primary": false
}
```


### normalization_loss_v1

유형: saved_record_analysis. 완료 태그 `normalization-loss-v1` → `f1ada01a904283a1036b4dc77b10fb3882194750`. 최초 소스 `d9152f2faeb4366ed349a9d23aac8167648def08`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17401,
    17402,
    17403,
    17404,
    17405,
    17406,
    17407,
    17408,
    17409,
    17410
  ],
  "training_trials": 400,
  "parent_learning_result": "results/selective_spillover_v1"
}
```

[등록](../results/registration_normalization_loss_v1.json) · [요약](../results/normalization_loss_v1/summary.json) · [독립 검산](../results/normalization_loss_v1/verification.json) · [보존 감사](../results/normalization_loss_archive_validation.json) · [재현](../docs/normalization-loss-reproduction.md)

원 요약 위치 `/validation/all_reconstruction_checks_passed`:
```json
true
```

원 요약 위치 `/validation/all_stage_and_partition_balances_passed`:
```json
true
```


### cone_frontier_v1

유형: saved_record_analysis. 완료 태그 `cone-frontier-v1` → `56e1a31537083825e83d4402cfa9445e7e519b9b`. 최초 소스 `70e13961dd17d6f5b70d75bace03c398ec1bc363`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17401,
    17402,
    17403,
    17404,
    17405,
    17406,
    17407,
    17408,
    17409,
    17410
  ],
  "training_trials": 400,
  "parent_learning_result": "results/selective_spillover_v1"
}
```

[등록](../results/registration_cone_frontier_v1.json) · [요약](../results/cone_frontier_v1/summary.json) · [독립 검산](../results/cone_frontier_v1/verification.json) · [보존 감사](../results/cone_frontier_archive_validation.json) · [재현](../docs/cone-frontier-reproduction.md)

원 요약 위치 `/validation/all_unit_caps_passed`:
```json
true
```

원 요약 위치 `/validation/all_observed_and_oracle_constraints_passed`:
```json
true
```

원 요약 위치 `/validation/all_case_certificates_passed`:
```json
true
```

원 요약 위치 `/validation/all_phase_balances_passed`:
```json
true
```


### balanced_learning_v1

유형: actual_learning. 완료 태그 `balanced-learning-v1` → `b836a780b91d81425e534dad7284485f284799ce`. 최초 소스 `f23a94725f5e76378e5dcc7a2c36552a309947b5`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17501,
    17502,
    17503,
    17504,
    17505,
    17506,
    17507,
    17508,
    17509,
    17510
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_balanced_learning_v1.json) · [요약](../results/balanced_learning_v1/summary.json) · [독립 검산](../results/balanced_learning_v1/verification.json) · [보존 감사](../results/balanced_learning_archive_validation.json) · [재현](../docs/balanced-learning-reproduction.md)

원 요약 위치 `/H_gain`:
```json
{
  "supported": false,
  "endpoint_comparisons": {
    "feedback_selective": {
      "mean_effect": 0.0018710252240668468,
      "positive_seeds": 5,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": 0.006715572422411775,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```

원 요약 위치 `/H_acquisition`:
```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_gain": false
}
```

원 요약 위치 `/H_unknown`:
```json
{
  "supported": false,
  "mean_balanced_minus_feedback_damage": 0.011364622409407766,
  "nonworsening_seeds": 0,
  "empirical_observed_sample_only": true,
  "population_equivalence_claimed": false,
  "fixed_400_trial_total_is_primary": true,
  "all_numeric_and_information_gates_passed": true
}
```


### balanced_pathways_v1

유형: saved_record_analysis. 완료 태그 `balanced-pathways-v1` → `f291f1f22e7ce83a0c836924530ca93e9bc12d33`. 최초 소스 `6c60a36e45bb4f0fff6c48f4c5df766b92489a69`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17501,
    17502,
    17503,
    17504,
    17505,
    17506,
    17507,
    17508,
    17509,
    17510
  ],
  "training_trials": 400,
  "parent_learning_result": "results/balanced_learning_v1"
}
```

[등록](../results/registration_balanced_pathways_v1.json) · [요약](../results/balanced_pathways_v1/summary.json) · [독립 검산](../results/balanced_pathways_v1/verification.json) · [보존 감사](../results/balanced_pathways_archive_validation.json) · [재현](../docs/balanced-pathways-reproduction.md)

원 요약 위치 `/validation/all_exact_accounting_gates_passed`:
```json
true
```

원 요약 위치 `/validation/last_discovery_trial_assigned_to_partial`:
```json
true
```

원 요약 위치 `/validation/zero_effects_and_roundoff_signs_preserved`:
```json
true
```

원 요약 위치 `/validation/both_Go_only_bridge_no_NoGo_zero_direction_ambiguity`:
```json
true
```

원 요약 위치 `/validation/full_known_objectives_compare_same_information_constraints`:
```json
true
```


### gated_learning_v1

유형: actual_learning. 완료 태그 `gated-learning-v1` → `b89e9f2baf1aed91b4451538c85ba3e9458df53b`. 최초 소스 `6c3f709263c8fc7bea8ad52efbafca96afc5d188`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17601,
    17602,
    17603,
    17604,
    17605,
    17606,
    17607,
    17608,
    17609,
    17610
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_gated_learning_v1.json) · [요약](../results/gated_learning_v1/summary.json) · [독립 검산](../results/gated_learning_v1/verification.json) · [보존 감사](../results/gated_learning_archive_validation.json) · [재현](../docs/gated-learning-reproduction.md)

원 요약 위치 `/H_gain`:
```json
{
  "supported": false,
  "endpoint_comparisons": {
    "balanced": {
      "mean_effect": 0.0018647163414324819,
      "positive_seeds": 9,
      "practical_advantage_pass": false
    },
    "feedback_selective": {
      "mean_effect": 0.004603776042436947,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": 0.006808307524882362,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```

원 요약 위치 `/H_acquisition`:
```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_gain": false
}
```

원 요약 위치 `/H_protection`:
```json
{
  "supported": true,
  "maximum_primary_seed_unknown_damage": 0.0,
  "absolute_tolerance": 1e-08,
  "comparisons": {
    "balanced": {
      "mean_effect": -0.05108806781749028,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    },
    "feedback_selective": {
      "mean_effect": -0.038461854820264735,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    }
  },
  "operational_protection_expected_by_design": true,
  "novel_learning_mechanism_demonstrated": false,
  "all_numeric_and_information_gates_passed": true
}
```


### gated_pathways_v1

유형: saved_record_analysis. 완료 태그 `gated-pathways-v1` → `e74604a67479299825e2ab5572fd3efd8f9970d6`. 최초 소스 `856963ed7b3699f34d392ffc93a842507319a95a`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17601,
    17602,
    17603,
    17604,
    17605,
    17606,
    17607,
    17608,
    17609,
    17610
  ],
  "training_trials": 400,
  "parent_learning_result": "results/gated_learning_v1"
}
```

[등록](../results/registration_gated_pathways_v1.json) · [요약](../results/gated_pathways_v1/summary.json) · [독립 검산](../results/gated_pathways_v1/verification.json) · [보존 감사](../results/gated_pathways_archive_validation.json) · [재현](../docs/gated-pathways-reproduction.md)

원 요약 위치 `/validation/all_exact_accounting_gates_passed`:
```json
true
```

원 요약 위치 `/validation/common_reference_phase_uses_prefeedback_memory`:
```json
true
```

원 요약 위치 `/validation/last_discovery_trial_assigned_to_pre`:
```json
true
```

원 요약 위치 `/validation/each_method_categories_use_own_observations`:
```json
true
```

원 요약 위치 `/validation/zero_effects_and_roundoff_signs_preserved`:
```json
true
```

원 요약 위치 `/validation/no_counterfactual_policy_or_direction_evaluated`:
```json
true
```


### gated_factors_v1

유형: saved_record_analysis. 완료 태그 `gated-factors-v1` → `d94922361e8a17e156bed0d5c4f240750f75f5ef`. 최초 소스 `6ee94716af68f22c9feef3b1e3b7085e57b31109`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17601,
    17602,
    17603,
    17604,
    17605,
    17606,
    17607,
    17608,
    17609,
    17610
  ],
  "training_trials": 400,
  "parent_learning_result": "results/gated_learning_v1"
}
```

[등록](../results/registration_gated_factors_v1.json) · [요약](../results/gated_factors_v1/summary.json) · [독립 검산](../results/gated_factors_v1/verification.json) · [보존 감사](../results/gated_factors_archive_validation.json) · [재현](../docs/gated-factors-reproduction.md)

원 요약 위치 `/validation/all_exact_accounting_gates_passed`:
```json
true
```

원 요약 위치 `/validation/common_post_uses_prefeedback_memory`:
```json
true
```

원 요약 위치 `/validation/only_actual_both_Go_crossed`:
```json
true
```

원 요약 위치 `/validation/own_other_partition`:
```json
true
```

원 요약 위치 `/validation/all_action_cases_retained`:
```json
true
```

원 요약 위치 `/validation/tiny_signs_preserved`:
```json
true
```

원 요약 위치 `/validation/same_recorded_fact_corners_match`:
```json
true
```

원 요약 위치 `/validation/no_future_policy_or_new_training`:
```json
true
```


### self_blend_learning_v1

유형: actual_learning. 완료 태그 `self-blend-learning-v1` → `0d1a7abaf35e564eb397a52bb9f44a4f1b723907`. 최초 소스 `7507ba8321b556652d3f2fd33b0b4dc7d3798e6a`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17701,
    17702,
    17703,
    17704,
    17705,
    17706,
    17707,
    17708,
    17709,
    17710
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_self_blend_learning_v1.json) · [요약](../results/self_blend_learning_v1/summary.json) · [독립 검산](../results/self_blend_learning_v1/verification.json) · [보존 감사](../results/self_blend_archive_validation.json) · [재현](../docs/self-blend-learning-reproduction.md)

원 요약 위치 `/H_gain`:
```json
{
  "supported": false,
  "endpoint_comparisons": {
    "gated_balanced": {
      "mean_effect": -0.0043857329099483144,
      "positive_seeds": 1,
      "practical_advantage_pass": false
    },
    "balanced": {
      "mean_effect": -0.0026480081884590057,
      "positive_seeds": 3,
      "practical_advantage_pass": false
    },
    "isolated": {
      "mean_effect": 0.0029544817620496765,
      "positive_seeds": 10,
      "practical_advantage_pass": false
    }
  },
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```

원 요약 위치 `/H_acquisition`:
```json
{
  "supported": false,
  "acquisition_pass_seeds": 0,
  "all_numeric_and_information_gates_passed": true,
  "requires_gain": false
}
```

원 요약 위치 `/H_protection`:
```json
{
  "supported": true,
  "maximum_primary_seed_unknown_damage": 0.0,
  "absolute_tolerance": 1e-08,
  "comparisons": {
    "gated_balanced": {
      "mean_effect": 0.0,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    },
    "isolated": {
      "mean_effect": 0.0,
      "nonworsening_seeds": 10,
      "all_seeds_nonworsening": true
    }
  },
  "operational_protection_expected_by_design": true,
  "novel_learning_mechanism_demonstrated": false,
  "all_numeric_and_information_gates_passed": true
}
```


### self_blend_factors_v1

유형: saved_record_analysis. 완료 태그 `self-blend-factors-v1` → `2715f1eada36207ce77709f08874688b1425d676`. 최초 소스 `a050fda72e31683961872c8d6c2c7d197d4face3`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17701,
    17702,
    17703,
    17704,
    17705,
    17706,
    17707,
    17708,
    17709,
    17710
  ],
  "training_trials": 400,
  "parent_learning_result": "results/self_blend_learning_v1"
}
```

[등록](../results/registration_self_blend_factors_v1.json) · [요약](../results/self_blend_factors_v1/summary.json) · [독립 검산](../results/self_blend_factors_v1/verification.json) · [보존 감사](../results/self_blend_factors_archive_validation.json) · [재현](../docs/self-blend-factors-reproduction.md)

원 요약 위치 `/validation/all_exact_accounting_gates_passed`:
```json
true
```

원 요약 위치 `/validation/common_post_uses_prefeedback_memory`:
```json
true
```

원 요약 위치 `/validation/only_actual_both_Go_crossed`:
```json
true
```

원 요약 위치 `/validation/own_other_partition`:
```json
true
```

원 요약 위치 `/validation/all_action_cases_retained`:
```json
true
```

원 요약 위치 `/validation/tiny_signs_preserved`:
```json
true
```

원 요약 위치 `/validation/same_recorded_fact_corners_match`:
```json
true
```

원 요약 위치 `/validation/no_future_policy_or_new_training`:
```json
true
```


### probability_alignment_v1

유형: saved_record_analysis. 완료 태그 `probability-alignment-v1` → `9159aa764d44183c21468974ba3b0f208d676731`. 최초 소스 `104d4a2038ae33cea0ae774d896380353af8b7dc`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17701,
    17702,
    17703,
    17704,
    17705,
    17706,
    17707,
    17708,
    17709,
    17710
  ],
  "training_trials": 400,
  "parent_learning_result": "results/self_blend_learning_v1"
}
```

[등록](../results/registration_probability_alignment_v1.json) · [요약](../results/probability_alignment_v1/summary.json) · [독립 검산](../results/probability_alignment_v1/verification.json) · [보존 감사](../results/probability_alignment_archive_validation.json) · [재현](../docs/probability-alignment-reproduction.md)

원 요약 위치 `/validation/all_numeric_information_and_support_checks_passed`:
```json
true
```

원 요약 위치 `/validation/new_objectives_use_own_observed_signs_only`:
```json
true
```

원 요약 위치 `/validation/same_factual_state_action_reward_and_scalar`:
```json
true
```

원 요약 위치 `/validation/all_eligible_logits_strictly_inside_clipping_domain`:
```json
true
```


### objective_learning_v1

유형: actual_learning. 완료 태그 `objective-learning-v1` → `b96a6c9d88a9f83cbe4e093272eea0ca77f245fe`. 최초 소스 `adeed238058cfb9fb9d2e186846d310bf1f0c296`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17801,
    17802,
    17803,
    17804,
    17805,
    17806,
    17807,
    17808,
    17809,
    17810
  ],
  "training_trials": 400,
  "parent_learning_result": null
}
```

[등록](../results/registration_objective_learning_v1.json) · [요약](../results/objective_learning_v1/summary.json) · [독립 검산](../results/objective_learning_v1/verification.json) · [보존 감사](../results/objective_learning_archive_validation.json) · [재현](../docs/objective-learning-reproduction.md)

원 요약 위치 `/H_gain`:
```json
{
  "gated_uniform": {
    "supported": false,
    "endpoint_comparisons": {
      "gated_balanced": {
        "mean_effect": -0.0018705134196065677,
        "positive_seeds": 2,
        "practical_advantage_pass": false
      },
      "isolated": {
        "mean_effect": 0.006338476769223856,
        "positive_seeds": 10,
        "practical_advantage_pass": false
      },
      "gated_self_blend": {
        "mean_effect": 0.0026083562276982275,
        "positive_seeds": 9,
        "practical_advantage_pass": false
      }
    },
    "all_numeric_and_information_gates_passed": true,
    "requires_acquisition": false
  },
  "gated_probability": {
    "supported": false,
    "endpoint_comparisons": {
      "gated_balanced": {
        "mean_effect": 0.0006817172395877757,
        "positive_seeds": 8,
        "practical_advantage_pass": false
      },
      "isolated": {
        "mean_effect": 0.008890707428418198,
        "positive_seeds": 10,
        "practical_advantage_pass": false
      },
      "gated_self_blend": {
        "mean_effect": 0.005160586886892571,
        "positive_seeds": 10,
        "practical_advantage_pass": false
      }
    },
    "all_numeric_and_information_gates_passed": true,
    "requires_acquisition": false
  }
}
```

원 요약 위치 `/H_acquisition`:
```json
{
  "gated_uniform": {
    "supported": false,
    "acquisition_pass_seeds": 0,
    "all_numeric_and_information_gates_passed": true,
    "requires_gain": false
  },
  "gated_probability": {
    "supported": false,
    "acquisition_pass_seeds": 0,
    "all_numeric_and_information_gates_passed": true,
    "requires_gain": false
  }
}
```

원 요약 위치 `/H_protection`:
```json
{
  "gated_uniform": {
    "supported": true,
    "maximum_primary_seed_unknown_damage": 0.0,
    "absolute_tolerance": 1e-08,
    "comparisons": {
      "gated_balanced": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      },
      "isolated": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      }
    },
    "operational_protection_expected_by_design": true,
    "novel_learning_mechanism_demonstrated": false,
    "all_numeric_and_information_gates_passed": true
  },
  "gated_probability": {
    "supported": true,
    "maximum_primary_seed_unknown_damage": 0.0,
    "absolute_tolerance": 1e-08,
    "comparisons": {
      "gated_balanced": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      },
      "isolated": {
        "mean_effect": 0.0,
        "nonworsening_seeds": 10,
        "all_seeds_nonworsening": true
      }
    },
    "operational_protection_expected_by_design": true,
    "novel_learning_mechanism_demonstrated": false,
    "all_numeric_and_information_gates_passed": true
  }
}
```

원 요약 위치 `/H_state`:
```json
{
  "supported": false,
  "comparison": "gated_probability_minus_gated_uniform",
  "mean_effect": 0.002552230659194343,
  "positive_seeds": 10,
  "all_numeric_and_information_gates_passed": true,
  "requires_acquisition": false
}
```


### objective_pathways_v1

유형: saved_record_analysis. 완료 태그 `objective-pathways-v1` → `beec7eb421b4574231f74ea0b2ede4dbc99e1ecc`. 최초 소스 `3783491b82f61f1d956756b356e09cbae1dcbd58`.

코호트/원 경로:
```json
{
  "model_seeds": [
    17801,
    17802,
    17803,
    17804,
    17805,
    17806,
    17807,
    17808,
    17809,
    17810
  ],
  "training_trials": 400,
  "parent_learning_result": "results/objective_learning_v1"
}
```

[등록](../results/registration_objective_pathways_v1.json) · [요약](../results/objective_pathways_v1/summary.json) · [독립 검산](../results/objective_pathways_v1/verification.json) · [보존 감사](../results/objective_pathways_archive_validation.json) · [재현](../docs/objective-pathways-reproduction.md)

원 요약 위치 `/validation/all_exact_accounting_gates_passed`:
```json
true
```

원 요약 위치 `/validation/same_prefeedback_full_gate_and_prefix_checked`:
```json
true
```

원 요약 위치 `/validation/only_actual_both_Go_crossed`:
```json
true
```

원 요약 위치 `/validation/one_sided_actual_choices_retained`:
```json
true
```

원 요약 위치 `/validation/recorded_common_uniform_and_own_probabilities_checked`:
```json
true
```

원 요약 위치 `/validation/both_Go_rewards_identical_and_scalar_difference_from_probability`:
```json
true
```


## 보존 앵커

- original_phase5: [results/phase5_synthesis_v1/summary.json](../results/phase5_synthesis_v1/summary.json), [results/phase5_synthesis_v1/verification.json](../results/phase5_synthesis_v1/verification.json), [docs/phase5-final-report.md](../docs/phase5-final-report.md), [docs/phase5-evidence-appendix.md](../docs/phase5-evidence-appendix.md), [docs/phase5-reproduction.md](../docs/phase5-reproduction.md), [results/phase5_synthesis_archive_validation.json](../results/phase5_synthesis_archive_validation.json)
- drosomem: [docs/drosomem-audit.json](../docs/drosomem-audit.json), [docs/drosomem-audit.md](../docs/drosomem-audit.md), [data/flywire_783_mb_left_kc512_s701/provenance.json](../data/flywire_783_mb_left_kc512_s701/provenance.json)
- human_comparison: [results/human_shell_rng_v2/summary.json](../results/human_shell_rng_v2/summary.json), [results/human_shell_rng_v2/verification.json](../results/human_shell_rng_v2/verification.json), [docs/human-shell-findings.md](../docs/human-shell-findings.md), [docs/human-shell-rng-design-note.md](../docs/human-shell-rng-design-note.md), [docs/human-shell-reproduction.md](../docs/human-shell-reproduction.md), [results/human_shell_comparison_archive_validation.json](../results/human_shell_comparison_archive_validation.json)
- pattern: [results/pattern_rule_v1/summary.json](../results/pattern_rule_v1/summary.json), [results/pattern_rule_v1/verification.json](../results/pattern_rule_v1/verification.json), [docs/pattern-rule-findings.md](../docs/pattern-rule-findings.md), [docs/pattern-rule-protocol.md](../docs/pattern-rule-protocol.md)
- interference_order: [results/interference_order_v1/summary.json](../results/interference_order_v1/summary.json), [results/interference_order_v1/verification.json](../results/interference_order_v1/verification.json), [docs/interference-order-findings.md](../docs/interference-order-findings.md), [docs/interference-order-protocol.md](../docs/interference-order-protocol.md)
- biological_literature_limits: [docs/literature-review.md](../docs/literature-review.md), [docs/human-learning-comparison.md](../docs/human-learning-comparison.md), [LIMITATIONS.md](../LIMITATIONS.md)
