# docs/ 색인

`docs/` 문서 지도. 아래 상세 목록의 파일 수·줄 수는 당시 조사 기록이며 현재 전체 수가 아니다.
**이 색인은 삭제를 돕기 위한 것이 아니라 삭제를 막기 위한 것이다.**

🔒 = **소스 코드·테스트가 이 경로를 문자열로 인용한다.** 이동·이름변경·삭제하면 docstring, 실행 계약,
게이트 검증기가 깨진다. 당시 조사에서는 82개 중 **41개**가 여기 해당했다.

## 먼저 읽을 것

독립 일반 물체 렌더러: [R1 정적 경로 감사](renderer_r1_audit_2026-09-11.md) ·
[근거 소스 SHA-256](renderer_r1_audit_2026-09-11.sha256).
R1 문서화 완료 · [R2 독립 shading prototype / 인수인계](renderer_r2_prototype_2026-09-11.md).
R2 구현 이후 독립 Warp/GPU 검증을 실행했습니다. 기존 perception·policy 경로와 분리합니다.
[R3 renderer-only validation 사전등록](preregistration_renderer_r3_2026-09-11.md)은 seed 173의 두 독립 실행을 고정합니다.
[R3 결과: PASS](../results/renderer_r3_seed173_comparison_2026-09-11/README.md). 학습·perception·policy 평가는 포함하지 않습니다.
[R4/R4b FAIL 보존과 재검증](../results/renderer_reverification_2026-09-11_1951/README.md) ·
[독립 배경 기술 검증·원배열 비교](../results/renderer_background_v1_2026-09-12/README.md) ·
[V1 probe 감사: 픽셀 수 일치, 자산 계약 불일치](../results/target_appearance_in_sim_2026-09-12/AUDIT.md).
[새 checkout·NumPy-only 환경의 CPU 근거 재현](public_evidence_reproduction_2026-09-12.md).
[독립 renderer 전체 CPU 신규 설치](renderer_cpu_quickstart_2026-09-12.md) ·
[설치 검증 기준](renderer_cpu_install_plan_2026-09-12.md) ·
[커밋 후 회귀·현재 상태 정정](repository_followup_2026-09-12.md).

논문형 블록 다이어그램(2026-09-11): [README 그림 9개 전체](assets/paper/) ·
[SVG + 4K PNG + 벡터 PDF ZIP](assets/paper/motar-paper-block-diagrams.zip).

이전 결과 요약 카드(2026-09-10, 보관): [9개 파트 미리보기](assets/presentation/) ·
[16:9 SVG + 4K PNG ZIP](assets/presentation/motar-presentation-2026-09-10.zip).
각 파트의 완료 범위와 한계를 그림 안에 함께 표기합니다.

ETH ds5 원본·파생 이미지는 current tree에 배포하지 않습니다:
[외부 데이터 취득·재현 안내](external_data/ETH_DS5.md), [공개 배포 상태](public_release_status_2026-09-14.md).

2026-09-16 사이트 근거 정리:
[외부 공개 시스템과의 관계·비교 가능성](relation_to_published_systems_2026-09-16.md) ·
[TM-E0…TM-E4 target behavior ladder와 현재 motion audit](target_behavior_ladder_2026-09-16.md) ·
[machine-readable component lifecycle](research_status_registry.json).

ETH ds5 실촬영 거리 측정: [E1–E5 수집·검증 계획](plans/eth_ds5_execution_2026-09-10.md),
[E3 분리 계획(E3-S/E3-P)](plans/eth_ds5_e3_2026-09-10.md),
[E3-S 사전등록](../results/eth_ds5_e3s_2026-09-10/PREREGISTRATION.md) ·
[E3-S 결과](../results/eth_ds5_e3s_2026-09-10/README.md) ·
[E3-P 게이트(BLOCKED)](../results/eth_ds5_e3p_gate_2026-09-10/README.md) ·
[왜곡 조사와 철회](../results/eth_ds5_intake_2026-09-10/distortion_investigation_2026-09-10/README.md).

실사 인지 최신 구현: [streaming API](specs/perception_streaming_v1.md),
[P7e/streaming 계약](plans/perception_p7e_streaming_2026-09-08.md),
[결과와 미통과 gate](../results/perception_p7e_streaming_2026-09-08/README.md).

| 무엇을 알고 싶은가 | 어디를 보는가 |
|---|---|
| 지금 상태와 다음 실험 | [`VERIFICATION.md`](../VERIFICATION.md) |
| 무슨 일이 있었나 (시간순) | [`WORKLOG.md`](../WORKLOG.md) 맨 아래 |
| 가설·방법 (charter) | [`RESEARCH_PLAN.md`](../RESEARCH_PLAN.md) |
| 실행 방법 | [`OPERATIONS.md`](../OPERATIONS.md) |
| 공개용 요약 | [`README.md`](../README.md) · [라이브 대시보드](status/) |

## 경로가 계약인 문서 — 손대면 안 되는 둘

| 파일 | 왜 |
|---|---|
| [`reference_platform_proposal_2026-08.md`](reference_platform_proposal_2026-08.md) | `navrl_ref5in_quad_config.py`가 **provenance-frozen**이고 그 docstring이 이 경로를 가리킨다. 바이트 하나만 바뀌어도 `eval_navrl_v2_density_sweep.sh`가 `exit 2`. 커밋 `921fb1d`가 '정리'했다가 8개 브랜치를 깼다 |
| [`prereg_2026-08-13_detector_coupling.md`](prereg_2026-08-13_detector_coupling.md) | **스텁**이다. 원본은 archive에 있고 소스 5곳이 이 경로를 인용한다. 2026-08-20에 원본이 옮겨지며 깨졌던 것을 2026-09-05에 복구했다 |



## 사전등록 — 측정 전에 규칙을 고정한 문서

날짜순이다. 파일명 규칙이 두 가지로 갈려 있으나 **이름을 바꾸지 않는다** — 절반 이상이 소스에서 절대경로로 인용된다.

| 파일 | 줄 | 최종 커밋 | |
|---|---:|---|---|
| [`prereg_2026-08-13_detector_coupling.md`](prereg_2026-08-13_detector_coupling.md) | 20 | (신규) | 🔒 |
| [`prereg_2026-08-21_n1_real_frame_reflection_audit.md`](prereg_2026-08-21_n1_real_frame_reflection_audit.md) | 246 | 2026-08-21 | 🔒 |
| [`prereg_2026-08-22_detection_range_2stage.md`](prereg_2026-08-22_detection_range_2stage.md) | 97 | 2026-08-22 | 🔒 |
| [`prereg_2026-08-22_honest_sensor_adaptation.md`](prereg_2026-08-22_honest_sensor_adaptation.md) | 112 | 2026-08-22 |  |
| [`prereg_2026-08-22_paired_reflection_consistency.md`](prereg_2026-08-22_paired_reflection_consistency.md) | 166 | 2026-08-22 |  |
| [`prereg_2026-08-22_sensor_fidelity.md`](prereg_2026-08-22_sensor_fidelity.md) | 225 | 2026-08-22 | 🔒 |
| [`prereg_2026-08-24_distance_fidelity.md`](prereg_2026-08-24_distance_fidelity.md) | 34 | 2026-08-24 |  |
| [`prereg_2026-08-24_physical_target_speed_envelope.md`](prereg_2026-08-24_physical_target_speed_envelope.md) | 54 | 2026-08-24 |  |
| [`prereg_2026-09-01_distractor_envelope.md`](prereg_2026-09-01_distractor_envelope.md) | 197 | 2026-09-02 | 🔒 |
| [`prereg_2026-09-02_speed_governor_stopcap_screen.md`](prereg_2026-09-02_speed_governor_stopcap_screen.md) | 110 | 2026-09-02 | 🔒 |
| [`prereg_2026-09-03_s1_structure_fix_shadow.md`](prereg_2026-09-03_s1_structure_fix_shadow.md) | 91 | 2026-09-04 | 🔒 |
| [`prereg_2026-09-04_contact_corridor_forensics.md`](prereg_2026-09-04_contact_corridor_forensics.md) | 284 | 2026-09-05 | 🔒 |
| [`prereg_2026-09-04_depth_noise_model_order.md`](prereg_2026-09-04_depth_noise_model_order.md) | 151 | 2026-09-04 | 🔒 |
| [`preregistration_active_search_geofence_2026-08-21.md`](preregistration_active_search_geofence_2026-08-21.md) | 62 | 2026-08-22 | 🔒 |
| [`preregistration_braking_aware_route_v3_2026-09-01.md`](preregistration_braking_aware_route_v3_2026-09-01.md) | 97 | 2026-09-01 | 🔒 |
| [`preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md`](preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md) | 66 | 2026-09-01 | 🔒 |
| [`preregistration_braking_aware_route_v3_lower1p25_matched_spawn_2026-09-01.md`](preregistration_braking_aware_route_v3_lower1p25_matched_spawn_2026-09-01.md) | 66 | 2026-09-01 | 🔒 |
| [`preregistration_braking_aware_route_v3_lower1p25_matched_spawn_gpu_authority_2026-09-01.md`](preregistration_braking_aware_route_v3_lower1p25_matched_spawn_gpu_authority_2026-09-01.md) | 53 | 2026-09-01 |  |
| [`preregistration_corrected_nonoverlap_physical_off_curriculum_2026-09-01.md`](preregistration_corrected_nonoverlap_physical_off_curriculum_2026-09-01.md) | 51 | 2026-09-01 |  |
| [`preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md`](preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md) | 59 | 2026-09-02 | 🔒 |
| [`preregistration_corrected_nonoverlap_physical_off_smoke_2026-09-01.md`](preregistration_corrected_nonoverlap_physical_off_smoke_2026-09-01.md) | 53 | 2026-09-01 |  |
| [`preregistration_corrected_nonoverlap_route_gate_2026-08-31.md`](preregistration_corrected_nonoverlap_route_gate_2026-08-31.md) | 64 | 2026-08-31 | 🔒 |
| [`preregistration_corrected_nonoverlap_route_gate_r2_2026-08-31.md`](preregistration_corrected_nonoverlap_route_gate_r2_2026-08-31.md) | 51 | 2026-08-31 | 🔒 |
| [`preregistration_navrl_physical_target_braking_2026-08-25.md`](preregistration_navrl_physical_target_braking_2026-08-25.md) | 58 | 2026-08-25 | 🔒 |
| [`preregistration_navrl_physical_target_braking_lower1p25_2026-08-26.md`](preregistration_navrl_physical_target_braking_lower1p25_2026-08-26.md) | 15 | 2026-08-26 | 🔒 |
| [`preregistration_navrl_v2_corrected_density_geometry_2026-08-27.md`](preregistration_navrl_v2_corrected_density_geometry_2026-08-27.md) | 81 | 2026-08-27 | 🔒 |
| [`preregistration_physical_target_global_route_2026-08-25.md`](preregistration_physical_target_global_route_2026-08-25.md) | 77 | 2026-08-25 |  |
| [`preregistration_physical_target_recovery_v2_gate_2026-08-25.md`](preregistration_physical_target_recovery_v2_gate_2026-08-25.md) | 70 | 2026-08-26 | 🔒 |
| [`preregistration_physical_target_recovery_v2_lower1p25_gate_2026-08-26.md`](preregistration_physical_target_recovery_v2_lower1p25_gate_2026-08-26.md) | 15 | 2026-08-26 | 🔒 |
| [`preregistration_physical_target_recovery_v2_no_connector_forensics_2026-08-26.md`](preregistration_physical_target_recovery_v2_no_connector_forensics_2026-08-26.md) | 128 | 2026-08-26 | 🔒 |
| [`preregistration_physical_target_route_recovery_forensics_2026-08-25.md`](preregistration_physical_target_route_recovery_forensics_2026-08-25.md) | 124 | 2026-08-25 |  |
| [`preregistration_physical_target_routed_simulator_gate_2026-08-25.md`](preregistration_physical_target_routed_simulator_gate_2026-08-25.md) | 129 | 2026-08-25 | 🔒 |
| [`preregistration_physical_target_speed_controller_calibration_2026-08-26.md`](preregistration_physical_target_speed_controller_calibration_2026-08-26.md) | 41 | 2026-08-26 | 🔒 |
| [`preregistration_physical_target_speed_controller_calibration_stage2_2026-08-26.md`](preregistration_physical_target_speed_controller_calibration_stage2_2026-08-26.md) | 33 | 2026-08-26 | 🔒 |
| [`preregistration_physical_target_two_envelope_recovery_2026-08-25.md`](preregistration_physical_target_two_envelope_recovery_2026-08-25.md) | 132 | 2026-08-25 | 🔒 |
| [`preregistration_s1_blind_search_state_2026-09-03.md`](preregistration_s1_blind_search_state_2026-09-03.md) | 257 | 2026-09-03 | 🔒 |
| [`preregistration_sam_instance_adapter_offline_2026-09-03.md`](preregistration_sam_instance_adapter_offline_2026-09-03.md) | 72 | 2026-09-03 |  |

## 결과 영수증

판정의 근거 자체다.

| 파일 | 줄 | 최종 커밋 | |
|---|---:|---|---|
| [`braking_aware_route_v3_lower1p25_matched_spawn_result_2026-09-01.md`](braking_aware_route_v3_lower1p25_matched_spawn_result_2026-09-01.md) | 73 | 2026-09-01 |  |
| [`corrected_nonoverlap_route_gate_r2_result_2026-08-31.md`](corrected_nonoverlap_route_gate_r2_result_2026-08-31.md) | 72 | 2026-08-31 | 🔒 |
| [`physical_target_recovery_v2_lower1p25_result_2026-08-26.md`](physical_target_recovery_v2_lower1p25_result_2026-08-26.md) | 95 | 2026-08-26 | 🔒 |
| [`physical_target_recovery_v2_no_connector_forensics_result_2026-08-26.md`](physical_target_recovery_v2_no_connector_forensics_result_2026-08-26.md) | 79 | 2026-08-26 |  |
| [`physical_target_route_recovery_result_2026-08-25.md`](physical_target_route_recovery_result_2026-08-25.md) | 49 | 2026-08-26 |  |

## 계획

SUPERSEDED 배너가 있는 것은 그 배너가 곧 provenance다.

| 파일 | 줄 | 최종 커밋 | |
|---|---:|---|---|
| [`plans/perception_final_implementation_plan_2026-09-07.md`](plans/perception_final_implementation_plan_2026-09-07.md) | — | 2026-09-07 | **IMPLEMENTATION AUTHORITY** |
| [`plans/perception_p3_detfly_execution_2026-09-07.md`](plans/perception_p3_detfly_execution_2026-09-07.md) | — | 2026-09-07 | **P3 COMPLETE** |
| [`plans/perception_joint_dataset_v1_2026-09-07.md`](plans/perception_joint_dataset_v1_2026-09-07.md) | — | 2026-09-07 | **SPLIT + DATASET COMPLETE** |
| [`plans/perception_joint_detector_training_2026-09-08.md`](plans/perception_joint_detector_training_2026-09-08.md) | — | 2026-09-08 | **JOINT DETECTOR COMPLETE** |
| [`plans/perception_p4_p5_execution_2026-09-08.md`](plans/perception_p4_p5_execution_2026-09-08.md) | — | 2026-09-08 | **P4 PASS · P5 KF REJECTED** |
| [`plans/perception_p6_p7_execution_2026-09-08.md`](plans/perception_p6_p7_execution_2026-09-08.md) | — | 2026-09-08 | **P6 COMPLETE · P7 TRANSFORMER SELECTED ON VAL** |
| [`plans/perception_p7b_p7c_execution_2026-09-08.md`](plans/perception_p7b_p7c_execution_2026-09-08.md) | — | 2026-09-08 | **COMPLETE · corrected P7c v2 SELECTED** |
| [`plans/perception_p7d_identity_boundary_2026-09-08.md`](plans/perception_p7d_identity_boundary_2026-09-08.md) | — | 2026-09-08 | **TRACK-ID DATA REQUIRED** |
| [`SAM3_PERCEPTION_VERIFICATION_PLAN_2026-09-03.md`](SAM3_PERCEPTION_VERIFICATION_PLAN_2026-09-03.md) | 266 | 2026-09-04 | 🔒 |
| [`SIM2REAL_3DAY_EXECUTION_PLAN.md`](SIM2REAL_3DAY_EXECUTION_PLAN.md) | 343 | 2026-08-26 | 🔒 |
| [`plans/perception_shape_temporal_redesign_2026-09-03.md`](plans/perception_shape_temporal_redesign_2026-09-03.md) | 207 | 2026-09-04 | 🔒 |
| [`plans/target_search_and_adversarial_evader.md`](plans/target_search_and_adversarial_evader.md) | 391 | 2026-09-03 |  |

## 사양

2026-08 PPT/논문 브리프 묶음은 2026-09-05에 삭제했다 — 판정이 뒤집혀 낡은 수치를 담고 있었고 소스 잠금이 없었다.

| 파일 | 줄 | 최종 커밋 | |
|---|---:|---|---|
| [`MOTAR_SYSTEM_SPEC_2026-08-24.md`](MOTAR_SYSTEM_SPEC_2026-08-24.md) | 295 | 2026-08-31 |  |

## 감사·조사

| 파일 | 줄 | 최종 커밋 | |
|---|---:|---|---|
| [`diagnostic_synthesis_2026-08-21.md`](diagnostic_synthesis_2026-08-21.md) | 70 | 2026-08-22 | 🔒 |
| [`discipline_review_2026-08-22.md`](discipline_review_2026-08-22.md) | 93 | 2026-08-24 |  |
| [`navrl_import_origin_audit_2026-08-22.md`](navrl_import_origin_audit_2026-08-22.md) | 53 | 2026-08-22 |  |
| [`navrl_physical_target_audit_2026-08-21.md`](navrl_physical_target_audit_2026-08-21.md) | 84 | 2026-08-22 |  |
| [`safety_filter_survey_2026-09-02.md`](safety_filter_survey_2026-09-02.md) | 264 | 2026-09-02 |  |
| [`target_motion_training_environment_audit_2026-08-25.md`](target_motion_training_environment_audit_2026-08-25.md) | 258 | 2026-08-26 |  |

## 기타

| 파일 | 줄 | 최종 커밋 | |
|---|---:|---|---|
| [`README.md`](README.md) | 156 | (신규) | 🔒 |
| [`execution_plan_2026-08-23_detection_range_stage1.md`](execution_plan_2026-08-23_detection_range_stage1.md) | 101 | 2026-08-23 |  |
| [`navrl_frame_selection_2026-08-23.md`](navrl_frame_selection_2026-08-23.md) | 66 | 2026-08-23 |  |
| [`navrl_joint_speed_preregistration_2026-08-21.md`](navrl_joint_speed_preregistration_2026-08-21.md) | 46 | 2026-08-22 |  |
| [`navrl_ref5in_bom_direction_2026-08-23.md`](navrl_ref5in_bom_direction_2026-08-23.md) | 76 | 2026-08-23 |  |
| [`navrl_ref5in_carrier_screen_2026-08-23.md`](navrl_ref5in_carrier_screen_2026-08-23.md) | 72 | 2026-08-23 |  |
| [`navrl_ref5in_component_screen_2026-08-23.md`](navrl_ref5in_component_screen_2026-08-23.md) | 63 | 2026-08-23 |  |
| [`navrl_ref5in_payload_packaging_contract_2026-08-23.md`](navrl_ref5in_payload_packaging_contract_2026-08-23.md) | 65 | 2026-08-23 |  |
| [`navrl_ref5in_thrust_stand_protocol_2026-08-23.md`](navrl_ref5in_thrust_stand_protocol_2026-08-23.md) | 73 | 2026-08-23 |  |
| [`navrl_ref5in_vendor_request_2026-08-23.md`](navrl_ref5in_vendor_request_2026-08-23.md) | 38 | 2026-08-23 |  |
| [`reference_platform_proposal_2026-08.md`](reference_platform_proposal_2026-08.md) | 15 | 2026-08-21 | 🔒 |
| [`topology_layout_snapshot_contract.md`](topology_layout_snapshot_contract.md) | 71 | 2026-08-22 |  |

## 아카이브 — 역사 기록

판정과 다음 단계는 `VERIFICATION.md`/`WORKLOG.md`를 본다. 🔒는 소스가 여전히 참조한다.

| 파일 | 줄 | 최종 커밋 | |
|---|---:|---|---|
| [`archive/CLAUDE_PPT_REVIEW_REQUEST_VERIFICATION5B_2026-08-13.md`](archive/CLAUDE_PPT_REVIEW_REQUEST_VERIFICATION5B_2026-08-13.md) | 412 | 2026-08-20 |  |
| [`archive/NEXT_WEEK_HANDOFF_2026-08-10.md`](archive/NEXT_WEEK_HANDOFF_2026-08-10.md) | 169 | 2026-08-20 |  |
| [`archive/README.md`](archive/README.md) | 38 | 2026-08-20 | 🔒 |
| [`archive/RESEARCH_PLAN_v2_history.md`](archive/RESEARCH_PLAN_v2_history.md) | 579 | 2026-08-20 |  |
| [`archive/codex_review_2026-08-10.md`](archive/codex_review_2026-08-10.md) | 220 | 2026-08-20 |  |
| [`archive/codex_review_2026-08-12.md`](archive/codex_review_2026-08-12.md) | 162 | 2026-08-20 |  |
| [`archive/development_directions_2026-08.md`](archive/development_directions_2026-08.md) | 150 | 2026-08-20 | 🔒 |
| [`archive/midterm_summary_2026-08.md`](archive/midterm_summary_2026-08.md) | 278 | 2026-08-20 |  |
| [`archive/prereg_2026-08-13_detector_coupling.md`](archive/prereg_2026-08-13_detector_coupling.md) | 163 | 2026-08-20 | 🔒 |
| [`archive/prereg_2026-08-14_detector_coupling_binbias.md`](archive/prereg_2026-08-14_detector_coupling_binbias.md) | 61 | 2026-08-20 | 🔒 |
| [`archive/presentation_followup_2026-08-14.md`](archive/presentation_followup_2026-08-14.md) | 95 | 2026-08-20 |  |
| [`archive/ref5in_audit_and_next_steps_2026-08-13.md`](archive/ref5in_audit_and_next_steps_2026-08-13.md) | 89 | 2026-08-20 |  |
| [`archive/reference_platform_proposal_2026-08.md`](archive/reference_platform_proposal_2026-08.md) | 252 | 2026-08-20 | 🔒 |
| [`archive/review_brief_2026-08-10.md`](archive/review_brief_2026-08-10.md) | 146 | 2026-08-20 |  |
| [`archive/review_brief_2026-08-12_verification_1_2.md`](archive/review_brief_2026-08-12_verification_1_2.md) | 149 | 2026-08-20 |  |
| [`archive/review_brief_2026-08-12_verification_3_4.md`](archive/review_brief_2026-08-12_verification_3_4.md) | 117 | 2026-08-20 |  |
| [`archive/sim_vs_hardware_gap_2026-08.md`](archive/sim_vs_hardware_gap_2026-08.md) | 174 | 2026-08-23 | 🔒 |
