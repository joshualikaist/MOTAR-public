import torch


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_velocity_control"
    args = {}
    num_envs = 4096
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 13
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 1000  # sim dt≈0.01 s → ~10 s
    return_state_before_reset = False

    # 요격 성공: 표적 접촉 + dist≤radius. URDF base_link 구 R=0.05 m → 첫 접촉 ≈0.10 m.
    intercept_success_radius = 0.11
    # empty_env.collision_force_threshold 와 동일 — 닿으면 ep 종료 = 성공 (데스존 방지).
    intercept_min_contact_force = 0.01
    no_penalty_on_intercept_contact = True
    # 정규화 기준(패치 #10): "순간·정면(ref 속도) body-center 요격" = 1000 점이 이론적 최댓값.
    #   terminal 예산 = collision_bonus(550) + hit_time_amp(350) + impact_speed_amp(100) = 1000.
    #   dense shaping 은 접근 유도용 소량(접촉 기하가 코앞 호버를 막아 누적이 제한됨).
    #   reward_global_scale=1.0 → 보고되는 ep return 이 "1000점 만점"으로 직접 읽힘.
    reward_episode_design_max = 1000.0

    reward_parameters = {
        # ---- 접근 dense shaping (거리·자세 그래디언트; 양수, 매 스텝 누적) ----
        "pos_gain_1": 1.7,
        "pos_exp_1": 10.5,
        "pos_gain_2": 1.0,
        "pos_exp_2": 5.8,
        "dist_far_ref_m": 22.0,
        "dist_far_div": 24.0,
        "dist_switch_m": 1.75,
        # 3-zone distance shaping: set dist_zone_mid_m > dist_switch_m (e.g. 1.75 / 1.2). 0 = legacy 2-zone.
        "dist_zone_mid_m": 0.0,
        "dist_mid_div": 24.0,
        # near-quad gain ↓ (2.5→1.0): 코앞 호버 dense 절대값을 낮춰 빠른 요격이 점수상 유리하게.
        "dist_near_quad_gain": 1.0,
        "lin_speed_pen_amp": 0.015,
        "lin_speed_pen_tanh_scale": 0.18,
        "upright_num": 0.22,
        "upright_eps": 0.1,
        "ang_vel_weight": 1.05,
        # posture 결합 ↓ (0.5→0.35): 코앞 최대 dense 항(pos·posture) 축소 → 누적 상한 하향.
        "posture_coupling_scale": 0.35,
        # ---- 요격 terminal 보너스 (접촉 성공 스텝, 합=1000 = 정규화 만점) ----
        "intercept_collision_bonus": 550.0,   # 기본: 깨끗한 body-center 접촉
        "intercept_hit_time_bonus_amp": 350.0, # 시간효율: 즉시→+350, 느리면→0
        "intercept_hit_time_norm_steps": 1000.0,
        "intercept_min_contact_force": 0.01,
        "intercept_impact_speed_bonus_amp": 100.0,  # "차 사고": ref 속도 정면 돌진→+100
        "intercept_impact_speed_ref_mps": 1.5,
        # ---- 약한 보조 항 (Massoud 논문 대응; 소량) ----
        "massoud_linear_dist_coef": 0.008,
        "massoud_speed_caution_coef": 0.010,
        "massoud_caution_dist_m": 3.0,
        # ---- dwell 벌점 (코앞 호버 누적 억제; ramp=0 권장) ----
        # 상수 0.25→0.5: 호버 누적 상한 ↓ (접근은 terminal 보너스가 충분히 상회해 자폭 유인 없음).
        "time_step_penalty": 0.5,
        "time_ramp_penalty_coef": 0.0,   # 비활성 — 이차 램프는 reward 붕괴 유발(패치 #6).
        "time_penalty_norm_steps": 1000.0,
        # ---- 사거리 이탈 leash (안전 경계) ----
        "out_of_range_dist": 8.0,
        "far_overshoot_pen_start_m": 3.5,
        "far_overshoot_pen_coef": 0.45,
        "far_overshoot_pen_ref_m": 5.0,
        # 거의 비활성 (드론이 표적 근처 유지 → 트리거 안 됨). 먼 이탈 안전망으로만 유지.
        "far_time_dist_penalty_dist_m": 5.0,
        "far_time_dist_penalty_min_steps": 350.0,
        "far_time_dist_penalty_coef": 0.11,
        # ---- 비활성 파라미터 (값 0 = 효과 없음; 현재 태스크에서 불필요) ----
        "crash_penalty": 0.0,                    # 접촉=성공이라 명시적 crash 벌점 없음
        "off_target_contact_reward": 0.0,        # 미달 접촉 보상 없음
        "non_intercept_crash_penalty_scale": 0.0, # 비요격 종료 벌점 없음
        # ---- 전역 스케일 (1.0 = ep return 을 1000점 만점으로 직접 해석) ----
        "reward_global_scale": 1.0,
        # Dense bonus for flying toward target (world-frame closing speed); 0 = off.
        "closing_speed_bonus_coef": 0.0,
        # 0 = legacy (any physics contact ends episode); 1 = intercept or out_of_range only.
        "terminate_only_qualifying_intercept": 0.0,
        # One-shot distance milestones (0 bonus = disabled); moving task overrides.
        "milestone_dist_1_m": 0.0,
        "milestone_dist_2_m": 0.0,
        "milestone_dist_3_m": 0.0,
        "milestone_bonus_1": 0.0,
        "milestone_bonus_2": 0.0,
        "milestone_bonus_3": 0.0,
        # Hyperbolic dist guidance: r = g/(d+r0) - g/(oor+r0); 0 = disabled (use zone shaping).
        "dist_hype_gain": 0.0,
        "dist_hype_r0": 0.25,
        # Distance-proportional time penalty: cost_per_step += coef * dist; 0 = disabled.
        "time_dist_pen_coef": 0.0,
        # Moving intercept: penalize when range is not shrinking (stall / tailgating).
        "range_stall_penalty_coef": 0.0,
        # Bonus per meter of body-center distance reduced in one RL step.
        "dist_progress_bonus_coef": 0.0,
        # Extra penalty when range grows: coef * (-range_rate) for range_rate < 0.
        "range_open_penalty_coef": 0.0,
        # No stall penalty when dist <= this (m); 0 = always apply stall penalty.
        "range_stall_exempt_dist_m": 0.0,
        # Per-step clamp on dense shaping only (intercept terminal steps are exempt).
        "reward_dense_step_floor": 0.0,
        "reward_dense_step_ceil": 0.0,
        "reward_dense_terminal_threshold": 400.0,
    }
