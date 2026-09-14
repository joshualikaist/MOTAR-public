"""Phase-3 runtime smoke: static regression + moving-target behavior, one env build.

Run (GPU free):  PYTHONNOUSERSITE=1 python tools/test_navrl_p3_smoke.py
Companion regression eval (strongest gate):
  NAVRL_NUM_BARS=110 NUM_ENVS=512 HEADLESS=True PLAY_GAMES_NUM=3000 \
    ./play_navrl.sh runs/ppo_260716_1223_navrl/nn/gen_ppo.pth   # expect captured ~0.93+
"""
from aerial_gym.registry.task_registry import task_registry  # isaacgym before torch
import torch

N = 16
FAILS = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)

task = task_registry.make_task("navrl_task", seed=7, num_envs=N, headless=True, use_warp=True)
print(f"[smoke] built: obs={task.observation_space['observations'].shape} act={task.action_space.shape} "
      f"step_dt={task.step_dt:.3f}s")
check("obs dim 156", task.observation_space["observations"].shape[0] == 156)
check("act dim 4", task.action_space.shape[0] == 4)
check("step_dt == 0.1", abs(task.step_dt - 0.1) < 1e-9)

# ---------- Phase A: static (v_t = 0 default) ----------
task.reset()
tgt0 = task.target_position.clone()
act = torch.zeros(N, 4, device=task.device); act[:, 0] = 0.5
ok_static = True; ok_finite = True
for _ in range(20):
    obs, rew, term, trunc, info = task.step(act)
    ok_static &= bool(torch.equal(task.target_position, tgt0) or (task.target_position - tgt0).abs().max() < 1e-9)
    ok_finite &= bool(torch.isfinite(rew).all() and torch.isfinite(obs["observations"]).all())
    # resets change targets; re-baseline on any reset
    done = (term > 0) | (trunc > 0)
    if bool(done.any()):
        tgt0 = task.target_position.clone()
check("static target never moves", ok_static)
check("static rewards/obs finite", ok_finite)
check("static target_vel_w all zero", bool((task.target_vel_w == 0).all()))

# ---------- Phase B: moving target, fixed speed (dt-bug detector) ----------
for pattern, code in (("cv", 0), ("waypoint", 1), ("circle", 2)):
    task.tm.speed_fixed = 1.5
    task.tm.pattern = pattern
    task.reset()
    check(f"{pattern}: sampled speed == 1.5", bool((task._tm_speed - 1.5).abs().max() < 1e-6))
    prev = task.target_position[:, 0:2].clone()
    speeds = []
    viol_wall = 0; viol_bar = 0; soft_n = 0; soft_total = 0
    m = float(task.cur.wall_margin)
    b_min = task.obs_dict["env_bounds_min"][:, 0:2]; b_max = task.obs_dict["env_bounds_max"][:, 0:2]
    for i in range(30):
        obs, rew, term, trunc, info = task.step(act)
        cur = task.target_position[:, 0:2]
        done = (term > 0) | (trunc > 0)
        live = ~done
        if bool(live.any()):
            step_d = (cur[live] - prev[live]).norm(dim=1)
            speeds.append((step_d / task.step_dt))
        prev = cur.clone()
        # bounds check (allow tiny numeric slack)
        viol_wall += int(((cur < b_min + m - 1e-4) | (cur > b_max - m + 1e-4)).sum())
        bars = task.obs_dict["obstacle_position"][:, : task.n_bars_active, 0:2]
        dbar = torch.cdist(cur.unsqueeze(1), bars).squeeze(1).min(dim=1).values
        viol_bar += int((dbar < 0.9).sum())          # HARD: capture sphere touches bar surface
        soft_n += int((dbar < 1.0 - 1e-3).sum())     # SOFT: nominal clearance not met
        soft_total += N
        if not bool(torch.isfinite(rew).all()):
            check(f"{pattern}: rewards finite", False); break
    sp = torch.cat(speeds)
    med = float(sp.median())
    # cv/waypoint should track 1.5 closely; circle chord speed slightly under arc speed;
    # reflections/push-outs/waypoint-arrivals cause occasional deviations -> use the median.
    tol = 0.10 if pattern != "circle" else 0.15
    check(f"{pattern}: realized speed ~= 1.5 (median {med:.3f})", abs(med - 1.5) < tol,
          f"(min {float(sp.min()):.2f} max {float(sp.max()):.2f})")
    check(f"{pattern}: inside wall margins (violations={viol_wall})", viol_wall == 0)
    check(f"{pattern}: capture sphere never overlaps a bar (d>=0.9, hard viol={viol_bar})", viol_bar == 0)
    soft_rate = soft_n / max(1, soft_total)
    # Residual 0.9-1.0 m band is structural: a bar ON the wall margin line forces the final wall
    # clamp to pull a pushed target slightly back in. The HARD invariant (capture sphere never
    # overlaps a bar, d >= 0.9) is what correctness requires and is asserted to be violation-free.
    check(f"{pattern}: nominal 1.0 m clearance mostly held (soft rate {soft_rate:.3f})", soft_rate < 0.10)
    check(f"{pattern}: target_vel_w magnitude sane", bool(task.target_vel_w[:, 0:2].norm(dim=1).max() <= 1.5 * 1.2 + 1e-3))

# ---------- Phase C: curriculum + env-state ----------
task.tm.speed_fixed = -1.0
task.tm.speed_final = 1.5
task.num_task_steps = 0
check("v_max(epoch 0) == 0", task._target_speed_max() == 0.0)
task.num_task_steps = 1500 * 32
check("v_max(epoch 1500) == 0.75", abs(task._target_speed_max() - 0.75) < 1e-9)
task.num_task_steps = 6000 * 32
check("v_max(epoch 6000) == 1.5", abs(task._target_speed_max() - 1.5) < 1e-9)
st = task.get_env_state()
check("env_state keys", st.get("num_task_steps") == 6000 * 32 and "n_bars_active" in st)

print("\n[smoke] RESULT:", "ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
raise SystemExit(0 if not FAILS else 1)
