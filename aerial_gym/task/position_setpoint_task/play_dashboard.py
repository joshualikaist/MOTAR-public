"""PLAY 전용: N개 에피소드를 한 배치로 돌린 뒤 종료 시 요약만 출력."""

from __future__ import annotations

import atexit
import math
import os

_EP_DONE_TOTAL = 0
_EP_SUCC_TOTAL = 0
_EP_REWARD_SUM = 0.0
_PLAY_WALL_S = 0.0
_ATEXIT_DONE = False


def play_compact_dashboard_enabled() -> bool:
    if os.environ.get("AERIAL_RL_PLAY", "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    v = os.environ.get("AERIAL_PLAY_COMPACT", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def play_episode_target() -> int:
    try:
        n = int(os.environ.get("PLAY_GAMES_NUM", "64"))
    except ValueError:
        n = 64
    return max(1, n)


def _target_success_fraction() -> float:
    raw = (
        os.environ.get("PLAY_TARGET_SUCCESS_RATE")
        or os.environ.get("PLAY_SUCCESS_TARGET_RATE")
        or "0.9"
    ).strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.9
    if v > 1.0:
        return min(max(v / 100.0, 0.0), 1.0)
    return min(max(v, 0.0), 1.0)


def _heuristic_checklist_lines(rate: float, n_eps: int, pct: float) -> list[str]:
    thr = _target_success_fraction()
    lines = [
        " Heuristic checklist (automatic hints only — verify with TensorBoard, rewards, collisions):",
    ]
    if n_eps < 40:
        lines.append(
            f"   • {n_eps} episode(s) in this batch — rate can be noisy. Run play again or raise "
            f"PLAY_GAMES_NUM for a steadier estimate vs your {thr * 100:.0f}% gate."
        )
    if rate < 0.3:
        lines.append(
            "   • Very low success: train longer / pick a checkpoint from a newer run "
            "(best gen_ppo.pth); check truncation vs crash-heavy episodes in intercept-long mode."
        )
        lines.append(
            "   • YAML: learning_rate −30–50%; try entropy_coef ↑ slightly; stabilize with save_frequency checkpoints."
        )
        lines.append(
            "   • Task: tweak position_setpoint_task_config.reward_parameters (pos gains, "
            "dist_far_*/dist_near_*/dist_switch_m, intercept bonuses, lin_speed_pen_amp, "
            "crash_penalty, out_of_range_dist)."
        )
    elif rate < 0.65:
        lines.append(
            "   • Mid-band: extend max_epochs finish; LR schedule / smaller LR; watch reward variance collapse."
        )
        lines.append(
            "   • Reward mix: reinforce near-target terms vs speed/action penalties until "
            "TensorBoard episodic intercept trend matches intuition."
        )
        lines.append(
            "   • INTERCEPT_SUCCESS_RADIUS vs training intercept_success_radius must match your "
            "\"good intercept\" semantics."
        )
    elif rate < thr:
        lines.append(
            "   • Close to gate: marginal gains from final training epochs, mildly lower LR, "
            "or narrower success radius calibration."
        )

    lines.append(
        "   • Target gate: {:.0f}% aggregate success (PLAY_TARGET_SUCCESS_RATE; default 0.9)."
        "".format(thr * 100.0)
    )
    lines.append("   • Current aggregate: {:.1f}%.".format(pct))
    return lines


def _print_overall_summary() -> None:
    global _EP_DONE_TOTAL, _EP_SUCC_TOTAL, _EP_REWARD_SUM, _PLAY_WALL_S
    if _EP_DONE_TOTAL <= 0:
        return

    rate = float(_EP_SUCC_TOTAL) / float(_EP_DONE_TOTAL)
    pct = rate * 100.0 if math.isfinite(rate) else 0.0
    mean_r = _EP_REWARD_SUM / max(_EP_DONE_TOTAL, 1)
    target = play_episode_target()
    bar = "\u2501" * 52

    print(
        "\n"
        + bar
        + "\n"
        + "  Aerial RL  ·  PLAY (batch)\n"
        + "\n"
        + f"  episodes (target) : {target}\n"
        + f"  episodes finished : {_EP_DONE_TOTAL}\n"
        + f"  body-center intercept : {_EP_SUCC_TOTAL} / {_EP_DONE_TOTAL}  ({pct:.1f}%)\n"
        + f"    (base_link contact while center dist ≤ radius; not wing-edge graze)\n"
        + f"  mean reward       : {mean_r:,.4f}\n"
        + f"  total time (s)    : {_PLAY_WALL_S:.4f}\n"
        + bar
        + "\n",
        flush=True,
    )

    thr = _target_success_fraction()
    tgt = thr * 100.0
    if rate + 1e-9 < thr:
        print(
            " >> Evaluation did not reach the {:.0f}% intercept-success bar — treat learning as unsuccessful "
            "for this criterion.\n"
            " >> Please resume or restart training, then evaluate again until aggregate play success clears "
            "~{:.0f}%+ (unless you redefine success).\n".format(tgt, tgt),
            flush=True,
        )
        for ln in _heuristic_checklist_lines(rate, _EP_DONE_TOTAL, pct):
            print(ln, flush=True)


def _ensure_atexit() -> None:
    global _ATEXIT_DONE
    if _ATEXIT_DONE:
        return
    atexit.register(_print_overall_summary)
    _ATEXIT_DONE = True


def record_episode_outcomes_this_step(
    num_finished: int,
    num_success: int,
    *,
    ep_return_sum: float = 0.0,
    step_wall_s: float = 0.0,
) -> tuple[int, int]:
    """누적만 하고 per-step 대시보드는 출력하지 않음. (총 완료, 총 성공) 반환."""
    global _EP_DONE_TOTAL, _EP_SUCC_TOTAL, _EP_REWARD_SUM, _PLAY_WALL_S
    if num_finished <= 0:
        return _EP_DONE_TOTAL, _EP_SUCC_TOTAL
    _ensure_atexit()
    _EP_DONE_TOTAL += int(num_finished)
    _EP_SUCC_TOTAL += int(num_success)
    _EP_REWARD_SUM += float(ep_return_sum)
    _PLAY_WALL_S += max(0.0, float(step_wall_s))
    return _EP_DONE_TOTAL, _EP_SUCC_TOTAL
