"""
Per-run training recorder + terminal post-run summary.

Layout (under each rl-games run folder, e.g. runs/ppo_YYMMDD_HHMM/):
  aerial_run/epoch_metrics.csv   — one row per PPO epoch (append/resume-safe)
  aerial_run/run_summary.json    — last exit diagnostics (overwritten each finalize)

Disable boxed summary: AERIAL_RUN_SUMMARY=0
"""

from __future__ import annotations

import csv
import importlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_EPOCH_CSV = "epoch_metrics.csv"
_SUMMARY_JSON = "run_summary.json"
_SUBDIR = "aerial_run"
_NA = "\u2014"


def _or_na(v: Any) -> str:
    if v is None:
        return _NA
    return str(v)

_CSV_FIELDS = (
    "epoch",
    "mean_reward",
    "mean_episode_length",
    "mean_target_dist_m",
    "mean_closest_target_dist_m",
    "success_closest_target_dist_m",
    "failure_closest_target_dist_m",
    "mean_surface_gap_m",
    "failure_surface_gap_m",
    "near_miss_count",
    "near_miss_rate",
    "radius_no_contact_count",
    "radius_no_contact_rate",
    "contact_no_radius_count",
    "contact_no_radius_rate",
    "intercept_succ",
    "intercept_done",
    "intercept_envs_hit",
    "num_parallel_envs",
    # navrl_task per-epoch metrics (empty for the intercept task)
    "captured_rate",
    "crash_rate",
    "timeout_rate",
    "closest_approach_m",
    "closest_min_m",
    "curriculum_max_m",
    "n_bars_active",
)


def _summary_enabled() -> bool:
    return os.environ.get("AERIAL_RUN_SUMMARY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _episode_len_max_hint() -> float:
    raw_ep_max = os.environ.get("AERIAL_TRAIN_EP_LEN_MAX", "").strip()
    if raw_ep_max:
        try:
            return float(raw_ep_max)
        except ValueError:
            pass
    cfg_mod = os.environ.get("AERIAL_TASK_CONFIG_MODULE", "").strip()
    if cfg_mod:
        try:
            mod = importlib.import_module(cfg_mod)
            return float(mod.task_config.episode_len_steps)
        except Exception:
            pass
    try:
        from aerial_gym.config.task_config.position_setpoint_task_config import task_config

        return float(task_config.episode_len_steps)
    except Exception:
        return 1000.0


def run_root_from_nn_dir(nn_dir: str | Path) -> Path:
    return Path(nn_dir).resolve().parent


@dataclass
class EpochRow:
    epoch: int
    mean_reward: Optional[float] = None
    mean_episode_length: Optional[float] = None
    mean_target_dist_m: Optional[float] = None
    mean_closest_target_dist_m: Optional[float] = None
    success_closest_target_dist_m: Optional[float] = None
    failure_closest_target_dist_m: Optional[float] = None
    mean_surface_gap_m: Optional[float] = None
    failure_surface_gap_m: Optional[float] = None
    near_miss_count: int = 0
    near_miss_rate: Optional[float] = None
    radius_no_contact_count: int = 0
    radius_no_contact_rate: Optional[float] = None
    contact_no_radius_count: int = 0
    contact_no_radius_rate: Optional[float] = None
    intercept_succ: int = 0
    intercept_done: int = 0
    intercept_envs_hit: int = 0
    num_parallel_envs: int = 0
    # navrl_task per-epoch metrics (None for the intercept task)
    captured_rate: Optional[float] = None
    crash_rate: Optional[float] = None
    timeout_rate: Optional[float] = None
    closest_approach_m: Optional[float] = None  # mean over NON-CRASH finished episodes
    closest_min_m: Optional[float] = None  # best (min) closest approach
    curriculum_max_m: Optional[float] = None
    n_bars_active: Optional[int] = None


@dataclass
class RunDiagnostics:
    exit_reason: str = "unknown"
    run_root: str = ""
    epochs_logged: int = 0
    first_epoch: Optional[int] = None
    last_epoch: Optional[int] = None
    last_mean_reward: Optional[float] = None
    last_mean_episode_length: Optional[float] = None
    last_mean_target_dist_m: Optional[float] = None
    last_mean_closest_target_dist_m: Optional[float] = None
    last_failure_closest_target_dist_m: Optional[float] = None
    last_failure_surface_gap_m: Optional[float] = None
    last_near_miss_rate: Optional[float] = None
    last_radius_no_contact_rate: Optional[float] = None
    last_contact_no_radius_rate: Optional[float] = None
    peak_mean_reward: Optional[float] = None
    peak_epoch: Optional[int] = None
    reward_at_end: Optional[float] = None
    reward_collapse: bool = False
    collapse_detail: str = ""
    total_intercept_succ: int = 0
    total_intercept_done: int = 0
    total_near_miss_count: int = 0
    total_radius_no_contact_count: int = 0
    total_contact_no_radius_count: int = 0
    max_intercept_envs_hit: int = 0
    first_intercept_epoch: Optional[int] = None
    first_short_episode_epoch: Optional[int] = None
    mean_dist_start: Optional[float] = None
    mean_dist_end: Optional[float] = None
    # navrl_task summary (populated only for navrl runs)
    is_navrl: bool = False
    last_captured_rate: Optional[float] = None
    last_crash_rate: Optional[float] = None
    last_timeout_rate: Optional[float] = None
    last_closest_approach_m: Optional[float] = None  # mean over non-crash episodes
    last_closest_min_m: Optional[float] = None
    last_curriculum_max_m: Optional[float] = None
    last_n_bars_active: Optional[int] = None
    peak_captured_rate: Optional[float] = None
    peak_captured_epoch: Optional[int] = None
    hints: list[str] = field(default_factory=list)


class TrainRunRecorder:
    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root).resolve()
        self.metrics_dir = self.run_root / _SUBDIR
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir / _EPOCH_CSV
        self.summary_path = self.metrics_dir / _SUMMARY_JSON
        self._rows: list[EpochRow] = []
        self._load_existing_csv()

    def _load_existing_csv(self) -> None:
        if not self.csv_path.is_file():
            return
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            for raw in csv.DictReader(f):
                try:
                    ep = int(raw["epoch"])
                except (KeyError, ValueError):
                    continue
                self._rows.append(
                    EpochRow(
                        epoch=ep,
                        mean_reward=_f(raw.get("mean_reward")),
                        mean_episode_length=_f(raw.get("mean_episode_length")),
                        mean_target_dist_m=_f(raw.get("mean_target_dist_m")),
                        mean_closest_target_dist_m=_f(raw.get("mean_closest_target_dist_m")),
                        success_closest_target_dist_m=_f(raw.get("success_closest_target_dist_m")),
                        failure_closest_target_dist_m=_f(raw.get("failure_closest_target_dist_m")),
                        mean_surface_gap_m=_f(raw.get("mean_surface_gap_m")),
                        failure_surface_gap_m=_f(raw.get("failure_surface_gap_m")),
                        near_miss_count=_i(raw.get("near_miss_count")),
                        near_miss_rate=_f(raw.get("near_miss_rate")),
                        radius_no_contact_count=_i(raw.get("radius_no_contact_count")),
                        radius_no_contact_rate=_f(raw.get("radius_no_contact_rate")),
                        contact_no_radius_count=_i(raw.get("contact_no_radius_count")),
                        contact_no_radius_rate=_f(raw.get("contact_no_radius_rate")),
                        intercept_succ=_i(raw.get("intercept_succ")),
                        intercept_done=_i(raw.get("intercept_done")),
                        intercept_envs_hit=_i(raw.get("intercept_envs_hit")),
                        num_parallel_envs=_i(raw.get("num_parallel_envs")),
                        captured_rate=_f(raw.get("captured_rate")),
                        crash_rate=_f(raw.get("crash_rate")),
                        timeout_rate=_f(raw.get("timeout_rate")),
                        closest_approach_m=_f(raw.get("closest_approach_m")),
                        closest_min_m=_f(raw.get("closest_min_m")),
                        curriculum_max_m=_f(raw.get("curriculum_max_m")),
                        n_bars_active=_i_opt(raw.get("n_bars_active")),
                    )
                )

    def record_epoch(
        self,
        *,
        epoch: int,
        mean_reward: Optional[float] = None,
        mean_episode_length: Optional[float] = None,
        mean_target_dist_m: Optional[float] = None,
        mean_closest_target_dist_m: Optional[float] = None,
        intercept_succ: int = 0,
        intercept_done: int = 0,
        intercept_envs_hit: int = 0,
        num_parallel_envs: int = 0,
        extra_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        extra = extra_metrics or {}
        row = EpochRow(
            epoch=int(epoch),
            mean_reward=mean_reward,
            mean_episode_length=mean_episode_length,
            mean_target_dist_m=mean_target_dist_m,
            mean_closest_target_dist_m=mean_closest_target_dist_m,
            success_closest_target_dist_m=_f(extra.get("success_closest_target_dist_m")),
            failure_closest_target_dist_m=_f(extra.get("failure_closest_target_dist_m")),
            mean_surface_gap_m=_f(extra.get("mean_surface_gap_m")),
            failure_surface_gap_m=_f(extra.get("failure_surface_gap_m")),
            near_miss_count=_i(extra.get("near_miss_count")),
            near_miss_rate=_f(extra.get("near_miss_rate")),
            radius_no_contact_count=_i(extra.get("radius_no_contact_count")),
            radius_no_contact_rate=_f(extra.get("radius_no_contact_rate")),
            contact_no_radius_count=_i(extra.get("contact_no_radius_count")),
            contact_no_radius_rate=_f(extra.get("contact_no_radius_rate")),
            intercept_succ=int(intercept_succ),
            intercept_done=int(intercept_done),
            intercept_envs_hit=int(intercept_envs_hit),
            num_parallel_envs=int(num_parallel_envs),
            captured_rate=_f(extra.get("navrl/captured_rate")),
            crash_rate=_f(extra.get("navrl/crash_rate")),
            timeout_rate=_f(extra.get("navrl/timeout_rate")),
            closest_approach_m=_f(extra.get("navrl/closest_nocrash_m")),
            closest_min_m=_f(extra.get("navrl/closest_min_m")),
            curriculum_max_m=_f(extra.get("navrl/curriculum_goal_dist_max_m")),
            n_bars_active=_i_opt(extra.get("navrl/n_bars_active")),
        )
        # Replace same epoch on resume overlap
        self._rows = [r for r in self._rows if r.epoch != row.epoch]
        self._rows.append(row)
        self._rows.sort(key=lambda r: r.epoch)

        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            w.writeheader()
            for r in self._rows:
                w.writerow(
                    {
                        "epoch": r.epoch,
                        "mean_reward": _cell(r.mean_reward),
                        "mean_episode_length": _cell(r.mean_episode_length),
                        "mean_target_dist_m": _cell(r.mean_target_dist_m),
                        "mean_closest_target_dist_m": _cell(r.mean_closest_target_dist_m),
                        "success_closest_target_dist_m": _cell(r.success_closest_target_dist_m),
                        "failure_closest_target_dist_m": _cell(r.failure_closest_target_dist_m),
                        "mean_surface_gap_m": _cell(r.mean_surface_gap_m),
                        "failure_surface_gap_m": _cell(r.failure_surface_gap_m),
                        "near_miss_count": r.near_miss_count,
                        "near_miss_rate": _cell(r.near_miss_rate),
                        "radius_no_contact_count": r.radius_no_contact_count,
                        "radius_no_contact_rate": _cell(r.radius_no_contact_rate),
                        "contact_no_radius_count": r.contact_no_radius_count,
                        "contact_no_radius_rate": _cell(r.contact_no_radius_rate),
                        "intercept_succ": r.intercept_succ,
                        "intercept_done": r.intercept_done,
                        "intercept_envs_hit": r.intercept_envs_hit,
                        "num_parallel_envs": r.num_parallel_envs,
                        "captured_rate": _cell(r.captured_rate),
                        "crash_rate": _cell(r.crash_rate),
                        "timeout_rate": _cell(r.timeout_rate),
                        "closest_approach_m": _cell(r.closest_approach_m),
                        "closest_min_m": _cell(r.closest_min_m),
                        "curriculum_max_m": _cell(r.curriculum_max_m),
                        "n_bars_active": _cell(r.n_bars_active),
                    }
                )

    def analyze(self, exit_reason: str) -> RunDiagnostics:
        diag = RunDiagnostics(exit_reason=exit_reason, run_root=str(self.run_root))
        rows = [r for r in self._rows if r.mean_reward is not None]
        if not rows:
            diag.hints.append("epoch 지표가 없습니다 — 학습이 1 epoch도 끝나지 않았을 수 있습니다.")
            return diag

        diag.epochs_logged = len(self._rows)
        diag.first_epoch = rows[0].epoch
        diag.last_epoch = rows[-1].epoch
        diag.last_mean_reward = rows[-1].mean_reward
        diag.last_mean_episode_length = rows[-1].mean_episode_length
        diag.last_mean_target_dist_m = rows[-1].mean_target_dist_m
        diag.last_mean_closest_target_dist_m = rows[-1].mean_closest_target_dist_m
        diag.last_failure_closest_target_dist_m = rows[-1].failure_closest_target_dist_m
        diag.last_failure_surface_gap_m = rows[-1].failure_surface_gap_m
        diag.last_near_miss_rate = rows[-1].near_miss_rate
        diag.last_radius_no_contact_rate = rows[-1].radius_no_contact_rate
        diag.last_contact_no_radius_rate = rows[-1].contact_no_radius_rate
        diag.reward_at_end = rows[-1].mean_reward

        peak = max(rows, key=lambda r: r.mean_reward or float("-inf"))
        diag.peak_mean_reward = peak.mean_reward
        diag.peak_epoch = peak.epoch

        diag.total_intercept_succ = sum(r.intercept_succ for r in self._rows)
        diag.total_intercept_done = sum(r.intercept_done for r in self._rows)
        diag.total_near_miss_count = sum(r.near_miss_count for r in self._rows)
        diag.total_radius_no_contact_count = sum(r.radius_no_contact_count for r in self._rows)
        diag.total_contact_no_radius_count = sum(r.contact_no_radius_count for r in self._rows)
        diag.max_intercept_envs_hit = max((r.intercept_envs_hit for r in self._rows), default=0)
        ep_max = _episode_len_max_hint()

        for r in self._rows:
            if diag.first_intercept_epoch is None and (
                r.intercept_succ > 0 or r.intercept_envs_hit > 0
            ):
                diag.first_intercept_epoch = r.epoch
            if diag.first_short_episode_epoch is None and r.mean_episode_length is not None:
                if r.mean_episode_length < ep_max - 1.0:
                    diag.first_short_episode_epoch = r.epoch

        dist_rows = [r for r in rows if r.mean_target_dist_m is not None]
        if dist_rows:
            n = max(1, len(dist_rows) // 10)
            diag.mean_dist_start = sum(r.mean_target_dist_m for r in dist_rows[:n]) / n
            diag.mean_dist_end = sum(r.mean_target_dist_m for r in dist_rows[-n:]) / n

        # Collapse: peak then sustained drop (like ppo_260530_1604: 785 → ~280)
        if peak.mean_reward is not None and peak.epoch is not None:
            after = [r for r in rows if r.epoch > peak.epoch and r.mean_reward is not None]
            if after:
                trough = min(after, key=lambda r: r.mean_reward or float("inf"))
                if (
                    peak.mean_reward > 400.0
                    and trough.mean_reward is not None
                    and trough.mean_reward < peak.mean_reward * 0.65
                ):
                    diag.reward_collapse = True
                    diag.collapse_detail = (
                        f"peak {peak.mean_reward:.1f} @ epoch {peak.epoch} "
                        f"→ trough {trough.mean_reward:.1f} @ epoch {trough.epoch}"
                    )

        nav_rows = [r for r in self._rows if r.captured_rate is not None]
        if nav_rows:
            diag.is_navrl = True
            last_nav = nav_rows[-1]
            diag.last_captured_rate = last_nav.captured_rate
            diag.last_crash_rate = last_nav.crash_rate
            diag.last_timeout_rate = last_nav.timeout_rate
            diag.last_closest_approach_m = last_nav.closest_approach_m
            diag.last_closest_min_m = last_nav.closest_min_m
            diag.last_curriculum_max_m = last_nav.curriculum_max_m
            diag.last_n_bars_active = last_nav.n_bars_active
            peak_nav = max(nav_rows, key=lambda r: r.captured_rate or float("-inf"))
            diag.peak_captured_rate = peak_nav.captured_rate
            diag.peak_captured_epoch = peak_nav.epoch

        diag.hints = _build_hints(diag)
        return diag

    def finalize(self, exit_reason: str) -> RunDiagnostics:
        diag = self.analyze(exit_reason)
        payload = {
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            **asdict(diag),
        }
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if _summary_enabled():
            print_run_summary_box(diag, self.csv_path, self.summary_path)
        return diag


def _f(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int:
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _i_opt(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _cell(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.6g}"


def _rate_count(succ: int, done: int) -> str:
    if done <= 0:
        return "n/a (0 finished)"
    return f"{100.0 * float(succ) / float(done):.1f}% ({succ}/{done})"


def _exit_reason_label(reason: str) -> str:
    return {
        "early_stop_stable": "조기 종료 (reward plateau)",
        "early_stop_collapse": "조기 종료 (reward collapse)",
        "early_stop_density_capture_collapse": "실패 (same-density capture collapse)",
        "early_stop_nan": "실패 (NaN/Inf reward)",
        "nonfinite_ppo": "실패 (NaN/Inf PPO state)",
        "score_to_win": "score_to_win 도달",
        "max_epochs": "max_epochs 도달",
        "max_frames": "max_frames 도달",
        "interrupted": "사용자 중단 (Ctrl+C 등)",
        "completed": "정상 종료",
        "unknown": "알 수 없음",
    }.get(reason, reason)


def _build_navrl_hints(d: RunDiagnostics) -> list[str]:
    hints: list[str] = []
    cap = d.last_captured_rate
    if cap is not None:
        if cap >= 0.9:
            hints.append(f"captured {100 * cap:.1f}% — 목표를 안정적으로 포획 (M1 달성권).")
        elif cap >= 0.6:
            hints.append(
                f"captured {100 * cap:.1f}% (peak {100 * (d.peak_captured_rate or 0.0):.1f}%) — "
                "포획되나 개선 여지."
            )
        elif cap >= 0.1:
            hints.append(f"captured {100 * cap:.1f}% — 낮음. 리워드 균형/커리큘럼 점검.")
        else:
            hints.append(
                f"captured {100 * cap:.1f}% — 거의 실패. loiter farming(안전항 개방공간 수입) 또는 "
                "리워드 최적해가 '배회'인지 확인."
            )
    if d.last_timeout_rate is not None and d.last_timeout_rate > 0.3:
        hints.append(
            f"timeout {100 * d.last_timeout_rate:.1f}% — 목표 근처 배회(farming) 의심. "
            "capture 보너스 대비 스텝 수입(안전+생존)을 점검."
        )
    if d.last_crash_rate is not None and d.last_crash_rate > 0.1:
        hints.append(
            f"실패 주원인이 crash {100 * d.last_crash_rate:.1f}% — safety weight↑ 또는 근접 마진 "
            "페널티(clearance_weight)로 충돌 저감."
        )
    if d.last_closest_approach_m is not None:
        best = f", best {d.last_closest_min_m:.2f} m" if d.last_closest_min_m is not None else ""
        hints.append(
            f"closest(no-crash) {d.last_closest_approach_m:.2f} m{best} — crash 제외 평균이라 "
            "실제 접근 정밀도를 반영(과거 all-episode 평균의 crash 왜곡 제거)."
        )
    if d.reward_collapse:
        hints.append(f"reward 붕괴 의심: {d.collapse_detail}.")
    return hints


def _build_hints(d: RunDiagnostics) -> list[str]:
    if d.is_navrl:
        return _build_navrl_hints(d)
    hints: list[str] = []
    ep_max = _episode_len_max_hint()

    if d.total_intercept_succ == 0 and d.max_intercept_envs_hit == 0:
        hints.append("요격 0건 — intercept 성공/접촉이 한 번도 없었습니다.")
    else:
        hints.append(
            f"요격 발생 — finished 에피소드 {d.total_intercept_succ}/{d.total_intercept_done}."
        )

    if d.last_mean_episode_length is not None:
        if d.last_mean_episode_length >= ep_max - 0.5:
            hints.append(
                f"mean ep length ≈ {d.last_mean_episode_length:.0f}/{ep_max:.0f} — "
                "거의 전부 타임아웃(접촉 없이 끝)."
            )
        elif d.first_short_episode_epoch is not None:
            hints.append(
                f"mean ep length < {ep_max:.0f} 첫 등장: epoch {d.first_short_episode_epoch} "
                "(접촉/조기 종료 시작)."
            )

    if d.mean_dist_start is not None and d.mean_dist_end is not None:
        delta = d.mean_dist_end - d.mean_dist_start
        if delta < -0.15:
            hints.append(
                f"표적 거리 감소 ({d.mean_dist_start:.2f}m → {d.mean_dist_end:.2f}m) — 접근은 진행됨."
            )
        elif delta > 0.15:
            hints.append(
                f"표적 거리 증가 ({d.mean_dist_start:.2f}m → {d.mean_dist_end:.2f}m) — "
                "멀어지거나 불안정."
            )
        else:
            hints.append(
                f"표적 거리 정체 ({d.mean_dist_start:.2f}m → {d.mean_dist_end:.2f}m) — "
                "중간 거리 호버 가능성."
            )

    if d.last_failure_surface_gap_m is not None:
        if d.last_failure_surface_gap_m <= 0.08:
            hints.append(
                f"실패 surface gap이 작음({d.last_failure_surface_gap_m:.3f}m) — 마지막 접촉/조준 구간을 우선 점검."
            )
        elif d.last_failure_closest_target_dist_m is not None:
            hints.append(
                f"실패 closest가 아직 큼({d.last_failure_closest_target_dist_m:.3f}m) — 접근 basin/탐색 지도가 우선."
            )

    if d.last_radius_no_contact_rate is not None and d.last_radius_no_contact_rate > 0.20:
        hints.append(
            f"radius 안이지만 contact 없음({100.0 * d.last_radius_no_contact_rate:.1f}%) — "
            "collision/렌더 기준 또는 접촉 판정을 확인."
        )
    if d.last_contact_no_radius_rate is not None and d.last_contact_no_radius_rate > 0.05:
        hints.append(
            f"contact는 있으나 성공 반경 밖({100.0 * d.last_contact_no_radius_rate:.1f}%) — "
            "성공 반경/표면 gap 기준 차이를 확인."
        )

    if d.reward_collapse:
        hints.append(
            f"reward 붕괴 의심: {d.collapse_detail}. entropy 과다·탐색 과다를 점검하세요."
        )
    elif d.peak_mean_reward is not None and d.last_mean_reward is not None:
        if d.last_mean_reward >= 650:
            hints.append("reward가 높은 구간 — 근접 shaping은 작동 중일 수 있으나 요격 여부는 intercept 지표로 확인.")
        elif d.last_mean_reward >= 500:
            hints.append("reward 중간대 — 접근 basin/속도 벌점/entropy 균형을 조정할 여지 있음.")
        else:
            hints.append("reward 낮음 — 정책 붕괴 또는 원거리 정체 상태일 수 있음.")

    if d.first_intercept_epoch is None:
        hints.append("다음 관찰: mean ep length 하락 → intercept > 0 순서로 진행되는지 확인.")

    return hints


def print_run_summary_box(
    diag: RunDiagnostics,
    csv_path: Path,
    summary_path: Path,
) -> None:
    bar = "\u2500" * 56
    lines = [
        "",
        bar,
        "  Aerial RL  \u00b7  Run summary (training ended)",
        "",
        f"  exit reason    : {_exit_reason_label(diag.exit_reason)}",
        f"  run folder     : {diag.run_root}",
        f"  epochs logged  : {diag.epochs_logged}  "
        f"(epoch {_or_na(diag.first_epoch)} \u2192 {_or_na(diag.last_epoch)})",
        "",
        f"  last reward    : {_fmt(diag.last_mean_reward)}",
        f"  peak reward    : {_fmt(diag.peak_mean_reward)}  @ epoch {_or_na(diag.peak_epoch)}",
        f"  last ep length : {_fmt(diag.last_mean_episode_length)}",
    ]
    if diag.is_navrl:
        lines += [
            "",
            f"  captured (success): {_pct(diag.last_captured_rate)}"
            f"    peak {_pct(diag.peak_captured_rate)} @ epoch {_or_na(diag.peak_captured_epoch)}",
            f"  crash             : {_pct(diag.last_crash_rate)}",
            f"  timeout (no cap)  : {_pct(diag.last_timeout_rate)}",
            f"  closest (no crash): {_fmt(diag.last_closest_approach_m, 'm')}"
            f"   best {_fmt(diag.last_closest_min_m, 'm')}",
            f"  curriculum max    : {_fmt(diag.last_curriculum_max_m, 'm')}",
            f"  density bars      : {_or_na(diag.last_n_bars_active)}",
        ]
    else:
        lines += [
            f"  last target dist: {_fmt(diag.last_mean_target_dist_m, 'm')}",
            f"  closest dist/ep: {_fmt(diag.last_mean_closest_target_dist_m, 'm')}",
            f"  fail closest   : {_fmt(diag.last_failure_closest_target_dist_m, 'm')}",
            f"  fail surf gap  : {_fmt(diag.last_failure_surface_gap_m, 'm')}",
            "",
            f"  intercept success: {_rate_count(diag.total_intercept_succ, diag.total_intercept_done)}",
            f"  near misses       : {diag.total_near_miss_count}",
            f"  first intercept epoch   : {_or_na(diag.first_intercept_epoch)}",
            f"  first ep_len drop epoch : {_or_na(diag.first_short_episode_epoch)}",
        ]
    lines += [
        "",
        "  Interpretation:",
    ]
    for h in diag.hints:
        lines.append(f"    \u2022 {h}")
    lines.extend(
        [
            "",
            f"  metrics CSV    : {csv_path}",
            f"  summary JSON   : {summary_path}",
            bar,
            "",
        ]
    )
    print("\n".join(lines), flush=True)


def _fmt(v: Optional[float], suffix: str = "") -> str:
    if v is None:
        return _NA
    s = f"{v:,.3f}" if suffix == "m" else f"{v:,.4f}"
    return f"{s}{suffix}" if suffix else s


def _pct(v: Optional[float]) -> str:
    if v is None:
        return _NA
    return f"{100.0 * v:.1f}%"


def get_or_create_recorder(agent) -> Optional[TrainRunRecorder]:
    nn_dir = getattr(agent, "nn_dir", None)
    if not nn_dir:
        return None
    rec = getattr(agent, "_aerial_train_run_recorder", None)
    if rec is None:
        rec = TrainRunRecorder(run_root_from_nn_dir(nn_dir))
        agent._aerial_train_run_recorder = rec
    return rec


def record_agent_epoch(
    agent,
    *,
    epoch_num: int,
    mean_reward: Optional[float],
    mean_episode_length: Optional[float],
    mean_target_dist_m: Optional[float],
    mean_closest_target_dist_m: Optional[float],
    intercept_extra_metrics: Optional[dict[str, Any]],
    intercept_succ: int,
    intercept_done: int,
    intercept_envs_hit: int,
    num_parallel_envs: int,
) -> None:
    if getattr(agent, "global_rank", 0) != 0:
        return
    rec = get_or_create_recorder(agent)
    if rec is None:
        return
    rec.record_epoch(
        epoch=epoch_num,
        mean_reward=mean_reward,
        mean_episode_length=mean_episode_length,
        mean_target_dist_m=mean_target_dist_m,
        mean_closest_target_dist_m=mean_closest_target_dist_m,
        intercept_succ=intercept_succ,
        intercept_done=intercept_done,
        intercept_envs_hit=intercept_envs_hit,
        num_parallel_envs=num_parallel_envs,
        extra_metrics=intercept_extra_metrics,
    )


def finalize_agent_run(agent, exit_reason: str) -> None:
    if getattr(agent, "global_rank", 0) != 0:
        return
    rec = get_or_create_recorder(agent)
    if rec is None:
        return
    rec.finalize(exit_reason)


def print_summary_for_run_folder(run_root: str | Path) -> None:
    """CLI helper: re-print summary from saved CSV/JSON."""
    root = Path(run_root).resolve()
    rec = TrainRunRecorder(root)
    json_path = rec.summary_path
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        diag = RunDiagnostics(**{k: data[k] for k in asdict(RunDiagnostics()).keys() if k in data})
        print_run_summary_box(diag, rec.csv_path, rec.summary_path)
    elif rec._rows:
        rec.finalize("replay")
    else:
        print(f"[aerial RL] no metrics under {root / _SUBDIR}", flush=True)
