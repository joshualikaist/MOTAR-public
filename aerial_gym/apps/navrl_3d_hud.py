"""OpenCV HUD panel for the NavRL 3-D interactive viewer."""

from __future__ import annotations

import time

import cv2
import numpy as np


class NavRL3DHud:
    """Side panel rendered with OpenCV while Isaac Gym draws the 3-D scene."""

    WIDTH = 440
    HEIGHT = 760
    BG = (18, 22, 30)
    FG = (230, 236, 245)
    MUTED = (130, 145, 165)
    ACCENT = (80, 170, 255)
    OK = (70, 190, 110)
    WARN = (240, 180, 70)
    BAD = (240, 90, 90)

    def __init__(self, title: str = "NavRL 3D HUD"):
        self._enabled = True
        self._title = title
        self._flash_text = ""
        self._flash_color = self.ACCENT
        self._flash_until = 0.0
        self._summary_lines: list[str] = []
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, self.WIDTH, self.HEIGHT)
            cv2.moveWindow(title, 24, 24)
        except cv2.error:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def flash(self, text: str, color: tuple[int, int, int] | None = None, seconds: float = 2.5):
        self._flash_text = str(text)
        self._flash_color = color or self.ACCENT
        self._flash_until = time.time() + max(0.5, float(seconds))

    def set_summary(self, lines: list[str]):
        self._summary_lines = list(lines)

    def update(self, lines: list[str], pip_images: list[tuple[str, np.ndarray]] | None = None):
        if not self._enabled:
            return
        canvas = np.full((self.HEIGHT, self.WIDTH, 3), self.BG, dtype=np.uint8)
        y = 28
        cv2.putText(
            canvas,
            "NavRL 3D",
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self.ACCENT,
            2,
            cv2.LINE_AA,
        )
        y += 28
        for line in lines:
            color = self.FG
            if line.startswith("---"):
                y += 6
                cv2.line(canvas, (14, y), (self.WIDTH - 14, y), (45, 55, 72), 1)
                y += 16
                continue
            if line.startswith("#"):
                color = self.MUTED
                line = line[1:].lstrip()
            elif line.startswith("+"):
                color = self.OK
                line = line[1:].lstrip()
            elif line.startswith("!"):
                color = self.WARN
                line = line[1:].lstrip()
            elif line.startswith("x"):
                color = self.BAD
                line = line[1:].lstrip()
            cv2.putText(
                canvas,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 22
            if y > self.HEIGHT - 180:
                break

        if time.time() < self._flash_until and self._flash_text:
            box_y0 = min(max(y + 8, 120), self.HEIGHT - 120)
            cv2.rectangle(
                canvas,
                (12, box_y0),
                (self.WIDTH - 12, box_y0 + 34),
                self._flash_color,
                -1,
            )
            cv2.putText(
                canvas,
                self._flash_text,
                (20, box_y0 + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self._summary_lines:
            box_y0 = self.HEIGHT - 150
            cv2.rectangle(canvas, (10, box_y0), (self.WIDTH - 10, self.HEIGHT - 10), (28, 34, 46), -1)
            sy = box_y0 + 22
            for line in self._summary_lines[:5]:
                cv2.putText(
                    canvas,
                    line,
                    (18, sy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.FG,
                    1,
                    cv2.LINE_AA,
                )
                sy += 22

        if pip_images:
            px = 16
            py = self.HEIGHT - 132
            for label, image in pip_images[:2]:
                if image is None or image.size == 0:
                    continue
                thumb = self._to_thumbnail(image, 180, 96)
                cv2.imshow("%s | %s" % (self._title, label), thumb)
                cv2.moveWindow("%s | %s" % (self._title, label), px, py)
                px += 196

        cv2.imshow(self._title, canvas)
        cv2.waitKey(1)

    @staticmethod
    def _to_thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            mono = arr.astype(np.float32)
            if mono.ndim == 3:
                mono = mono[..., 0]
            if mono.max() > 1.5:
                mono = mono / max(mono.max(), 1.0)
            mono = np.clip(mono, 0.0, 1.0)
            gray = (mono * 255.0).astype(np.uint8)
            inv = 255 - gray
            bgr = cv2.applyColorMap(inv, cv2.COLORMAP_TURBO)
        return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

    def close(self):
        if not self._enabled:
            return
        try:
            cv2.destroyWindow(self._title)
        except cv2.error:
            pass
        for suffix in ("depth", "rgb"):
            try:
                cv2.destroyWindow("%s | %s" % (self._title, suffix))
            except cv2.error:
                pass


def build_hud_lines(task, env_id: int = 0) -> list[str]:
    """Build HUD text lines from a live NavRL task instance."""
    lines: list[str] = []
    total = int(getattr(task, "general_num_trials", 10))
    trial = int(getattr(task, "general_trial_index", 0))
    completed = int(getattr(task, "general_completed_trials", 0))
    if task.general_eval_mode:
        lines.append("Trial %d/%d  (done %d)" % (max(1, trial), total, completed))
    else:
        lines.append("Interactive preview")
    lines.append("Bars %d" % int(task.n_bars_active))
    target_speed = (
        float(task._runtime_target_speed)
        if task._runtime_target_speed is not None
        else float(task._tm_speed[env_id].item())
    )
    lines.append("Target %.2f m/s  Drone max %.2f m/s" % (target_speed, float(task.task_config.max_velocity)))
    lines.append("Mode %s" % ("MANUAL" if task._interactive_manual else "POLICY"))
    lines.append("---")

    pos = task.obs_dict["robot_position"][env_id]
    target = task.target_position[env_id]
    dist = float(torch_norm(target - pos))
    closest = float(task.ep_min_goal_dist[env_id].item())
    lines.append("GT dist %.2f m  closest %.2f m" % (dist, closest))
    lines.append("Step %d / %d" % (int(task.sim_env.sim_steps[env_id].item()), int(task.task_config.episode_len_steps)))
    lines.append("LiDAR overlay %s" % ("ON" if task._interactive_show_lidar else "OFF"))

    if task.general_eval_mode:
        lines.append(
            "Score cap=%d crash=%d to=%d"
            % (
                int(task.general_successes),
                int(task.general_crashes),
                int(task.general_timeouts),
            )
        )

    lines.append("---")
    lines.append("# Perception")
    visible = bool(task._visible_now[env_id].item())
    lines.append(("+ visible" if visible else "x not visible"))
    if task.perception_mode:
        conf = float(task.obs_dict.get("navrl_track_confidence", task._visible_now)[env_id].item())
        age = float(task.obs_dict.get("navrl_track_age", task._visible_now)[env_id].item())
        cov = task.obs_dict.get("navrl_track_covariance")
        cov_val = float(cov[env_id, 0, 0].item()) if cov is not None else float("nan")
        lines.append("track conf=%.2f age=%.1fs cov=%.3f" % (conf, age, cov_val))
    elif task.vision_mode and task.detector is not None:
        lines.append("legacy vision detector active")

    lines.append("---")
    lines.append("# Keys")
    lines.append(",/. target speed")
    lines.append("-/= drone speed")
    lines.append("G LiDAR  N new trial")
    lines.append("M manual  IJKL move  UO yaw")
    return lines


def build_hud_pip(task, env_id: int = 0) -> list[tuple[str, np.ndarray]]:
    depth = task.obs_dict.get("navrl_raw_depth")
    rgb = task.obs_dict.get("navrl_raw_rgb")
    images: list[tuple[str, np.ndarray]] = []
    if depth is not None:
        images.append(("depth", depth[env_id].detach().cpu().numpy()))
    elif task.obs_dict.get("obstacle_camera_depth") is not None:
        images.append(
            (
                "depth",
                task.obs_dict["obstacle_camera_depth"][env_id].detach().cpu().numpy(),
            )
        )
    if rgb is not None:
        images.append(("rgb", rgb[env_id].detach().cpu().numpy()))
    return images


def torch_norm(vec) -> float:
    return float(vec.norm().item())
