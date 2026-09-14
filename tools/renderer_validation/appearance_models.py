"""The three R4 appearance models. Each maps one shared G-buffer to linear RGB.

No instance ID, semantic label, ground truth or learned model enters any of them. The depth
arm is written here from its own definition; it is not a copy or re-run of the NavRL renderer,
and its numbers are not compared to NavRL's.
"""
import numpy as np
import torch

from .shading import shade

BT709 = (0.2126, 0.7152, 0.0722)
DEPTH_BASE = 0.15
DEPTH_GAIN = 0.65


def scene_mean_color(appearance):
    """One colour per scene: the mean over that scene's material colours."""
    return appearance.base_color.mean(axis=1).astype(np.float32)


def uniform_appearance_from(appearance):
    """The same lighting, with a single colour and kd shared by every material.

    All arms then run at the same colour scale, so a difference between them cannot be a
    difference in overall brightness.
    """
    from dataclasses import replace
    count, materials = appearance.base_color.shape[:2]
    colors = np.repeat(scene_mean_color(appearance)[:, None, :], materials, axis=1)
    kd = np.repeat(appearance.kd.mean(axis=1, keepdims=True), materials, axis=1)
    return replace(appearance, base_color=colors, kd=kd.astype(np.float32))


@torch.no_grad()
def depth_gradient_rgb(gbuffer, appearance, camera, background=(0.0, 0.0, 0.0)):
    """Appearance as a function of ray range alone; normal, material and face never enter."""
    if not 0.0 < camera.far_range_m:
        raise ValueError("far_range_m must be positive")
    device = gbuffer.range_m.device
    count = gbuffer.range_m.shape[0]
    if appearance.base_color.shape[0] != count:
        raise ValueError("Appearance must match the batch")
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    if bg.shape != (3,) or not torch.isfinite(bg).all() or ((bg < 0) | (bg > 1)).any():
        raise ValueError("background must be three finite values in [0,1]")
    tint = torch.tensor(scene_mean_color(appearance), device=device)[:, None, None, :]
    proximity = (1.0 - gbuffer.range_m / camera.far_range_m).clamp(0.0, 1.0)
    luminance = DEPTH_BASE + DEPTH_GAIN * proximity
    rgb = (luminance[..., None] * tint).clamp(0.0, 1.0)
    return torch.where(gbuffer.valid[..., None], rgb, bg).contiguous()


def render_arms(gbuffer, scene, appearance, camera):
    """The three preregistered arms plus one diagnostic, all from the same G-buffer."""
    uniform = uniform_appearance_from(appearance)
    return {
        "flat": shade(gbuffer, scene, uniform, "flat"),
        "depth_gradient": depth_gradient_rgb(gbuffer, appearance, camera),
        "lambertian": shade(gbuffer, scene, appearance, "lambertian"),
        # Diagnostic only: isolates shading from material diversity. Not a preregistered arm.
        "lambertian_uniform_color": shade(gbuffer, scene, uniform, "lambertian"),
    }
