"""Linear-RGB appearance only. No instance ID, depth, labels or learned models enter shading."""
import torch


@torch.no_grad()
def shade(gbuffer, scene, appearance, mode="lambertian", background=(0.0, 0.0, 0.0)):
    if mode not in ("flat", "lambertian"):
        raise ValueError("mode must be flat or lambertian")
    device = gbuffer.normal_world.device
    n = gbuffer.face_id.shape[0]
    if appearance.base_color.shape[:2] != (n, scene.material_count):
        raise ValueError("Appearance must match batch and material table")
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    if bg.shape != (3,) or not torch.isfinite(bg).all() or ((bg < 0) | (bg > 1)).any():
        raise ValueError("background must be three finite values in [0,1]")
    table = torch.tensor(scene.face_material.copy(), dtype=torch.long, device=device)
    material = table[gbuffer.face_id.clamp_min(0).long()]
    batch = torch.arange(n, device=device)[:, None, None]
    colors = torch.tensor(appearance.base_color.copy(), device=device)[batch, material]
    if mode == "lambertian":
        direction = torch.tensor(appearance.light_direction.copy(), device=device)[:, None, None, :]
        cosine = (gbuffer.normal_world * direction).sum(dim=-1).clamp_min(0.0)
        kd = torch.tensor(appearance.kd.copy(), device=device)[batch, material]
        ambient = torch.tensor(appearance.ambient.copy(), device=device)[:, None, None]
        gain = torch.tensor(appearance.directional.copy(), device=device)[:, None, None]
        colors = colors * (ambient + kd * gain * cosine)[..., None]
    return torch.where(gbuffer.valid[..., None], colors.clamp(0.0, 1.0), bg).contiguous()
