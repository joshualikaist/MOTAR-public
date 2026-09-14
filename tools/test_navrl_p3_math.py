"""Phase-3 reward/capture math unit tests (pure torch CPU — validates formulas BEFORE implementation).

Formulas under test (exactly as they will be implemented in navrl_task.py):
  range-rate  : r_vel = ((v_drone - v_target) . dir_to_target)
  PBRS anchor : progress = ||prev_pos - target_new|| - gamma * ||pos_new - target_new||
  segment cap : rel segment a=prev_rel -> b=cur_rel vs origin, radius 0.5
                prev_rel = prev_pos - target_prev,  cur_rel = pos_new - target_new
"""
import torch

R = 0.5
GAMMA = 0.99

def seg_capture(prev_rel, cur_rel, radius=R):
    a, b = prev_rel, cur_rel
    ab = b - a
    denom = (ab * ab).sum(-1).clamp(min=1e-9)
    t = (-(a * ab).sum(-1) / denom).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return closest.norm(dim=-1) < radius

def point_capture(cur_rel, radius=R):
    return cur_rel.norm(dim=-1) < radius

ok = True
def check(name, cond):
    global ok
    s = "PASS" if cond else "FAIL"
    if not cond: ok = False
    print(f"  [{s}] {name}")

print("== 1) v_t=0 identity: range-rate == old vel term ==")
v_drone = torch.randn(1000, 3); v_t0 = torch.zeros(1000, 3)
d = torch.randn(1000, 3); d = d / d.norm(dim=1, keepdim=True)
old = (v_drone * d).sum(1)
new = ((v_drone - v_t0) * d).sum(1)
check("max |new-old| == 0", torch.equal(old, new))

print("== 2) v_t=0 identity: PBRS anchor == old prev_dist form ==")
prev_pos = torch.randn(1000, 3); pos = prev_pos + 0.2 * torch.randn(1000, 3)
target = torch.randn(1000, 3) * 5
old_prev_dist = (prev_pos - target).norm(dim=1)              # what the old code stored last step
old_prog = old_prev_dist - GAMMA * (pos - target).norm(dim=1)
new_prog = (prev_pos - target).norm(dim=1) - GAMMA * (pos - target).norm(dim=1)
check("identical", torch.allclose(old_prog, new_prog, atol=0, rtol=0))

print("== 3) PBRS anchor credits ONLY drone motion (drone frozen, target flees) ==")
prev_pos = torch.zeros(1, 3); pos = prev_pos.clone()          # drone did not move
t_prev = torch.tensor([[5.0, 0, 0]]); t_new = torch.tensor([[6.0, 0, 0]])  # target fled 1 m
naive = (prev_pos - t_prev).norm(dim=1) - GAMMA * (pos - t_new).norm(dim=1)   # old formula w/ moving target
anchored = (prev_pos - t_new).norm(dim=1) - GAMMA * (pos - t_new).norm(dim=1)
# naive punishes the drone for the target's move (5 - 0.99*6 = -0.94); anchored gives only the
# gamma-shrink residual (+0.01*dist = 0.06), i.e., ~zero credit for standing still.
check(f"naive={naive.item():.3f} punishes; anchored={anchored.item():.3f} ~= (1-gamma)*dist",
      abs(anchored.item() - (1-GAMMA)*6.0) < 1e-6 and naive.item() < -0.9)

print("== 4) segment capture: static-target reduction & tunneling ==")
# 4a. grazing pass the point-test misses: endpoints at 0.55, midpoint at 0.30
a = torch.tensor([[-0.46, 0.30, 0.0]]); b = torch.tensor([[0.46, 0.30, 0.0]])
check("point-test misses both endpoints", bool(~point_capture(a) & ~point_capture(b)))
check("segment-test catches the pass", bool(seg_capture(a, b)))
# 4b. far pass stays uncaptured
a2 = torch.tensor([[-1.0, 0.8, 0.0]]); b2 = torch.tensor([[1.0, 0.8, 0.0]])
check("far pass not captured", bool(~seg_capture(a2, b2)))
# 4c. zero-length segment (hover, static target) == point test
c = torch.tensor([[0.3, 0.2, 0.0]])
check("zero-length == point test (inside)", bool(seg_capture(c, c) == point_capture(c)))
c2 = torch.tensor([[3.0, 2.0, 0.0]])
check("zero-length == point test (outside)", bool(seg_capture(c2, c2) == point_capture(c2)))
# 4d. v_t=0, 0.2 m steps: segment ⊇ point (never loses captures)
torch.manual_seed(0)
prev = torch.randn(20000, 3); step = 0.2 * torch.nn.functional.normalize(torch.randn(20000, 3), dim=1)
cur = prev + step
pt = point_capture(cur); sg = seg_capture(prev, cur)
check("segment never drops a point capture", bool((pt & ~sg).sum() == 0))
extra = int((sg & ~pt).sum())
print(f"      (segment adds {extra}/20000 grazing captures at v=0 — expected small)")

print("== 5) moving-target relative segment: head-on fly-through at 4 m/s closing ==")
# drone at x=-0.3 moving +x 2 m/s, target at x=+0.1 moving -x 2 m/s (rel step 0.4 m), passes through origin region
p_prev = torch.tensor([[-0.3, 0.0, 0.0]]); p_new = torch.tensor([[-0.1, 0.0, 0.0]])
t_prev = torch.tensor([[0.35, 0.0, 0.0]]); t_new = torch.tensor([[0.15, 0.0, 0.0]])
prev_rel = p_prev - t_prev   # -0.65
cur_rel = p_new - t_new      # -0.25 -> point test catches (0.25 < 0.5). Make a harder crossing case:
p_prev2 = torch.tensor([[-0.62, 0.05, 0.0]]); p_new2 = torch.tensor([[-0.22, 0.05, 0.0]])
t_prev2 = torch.tensor([[0.0, 0.0, 0.0]]);  t_new2 = torch.tensor([[-0.42, 0.0, 0.0]])  # target dashes past drone
pr = p_prev2 - t_prev2       # (-0.62, 0.05)  |.|=0.622  outside
cr = p_new2 - t_new2         # (+0.20, 0.05)  |.|=0.206  inside -> fine. True tunneling needs both outside:
t_new3 = torch.tensor([[-1.35, 0.0, 0.0]])   # target overshoots far behind the drone in one step
cr3 = p_new2 - t_new3        # (1.13, 0.05) outside; relative path swept THROUGH the origin
check("point-test misses both ends", bool(~point_capture(pr) & ~point_capture(cr3)))
check("relative segment catches the fly-through", bool(seg_capture(pr, cr3)))

print("\nALL:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
