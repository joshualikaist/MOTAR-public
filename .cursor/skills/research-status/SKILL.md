---
name: research-status
description: >-
  Update and publish the MOTAR (NavRL-in-AerialGym) research status: analyze the
  latest RL training runs and the collision-rate curve, audit the density/distance
  curriculum, propose the next training run, refresh the 3D web dashboard, and
  publish it to GitHub Pages. Use when the user asks to check the latest training,
  analyze crash/collision rate, review the curriculum, decide the next run, or
  update/publish the research status page or dashboard. Delegates heavy exploration
  to subagents to keep token use low.
disable-model-invocation: true
---

# MOTAR Research-Status Update

Orchestrates a token-efficient research-status update for the MOTAR project
(RL drone target-interception in obstacle fields). Heavy reading is delegated to
subagents; the main agent only synthesizes and edits.

Repo: `joshualikaist/MOTAR`, branch `research/navrl-env` (also the default branch).
Both scripts derive the repo root from their own location, so they work in any
clone — on the workstation it is `~/workspaces/aerial_gym_ws/src/aerial_gym_simulator`
(nested inside a separate, remote-less workspace repo), and on a Cursor cloud/mobile
agent it is the clone root. Dashboard lives in `docs/status/` and is served by GitHub
Pages **straight from the research branch's `/docs` folder** (no gh-pages branch).
Live at `https://joshualikaist.github.io/MOTAR/status/`.

**GPU work cannot run on a cloud/mobile agent** — no Isaac Gym and no GPU there.
Steps 1 and 4–6 are safe anywhere; the subagent analysis in step 2 only reads files.
Training and held-out evaluation must run on the RTX 3070 workstation.

## Workflow (copy this checklist, run in order)

```
- [ ] 1. Snapshot metrics       (tools/update_status_snapshot.py)
- [ ] 2. Analyze (subagents)    (runs+crash, curriculum, ops)  ← parallel
- [ ] 3. Synthesize             (diagnosis + next-run recipe)
- [ ] 4. Refresh dashboard      (docs/status/index.html, app.js, status.json)
- [ ] 5. Publish                (scripts/publish_dashboard.sh)
- [ ] 6. Log                    (append to WORKLOG.md)
```

### Step 1 — Snapshot metrics (cheap, do first)
`tools/update_status_snapshot.py` is the ONLY thing that may write
`docs/status/status.json`. It applies the evidence gates, emits `research_update`,
and rewrites the inline `index.html` fallback from the same object so the two
cannot drift:

```bash
python3 tools/update_status_snapshot.py
# -> docs/status/status.json  +  the fallback block in docs/status/index.html
```

On a fresh clone this exits 2 with "refusing to write: no runs found" — that is
correct, not a bug. `runs/` is gitignored, so a cloud/mobile agent has no run
evidence and writing would erase the published history. Steps 1 and 5 therefore
belong on the training workstation.

Then read `status.json` (small) for the latest run + curves. Do NOT read individual
run files yourself — that is the subagents' job.

`scripts/collect_status.py` in this skill is a **legacy** scraper kept only for
cheap run/CSV dumps; it omits `research_update` and undercounts runs, so it writes
`status.legacy.json` and must never produce the published `status.json`.

### Step 2 — Analyze via subagents (parallel, `explore`, `run_in_background: true`)
Launch these three in ONE message. Give each the repo path and tell it to return
tight, numeric, file:line-cited reports (they lack your context):

- **Runs + crash spike**: enumerate runs, chronological table, latest-run
  peak→final trajectory, characterize the collision-rate spike vs density /
  distance / target-speed, and rule artifact-vs-real (checkpoint-selection
  `gen_ppo.pth`=best-reward=low-density vs `last_gen`; two-curricula-collide;
  4GB minibatch confound — cross-check `WORKLOG.md`).
- **Curriculum audit**: exact knobs/defaults/formulas for the density curriculum
  (`NAVRL_DENSITY_CURRICULUM/WARMUP/THRESHOLD`, `NAVRL_NUM_BARS/MAX_BARS`), the
  distance curriculum (`NAVRL_K_FINAL/K_MIN_FINAL/K_WARMUP`), target-speed
  (`NAVRL_TARGET_SPEED*`), placement geometry (random rejection, min spacing,
  saturation relaxation ~115-120 & ~148 bars), the crash-cliff geometry math,
  and a concrete "sequential density" command recipe.
- **Repo/ops**: doc inventory + staleness, `git status`/diffs (safe-to-commit?),
  gh-pages publishing recommendation.

### Step 3 — Synthesize
From the subagent reports, write two things (concise, evidence-cited):
1. **Diagnosis** of the collision-rate behavior (real geometric limit vs curriculum/
   checkpoint artifact).
2. **Next-run recipe**: exact `./train_navrl.sh` command(s). Default principle the
   user asked for: raise bar density in *smaller sequential steps*, hold the
   distance curriculum shallow, snapshot a checkpoint per density level.

### Step 4 — Refresh the dashboard
Edit the content blocks in `docs/status/app.js`:
- `renderStatic()` → `#diagnosis` and `#nextplan` HTML with the new synthesis.
- roadmap phase states in the `phases` array if a phase changed.
`status.json` is already refreshed by Step 1; the 3D scene and data panels are
data-driven and need no edits. Update the fallback JSON in `index.html` only if a
big result changed. Validate: `node --check docs/status/app.js`.

### Step 5 — Publish
```bash
bash .cursor/skills/research-status/scripts/publish_dashboard.sh
# -> commits ONLY docs/ on the current (research) branch and pushes.
# site: https://joshualikaist.github.io/MOTAR/status/
```
Requires SSH push access to `joshualikaist/MOTAR` (origin is SSH). One-time GitHub
setting: repo Settings -> Pages -> Source "Deploy from a branch", Branch
`research/navrl-env`, Folder `/docs`. No gh-pages branch, no `gh` CLI needed.

### Step 6 — Log
Append a dated entry to `src/aerial_gym_simulator/WORKLOG.md` (newest at bottom):
what was analyzed, the verdict, and the next-run command. Keep it terse.

## Conventions (MOTAR-specific facts the agent must not re-derive)
- Metrics: `captured` (포획, dist<0.5 m), `crash` (충돌), `timeout`. Success radius 0.5 m.
- **v2 arena (current): 40×40×3 m**, full-width `navrl_band`, 3 m bars so there is no
  fly-over, goal 6–28 m, target speed 0.3–1.5 m/s, 600-step episodes.
  Placement area **1600 m²** → density = bars/1600×100 per 100 m².
  `NAVRL_ARENA_XY` still *defaults* to 24.0 in code; the v2 launchers export 40.
  Do not quote the legacy 24×24 / 478 m² numbers as v2 evidence.
- Two obs regimes: **GT-injected** (actor sees target GT) vs **sensor-only** (LiDAR+detector, no GT).
- **Checkpoint trap**: for a density-curriculum run, evaluate high density with the
  **last** checkpoint (`last_gen_ppo_ep_*`), NOT `gen_ppo.pth` (best-reward = low-density policy).
- Training-time `captured` mixes densities during the ramp; held-out per-density eval is the real curve.
- Prefer `docs/status/status.json` (`arena_geometry`, `placement_area_m2`) over anything
  written here — it is regenerated from the code and cannot go stale the way prose does.

## Files
- `tools/update_status_snapshot.py` — canonical dashboard writer (`status.json` + HTML
  fallback, with the evidence gates). **Execute.**
- `scripts/publish_dashboard.sh` — commit ONLY `docs/` and push. **Execute.**
- `scripts/collect_status.py` — legacy scraper → `status.legacy.json`. Never publish from it.
- `docs/status/index.html`, `app.js` — the dashboard (edit `renderStatic()` for prose).
