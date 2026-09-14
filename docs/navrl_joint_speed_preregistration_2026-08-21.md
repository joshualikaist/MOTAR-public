# NavRL joint speed-allocation diagnosis — preregistration

This is an evaluation-only, single-cell diagnostic of the frozen ep25000+riskcap candidate.  It
does not alter or search riskcap parameters, and it does not train PPO.

## Frozen condition

- 205 bars, deterministic original action, seed 379, 4,097 held-out episodes
- checkpoint SHA-256 `f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40`
- riskcap and braking constants are identical to the completed campaign
- instrumentation is opt-in only (`NAVRL_JOINT_SPEED_TELEMETRY=1`); ordinary bulk evaluations do
  not construct the recorder or export its key
- source bytes, checkpoint bytes, evaluator, Python environment and result are receipt-bound by
  `eval_navrl_v2_density_sweep.sh`
- when an isolated Git worktree lacks ignored `runs/`, the launcher resolves the pinned checkpoint
  from the primary worktree via `git rev-parse --git-common-dir`; SHA enforcement is unchanged,
  while the default result directory remains under the diagnostic worktree's own repository root

## Measurements

Every action-selection step records actual horizontal speed, requested and executed command, plus
three independently queried actor-safe directional LiDAR minima: along actual velocity, requested
command and executed command.  Actual/requested/executed braking distance and margin always use
the clearance from the same direction as their speed.  The primary risk variable is therefore the
decision-time actual-velocity-direction stopping margin, not a command/velocity hybrid.  Normalized
XY action delta and requested/realized heading-rate and curvature proxies are also recorded.  Steps
are attributed after termination to capture/crash/timeout and binned by actual-direction margin:

`<0`, `0–0.5`, `0.5–1.5`, `>=1.5 m`.

The last 1.0 s before every cause-attributed bar contact is aggregated separately.  Heading rate is
a finite difference of command or actual velocity bearing, and curvature is heading-rate divided
by speed only when both adjacent speeds are at least 0.25 m/s.  They are explicitly proxies, not a
planner's required curvature or a reconstructed continuous path.

## Fixed gates

Quality requires at least 100 bar-contact episodes, 500 pre-contact steps and 1,000 capture steps.
Conditional on quality, the descriptive association gate passes only when:

1. at least 50% of the last-1-s bar-contact steps have negative actual-direction stopping margin; and
2. that rate exceeds the capture-episode step rate by at least 10 percentage points.

A pass supports only an association between unsafe speed margin and contact.  It does **not** show
that speed caused the contact, identify a controller change, or authorize post-hoc riskcap tuning.
Failure leaves the adaptive-speed hypothesis unconfirmed; proxy distributions remain descriptive.
