---
name: session-handoff
description: >-
  Budget a long autonomous session so it ends on a clean, committed, documented state instead of
  being cut off mid-run — and hand off with an explicit "resume exactly here" instruction. Invoke
  when the user asks for autonomous work ("세션 종료까지", "자율적으로", "알아서 진행"), when a
  session is expected to run for hours, or whenever you notice you are past ~70% of the available
  time. Also invoke at the moment you decide to stop.
---

# Session handoff — finish clean, hand off explicitly

A long autonomous session that gets cut off mid-run is worse than one that stopped earlier on
purpose: the GPU work is lost, the worktree is dirty, and the next session has to reconstruct what
happened from logs. This skill makes the ending a planned step, not an accident.

## The rule

**Reserve the last ~20% of the session for closing out.** For a 5-hour session that is the final
hour. Do not start anything in that window that cannot finish inside it.

Compute the deadline at the START of the session, not when you feel rushed:

```bash
date '+session start %Y-%m-%d %H:%M:%S'
```

Write the wrap-up time into your todo list as its own item, e.g. `[HH:MM] 마무리 시작 — 커밋·문서·핸드오프`.
Re-check the clock whenever you finish a work unit; a long GPU run can silently eat the margin.

## Before starting any new unit of work, ask three questions

1. **How long does this take?** Use a measured number, not a guess — grep the mtimes of a previous
   campaign's cells to get real per-cell wall clock.
2. **Does it fit before the wrap-up deadline, with slack?** If a 25-minute eval has 30 minutes left,
   it does not fit — verification, summary and commit come after it.
3. **If I run out of time mid-way, what state does that leave?** Anything that leaves a half-written
   result directory, a partially edited source file, or an unverified claim is not safe to start.

If the answer to (2) is no, do the largest piece that DOES fit, or spend the time on documentation
and preregistration for the next session instead. Preregistering the next experiment is real work,
costs no GPU, and is exactly what makes the handoff cheap.

## What "finished cleanly" means

Before you write the final report:

- [ ] No training/eval process still running that you started (`pgrep -af "runner.py|eval_navrl"`)
- [ ] Every claim in the report traces to a file on disk, not to scrollback
- [ ] `WORKLOG.md` entry written — including hypotheses that were rejected and runs that were voided
- [ ] Work committed in logical units and pushed
- [ ] `git status` shows only other sessions' WIP; nothing of yours is left uncommitted
- [ ] Any VOID'd run has a `VOID.md` next to it saying why
- [ ] Generated artifacts regenerated and their linters run

## The handoff message

End with these, in this order. Be concrete — the next session should not need to re-derive anything.

1. **어디까지 했는가** — units of work completed, with the actual numbers, not adjectives.
2. **판정** — against the preregistered gate, stated as pass/fail/inconclusive. If a gate failed,
   say so even when the informal reading is favourable, and say which direction the failure leans.
3. **말할 수 있는 것 / 말할 수 없는 것** — the defensible claim and its boundary.
4. **커밋** — hashes and one line each.
5. **다음에 여기서부터 시작하면 됩니다** — exactly ONE next action, with the command or file to open
   first, and the gate it is being run against. Not a menu of options.
6. **차단 사항** — anything the next session must not do (frozen verdicts, forbidden retraining,
   contracts that must not be edited while a campaign runs).

## Anti-patterns

- Starting a multi-hour run in the last hour "because there might be time".
- Reporting a result that is still being written to disk.
- Ending with "next steps: A, B, or C" — pick one and say why.
- Leaving a source file edited but uncommitted because a campaign is mid-flight; either finish the
  campaign or revert the edit, since the next session cannot tell a WIP edit from a deliberate one.
- Silently dropping a scoped item that ran out of time. Say what was cut and why.

## Interaction with running campaigns

Source edits and running evaluations are mutually exclusive in this repo: the per-cell source
manifest guard rejects cells whose runtime bytes changed mid-run. So in the wrap-up window, if a
campaign is running, restrict yourself to files it does not hash (docs, new tools, preregistration)
and say so in the handoff.
