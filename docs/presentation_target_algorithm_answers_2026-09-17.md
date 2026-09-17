# 발표용 답변 세트 — "타겟은 어떤 알고리즘으로 움직이나요?"

Canonical source: [`docs/target_motion_algorithm_2026-09-17.md`](target_motion_algorithm_2026-09-17.md).
이 문서는 같은 내용을 **말로 답할 수 있는 길이**로 나눠 둔 것이다. 새 주장은 없다.

> 답변에서 절대 하면 안 되는 것: **browser preview와 research TM-E2를 같은 알고리즘처럼
> 말하는 것.** browser는 global route를 쓰고 TM-E2는 안 쓴다.
>
> 두 계보 모두 obstacle-aware but **pursuer-independent** 다. NOT PPO, NOT PhysX.

---

## 15초 답변

**한국어**

> 사이트 데모의 타겟은 장애물을 무시하고 아무 방향으로 움직이는 게 아닙니다. 갈 수 있는
> 목표점을 고르고, 장애물을 피하는 경로를 계획한 뒤, 속도·가속도·회전 제한을 지키면서
> 그 경로를 따라갑니다. 도착하면 다음 목표를 고릅니다.

**English**

> The target picks a reachable destination, plans a route that avoids the
> obstacles, and follows it under speed, acceleration and turn-rate limits. When
> it arrives it picks another one.

---

## 45초 답변 — 비전공 교수 / 일반 청중

**한국어**

> 사이트의 타겟은 장애물을 무시하고 임의 방향으로 움직이는 것이 아닙니다. 먼저 실제로
> 도달 가능한 목표점을 고르고, 장애물을 자기 몸 크기만큼 부풀린 지도 위에서 장애물을
> 통과하지 않는 전역 경로를 찾습니다. 그다음 그 경로로 순간이동하는 게 아니라, 속도와
> 가속도, 방향 변화에 제한을 둔 추종기가 경로를 따라갑니다. 목적지에 도착하면 새 목적지를
> 골라 계속 움직이고, 안전한 경로가 없으면 장애물을 뚫지 않고 멈춰서 다시 계획합니다.
>
> 다만 이건 사이트에서 환경을 **설명하기 위한** GT 기반 motion generator입니다. 타겟이
> 장애물 위치를 안다는 뜻이지, 추적하는 드론의 정책이 그 정보를 받는다는 뜻이 아닙니다.
> 연구 시뮬레이터에는 정지, 등속, 장애물 인지형 TM-E2 같은 별도의 타겟 계보가 따로 있고,
> 그쪽은 전역 경로를 쓰지 않습니다.

**English**

> The target on the site does not move in an arbitrary direction ignoring
> obstacles. It first chooses a destination it can actually reach, then finds a
> global route over a map where the obstacles have been inflated by its own body
> size, so the route never passes through one. It does not teleport along that
> route: a follower with speed, acceleration and heading-rate limits tracks it.
> On arrival it selects a new destination; if no safe route exists it stops and
> replans rather than clipping through a bar.
>
> That said, this is a GT-based motion generator for *explaining* the
> environment. The target knowing where the obstacles are does not mean the
> tracking policy receives that information. The research simulator has separate
> target lineages — static, constant velocity, and the obstacle-aware TM-E2 —
> and those do not use a global route.

---

## Technical 답변 — 1~2분, 연구자 대상

**한국어**

> 두 가지를 먼저 구분해야 합니다. 사이트 browser preview와 연구 시뮬레이터의 TM-E2는
> **다른 알고리즘**입니다.
>
> browser preview는 계층형 motion generator입니다. 목표점 생성 → configuration-space
> 장애물 팽창 → 8-connected A\* → line-of-sight 기반 경로 단순화 → bounded follower →
> fail-closed 재계획 순서입니다. 팽창은 바 half-extent에 타겟 support 반경과 tracking
> margin을 더하는 방식이고, 대각 이동은 양쪽 직교 이웃이 모두 비어 있을 때만 허용해서
> 맞닿은 바 사이로 빠져나가지 못하게 합니다. 경로 단순화는 spline이 아니라 farthest-visible
> shortcut이고, 채택되는 모든 구간은 닫힌 AABB 검사로 다시 검증합니다. AUTO ROAM에서는
> 목표점을 무작위로 뽑고 A\*를 도는 게 아니라, 현재 위치에서 free grid를 flood해서 **실제로
> 도달된 셀 중에서** 목표를 고릅니다. 그래서 reachability가 rejection sampling이 아니라
> 구성적으로 보장됩니다. A\*는 목표가 고정된 경우, 즉 click/tap goal과 추적 드론 자신의
> 경로에 씁니다.
>
> 반면 TM-E2는 전역 경로가 없습니다. nominal waypoint 방향을 기준으로 heading offset과
> cruise speed scale 조합에 정지 후보를 더해 만든 후보군을, 동일한 가속도·선회 제한으로
> lookahead 동안 rollout하고, workspace와 장애물 여유를 모두 만족하는 후보를 골라 **첫
> 스텝만** 실행한 뒤 다음 RL step에서 다시 계산합니다. 경로를 저장하지 않고, lookahead
> 너머의 모퉁이를 돌아볼 수 없습니다. 전 horizon을 만족하는 후보가 없으면 성공한 척하지
> 않고 가장 긴 안전 prefix를 고른 뒤 infeasible 플래그를 telemetry에 노출합니다.
>
> 연구 시뮬레이터에서 A\*가 등장하는 곳은 physical 계보의 옵션 route mode 하나뿐이고,
> 기본값은 off이며 virtual/bounded 타겟에는 거부됩니다. 그 계보의 route mechanism gate는
> `FAIL_ROUTE_MECHANISM`으로 기록돼 있습니다.
>
> GT 경계는 이렇습니다. 타겟은 perception agent가 아니라 benchmark motion generator이기
> 때문에 시뮬레이터 geometry를 쓸 수 있습니다. 타겟이 바 안으로 들어가거나 teleport하면
> benchmark 자체가 오염되니까요. 하지만 그 정보는 PPO actor observation에 들어가지 않고,
> behavior contract에서 `uses_privileged_pursuer_gt`는 모든 레벨에서 false이며 테스트가
> 이를 강제합니다.

**English**

> Two things have to be separated first: the browser preview and the research
> TM-E2 are **different algorithms**.
>
> The browser preview is a hierarchical motion generator: goal generation →
> configuration-space obstacle inflation → 8-connected A\* → line-of-sight route
> simplification → bounded route follower → fail-closed replanning. Inflation
> adds the target's support radius and a tracking margin to each bar's
> half-extent, and a diagonal move is admitted only when both orthogonal
> neighbours are free, so a path cannot slip between two touching bars.
> Simplification is a farthest-visible shortcut, not a spline, and every accepted
> segment is re-verified against the closed-AABB test. In auto-roam the goal is
> not sampled and then planned to: a flood from the current cell determines which
> cells are reachable and the destination is chosen from those, so reachability is
> constructive rather than rejection-sampled. A\* is used when the destination is
> fixed — the click/tap goal, and the tracking UAV's own route.
>
> TM-E2 has no global route at all. From the nominal waypoint direction it builds
> candidates as heading offsets crossed with cruise-speed scales plus one stop
> command, rolls each forward over the lookahead under the same acceleration and
> turn-rate bounds, keeps those that stay inside the workspace and clear of the
> bars, and executes only the **first** step before recomputing next RL step. It
> stores no route and cannot see around a corner beyond its lookahead. When no
> candidate survives the full horizon it does not pretend otherwise: it takes the
> longest safe prefix and exposes an infeasible flag to telemetry.
>
> A\* appears in the research simulator only in the physical lineage's optional
> route mode, which defaults to off and is refused for virtual or bounded
> targets; that lineage's route mechanism gate is recorded as
> `FAIL_ROUTE_MECHANISM`.
>
> On the GT boundary: the target may use simulator geometry because it is a
> benchmark motion generator, not a perception agent — a target that walks into
> bars or teleports corrupts the benchmark. That information is never passed to
> the PPO actor observation; `uses_privileged_pursuer_gt` is false at every
> behaviour level and a test enforces it.

---

## 예상 후속 질문과 한 문장 답

| 질문 | 한 문장 답 |
|---|---|
| 타겟이 pursuer를 보고 도망가나요? | 아니요. 구현된 어떤 타겟도 pursuer 위치에 반응하지 않습니다. obstacle-aware는 adversarial과 다릅니다. |
| A\*면 너무 쉬운 거 아닌가요? | A\*는 타겟이 **자기** 경로를 만드는 데 쓰이지, pursuer에게 타겟까지의 정답 경로를 주지 않습니다. |
| 왜 타겟에게 GT를 줬나요? | 타겟은 perception agent가 아니라 benchmark motion generator이고, 타겟이 벽을 뚫으면 benchmark가 오염되기 때문입니다. |
| 왜 random walk를 안 쓰나요? | 밀도가 올라가면 벽에 막혀 비현실적인 반사·teleport 보정이 필요해지고, 그게 benchmark를 망칩니다. |
| 길이 막히면요? | zero command로 멈추고 재계획합니다. 움직여 보이려고 장애물을 통과시키지 않습니다. |
| 막힌 공간에 spawn되면요? | spawn이 free-space connectivity 조건을 만족해야 하고, 닫힌 pocket이면 다른 spawn을 고릅니다. controller를 억지로 통과시키지 않습니다. |
| 재현되나요? | 예. seed와 config가 같으면 동일하게 재생되고, 30/60/120 FPS에서 동일한 trajectory임을 검증했습니다. |
| 사이트 움직임이 연구 결과인가요? | 아니요. browser GT preview는 설명용이고 PPO/PhysX 성능 증거가 아닙니다. |
| TM-E3는요? | PLANNED이고 미구현입니다. 선택하면 조용히 scripted motion으로 대체하지 않고 실패합니다. |

---

## 말할 때 피해야 할 표현

| 쓰지 말 것 | 대신 |
|---|---|
| "MOTAR 타겟은 A\*를 씁니다" | "사이트 browser preview가 A\*를 씁니다. 연구 TM-E2는 local receding-horizon입니다." |
| "경로를 spline으로 부드럽게 합니다" | "farthest-visible shortcut으로 불필요한 waypoint를 제거합니다." |
| "타겟이 도망갑니다 / 회피 기동합니다" | "타겟은 장애물을 피하지만 pursuer에는 반응하지 않습니다." |
| "browser 데모가 연구 환경입니다" | "browser는 설명용 GT preview이고, 연구 시뮬레이터에는 별도 계보가 있습니다." |
| "타겟이 GT를 쓰니까 정책도 GT를 씁니다" | "타겟 생성기의 GT와 정책 관측은 분리돼 있고 테스트로 강제됩니다." |
