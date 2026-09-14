"""A5: assemble the paper's ablation table from the preregistered runs.

Reads only committed result JSON -- no simulation, no GPU. Every number traces to a run
directory named in the output, so the table can be regenerated and checked.

The table the plan sketched crossed "trained with" against "evaluated under", which suited a
Swift-style observation-model story. Our measurements went somewhere else: the filters are all
inference-time and the frozen policy is common to every arm. Both the clearance geometry AND
the cap law vary, so a geometry-only comparison must hold the law fixed (stopcap vs dwa_arc).
The missing riskcap-law/arc-geometry cell and cross-condition replication belong to A7.
"""

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A4 = REPO / "results" / "navrl_contact_geometry_seed509"
FORENSICS = {
    491: REPO / "results" / "navrl_contact_geometry_seed491",
    497: REPO / "results" / "navrl_contact_geometry_seed497",
}
A3 = REPO / "results" / "navrl_contact_geometry_seed503"

ARMS = ("off", "fixed2p0", "riskcap", "stopcap", "omni", "dwa_arc")
GEOMETRY = {
    "off": "—", "fixed2p0": "—",
    "riskcap": "직선 회랑 ±0.45 m", "stopcap": "직선 회랑 ±0.45 m",
    "omni": "전방위", "dwa_arc": "원호 튜브 ±0.45 m",
}
LAW = {
    "off": "없음", "fixed2p0": "상수 2.0", "riskcap": "바닥 2.0 + 해제",
    "stopcap": "정지거리", "omni": "정지거리", "dwa_arc": "정지거리",
}


def _load(path):
    return json.loads((path / "70bars.json").read_text())


def _delta_ci(a, b, key):
    pa, na = a["outcome"][key], a["actual_episodes"]
    pb, nb = b["outcome"][key], b["actual_episodes"]
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = (pa - pb) * 100.0
    return d, d - 1.96 * se * 100.0, d + 1.96 * se * 100.0


def main():
    D = {a: _load(A4 / a) for a in ARMS}
    base = D["riskcap"]
    out = ["# A5 — 절제표 (논문용)", "",
           "생성: `tools/build_a5_ablation_table.py`. 시뮬레이션 없음 — 커밋된 결과 JSON만 읽는다.", ""]

    out += ["## 표 1 — 필터의 감시 기하와 상한 법칙 (seed 509, 6 arm, 요청 2,049 ep/arm, 동일 체크포인트·소스)", "",
            "실제 완료 수는 arm별 2,049~2,051 ep이며, 비율과 CI는 각 JSON의 `actual_episodes`를 사용한다.", "",
            "| arm | 감시 기하 | 상한 법칙 | capture | crash | timeout | Δcrash vs riskcap |",
            "|---|---|---|---:|---:|---:|---|"]
    for a in ARMS:
        o = D[a]["outcome"]
        if a == "riskcap":
            delta = "기준선"
        else:
            d, lo, hi = _delta_ci(D[a], base, "crash_rate")
            mark = "**" if (hi < 0 or lo > 0) else ""
            delta = f"{mark}{d:+.2f} pp{mark} [{lo:+.2f}, {hi:+.2f}]"
        out.append(
            f"| `{a}` | {GEOMETRY[a]} | {LAW[a]} | {o['capture_rate']:.2%} | "
            f"{o['crash_rate']:.2%} | {o['timeout_rate']:.2%} | {delta} |"
        )

    geometry_crash, geometry_lo, geometry_hi = _delta_ci(D["dwa_arc"], D["stopcap"], "crash_rate")
    law_crash, law_lo, law_hi = _delta_ci(D["stopcap"], base, "crash_rate")
    geometry_capture, _, _ = _delta_ci(D["dwa_arc"], D["stopcap"], "capture_rate")
    geometry_timeout, timeout_lo, timeout_hi = _delta_ci(D["dwa_arc"], D["stopcap"], "timeout_rate")
    out += ["",
            "조건: 70 bars·ref5in ep1900·brake 2.0. `riskcap`→`dwa_arc`는 법칙과 기하를 함께 바꾼다.",
            f"같은 정지법칙의 `dwa_arc − stopcap` crash는 **{geometry_crash:+.2f} pp** "
            f"CI[{geometry_lo:+.2f}, {geometry_hi:+.2f}] (비유의)이고, "
            f"같은 직선 기하의 `stopcap − riskcap`은 **{law_crash:+.2f} pp** "
            f"CI[{law_lo:+.2f}, {law_hi:+.2f}]다.",
            "09-02 스크린의 stopcap 21.31 %는 205 bars·ep25000·seed 49·brake 2.9608856678의 "
            "별도 실행이다. 이 표와는 플랫폼·초기 목표 거리도 달라, 부호 반전을 밀도 하나에 귀속할 수 없다."]

    out += ["", "## 표 2 — 보수성의 대가", "",
            "| arm | 경로 길이 | <3 m 체류 | 개입률 | 실행 속도 | 거버너 µs/env |",
            "|---|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        c, g = D[a]["contact_geometry"], D[a]["speed_governor"]
        out.append(
            f"| `{a}` | {c['path_length_mean_m']:.1f} m | "
            f"{c['corridor_clearance_below_3m_rate']:.2%} | {g['intervention_rate']:.1%} | "
            f"{g['mean_executed_speed_mps']:.2f} m/s | {c['governor_compute_us_mean_per_env']:.2f} |"
        )

    out += ["", "## 표 3 — 접촉이 회랑 밖이라는 사실의 재현 (2 시드)", "",
            "| seed | arm | lateral | no_return | 합계 | in_corridor | VERTICAL_OUT |",
            "|---:|---|---:|---:|---:|---:|---:|"]
    for seed, root in FORENSICS.items():
        for arm in ("off", "riskcap"):
            cm = _load(root / arm)["contact_geometry"]["commanded_direction"]
            out.append(
                f"| {seed} | `{arm}` | {cm['lateral_rate']:.1%} | {cm['no_return_rate']:.1%} | "
                f"**{cm['lateral_rate'] + cm['no_return_rate']:.1%}** | "
                f"{cm['in_corridor_rate']:.1%} | {cm['vertical_out_rate']:.1%} |"
            )

    out += ["", "## 표 4 — 처방의 반사실 (A3, seed 503, 정책 무변경)", "",
            "star-convex 자유공간이었다면 회랑이 놓친 접촉 중 몇 %가 사전 감지되었나.", "",
            "| arm | 대상 | 재분류 | 비율 | lateral | no_return |",
            "|---|---:|---:|---:|---:|---:|"]
    te = tr = 0
    for arm in ("off", "riskcap"):
        sc = _load(A3 / arm)["star_convex_shadow"]
        te += sc["eligible_contacts"]; tr += sc["reclassified_seen"]
        out.append(
            f"| `{arm}` | {sc['eligible_contacts']} | {sc['reclassified_seen']} | "
            f"**{sc['reclassification_rate']:.1%}** | {sc['lateral_rate']:.1%} | "
            f"**{sc['no_return_rate']:.1%}** |"
        )
    r = tr / te
    se = math.sqrt(r * (1 - r) / te)
    out.append(f"| 합산 | {te} | {tr} | **{r:.1%}** CI[{r-1.96*se:.1%}, {r+1.96*se:.1%}] | | |")

    out += ["", "## 읽는 법", "",
            "1. **표 3**: 충돌의 4분의 3 이상이 감시 회랑 밖이며 두 시드에서 재현된다. "
            "`VERTICAL_OUT`은 정확히 0 — 막대가 2 m로 높고 순항 고도가 1 m이므로 기하의 귀결이다.",
            "2. **표 1**: 그렇다고 회랑을 없애면(`omni`) 무너진다 — crash는 최저지만 timeout 49.8 %. "
            f"같은 정지법칙의 `stopcap`→`dwa_arc`는 capture {geometry_capture:+.2f} pp, "
            f"timeout {geometry_timeout:+.2f} pp CI[{timeout_lo:+.2f}, {timeout_hi:+.2f}]이며 "
            "crash 차이는 비유의다. riskcap 대비 −7.70 pp 전체를 원호 효과로 읽을 수 없다.",
            "3. **표 2**: `dwa_arc`는 이 조건에서 정지법칙의 비용을 걷어냈다 — `stopcap` 대비 "
            "3 m 미만 체류는 21.85→6.25 %, 개입률은 19.3→4.0 %다. "
            "riskcap 법칙 아래에서도 원호가 같은 역할을 하는지는 A7의 빠진 2×2 칸이 답한다.",
            "4. **표 4**: 회랑이 놓친 접촉의 68.9 %는 사전 감지 가능했고, 그중 무반환 범주는 96~97 %다.",
            "", "A4 사전등록의 역사적 판정 `GEOMETRY_MATTERS`는 보존한다. 그 규칙의 riskcap 대조는 "
            "법칙·기하의 결합 효과를 검출하므로, 독립적인 원호 안전 효과나 재현을 확정하는 근거로 "
            "쓰지 않는다. A7의 2×2 표와 판정은 `tools/build_a7_arc_attribution_table.py`가 별도로 생성한다.",
            "", "## 이 표가 지지하지 않는 것", "",
            "- 어떤 필터도 **실기에서** 검증되지 않았다. 전부 시뮬레이션이다.",
            "- 필터와 함께 **재적응 학습**하면 순위가 바뀔 수 있다. 모든 arm은 riskcap과 함께 "
            "적응한 동결 정책 위에서 추론 시점에만 작동했다 — `stopcap`·`omni`·`dwa_arc`는 "
            "이 정책에 분포 밖이다.",
            "- `riskcap`→`dwa_arc`에서 법칙과 기하를 함께 바꿨다. `stopcap`→`dwa_arc`의 "
            "동일 법칙 비교는 있으나, `riskcap_arc`가 없어 법칙과 독립적인 기하 효과·상호작용은 미판정이다.",
            "- 기하 비교는 70 bars·ref5in ep1900·seed 509에 한정된다. A7 P3는 기존 205 bars "
            "스크린에서 brake 축을 검사하지만 과거 스크린은 `runtime_git_dirty=true`이고 P3는 "
            "접촉 기하 계측도 추가한다. M5만으로 소스·계측의 동등성을 입증하지는 못한다. "
            "P1/P3 차이는 밀도 외 체크포인트·플랫폼·목표 거리·시드도 포함한다. "
            "A7 결과나 밀도의 단독 인과효과를 이 표에서 주장하지 않는다.", ""]
    (REPO / "docs" / "a5_ablation_table.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
