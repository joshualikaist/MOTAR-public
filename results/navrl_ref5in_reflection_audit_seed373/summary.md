# N1 real-frame reflection audit (seed 373)

**판정: `CHIRALITY_CONFIRMED_REAL_FRAME`**

| 항목 | 값 |
|---|---|
| median(conj_err_lat) | 1.4540 |
| lateral sign agreement | 0.0249 |
| 비교가능 행 | 15218 |
| conj_err_lat p90 / p95 / p99 | 1.6598 / 1.7033 / 1.7643 |
| 유효 프레임 / 전체 프레임 | 15,488 / 15,488 (게이트 4,096) |
| stride_eff / decimations | 16 / 4 |
| 에피소드 (요청/실제) | 1,024 / 1,024 |
| 품질 게이트 | 9개 평가, 실패 0개 |

- checkpoint SHA-256 `197ea26999d6bb9c…`, frames.npz SHA-256 `4af8252d23de5701…` (npz는 커밋하지 않는다)
- 실행된 `aerial_gym/__init__.py` sha256 `1ec098503bb19538…` = manifest 항목 (Q6 강제)
- 조건: 70 bars, seed 373, deterministic, reflection_mode=original, speed_governor=off, provenance override 미사용

## 권한

이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않으며 P3를 해제하지 않는다 (`p2_verdict_changed`/`d1_verdict_changed`/`p3_unlocked` 전부 false).

## 한계 (사전등록 §2)

- L1: 관측 반사는 세계 반사의 근사다. 장애물 토큰은 부호만 뒤집고 순서를 재배열하지 않는다 (ppo_update_safety.py:398-403). mode-probe가 이 구멍을 slot_permutation_max_abs = 0.0078로 무시 가능함을 보였으며, 본 실험은 그 값을 재측정하지 않고 인용한다.
- L2: 정규화기(running_mean_std)는 좌우 대칭이 아니다. 플레이어는 preproc(M o) 순서로 적용하며 (navrl_players.py:155) 이는 물리적으로 옳지만, 측정된 chirality의 일부는 네트워크가 아니라 정규화기 통계의 비대칭에서 올 수 있다. 보조 측정 S1로 분해하되 판정에는 쓰지 않는다.
- L3: 이 실험은 outcome을 측정하지 않는다. chirality가 있어도 대칭 아레나에서는 성능 손실이 없을 수 있다(2026-08-02). 따라서 어떤 결과도 'chirality가 성능을 해친다'는 주장의 근거가 될 수 없다.
- L4: 단일 checkpoint·단일 조건. 70막대 1셀, seed 1개. 계보 전반이나 밀도 전반으로 일반화하지 않는다.
