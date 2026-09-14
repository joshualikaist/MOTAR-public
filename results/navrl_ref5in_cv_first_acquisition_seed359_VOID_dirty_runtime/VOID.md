# VOID — 셀 실행 시 runtime tree가 dirty였음 (요약 미산출)

## 무슨 일이 있었나

toward/away 두 셀 모두 평가는 끝냈지만 요약 생성이 두 단계에서 막혔다.

1. 내 `verify_all`이 `verify_cell`의 두 번째 반환값을 runtime byte map으로 착각했다. 실제로는
   **receipt**이고 completion timestamp·evaluation nonce·셀별 log/result 해시를 담으므로 셀 간
   비교는 항상 실패한다. 기존 telemetry orchestrator처럼 receipt가 가리키는 manifest에서 map을
   뽑도록 고쳤다.
2. 그 뒤 `verify_runtime_clean_manifest`가 실제 계약 위반을 잡았다:
   `toward runtime source was dirty: ['?? aerial_gym/rl_training/rl_games/eval_navrl_v2_detector_coupling_binbias.sh']`
   — 다른 세션(Codex)의 untracked 런처가 `aerial_gym/` 아래 남아 있어 두 셀 모두 dirty tree에서
   실행됐다.

## 왜 강제로 넘기지 않았나

가드를 우회하면 이 결과의 runtime 바이트가 어떤 커밋과도 대응하지 않게 된다. 사전등록과 운영
규칙 모두 drift 시 VOID 후 원인 수정을 요구한다. 해당 런처를 커밋해 tree를 clean으로 만들고
재실행한다.

## 시드 재사용 근거

`summary.json`은 **생성되지 않았고**, 사전등록 primary screen(away never-acquired rate 차이)은
**한 번도 계산된 적이 없다**. 이 실행에서 관측한 수치는 outcome split뿐이다:
toward `94.29/5.61/0.10%`, away `37.19/9.22/53.59%` — 각각 seed 353의 `94.29/5.37/0.34%`,
`35.97/9.42/54.61%`를 재현하는 보조 지표이며 어떤 screen에도 들어가지 않는다.

즉 "판정을 보고 재시도"가 아니라 "요약이 산출되지 않은 실행의 재실행"이다. 동의하지 않으면 새
시드로 재사전등록하면 된다.
