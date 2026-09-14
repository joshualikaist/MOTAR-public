# VOID — A3 seed 503, source drift during evaluation

**무효 사유**: 평가 실행 중 `aerial_gym/task/navrl_task/navrl_task.py`가 변경됐다.
평가기가 `runtime source changed during evaluation`으로 exit 1 했고, `riskcap` arm은 시작조차
하지 못했다.

원인은 운영 실수다. A3(seed 503)를 백그라운드로 돌리는 동안 같은 파일에 A4 baseline
(`omni` / `dwa_arc`)의 clearance 분기를 배선했다. 실행 중인 파이썬 프로세스는 이미 임포트된
구 코드를 썼을 가능성이 높지만, **영수증이 그것을 보증할 수 없다.** 그것이 이 가드가 존재하는
이유다.

`off` arm의 수치(재분류 156/234 = 66.7 %, `no_return` 55/57, `lateral` 101/177)는 60-ep 스모크
(66.7 %)와 일치했으나 **채택하지 않는다.** 계보가 증명되지 않은 수치를 쓰면 그 뒤의 모든 판정이
같은 결함을 물려받는다.

**교훈**: GPU 평가가 도는 동안 `aerial_gym/**` 소스를 편집하지 않는다. 다음 단계 구현은 평가
완료 후에 하거나, 별도 worktree에서 한다.

재실행: seed 503 그대로, 작업트리를 커밋해 깨끗이 한 뒤 처음부터.
