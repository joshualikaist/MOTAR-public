# 독립 일반 렌더러 — CPU 신규 설치

대상은 일반 geometry/shading prototype이다. root package, Isaac Gym, detector, 정책·제어기는
설치하거나 실행하지 않는다. 기존 simulator/학습 conda 환경에 아래 패키지를 덮어쓰지 않는다.
검증 상태와 실패 이력은 [설치 결과](../results/renderer_cpu_install_2026-09-12/README.md)를 따른다.

## 전제조건

- Linux x86-64, Python 3.8의 `venv`, `git`, 인터넷 연결. 검증 호스트는 Ubuntu 20.04 계열이다.
- 약 2 GB 이상의 추가 공간을 확보한다. 캐시와 source build에 따라 더 필요할 수 있다.
- [requirements-renderer-cpu.txt](../requirements-renderer-cpu.txt)는 이 플랫폼 전용이다.
  Python 3.9 이상·Windows·macOS·CUDA 환경 호환성을 주장하지 않는다.
- runtime 버전과 주요 source를 고정했지만 OS/system libraries 및 격리 build 의존성까지
  고정한 hermetic 설치는 아니다. 전체 이력이 필요한 근거 비교를 위해 full checkout을 사용한다.

## 1. 새 환경 설치

저장소 루트에서 실행한다. `python3.8`이 없다면 먼저 Python 3.8 환경을 준비해야 한다.

```bash
renderer_env=$(mktemp -d /tmp/motar-renderer-env-XXXXXX)
python3.8 -m venv "$renderer_env"
PYTHONNOUSERSITE=1 "$renderer_env/bin/python" -m pip install --no-cache-dir pip==25.0.1
PYTHONNOUSERSITE=1 "$renderer_env/bin/python" -m pip install --timeout 60 --retries 3 \
  --cache-dir "$renderer_env/pip-cache" --report "$renderer_env/install.json" \
  -r requirements-renderer-cpu.txt
PYTHONNOUSERSITE=1 "$renderer_env/bin/python" -m pip check
```

PyPI `urdfpy==0.0.22`는 기존 테스트 146개 중 원기둥 처리 오류 8개를 낸다.
파일의 Git commit pin은 이 오류가 수정된 upstream source를 선택한다.
같은 버전이면 기존 환경의 VCS 설치가 **아무것도 교체하지 않고 성공 종료**할 수 있다.
그래서 새 환경을 사용하며 다음 출처 검사도 필수다. 기존 연구 환경에 강제 재설치를 권하지 않는다.

## 2. 실제 설치 출처와 격리 검사

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$renderer_env/bin/python" -B - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import site
import sys
import torch

assert sys.version_info[:2] == (3, 8)
assert sys.prefix != sys.base_prefix
assert site.ENABLE_USER_SITE is False
assert torch.__version__ == '2.4.1+cpu' and torch.version.cuda is None
prefix = Path(sys.prefix).resolve()
for package in metadata.distributions():
    location = Path(package.locate_file('')).resolve()
    assert location == prefix or prefix in location.parents, (package.metadata['Name'], location)
origin = json.loads(metadata.distribution('urdfpy').read_text('direct_url.json') or '{}')
assert origin.get('url') == 'https://github.com/mmatl/urdfpy.git'
assert origin.get('vcs_info', {}).get('commit_id') == '5466842899b33bd549e8f9e2a9a987bd5e37373b'
assert not origin.get('dir_info', {}).get('editable', False)
print('CPU profile/source/isolation: PASS')
PY
```

`pip check`만으로는 urdfpy의 잘못된 구현이나 같은 버전 no-op을 잡지 못한다.

## 3. 기존 독립 계약과 실제 일반 장면

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$renderer_env/bin/python" -B \
  -m unittest discover -s tests -p 'test_renderer*.py'

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$renderer_env/bin/python" -B \
  tools/run_renderer_validation.py --device cpu --seed 0 \
  --num-scenes 1 --width 160 --height 120 --frames 2 --output "$renderer_env/smoke1"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$renderer_env/bin/python" -B \
  tools/run_renderer_validation.py --device cpu --seed 0 \
  --num-scenes 1 --width 160 --height 120 --frames 2 --output "$renderer_env/smoke2"
```

테스트 구성은 기존 146개 + 새 패키징 계약 6개 = 152개다. 오류·실패·skip으로 줄여서 통과를 만들지
않는다. `--help`만 실행하는 것은 실제 패키지 import나 renderer 검증이 아니다.

실제 생성 receipt는 `RENDERED_UNASSESSED`, `experiment_verdict=NOT_EVALUATED`,
`benchmark=NOT_RUN`, `training=NOT_SUPPORTED`다. Warp는 초기화 중 CUDA 장치 미발견 메시지를
출력할 수 있으나 CPU 실행의 성공 여부는 프로세스 종료 코드와 receipt로 확인한다.
원본과 같은 검증에는 receipt에 기록된 파일 SHA 및 디코딩한 배열 SHA를 확인하고 두 실행의
배열 hash를 비교해야 한다. 압축 파일 전체 byte 일치만으로 대체하지 않는다.

## 한계

이 절차는 CPU 설치와 작은 정적 일반 장면의 실행성만 검증한다. GPU 성능, 전체 환경 step FPS,
shortcut 감소, detector/정책 성능은 측정하지 않는다. R4/R4b FAIL은 유지한다.
V1의 과거 CONTRACT_MISMATCH는 설치가 아니라 별도 테스트 수정 `5cea0e4`로 해소됐다
([후속 검사](repository_followup_2026-09-12.md)). 이를 검출기 통합 완료로 읽지 않는다.
