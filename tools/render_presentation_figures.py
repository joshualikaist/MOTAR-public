#!/usr/bin/env python3
"""Refresh documentation figures and export a Korean 16:9 presentation package.

This is a presentation-only build: reads existing diagrams, never runs experiments.
python3 tools/render_presentation_figures.py [--raster]
Raster export requires Chrome and websocket-client; SVG generation uses stdlib only.
"""
import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
OUT = ASSETS / 'presentation'
DATE = '2026-09-10'
INK, MUTED, TEAL, AMBER = '#142d3b', '#526775', '#007f73', '#a55316'


class Slide:
    def __init__(self, title, subtitle, status):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc><rect width="1600" height="900" fill="#f4f7fa"/><g font-family="Arial, Noto Sans CJK KR, sans-serif">']
        self.text(52, 48, 'MOTAR / RESEARCH EVIDENCE', 17, TEAL, 700)
        self.text(1548, 48, DATE + ' · ' + status, 16, MUTED, 700, 'end')
        self.text(52, 108, title, 39, INK, 700)
        self.text(52, 153, subtitle, 22)

    def text(self, x, y, text, size=23, color=MUTED, weight=400, anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(text)}</text>')

    def box(self, x, y, w, h, fill='white'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="#dce5ea"/>')

    def card(self, x, y, w, tag, title, rows, color=TEAL, h=300):
        self.box(x, y, w, h)
        self.text(x+26, y+42, tag, 17, color, 700)
        self.text(x+26, y+88, title, 29, INK, 700)
        for i, row in enumerate(rows):
            self.text(x+26, y+139+i*39, row, 22)

    def footer(self, note, source):
        self.parts.append('<path d="M52 804H1548" stroke="#cbd8df"/>')
        self.text(52, 839, note, 20, AMBER)
        self.text(52, 875, source, 16)

    def save(self, path):
        path.write_text('\n'.join(self.parts) + '\n</g></svg>\n', encoding='utf-8')


def overview():
    s = Slide('연구의 세 축 — 무엇을 측정했고, 어디까지 확인했는가',
              '시뮬레이션 진단 · 실영상 인지 · 공개 데이터 거리 측정은 서로 다른 검증 범위입니다.', 'EVIDENCE MAP')
    s.card(52, 204, 482, 'TRACK A / SIMULATION', '안전필터 진단',
           ['실험 계약과 실패 모드를 비교', '학습 시드 반복 검증 완료', '공동적응 주장은 철회', '실기 안전성을 입증한 것은 아님'], h=335)
    s.card(559, 204, 482, 'TRACK B / REAL-IMAGERY SOFTWARE', '인지 파이프라인',
           ['검출 · 시간 모델 · 스트리밍', 'P8–P10 오차 측정 및 시뮬 평가', '최종 selector test 평가 완료', '실영상 → 실기 정책 검증은 아님'], h=335)
    s.card(1066, 204, 482, 'TRACK C / PUBLIC REAL FOOTAGE', '크기 기반 거리 측정',
           ['ETH ds5 · 지상 카메라 1대', 'E3-S: SIZE_RANGE_USABLE', 'E3-P: 자세 신뢰성 gate 미달', '자세 효과의 원인 분해는 보류'], h=335)
    s.box(52, 572, 1496, 189, '#e4f1ee')
    s.text(80, 617, '완료된 측정과 남은 전제조건을 구분합니다', 29, TEAL, 700)
    s.text(80, 665, '기존 색 detector + 단일 KF는 알려진 실패 기준선이며, 새 인지 경로의 대표 그림이 아닙니다.', 24)
    s.text(80, 712, '각 축의 결과는 연결되어 있지만, 하나의 실기 end-to-end 성능으로 합산하지 않습니다.', 24)
    s.footer('공통 한계: 실기 미조립 · 자체 센서 로그 없음 · 실제 비행 성능 및 sim-to-real 전이 미입증',
             '근거: README.md / Track A·B·C 상태표 및 연결된 결과 보고서')
    s.save(ASSETS / 'motar-system-overview.svg')


def perception():
    s = Slide('실영상 인지 — 구현 경로와 검증 결과',
              'FINAL PERCEPTION PATH · 기존 결과의 요약이며 새로운 실험이나 구현 계약이 아닙니다.', 'SOFTWARE EVALUATED')
    s.card(52, 200, 482, '01 / DATA & DETECTOR', 'NPS-Drones · Det-Fly',
           ['joint + multi-scale 학습 완료', 'Det-Fly validation AP50: 0.7166', 'K = 5 후보를 유지', 'zero-shot 평가와 직접 전후 비교 금지'], h=318)
    s.card(559, 200, 482, '02 / TEMPORAL COMPARISON', '동일 검출기 위의 비교',
           ['CNN + KF: 채택하지 않음', 'GRU · T=8: 채택하지 않음', 'Transformer · T=16: P7c v2 선택', 'validation utility: 0.6855'], h=318)
    s.card(1066, 200, 482, '03 / FROZEN EVALUATION', '스트리밍과 최종 test',
           ['S1: 14.72 → 22.48 FPS', '2,296 validation 프레임 출력 일치', 'S4 test utility: 0.4701', 'test: 1,725 프레임 · 다른 영상'], h=318)
    s.box(52, 548, 1496, 218, '#e4f1ee')
    s.text(80, 590, 'P8 → P9 → P10 / Inject measured errors · PPO retraining', 25, TEAL, 700)
    s.text(80, 637, 'P8: 단일 GT 1,310프레임에서 측정 · 8 px 미만 / 64 px 이상은 지원 근거 없음', 23)
    s.text(80, 682, 'P10 사전등록 주 효과: +1.41 pp, 95% CI [−0.38, +3.21] — 0 포함', 27, INK, 700)
    s.text(80, 726, '8셀 평가 완료 · 학습 시드 1개 · 유의한 회복이나 clean 성능 보존을 입증하지 못함', 23)
    s.footer('센서 융합과 실제 비행은 별도 미검증 범위 · validation/test 차이를 짝지은 효과로 해석하지 않음',
             '근거: results/perception_{streaming_overlap_s1,s4,p8,p10}_2026-09-09/README.md')
    s.save(ASSETS / 'motar-perception-final.svg')


def range_result():
    s = Slide('E3 분리 — 거리 추정은 측정, 자세 효과는 보류',
              'ETH ds5 cam0 · 한 비행 / 한 카메라 · 공개 위치 GT와 영상 크기 측정에 한정', 'E3-S COMPLETE / E3-P BLOCKED')
    s.box(52, 202, 958, 553)
    s.text(80, 245, 'E3-S / SIZE_RANGE_USABLE', 20, TEAL, 700)
    s.text(80, 339, '6.2%', 83, TEAL, 700)
    s.text(360, 299, '블록별 절대 상대 거리 오차 중앙값의 중앙값', 24, INK, 700)
    s.text(360, 342, '프레임 전체 pooled 중앙값이나 자세 오차가 아님', 22)
    for i, row in enumerate([
        '3,107프레임 · 9개 15초 블록 · 관측 거리 31–108 m',
        '블록 제외 교차검증: 평가 블록은 해당 fitting에서 제외',
        '크기 척도: 어두운 픽셀 수의 제곱근 — GT bbox가 아님',
        '9.3 pp: 블록 중앙값의 p90−p10 산포 — 신뢰구간이 아님',
        '30–50 m 구간은 표본 부족 · 나머지 3개 구간 보고',
        '시간 이동 민감도: 6.2% → 6.5% · 원인은 분해하지 않음',
        '이미 탐색한 동일 비행 · 독립적인 신규 비행 검증이 아님',
    ]):
        s.text(80, 407+i*46, row, 23)
    s.card(1035, 202, 513, 'E3-P / E3P_GATE_BLOCKED', '상대 자세 효과: 미측정',
           ['규약 · 분리 가능성 · 시간 gate 확인', '남은 blocker: 자세 신뢰성 기준 미달', '기준을 통과할 때까지 바꾸지 않음', '별도 사전등록 전에는 분해하지 않음'], color=AMBER, h=328)
    s.box(1035, 555, 513, 200, '#fff0e3')
    s.text(1061, 598, '절대 기하: 미해결', 27, AMBER, 700)
    s.text(1061, 641, '5.6 px 재투영 잔차의 원인 미확정', 23)
    s.text(1061, 682, '0.196 px는 궤적 매끄러움 잔차', 23)
    s.text(1061, 723, '전체 측정 정확도의 인증값이 아님', 23)
    s.footer('거리와 시간이 교락 · 위치 GT 경고 상태 포함 · 6.9% 크기 산포를 거리/자세 오차로 읽지 않음',
             '근거: results/eth_ds5_e3s_2026-09-10/README.md · docs/plans/eth_ds5_e3_2026-09-10.md')
    s.save(ASSETS / 'motar-eth-e3-evidence.svg')


# Existing technical panels stay intact. New captions distinguish historical contracts
# from measured capabilities; no control parameters or runtime code are changed.
PARTS = [
    ('system-overview', '연구 전체 구조', None, []),
    ('arena-geometry', '환경과 센서 범위', 'SIMULATION / 기존 실험 기하',
     ['그림의 축척과 수치는 시뮬 계약입니다.', '센서 범위는 실제 기체의 실측값이 아닙니다.', '공개 ETH 영상의 촬영 조건과 구분합니다.']),
    ('platform-hardware', '플랫폼과 센서', 'CANDIDATE / 실기 미조립',
     ['기체와 센서는 설계 후보입니다.', 'BOM·전원·열·관성 실측이 없습니다.', '영상 평가 결과를 실기 성능으로 읽지 않습니다.']),
    ('control-stack', '정책과 고정 제어기', 'SIMULATION / 구현 계약',
     ['학습 정책과 고정 제어기를 구분한 그림입니다.', 'P10은 시뮬레이터 평가 결과입니다.', '실제 비행 및 전이 성능은 미검증입니다.']),
    ('safety-filter', '안전필터 진단', 'SIMULATION / 주장 범위',
     ['필터 구조와 실패를 설명하는 기존 그림입니다.', '학습 시드 반복 뒤 공동적응 주장은 철회됐습니다.', '구조 설명만으로 안전성을 보장하지 않습니다.']),
    ('perception-final', '실영상 인지와 평가', None, []),
    ('perception-detection', '색 검출기 기준선', 'KNOWN FAILURE / 기존 기준선',
     ['단일 중심점으로 후보를 합치는 실패가 있습니다.', '현재 실영상 인지의 대표 성능이 아닙니다.', '과거 실패 경로를 보존한 비교 자료입니다.']),
    ('perception-candidate', 'SAM 설계 기록', 'ARCHIVED / 채택하지 않은 후보',
     ['offline CPU adapter 범위만 구현됐습니다.', 'SAM worker 및 제어루프 통합은 없습니다.', '현재 인지 파이프라인과 혼동하지 않습니다.']),
    ('eth-e3-evidence', 'E3-S 결과와 E3-P gate', None, []),
]


def annotate(path, status, notes):
    """Idempotent caption strip, preserving original SVG bytes inside the wrapper."""
    original = path.read_text(encoding='utf-8')
    match = re.search(r'<!-- ORIGINAL_PANEL_START -->\n(.*?)\n<!-- ORIGINAL_PANEL_END -->', original, re.S)
    if match:
        original = match.group(1)
    root = ET.fromstring(original)
    _, _, w, h = map(float, root.attrib['viewBox'].split())
    new_h = h + 150
    caption = escape(status + ' · ' + DATE)
    rows = ''.join(f'<text x="40" y="{h+78+i*29:g}" font-size="18" fill="{MUTED}">{escape(t)}</text>' for i, t in enumerate(notes[:2]))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}" height="{new_h:g}" viewBox="0 0 {w:g} {new_h:g}" role="img" aria-labelledby="context-title context-desc"><title id="context-title">{caption}</title><desc id="context-desc">{escape(" ".join(notes))}</desc>\n<!-- ORIGINAL_PANEL_START -->\n{original.rstrip()}\n<!-- ORIGINAL_PANEL_END -->\n<rect y="{h:g}" width="{w:g}" height="150" fill="#e4f1ee"/><g font-family="Arial, Noto Sans CJK KR, sans-serif"><text x="40" y="{h+39:g}" font-size="20" fill="{TEAL}" font-weight="700">{caption}</text>{rows}</g></svg>\n', encoding='utf-8')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    overview()
    perception()
    range_result()
    entries = []
    for number, (name, title, status, notes) in enumerate(PARTS, 1):
        source = ASSETS / ('motar-' + name + '.svg')
        target = OUT / ('%02d-%s.svg' % (number, name))
        if status:
            annotate(source, status, notes)
            # Keep all text self-contained: nested SVG, no external image/font references.
            s = Slide(title, '기존 구조도와 최신 해석 범위를 함께 읽는 발표용 요약', status.split(' / ')[0])
            content = source.read_text(encoding='utf-8')
            content = re.sub(r'<svg\b', '<svg x="52" y="185"', content, count=1)
            content = re.sub(r'width="[\d.]+" height="[\d.]+"', 'width="960" height="600"', content, count=1)
            # Nested IDs must not shadow the slide accessibility title/description.
            content = content.replace('id="title"', 'id="panel-title"').replace('id="desc"', 'id="panel-desc"').replace('aria-labelledby="title desc"', 'aria-labelledby="panel-title panel-desc"')
            s.parts.append(content)
            s.box(1040, 202, 508, 557)
            s.text(1067, 250, '발표에서 강조할 범위', 27, TEAL, 700)
            # Manual Korean line wrapping keeps text readable at presentation scale.
            import textwrap
            y = 318
            for note in notes:
                for line in textwrap.wrap(note, 23):
                    s.text(1067, y, line, 24)
                    y += 39
                y += 30
            s.footer('원본 구조 및 실험 수치는 유지 · 추가된 설명은 검증 범위와 현재 상태를 구분합니다.',
                     '근거: README.md 해당 파트 및 연결된 결과 보고서 / 원본: motar-' + name + '.svg')
            s.save(target)
        else:
            shutil.copyfile(source, target)
        entries.append({'number': number, 'title': title, 'svg': target.name, 'png': target.with_suffix('.png').name,
                        'source': source.relative_to(ROOT).as_posix(), 'notes': notes})
    return entries


def rasterize(entries):
    import websocket
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:
        raise SystemExit('Chrome is required for --raster')
    with tempfile.TemporaryDirectory(prefix='motar-slides-') as tmp:
        process = subprocess.Popen([chrome, '--headless=new', '--no-sandbox', '--disable-gpu',
                                    '--disable-dev-shm-usage', '--remote-debugging-port=0',
                                    '--user-data-dir=' + tmp + '/profile', 'about:blank'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sock = None
        try:
            port_file = Path(tmp) / 'profile/DevToolsActivePort'
            for _ in range(100):
                if port_file.exists():
                    break
                if process.poll() is not None:
                    raise RuntimeError('Chrome exited before opening its debugging port')
                time.sleep(0.1)
            port = port_file.read_text().splitlines()[0]
            with urllib.request.urlopen('http://127.0.0.1:' + port + '/json', timeout=10) as response:
                target = next(t for t in json.load(response) if t['type'] == 'page')
            sock = websocket.create_connection(target['webSocketDebuggerUrl'], suppress_origin=True, timeout=20)
            sequence = 0

            def call(method, params=None):
                nonlocal sequence
                sequence += 1
                sock.send(json.dumps({'id': sequence, 'method': method, 'params': params or {}}))
                while True:
                    response = json.loads(sock.recv())
                    if response.get('id') == sequence:
                        if 'error' in response:
                            raise RuntimeError(response['error'])
                        return response['result']

            call('Emulation.setDeviceMetricsOverride', {'width': 1600, 'height': 900,
                                                       'deviceScaleFactor': 2.4, 'mobile': False})
            for item in entries:
                svg = OUT / item['svg']
                # setDocumentContent avoids navigation/load races and fixes the viewport,
                # unlike CLI screenshots whose window includes browser UI dimensions.
                frame = call('Page.getFrameTree')['frameTree']['frame']['id']
                html = '<!doctype html><meta charset="utf-8"><title>' + escape(svg.stem) + '</title><style>html,body{margin:0;width:1600px;height:900px;overflow:hidden}</style>' + svg.read_text(encoding='utf-8')
                call('Page.setDocumentContent', {'frameId': frame, 'html': html})
                check = call('Runtime.evaluate', {'expression': '''document.fonts.ready.then(() => {
                    const bad = [...document.querySelectorAll('text')].filter(t => {
                      const b=t.getBoundingClientRect();
                      return b.left < -1 || b.top < -1 || b.right > 1601 || b.bottom > 901;
                    }).map(t => t.textContent);
                    return bad;
                })''', 'awaitPromise': True, 'returnByValue': True})
                if check.get('exceptionDetails') or check['result'].get('value'):
                    raise RuntimeError('Text outside slide ' + item['svg'] + ': ' + str(check))
                block_check = call('Runtime.evaluate', {'expression': '''(() => {
                    const bad = [];
                    document.querySelectorAll('[data-block]').forEach(g => {
                      const r = g.querySelector('rect').getBoundingClientRect();
                      g.querySelectorAll('text').forEach(t => {
                        const b = t.getBoundingClientRect();
                        if (b.left < r.left + 6 || b.right > r.right - 6 || b.top < r.top || b.bottom > r.bottom)
                          bad.push(t.textContent);
                      });
                    });
                    return bad;
                })()''', 'returnByValue': True})
                if block_check.get('exceptionDetails') or block_check['result'].get('value'):
                    raise RuntimeError('Text outside block ' + item['svg'] + ': ' + str(block_check))
                shot = call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True,
                                                       'clip': {'x': 0, 'y': 0, 'width': 1600, 'height': 900, 'scale': 1}})
                (OUT / item['png']).write_bytes(base64.b64decode(shot['data']))
                if 'pdf' in item:
                    pdf = call('Page.printToPDF', {'printBackground': True, 'paperWidth': 1600/96,
                                                  'paperHeight': 900/96, 'marginTop': 0, 'marginBottom': 0,
                                                  'marginLeft': 0, 'marginRight': 0, 'scale': 1})
                    (OUT / item['pdf']).write_bytes(base64.b64decode(pdf['data']))
        finally:
            if sock:
                try:
                    sock.send(json.dumps({'id': sequence+1, 'method': 'Browser.close'}))
                except (OSError, websocket.WebSocketException):
                    pass
                sock.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def publish(entries, raster):
    files = []
    for item in entries:
        files.append(OUT / item['svg'])
        if raster:
            files.append(OUT / item['png'])
    manifest = {'date': DATE, 'canvas': [1600, 900], 'png_pixels': [3840, 2160] if raster else None,
                'scope': 'Documentation-only; existing results, no new experiments.', 'slides': entries,
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    evidence = [
        'results/perception_streaming_overlap_s1_2026-09-09/README.md',
        'results/perception_s4_2026-09-09/README.md',
        'results/perception_p8_2026-09-09/README.md',
        'results/perception_p10_2026-09-09/README.md',
        'results/eth_ds5_e3s_2026-09-10/README.md',
        'docs/plans/eth_ds5_e3_2026-09-10.md',
        'results/navrl_grid_r2_d4_trainseed_rep/README.md',
    ]
    manifest['evidence_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in evidence}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    intro = '# MOTAR 발표용 그림 · ' + DATE + '\n\n16:9 SVG 원본과 3840×2160 PNG입니다. PowerPoint에서 **삽입 → 그림**으로 추가하세요.\n'
    intro += 'PNG는 글꼴 호환성이 좋고, SVG는 확대해도 선명합니다. SVG 편집 시 Noto Sans CJK KR 글꼴이 필요할 수 있습니다.\n\n수치는 기존 결과의 요약이며 새 실험을 실행하지 않았습니다. 한계 문구를 잘라내지 마세요.\n\n'
    intro += '\n'.join('- %02d · %s — [%s](%s)' % (e['number'], e['title'], e['svg'], e['svg']) for e in entries)
    intro += '\n\n재생성: `python3 tools/render_presentation_figures.py --raster` (Chrome 및 websocket-client 필요).\n'
    (OUT / 'README.md').write_text(intro, encoding='utf-8')
    cards = ''.join(f'<article><h2>{e["number"]:02d} · {escape(e["title"])}</h2><img src="{e["svg"]}" alt="{escape(e["title"])}"><p><a href="{e["svg"]}" download>SVG 원본</a> · <a href="{e["png"]}" download>4K PNG</a></p></article>' for e in entries)
    (OUT / 'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MOTAR 발표용 그림</title><style>body{font:18px/1.6 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;background:#f4f7fa;color:#142d3b}img{width:100%;height:auto}article{margin:40px 0}a{color:#007f73}</style><h1>MOTAR 발표용 그림 · '+DATE+'</h1><p>9개 파트 · 16:9 SVG + 4K PNG · 기존 결과와 검증 한계 포함</p><p><a href="motar-presentation-2026-09-10.zip" download>전체 ZIP 다운로드</a> · <a href="README.md">사용 안내</a> · <a href="../../status/">연구 사이트</a></p>'+cards+'</html>\n', encoding='utf-8')
    if raster:
        with zipfile.ZipFile(OUT / 'motar-presentation-2026-09-10.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in files + [OUT / 'README.md', OUT / 'manifest.json']:
                info = zipfile.ZipInfo(path.name, (2026, 9, 10, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raster', action='store_true', help='export 4K PNG and ZIP with Chrome')
    args = parser.parse_args()
    slides = build()
    if args.raster:
        rasterize(slides)
        publish(slides, True)
    print('Built %d presentation figures%s' % (len(slides), ' + PNG / ZIP' if args.raster else ''))
