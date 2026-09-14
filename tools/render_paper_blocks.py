#!/usr/bin/env python3
"""Paper-style redrawing of existing documentation; no algorithm changes.

Run with detector_runs/venv/bin/python tools/render_paper_blocks.py.
Uses the existing Chrome exporter (websocket-client required).
"""
import hashlib
import json
import zipfile
from html import escape
from pathlib import Path
import render_presentation_figures as export

OUT = Path(__file__).resolve().parents[1] / 'docs/assets/paper'


class Diagram:
    def __init__(self, title, desc):
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#253341"/></marker></defs><rect width="1600" height="900" fill="white"/><g font-family="Arial, Noto Sans CJK KR, sans-serif" fill="#16232f">']
        self.text(55, 68, title, 35, bold=True)

    def text(self, x, y, t, size=25, bold=False, anchor='start', color='#253341'):
        self.s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{escape(t)}</text>')

    def block(self, x, y, w, h, title, rows=(), fill='white', planned=False):
        self.s.append('<g data-block="true">')
        dash = ' stroke-dasharray="8 6"' if planned else ''
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="#253341" stroke-width="2"{dash}/>')
        self.text(x+w/2, y+43, title, 27, True, 'middle')
        for i, row in enumerate(rows):
            self.text(x+w/2, y+83+i*33, row, 23, anchor='middle')
        self.s.append('</g>')

    def edge(self, points, dashed=False):
        d = 'M' + ' L'.join(f'{x} {y}' for x, y in points)
        self.s.append(f'<path d="{d}" fill="none" stroke="#253341" stroke-width="2.5" marker-end="url(#arrow)"' + (' stroke-dasharray="8 6"' if dashed else '') + '/>')

    def note(self, lines):
        self.s.append('<path d="M55 784H1545" stroke="#b6c0c8"/>')
        for i, line in enumerate(lines):
            self.text(55, 824+i*35, line, 22, color='#52606d')

    def junction(self, x, y):
        self.s.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#253341"/>')

    def save(self, name):
        (OUT / name).write_text('\n'.join(self.s) + '\n</g></svg>\n', encoding='utf-8')


def perception():
    d = Diagram('Streaming perception pipeline', 'Existing documented image-processing branches, temporal buffer and frame-local selection output. No camera-to-policy integration is implied.')
    d.block(55, 310, 215, 150, 'Frame input', ['BGR image', 'sequence · time'], '#f1f4f6')
    d.block(365, 170, 260, 150, 'Frozen detector', ['Per-frame candidates', 'Top-K, K = 5'], '#eaf1f8')
    d.block(365, 475, 260, 150, 'Optical flow', ['Previous / current', 'grayscale frames'], '#f1f4f6')
    d.edge([(270, 355), (315, 355), (315, 245), (365, 245)])
    d.edge([(315, 355), (315, 550), (365, 550)])
    d.junction(315, 355)
    d.text(380, 145, 'GPU branch', 22)
    d.text(380, 455, 'CPU branch', 22)
    d.block(715, 310, 250, 175, 'Feature assembly', ['Candidates + GMC', 'Motion features'], '#eaf1f8')
    d.edge([(625, 245), (665, 245), (665, 360), (715, 360)])
    d.edge([(625, 550), (665, 550), (665, 435), (715, 435)])
    d.text(652, 680, 'Parallel branches', 23, anchor='middle')
    d.block(1050, 310, 205, 175, 'History buffer', ['T = 16 frames', 'Sequence-local'])
    d.edge([(965, 397), (1050, 397)])
    d.block(1340, 310, 205, 175, 'Transformer', ['Temporal selector', 'P7c v2'], '#eaf1f8')
    d.edge([(1255, 397), (1340, 397)])
    d.block(1165, 610, 380, 115, 'Selection output', ['Candidate rank or NO_LOCK'], '#f1f4f6')
    d.edge([(1442, 485), (1442, 610)])
    d.text(1080, 550, 'State reset on sequence change', 21)
    d.note(['Solid arrows: implemented data flow. GMC: global motion compensation.',
            'Frame-local selection only; GT annotations are not inference inputs.'])
    d.save('perception-block-diagram.svg')


def safety():
    d = Diagram('Safety filter: geometry and speed-cap blocks', 'Module-level summary of the documented geometry and cap-law comparison; geometry arms are alternatives, not sequential filters. No new geometry implementation is specified.')
    d.text(55, 112, 'Direction-preserving speed filter · geometry arms are alternatives', 24, color='#52606d')
    d.block(55, 195, 260, 145, 'Task inputs', ['LiDAR returns', 'Ego state'], '#f1f4f6')
    d.block(440, 195, 310, 145, 'Geometry arm', ['Straight corridor', 'or arc-clearance'], '#eaf1f8')
    d.block(890, 195, 280, 145, 'Speed-cap law', ['Configured mode', 'Scalar speed limit'], '#eaf1f8')
    d.edge([(315, 268), (440, 268)])
    d.edge([(750, 268), (890, 268)])
    d.text(781, 245, 'clearance', 20)
    d.block(55, 495, 260, 145, 'Policy command', ['Horizontal velocity', 'Requested direction'], '#f1f4f6')
    d.block(890, 495, 280, 145, 'Magnitude scaling', ['Direction unchanged', 'Speed bounded by cap'], '#eaf1f8')
    d.block(1290, 495, 255, 145, 'Filtered command', ['Horizontal velocity', 'To fixed controller'], '#f1f4f6')
    d.edge([(315, 567), (890, 567)])
    d.text(610, 600, 'Original horizontal command', 23, anchor='middle')
    d.edge([(375, 567), (375, 405), (595, 405), (595, 340)])
    d.junction(375, 567)
    d.text(406, 385, 'command context', 22)
    d.edge([(1030, 340), (1030, 495)])
    d.text(1050, 421, 'speed cap', 22)
    d.edge([(1170, 567), (1290, 567)])
    d.text(55, 727, 'Geometry choice and cap-law choice are separate comparison axes.', 24)
    d.note(['Solid arrows: module-level data flow. No direction replanning or vertical regulation is shown.',
            'The arc-clearance filter is not a DWA planner or a collision-safety guarantee.'])
    d.save('safety-filter-block-diagram.svg')


def overview():
    d = Diagram('MOTAR research workflow', 'Three separate research workflows from data sources to evaluation artifacts; not a deployed closed-loop architecture.')
    d.text(55, 135, 'DATA / EXPERIMENT SOURCE', 21)
    d.text(585, 135, 'ANALYSIS', 21)
    d.text(1115, 135, 'RECORDED OUTPUT', 21)
    rows = [
        (185, 'A · Simulation', ['Fixed experiment contract'], 'Filter comparison', ['Geometry and cap-law arms'], 'Diagnostic evidence', ['Capture / crash / timeout']),
        (385, 'B · Real imagery', ['NPS-Drones / Det-Fly'], 'Perception evaluation', ['Detector / temporal selector'], 'Measured errors', ['Streaming / P8–P10 reports']),
        (585, 'C · ETH ds5', ['Video / published position GT'], 'Range study', ['E3-S analysis / E3-P gate'], 'Range-error report', ['Attitude analysis gated']),
    ]
    for y, a, ar, b, br, c, cr in rows:
        d.block(55, y, 420, 135, a, ar, '#f1f4f6')
        d.block(585, y, 420, 135, b, br, '#eaf1f8')
        d.block(1115, y, 430, 135, c, cr)
        d.edge([(475, y+68), (585, y+68)])
        d.edge([(1005, y+68), (1115, y+68)])
    d.note(['Arrows show study inputs and outputs, not an integrated flight-control connection.'])
    d.save('system-overview-block-diagram.svg')


def arena():
    d = Diagram('Simulation environment and observation sources', 'Component-level view of the existing arena and simulated sensor sources; not a to-scale map.')
    d.block(55, 330, 300, 155, 'Arena contract', ['Scene bounds', 'Episode configuration'], '#f1f4f6')
    d.block(480, 180, 320, 145, 'Static scene', ['Obstacle layout', 'Arena boundaries'])
    d.block(480, 515, 320, 145, 'Dynamic scene', ['Vehicle state', 'Moving-target state'])
    d.edge([(355, 407), (415, 407), (415, 253), (480, 253)])
    d.edge([(415, 407), (415, 588), (480, 588)])
    d.junction(415, 407)
    d.block(925, 330, 300, 155, 'Simulated sensing', ['Camera / depth', 'LiDAR returns'], '#eaf1f8')
    d.edge([(800, 253), (865, 253), (865, 375), (925, 375)])
    d.edge([(800, 588), (865, 588), (865, 443), (925, 443)])
    d.block(1315, 330, 230, 155, 'Observations', ['Sensor outputs', 'Ego-state fields'], '#f1f4f6')
    d.edge([(1225, 407), (1315, 407)])
    d.edge([(640, 660), (640, 720), (1430, 720), (1430, 485)])
    d.text(1100, 752, 'Ego state', 23, anchor='middle')
    d.note(['Block diagram, not spatial scale. Arena dimensions and sensor ranges remain in the text.'])
    d.save('arena-block-diagram.svg')


def platform():
    d = Diagram('Simulation platform: subsystem composition', 'Composition of the documented hardware-informed simulation candidate; not an assembled hardware wiring diagram.')
    d.block(535, 160, 530, 130, 'Hardware-informed simulation', ['navrl_ref5in_quad_v2 candidate'], '#f1f4f6')
    for x, title, rows in [
        (55, 'Vehicle model', ['Airframe / collision proxy']),
        (590, 'Sensor model', ['Camera / LiDAR / ego state']),
        (1125, 'Actuation model', ['Controller / motor dynamics']),
    ]:
        d.block(x, 400, 420, 130, title, rows, '#eaf1f8')
    d.edge([(800, 290), (800, 345), (265, 345), (265, 400)])
    d.edge([(800, 345), (800, 400)])
    d.edge([(800, 345), (1335, 345), (1335, 400)])
    d.junction(800, 345)
    for x, title, row in [(55, 'Physical parameters', 'Mass / inertia / geometry'),
                          (590, 'Observation interface', 'Images / returns / state'),
                          (1125, 'Simulation response', 'Forces / rigid-body motion')]:
        d.block(x, 625, 420, 115, title, [row])
        d.edge([(x+210, 530), (x+210, 625)])
    d.note(['Arrows show subsystem decomposition. Hardware unassembled; no measured hardware validation is implied.'])
    d.save('platform-block-diagram.svg')


def control():
    d = Diagram('Learned policy and fixed control stack', 'Simplified redrawing of the existing documented control chain; no gain, command-bound or controller modifications.')
    data = [
        ('Policy actor', ['Learned output']),
        ('Command mapping', ['Body-frame setpoints']),
        ('Altitude PI', ['Vertical-channel override']),
        ('Velocity control', ['Fixed Lee controller']),
        ('Force / attitude', ['Tilt-limited reference']),
        ('Attitude / rate', ['Torque command']),
        ('Motor allocation', ['Fixed motor dynamics']),
        ('Rigid-body physics', ['Simulation state']),
    ]
    xs = [55, 455, 855, 1255]
    for i, (title, rows) in enumerate(data):
        x = xs[i] if i < 4 else xs[7-i]
        y = 215 if i < 4 else 535
        d.block(x, y, 290, 140, title, rows, '#eaf1f8' if i == 0 else 'white')
        if i < 3:
            d.edge([(x+290, y+70), (x+400, y+70)])
        elif 4 <= i < 7:
            d.edge([(x, y+70), (x-110, y+70)])
    d.edge([(1400, 355), (1400, 535)])
    d.text(55, 150, 'LEARNED', 21)
    d.text(455, 150, 'FIXED CONTROLLER AND SIMULATION CONTRACT', 21)
    d.text(880, 751, 'Read top row left → right; bottom row right → left.', 24, anchor='middle')
    d.note(['The policy vertical output is overridden by the altitude PI. Detailed parameters remain in the text.'])
    d.save('control-block-diagram.svg')


def archived_sam():
    d = Diagram('Archived SAM candidate: proposed processing chain', 'Archived design; only the offline CPU instance adapter exists. Dashed blocks and arrows are unimplemented proposals.')
    d.text(55, 133, 'ARCHIVED · not in the control loop', 24, color='#8a4d20')
    specs = [
        ('Sensor packet', ['RGB-D / capture time'], True),
        ('SAM discovery', ['Proposed worker'], True),
        ('Instance adapter', ['Offline CPU boundary'], False),
        ('Association', ['Proposed track bank'], True),
        ('Decision output', ['Policy boundary'], True),
    ]
    for i, (title, rows, planned) in enumerate(specs):
        x = 55 + i*310
        d.block(x, 305, 250, 150, title, rows, '#f1f4f6' if planned else '#eaf1f8', planned)
        if i:
            d.edge([(x-60, 380), (x, 380)], dashed=True)
    d.block(675, 605, 250, 120, 'CC stub', ['Offline test input'], '#f1f4f6')
    d.edge([(800, 605), (800, 455)])
    d.note(['Dashed: proposed / unimplemented. Solid: existing offline CPU adapter and stub input.',
            'The SAM worker and downstream control integration were not implemented.'])
    d.save('sam-archive-block-diagram.svg')


def color_baseline():
    d = Diagram('Color-detector baseline: candidate collapse', 'Historical known-failure pipeline; same-color objects are combined into one centroid. This is not the selected real-imagery detector.')
    d.text(55, 133, 'KNOWN FAILURE · historical in-simulation baseline', 24, color='#8a4d20')
    specs = [
        ('RGB-D frame', ['Image input']),
        ('Pixel classifier', ['Color response']),
        ('Positive mask', ['Objects merged']),
        ('Single centroid', ['One candidate']),
        ('Association', ['LiDAR return', 'Baseline estimate']),
    ]
    for i, (title, rows) in enumerate(specs):
        x = 55 + i*310
        d.block(x, 305, 250, 150, title, rows, '#f1f4f6' if i in (0, 4) else '#fdf2eb')
        if i:
            d.edge([(x-60, 380), (x, 380)])
    d.block(365, 605, 350, 120, 'Same-color objects', ['Shared classifier response'], '#fdf2eb')
    d.edge([(490, 605), (490, 455)])
    d.text(1080, 581, 'Object identities are lost at the merge.', 25, anchor='middle')
    d.note(['This figure explains a measured failure; it does not describe an improved detector.'])
    d.save('color-baseline-block-diagram.svg')


def e3():
    d = Diagram('E3 analysis: separate range measurement from attitude attribution', 'Existing E3-S analysis and E3-P reliability gate, shown as separate workflows; no new analysis, criterion or cause attribution.')
    d.text(55, 149, 'E3-S · SIZE-BASED RANGE EVALUATION', 23)
    top = [('Video + position GT', ['Public ETH ds5']), ('Size / range samples', ['Paired measurements']),
           ('Block evaluation', ['Separate fitting / scoring']), ('Error report', ['Bias / absolute / relative'])]
    for i, (title, rows) in enumerate(top):
        x = 55+i*400
        d.block(x, 195, 290, 145, title, rows, '#eaf1f8' if i in (1, 2) else '#f1f4f6')
        if i:
            d.edge([(x-110, 268), (x, 268)])
    d.text(55, 499, 'E3-P · ATTITUDE ATTRIBUTION GATE', 23)
    d.block(55, 545, 290, 145, 'Published pose', ['Attitude / timestamps'], '#f1f4f6')
    d.block(455, 545, 390, 145, 'Reliability gate', ['Convention / timing / reliability'], '#eaf1f8')
    d.block(1095, 545, 450, 145, 'BLOCKED', ['No attitude-error decomposition'], '#fdf2eb')
    d.edge([(345, 618), (455, 618)])
    d.edge([(845, 618), (1095, 618)])
    d.text(960, 592, 'not passed', 23, anchor='middle')
    d.note(['Range-estimation error is not attitude error. Numerical results and limitations are in the caption.'])
    d.save('e3-analysis-block-diagram.svg')


FIGURES = [
    ('system-overview-block-diagram', '전체 연구 흐름', overview,
     '세 연구 축의 입력 자료 → 분석 → 기록 산출물을 구분했습니다. 실제 비행의 end-to-end 연결을 뜻하지 않습니다.'),
    ('arena-block-diagram', '환경과 관측 생성', arena,
     '환경 계약에서 정적·동적 장면으로 분기하고, 센서 출력과 ego state가 관측으로 이어지는 구성도입니다. 공간 축척도는 별도 보존했습니다.'),
    ('platform-block-diagram', '플랫폼 구성', platform,
     '시뮬레이션 후보를 vehicle·sensor·actuation model로 분해한 구성도입니다. 부품 배선도나 조립 완료된 기체가 아닙니다.'),
    ('control-block-diagram', '정책과 제어 체인', control,
     '기존 8단계 제어 체인을 큰 블록으로 다시 배치했습니다. 윗줄은 왼쪽→오른쪽, 아랫줄은 오른쪽→왼쪽입니다. gain·수식은 본문에 남겼습니다.'),
    ('safety-filter-block-diagram', 'Safety filter', safety,
     '직선·arc-clearance는 선택 가능한 비교군이며 연속 필터가 아닙니다. geometry와 cap-law 축, 정책 명령의 분기·합류를 표시했습니다. 방향 재계획이나 안전 보장으로 해석하지 않습니다.'),
    ('perception-block-diagram', '인지 처리 과정', perception,
     '검출·광류의 병렬 흐름 → 특징 결합 → 시간 이력 → Transformer → 후보 선택입니다. 출력은 프레임 안의 rank 또는 NO_LOCK이며 물리적 track ID나 거리 추정이 아닙니다.'),
    ('sam-archive-block-diagram', '보관된 SAM 후보', archived_sam,
     '점선 박스·화살표는 미구현 제안입니다. 실선은 기존 offline CPU instance adapter와 CC stub 입력뿐입니다. SAM worker나 제어 통합은 구현되지 않았습니다.'),
    ('color-baseline-block-diagram', '색 검출기 실패 경로', color_baseline,
     '동색 물체가 같은 픽셀 분류를 통과해 하나의 mask와 중심점으로 합쳐지는 기존 실패를 도식화했습니다. 현재 채택한 실영상 detector가 아닙니다.'),
    ('e3-analysis-block-diagram', 'E3 분석 절차', e3,
     'E3-S의 크기 기반 거리 평가와 E3-P의 자세 신뢰성 gate를 분리했습니다. 수치·GT 품질·시간 정합성 한계는 본문과 결과 보고서에 남겼고 새로운 원인 분해는 하지 않았습니다.'),
]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for _, _, builder, _ in FIGURES:
        builder()
    entries = [{'svg': name+'.svg', 'png': name+'.png', 'pdf': name+'.pdf'} for name, _, _, _ in FIGURES]
    export.OUT = OUT
    export.rasterize(entries)
    captions = '# README 논문형 블록 다이어그램 · 2026-09-11\n\n'
    captions += 'README 본문의 그림 9개 전체와 같은 파일입니다. 흰 배경·큰 블록·직교 화살표로 통일했습니다.\n'
    captions += 'SVG와 PDF는 벡터 원본이며, PNG는 3840×2160입니다. PPT에는 PNG 또는 SVG, 논문에는 PDF를 사용할 수 있습니다.\n'
    captions += '수치·긴 설명은 본문 캡션으로 분리했습니다. 새 실험·알고리즘·매개변수 변경은 없습니다.\n\n'
    for i, (name, title, _, caption) in enumerate(FIGURES, 1):
        captions += f'## Fig. {i}. {title}\n\n{caption}\n\n[SVG]({name}.svg) · [PNG]({name}.png) · [PDF]({name}.pdf)\n\n'
    captions += '근거: [현재 README](https://github.com/joshualikaist/MOTAR/blob/main/README.md) · [streaming 명세](../../specs/perception_streaming_v1.md) · [E3 계획](../../plans/eth_ds5_e3_2026-09-10.md).\n\n'
    captions += '재생성: `python tools/render_paper_blocks.py` (Chrome, websocket-client 필요).\n'
    (OUT / 'README.md').write_text(captions, encoding='utf-8')
    files = [OUT / e[k] for e in entries for k in ('svg', 'png', 'pdf')] + [OUT / 'README.md']
    (OUT / 'manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)+'\n')
    with zipfile.ZipFile(OUT / 'motar-paper-block-diagrams.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [OUT / 'manifest.json']:
            info = zipfile.ZipInfo(p.name, (2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    cards = '\n'.join(f'<section><h2>{i:02d} · {escape(title)}</h2><img src="{e["svg"]}" alt="{escape(title)}"><p>{escape(caption)}</p><p><a href="{e["svg"]}" download>SVG</a> · <a href="{e["png"]}" download>4K PNG</a> · <a href="{e["pdf"]}" download>Vector PDF</a></p></section>' for i, ((_, title, _, caption), e) in enumerate(zip(FIGURES, entries), 1))
    (OUT / 'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MOTAR 논문형 블록 다이어그램 9종</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:18px/1.6 system-ui;color:#253341}img{width:100%;height:auto}a{color:#25608c}section{margin:60px 0}p{max-width:1000px}</style><h1>README 블록 다이어그램 9종 · 2026-09-11</h1><p><a href="motar-paper-block-diagrams.zip">SVG + 4K PNG + 벡터 PDF 전체 다운로드</a> · <a href="README.md">그림 설명과 근거</a></p>\n'+cards+'</html>\n', encoding='utf-8')
    print('Built nine paper-style SVG + 4K PNG + vector PDF diagrams')


if __name__ == '__main__':
    build()
