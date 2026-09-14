#!/usr/bin/env python3
"""Generate a NEW dated explanatory figure set; never rewrite pinned paper assets.

SVG is the source of truth. --export derives PNG/vector PDF using the existing
Chrome exporter; no simulator imports or experiment execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import render_paper_blocks as blocks
import render_presentation_figures as exporter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/paper/overview-2026-09-13'
TRACK_D = json.loads((ROOT / 'docs/status_manifest.json').read_text())['track_d']
SENSOR = '#edf4f6'
LEARNED = '#e7edf9'
FIXED = '#f2f2ed'


def diagram(stem, title, desc, nodes, edges, notes):
    d = blocks.Diagram(title, desc)
    for x, y, label, rows, category, future in nodes:
        d.block(x, y, 425, 140, label, rows, category, planned=future)
    for points, future in edges:
        d.edge(points, dashed=future)
    d.note(notes)
    d.save(stem + '.svg')


def chain(stem, title, labels, notes):
    positions = [(110, 170), (1060, 170), (1060, 375), (110, 375), (110, 580), (1060, 580)]
    nodes = [(x, y, *item) for (x, y), item in zip(positions, labels)]
    routes = [[(535,240),(1060,240)], [(1272,310),(1272,375)],
              [(1060,445),(535,445)], [(322,515),(322,580)], [(535,650),(1060,650)]]
    edges = [(route, labels[i+1][-1]) for i, route in enumerate(routes[:len(labels)-1])]
    diagram(stem, title, 'Explanatory module map of existing research, not a deployment design.', nodes, edges, notes)


def system_overview():
    d = blocks.Diagram('MOTAR system and evidence boundaries',
        'Central simulation research path with separate real-image error and appearance-rendering evidence branches.')
    x, w, h = 610, 380, 64
    labels = [('3-D environment', SENSOR), ('Camera + LiDAR', SENSOR),
              ('Perception / tracking', LEARNED), ('Temporal policy · PPO', LEARNED),
              ('Safety filter', FIXED), ('Low-level control', FIXED),
              ('Simulated UAV', FIXED)]
    ys = [112, 198, 284, 370, 456, 542, 628]
    for y, (label, color) in zip(ys, labels):
        d.block(x, y, w, h, label, fill=color)
    for first, second in zip(ys, ys[1:]):
        d.edge([(x+w/2, first+h), (x+w/2, second)])
    d.block(65, 174, 350, 105, 'Real UAV video', ['Separate data lineage'], SENSOR)
    d.block(65, 354, 350, 105, 'Measured errors', ['Detector · association'], SENSOR)
    d.block(65, 534, 350, 105, 'Simulation injection', ['P8–P10 experiments'], SENSOR)
    d.edge([(240,279),(240,354)])
    d.edge([(240,459),(240,534)])
    d.edge([(415,586),(510,586),(510,316),(610,316)], dashed=True)
    d.block(1185, 222, 350, 105, 'Appearance research', ['Geometry · shading'], FIXED)
    d.block(1185, 402, 350, 105, 'Dynamic-mesh probe', ['Shadow-only integration'], SENSOR)
    d.block(1185, 582, 350, 105, 'Bounded evidence', ['D6 · D7'], SENSOR)
    d.edge([(1360,327),(1360,402)])
    d.edge([(1360,507),(1360,582)])
    d.note(['Solid arrows: documented flows within each lineage. Dashed arrow: measured-error injection, not live RGB integration.',
            'Simulation only. Appearance shadow output has no downstream consumer; D8 perception integration is not started.'])
    d.save('research-overview-block-diagram.svg')


def appearance_pipeline():
    d = blocks.Diagram('Appearance and dynamic-mesh research paths',
        'Production analytic sensing is distinct from the independent geometry-to-appearance research path.')
    d.block(110, 140, 430, 115, 'Analytic proxy', ['Existing detector input'], FIXED)
    d.block(1060, 140, 430, 115, 'Production detector', ['Observation unchanged'], FIXED)
    d.edge([(540,198),(1060,198)])
    labels = [('URDF geometry',['Mesh · local pose'],FIXED),
              ('Ray intersection',['Target-local query'],SENSOR),
              ('Geometry buffers',['Depth · normal · face ID'],SENSOR),
              ('Material + lighting',['Independent RGB output'],FIXED)]
    xs = [55,445,835,1225]
    for x,(label,rows,color) in zip(xs,labels):
        d.block(x, 390, 320, 125, label, rows, color)
    for x1,x2 in zip(xs,xs[1:]):
        d.edge([(x1+320,452),(x2,452)])
    d.block(585, 620, 430, 105, 'Mesh-derived observation', ['D8 · not started'], FIXED, planned=True)
    d.edge([(995,515),(995,570),(800,570),(800,620)], dashed=True)
    d.text(995, 550, f"D7 shadow: {TRACK_D['D7']['status']} · no downstream consumer", 20,
           anchor='end', color='#52606d')
    d.note(['D7 times a read-only shadow query beside production; its debug output is not detector input.',
            f"D6 {TRACK_D['D6']['status']}; D7 {TRACK_D['D7']['status']} is cost/non-interference only; shortcut reduction is unmeasured."])
    d.save('appearance-rendering-block-diagram.svg')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks.OUT = OUT
    system_overview()
    chain('system-control-block-diagram', 'Documented simulation control stack', [
        ('Camera + LiDAR', ['Existing simulation sensing'], SENSOR, False),
        ('Structured observation', ['Existing detector / tracker'], FIXED, False),
        ('Transformer PPO', ['Policy history encoding'], LEARNED, False),
        ('Safety filter', ['Configured diagnostic arm'], FIXED, False),
        ('Velocity controller', ['Fixed control stack'], FIXED, False),
        ('Rigid-body simulation', ['Physics state update'], FIXED, False)],
        ['High-level description of existing modules; filter activation is lineage-specific.',
         'The real-image temporal selector is a separate pipeline, not this simulator detector.'])
    diagram('perception-tracking-block-diagram', 'Real-image perception: separate evidence lineage',
        'Detector and optical flow run in parallel; selection does not imply metric state or policy integration.', [
        (110,170,'RGB frame',['Sequence · timestamp'],SENSOR,False),
        (1060,170,'Detector',['Top-K candidates'],LEARNED,False),
        (110,375,'Optical flow',['Inter-frame motion'],FIXED,False),
        (1060,375,'Feature history',['Candidates + motion'],FIXED,False),
        (1060,580,'Temporal association',['Candidate rank / no selection'],LEARNED,False),
        (110,580,'Policy observation link',['Not integrated here'],FIXED,True)], [
        ([(535,240),(1060,240)],False), ([(322,310),(322,375)],False),
        ([(1272,310),(1272,375)],False), ([(535,445),(1060,445)],False),
        ([(1272,515),(1272,580)],False), ([(1060,650),(535,650)],True)],
        ['Separate measured-error branch: real data → error model → simulation injection (P8–P10).',
         'No GT identifiers at inference. Dashed link: unimplemented integration, not an established state estimator.'])
    chain('observation-transformer-block-diagram', 'Observation and temporal policy: historical contract', [
        ('Sensor-derived features',['Scan · object · ego histories'],SENSOR,False),
        ('Structured history',['Canonical field contract'],FIXED,False),
        ('Token encoders',['Separate feature families'],LEARNED,False),
        ('Temporal Transformer',['Policy context representation'],LEARNED,False),
        ('Actor output',['Simulation action interface'],LEARNED,False),
        ('Recorded rollout',['Evaluation evidence'],SENSOR,False)],
        ['Policy Transformer and real-image association Transformer are different models.',
         'Privileged GT is excluded from actor input; this diagram does not specify a new policy.'])
    diagram('safety-filter-block-diagram', 'Safety filter: direction-preserving constraint',
        'Existing command and LiDAR context feed configured geometry and speed-cap calculations.', [
        (110,170,'Nominal action',['Policy velocity command'],LEARNED,False),
        (1060,170,'LiDAR geometry',['Clearance · ego state'],SENSOR,False),
        (110,375,'Risk / stopping estimate',['Configured comparison axis'],FIXED,False),
        (1060,375,'Arc-clearance inset',['Alternative geometry arm'],FIXED,False),
        (110,580,'Magnitude constraint',['Direction preserved'],FIXED,False),
        (1060,580,'Filtered command',['To fixed controller'],SENSOR,False)], [
        ([(322,310),(322,375)],False), ([(1060,240),(760,240),(760,415),(535,415)],False),
        ([(1272,310),(1272,375)],False), ([(1060,445),(535,445)],True),
        ([(322,515),(322,580)],False), ([(535,650),(1060,650)],False)],
        ['Arc clearance is a separately evaluated geometry arm, not an additional sequential filter.',
         'Filtered does not mean certified safe; this figure specifies no new planner or controller.'])
    appearance_pipeline()
    chain('evidence-reproducibility-block-diagram', 'Evidence before claims', [
        ('Preregister',['Question · criteria · scope'],FIXED,False),
        ('Run',['Pinned source · environment'],FIXED,False),
        ('Receipt',['Seeds · parameters · hashes'],SENSOR,False),
        ('Independent checks',['Recompute · regression'],FIXED,False),
        ('Bounded claim',['Result with limitations'],SENSOR,False),
        ('Public record',['Negative / withdrawn retained'],SENSOR,False)],
        ['A passing software test is not a positive scientific result.',
         'Public overview links the evidence; it does not replace original receipts.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    build()
    if args.export:
        exporter.OUT = OUT
        exporter.rasterize([dict(svg=p.name, png=p.stem+'.png', pdf=p.stem+'.pdf')
                            for p in sorted(OUT.glob('*.svg'))])
    files = sorted(p for p in OUT.iterdir() if p.suffix in ('.svg','.png','.pdf'))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(OUT/'research-overview-figures.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files + [OUT/'manifest.json']:
            archive.write(p, p.name)
    print(json.dumps({'figures':len(list(OUT.glob('*.svg'))), 'artifacts':len(files)}))


if __name__ == '__main__':
    main()
