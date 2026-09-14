"""Documentation-only export checks; no simulator, datasets or GPU required."""
import hashlib
import json
import re
import shutil
import struct
import subprocess
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/presentation'
NS = {'s': 'http://www.w3.org/2000/svg'}


class PresentationFiguresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((OUT / 'manifest.json').read_text())

    def test_nine_self_contained_widescreen_svgs(self):
        self.assertEqual(len(self.manifest['slides']), 9)
        for entry in self.manifest['slides']:
            svg = ET.parse(OUT / entry['svg']).getroot()
            self.assertEqual(svg.attrib['viewBox'], '0 0 1600 900')
            self.assertIsNotNone(svg.find('s:title', NS))
            self.assertIsNotNone(svg.find('s:desc', NS))
            ids = [node.attrib['id'] for node in svg.iter() if 'id' in node.attrib]
            self.assertEqual(len(ids), len(set(ids)), entry['svg'])
            self.assertFalse(svg.findall('.//s:image', NS), 'export must not depend on external images')

    def test_png_dimensions(self):
        for entry in self.manifest['slides']:
            data = (OUT / entry['png']).read_bytes()
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(struct.unpack('>II', data[16:24]), (3840, 2160))

    def test_manifest_hashes(self):
        self.assertEqual(len(self.manifest['sha256']), 18)
        for filename, expected in self.manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((OUT / filename).read_bytes()).hexdigest(), expected)

    def test_cited_evidence_hashes(self):
        for filename, expected in self.manifest['evidence_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), expected)

    def test_zip_matches_loose_exports(self):
        with zipfile.ZipFile(OUT / 'motar-presentation-2026-09-10.zip') as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 20)
            for filename in archive.namelist():
                self.assertEqual(archive.read(filename), (OUT / filename).read_bytes())

    def test_gallery_links_exist(self):
        html = (OUT / 'index.html').read_text()
        for link in re.findall(r'(?:href|src)="([^"]+)"', html):
            self.assertTrue((OUT / link).exists(), link)

    def test_evidence_limits_survive_export(self):
        eth = (OUT / '09-eth-e3-evidence.svg').read_text()
        for term in ['6.2%', 'GT bbox가 아님', '신뢰구간이 아님', 'E3P_GATE_BLOCKED', '원인은 분해하지 않음']:
            self.assertIn(term, eth)
        perception = (OUT / '06-perception-final.svg').read_text()
        for term in ['[−0.38, +3.21]', '0 포함', '학습 시드 1개', '실제 비행은 별도 미검증']:
            self.assertIn(term, perception)

    def test_paper_diagrams_and_export(self):
        paper = ROOT / 'docs/assets/paper'
        manifest = json.loads((paper / 'manifest.json').read_text())
        self.assertEqual(len(manifest), 28)  # 9 × SVG/PNG/PDF + captions
        for filename, expected in manifest.items():
            self.assertEqual(hashlib.sha256((paper / filename).read_bytes()).hexdigest(), expected)
        sources = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', (ROOT / 'docs/results_overview_2026-09-12.md').read_text())
        self.assertEqual(len(sources), 9)
        self.assertTrue(all(source.startswith('assets/paper/') for source in sources))
        for source in sources:
            stem = Path(source).stem
            svg = ET.parse(paper / (stem + '.svg')).getroot()
            self.assertEqual(svg.attrib['viewBox'], '0 0 1600 900')
            self.assertGreaterEqual(len(svg.findall('.//s:path[@marker-end]', NS)), 4)
            self.assertGreaterEqual(len(svg.findall('.//s:g[@data-block]', NS)), 5)
            ids = [node.attrib['id'] for node in svg.iter() if 'id' in node.attrib]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(svg.findall('.//s:image', NS))
            data = (paper / (stem + '.png')).read_bytes()
            self.assertEqual(struct.unpack('>II', data[16:24]), (3840, 2160))
            self.assertTrue((paper / (stem + '.pdf')).read_bytes().startswith(b'%PDF-'))
            for page in [ROOT / 'docs/results_overview_2026-09-12.md', ROOT / 'docs/status/archive-2026-09-13.html']:
                self.assertIn('paper/' + stem + '.svg', page.read_text())
        with zipfile.ZipFile(paper / 'motar-paper-block-diagrams.zip') as archive:
            self.assertEqual(len(archive.namelist()), 29)
            self.assertIsNone(archive.testzip())
            for filename in archive.namelist():
                self.assertEqual(archive.read(filename), (paper / filename).read_bytes())

    def test_paper_gallery_matches_all_readme_images(self):
        paper = ROOT / 'docs/assets/paper'
        gallery = (paper / 'index.html').read_text()
        sources = re.findall(r'<img src="([^"]+)"', gallery)
        readme_sources = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', (ROOT / 'docs/results_overview_2026-09-12.md').read_text())
        self.assertEqual(sources, [Path(p).name for p in readme_sources])
        for link in re.findall(r'(?:href|src)="([^"]+)"', gallery):
            self.assertTrue((paper / link).exists(), link)

    def test_paper_scope_and_archived_paths(self):
        paper = ROOT / 'docs/assets/paper'
        sam = (paper / 'sam-archive-block-diagram.svg').read_text()
        self.assertIn('ARCHIVED', sam)
        self.assertIn('stroke-dasharray=', sam)
        self.assertIn('Offline CPU boundary', sam)
        safety = (paper / 'safety-filter-block-diagram.svg').read_text()
        self.assertIn('geometry arms are alternatives', safety)
        self.assertIn('not a DWA planner', safety)
        e3 = (paper / 'e3-analysis-block-diagram.svg').read_text()
        self.assertIn('BLOCKED', e3)
        self.assertIn('No attitude-error decomposition', e3)

    @unittest.skipUnless(shutil.which('pdfinfo') and shutil.which('pdftotext') and shutil.which('pdfimages'),
                         'Poppler tools required for vector-PDF validation')
    def test_paper_pdfs_are_single_page_selectable_vectors(self):
        paper = ROOT / 'docs/assets/paper'
        pdfs = sorted(paper.glob('*.pdf'))
        self.assertEqual(len(pdfs), 9)
        for pdf in pdfs:
            info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
            self.assertRegex(info, r'Pages:\s+1\b')
            text = subprocess.check_output(['pdftotext', str(pdf), '-'], text=True)
            self.assertGreater(len(text.strip()), 100)
            images = subprocess.check_output(['pdfimages', '-list', str(pdf)], text=True)
            self.assertFalse(re.search(r'^\s*\d+\s+\d+\s+', images, re.M), 'PDF should not be a raster screenshot')


if __name__ == '__main__':
    unittest.main()
