"""Paper-page contracts; historical detail contracts remain separately tested."""
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import unittest
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'docs/status'
FIG = ROOT / 'docs/assets/paper/overview-2026-09-13'


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links, self.images, self.scripts, self.nav_links = [], [], [], [], []
        self._in_nav = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'nav':
            self._in_nav = True
        if 'id' in a:
            self.ids.append(a['id'])
        for name in ('href', 'src'):
            if name in a:
                self.links.append(a[name])
        if tag == 'a' and self._in_nav:
            self.nav_links.append(a.get('href',''))
        if tag == 'img':
            self.images.append(a)
        if tag == 'script' and 'src' in a:
            self.scripts.append(a['src'])

    def handle_endtag(self, tag):
        if tag == 'nav':
            self._in_nav = False


class ResearchOverviewTest(unittest.TestCase):
    def setUp(self):
        self.text = (SITE / 'index.html').read_text()
        self.css = (SITE / 'overview.css').read_text()
        self.page = Page(self.text)

    def test_paper_structure_and_accessibility(self):
        self.assertEqual(len(self.page.ids), len(set(self.page.ids)))
        for section in ('abstract','introduction','environment','arena','system','method',
                        'perception','safety-filter','appearance','experiments','algorithms',
                        'evidence','discussion','reproducibility','availability'):
            self.assertIn(section, self.page.ids)
        self.assertEqual(len(self.page.nav_links), 6)
        # Nine numbered figures plus Figure T1, a lettered supporting diagram in
        # 2.2 Target motion. T1 is drawn in CSS and carries no image asset, so the
        # hash-pinned dated figure package is unaffected and the image count is
        # 8 (Figure 2 is the interactive canvas, T1 is markup). Figure 9 is the
        # quantitative positioning diagram added with section 6.2; it is generated
        # by tools/build_quantitative_positioning_figure.py from the registry.
        self.assertEqual(len(re.findall(r'<figure\b', self.text)), 10)
        self.assertEqual(len(self.page.images), 8)
        self.assertEqual(len(re.findall(r'<figcaption>', self.text)), 10)
        for number in range(1, 10):
            self.assertIn(f'Figure {number}.', self.text)
        self.assertIn('Figure T1.', self.text)
        for image in self.page.images:
            self.assertGreater(len(image.get('alt','')), 15)
        self.assertIn('본문 바로가기', self.text)
        self.assertIn('<noscript>', self.text)

    def test_abstract_is_concise_and_title_is_not_marketing_copy(self):
        abstract = self.text.split('<section class="paper-section abstract"')[1].split('</section>',1)[0]
        prose = unescape(re.sub(r'<[^>]+>', ' ', abstract))
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", prose)
        self.assertGreaterEqual(len(words), 120)
        self.assertLessEqual(len(words), 180)
        self.assertIn('<h1>MOTAR</h1>', self.text)
        self.assertIn('Moving Object Tracking And Rendezvous', self.text)
        self.assertIn('Reinforcement Learning for UAV Tracking and Close Approach in Random Obstacle Fields', self.text)
        self.assertNotIn('Moving Object Tracking And Reinforcement Learning for UAV', self.text)
        self.assertNotIn('Observe. Measure.', self.text)
        self.assertNotIn('overview-card', self.text)

    def test_all_active_local_links_and_html_fragments(self):
        for link in self.page.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = (SITE / unquote(parsed.path or 'index.html')).resolve()
            self.assertIn(ROOT, target.parents, link)
            self.assertTrue(target.exists(), link)
            if parsed.fragment and target.suffix == '.html':
                self.assertIn(unquote(parsed.fragment), Page(target.read_text()).ids, link)

    def test_pages_landing_links_do_not_escape_the_docs_root(self):
        # GitHub Pages publishes /docs only. ../../ from status/ becomes github.io 404.
        self.assertNotIn('href="../../', self.text)
        self.assertIn('https://github.com/joshualikaist/MOTAR-public"', self.text)
        self.assertIn('MOTAR-public/blob/main/CITATION.cff', self.text)
        self.assertNotIn('https://github.com/joshualikaist/MOTAR"', self.text)
        prefix = 'https://github.com/joshualikaist/MOTAR-public/blob/main/'
        for link in self.page.links:
            if link.startswith(prefix):
                self.assertTrue((ROOT / link[len(prefix):]).exists(), link)

    def test_viewer_dom_and_script_order_preserved(self):
        archive = Page((SITE / 'archive-2026-09-13.html').read_text())
        for name in archive.ids:
            if name == 'stage' or name.startswith(('hud-','sl-','lbl-','btn-','cb-','sel-')):
                self.assertIn(name, self.page.ids)
        self.assertIn('id="motion-mode-note"', self.text)
        def strip_ver(src):
            return src.split('?')[0]
        current = [s for s in self.page.scripts if 'status_manifest' not in s]
        archive_scripts = [strip_ver(s) for s in archive.scripts if 'status_manifest' not in s]
        self.assertTrue(any('arena_demo_planner.js' in s for s in current))
        shared = [strip_ver(s) for s in current if 'arena_demo_planner' not in s]
        self.assertEqual(shared, archive_scripts)
        self.assertIn('value="gt-free-roam" selected', self.text)
        self.assertIn('value="gt-route-track" selected', self.text)
        self.assertIn('value="routed-preview"', self.text)
        self.assertIn('BROWSER GT PREVIEW', self.text)
        self.assertIn('NOT PPO · NOT PHYSX · NOT RESEARCH EVIDENCE', self.text)
        self.assertIn('NOT PhysX/PPO', self.text)
        self.assertIn('browser-only visualization', self.text.lower())

    def test_current_claim_boundaries(self):
        for phrase in ('simulation-only', 'interception', 'no-selection', 'persistent physical identity',
                       'size-proxy metric', 'net adaptation benefit is not established',
                       'shortcut reduction has not been measured', 'withdrawn C3',
                       'D8', 'has not started', 'no real-flight validation'):
            self.assertIn(phrase.lower(), self.text.lower())
        data = json.loads((ROOT/'docs/status_manifest.json').read_text())
        for key, status in [('D6','INCONCLUSIVE'),('D7','GO'),('D8','TECHNICAL_GO'),('D9','NOT_RUN')]:
            self.assertEqual(data['track_d'][key]['status'], status)
            self.assertIn(f'data-status-id="{key}"', self.text)
        self.assertNotIn('색만으로 표적을 찾을 수 있었습니다', self.text)

    def test_component_lifecycle_is_separate_from_evidence_verdict(self):
        registry = json.loads((ROOT / 'docs/research_status_registry.json').read_text())
        expected = {
            'P10': ('COMPLETED', 'INCONCLUSIVE'),
            'D8B': ('COMPLETED', 'MATERIAL_LOSS'),
            'D8C': ('PLANNED', 'NOT_STARTED'),
            'D9': ('PLANNED', 'NOT_RUN'),
            'SAM_IN_SIM': ('ARCHIVED_WITHDRAWN', 'SUPERSEDED'),
            'LIVE_RGB_POLICY': ('NOT_TESTED', 'NOT_INTEGRATED'),
            'BEARING_DEG': ('BLOCKED', 'BLOCKED_BY_INTRINSICS'),
        }
        for key, pair in expected.items():
            row = registry['components'][key]
            self.assertEqual((row['lifecycle_status'], row['evidence_status']), pair)
            self.assertTrue((ROOT / row['evidence']).is_file(), key)
            self.assertIn(f'data-component-id="{key}"', self.text)
        for status in ('COMPLETED', 'PLANNED', 'BLOCKED', 'NOT_TESTED',
                       'ARCHIVED_WITHDRAWN'):
            self.assertIn(status, registry['status_semantics'])
        self.assertIn('lifecycle is separate from evidence verdict',
                      (SITE / 'status_manifest.js').read_text())

    def test_matched_baseline_table_preserves_registered_numbers_and_limits(self):
        for phrase in (
            '−1.4903 pp', '[−1.8981, −1.0826]', '15/15 cells',
            '−5.60 pp; capture +4.44 pp',
            '+3.75 pp, 95% CI [+1.30, +6.19]',
            '37.82% → 78.04% (+40.21 pp)',
            '−0.015 pp; 95% CI [−1.752, +1.723]',
            '0.6655', '0.6725', '+0.00697 utility', '0.4701',
            'Mean +0.73 pp; 95% CI [−1.04, +2.50]',
            '−48.967 pp', '[−50.113, −47.821]',
            'They do not establish superiority over published external systems',
            'pp</abbr> means percentage points',
        ):
            self.assertIn(phrase, self.text)
        for boundary in (
            'No formal collision-safety guarantee',
            'crash not confirmed',
            'Does not establish that real-world latency is solved',
            'this is not superiority',
            'Not held-out superiority',
            'Net adaptation benefit is not established',
            'causality of any renderer defect is NOT_TESTED',
        ):
            self.assertIn(boundary.lower(), self.text.lower())

    def test_external_table_has_no_direct_numeric_ranking(self):
        section = self.text.split('<h3>6.3 Relation to published systems</h3>', 1)[1]
        section = section.split('<h3>6.4 Current evidence boundaries</h3>', 1)[0]
        for work in ('NavRL', 'NavRL++', 'YOPO', 'YOPOv2-Tracker', 'OPEN',
                     'AgilePE', 'Fast-Tracker', 'Elastic Tracker', 'MAD',
                     'FlowPilot', 'PILOT', 'Temporal Barrier'):
            self.assertIn(work, section)
        self.assertIn('No selected work is currently Class A', section)
        self.assertNotRegex(section, r'(?:outperform|better than|superior by)\s+\d')
        self.assertIn('relation_to_published_systems_2026-09-16.md', section)

    def test_target_behavior_ladder_is_explicit_and_fail_closed(self):
        for key in ('TM_E0', 'TM_E1', 'TM_E2', 'TM_E3', 'TM_E4'):
            self.assertIn(f'data-component-id="{key}"', self.text)
        self.assertIn('Evader may use privileged GT obstacle information', self.text)
        self.assertIn('pursuer may not', self.text)
        self.assertIn('No E0/E1/E2 policy-performance grid is reported', self.text)

    def test_d8b_result_is_not_promoted_to_perception_improvement(self):
        result = json.loads((ROOT/'results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/summary.json').read_text())
        self.assertEqual(result['verdict'], 'MATERIAL_LOSS')
        for phrase in ('MATERIAL_LOSS', 'D8c has not started', 'Historical pre-D8 diagram',
                       'not a detector-improvement verdict'):
            self.assertIn(phrase, self.text)
        self.assertIn('dynamic_mesh_policy_sensitivity_d8b_2026-09-13/README.md', self.text)

    def test_parameters_distinguish_defaults_and_lineages(self):
        for value in ('4 × 36 / 4 m','4 × 72 / 12 m','2.0 m/s','2.5 m/s',
                      '256 default / 128 D7','17 tokens','be89675','receipt'):
            self.assertIn(value, self.text)
        self.assertIn('source default XY is 24 m', self.text)
        self.assertIn('distinct from association Transformer', self.text)

    def test_paper_visual_contract(self):
        for contract in ('max(20px, calc((100vw - 960px) / 2))',
                         'Georgia, "Times New Roman", serif',
                         'width: min(calc(100% - 40px), 960px)',
                         'width: min(1120px, calc(100vw - 30px))',
                         'border-collapse: collapse'):
            self.assertIn(contract, self.css)
        for forbidden in ('linear-gradient', 'box-shadow: 0 ', '.overview-card'):
            self.assertNotIn(forbidden, self.css)
        self.assertIn('@media (max-width: 760px)', self.css)

    def test_new_figure_formats_hashes_and_zip(self):
        manifest = json.loads((FIG/'manifest.json').read_text())
        self.assertEqual(len(manifest), 21)
        self.assertEqual(len(list(FIG.glob('*.svg'))), 7)
        for name, digest in manifest.items():
            data = (FIG/name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)
            if name.endswith('.png'):
                self.assertEqual(struct.unpack('>II',data[16:24]), (3840,2160))
            if name.endswith('.pdf'):
                self.assertTrue(data.startswith(b'%PDF-'))
        with zipfile.ZipFile(FIG/'research-overview-figures.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()), 22)
            for name in z.namelist():
                self.assertEqual(z.read(name),(FIG/name).read_bytes())

    def test_svg_structure_and_unimplemented_boundary(self):
        ns = {'s':'http://www.w3.org/2000/svg'}
        for path in FIG.glob('*.svg'):
            svg = ET.parse(path).getroot()
            self.assertEqual(svg.get('role'),'img')
            self.assertIsNotNone(svg.find('s:title',ns))
            self.assertIsNotNone(svg.find('s:desc',ns))
            self.assertGreaterEqual(len(svg.findall('.//s:g[@data-block]',ns)),6)
            self.assertFalse(svg.findall('.//s:image',ns))
            ids = [n.get('id') for n in svg.iter() if n.get('id')]
            self.assertEqual(len(ids),len(set(ids)))
        for name in ('research-overview','perception-tracking','appearance-rendering'):
            self.assertIn('stroke-dasharray', (FIG/(name+'-block-diagram.svg')).read_text())
        for name in ('research-overview', 'perception-tracking', 'appearance-rendering'):
            text = (FIG/(name+'-block-diagram.svg')).read_text()
            self.assertIn('data-status="COMPLETED"', text)
        self.assertIn(
            'data-status="NOT_TESTED"',
            (FIG/'perception-tracking-block-diagram.svg').read_text(),
        )
        self.assertIn(
            'data-status="PLANNED"',
            (FIG/'appearance-rendering-block-diagram.svg').read_text(),
        )

    def test_old_hash_pinned_packages_and_complete_detail_preserved(self):
        for name, digest in {
            'docs/assets/paper/manifest.json':'6941d5497f6d3c20879ad85d215375d5b21c5bca09fa23b3993efb729ca20b5d',
            'docs/assets/paper/motar-paper-block-diagrams.zip':'c2698cd6278026696585c394b741fd7f2be8fd51db6ab3661bcf8d80d8eb157c',
            'docs/status/archive-2026-09-13.html':'a96982824b82293172b73ac226ade7396b8773c62366c21e32a5f4694accf2bb'
        }.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
        self.assertIn('archive-2026-09-13.html',self.text)


if __name__ == '__main__':
    unittest.main()
