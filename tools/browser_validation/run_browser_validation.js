'use strict';
/*
 * Real-Chrome validation of the MOTAR GT tracking preview at desktop, tablet
 * and mobile viewports.
 *
 * Serves the repository over loopback, injects tools/browser_validation/probe.js
 * into docs/status/index.html ONLY for the ?probe=1 request (the published file
 * is never modified), runs each viewport for the full duration, and collects the
 * probe's JSON report.
 *
 * Engineering validation of the browser preview. Not PPO, not PhysX, not
 * research performance evidence.
 *
 * Usage: node tools/browser_validation/run_browser_validation.js [--out FILE]
 *                                                               [--seconds N]
 */

const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const {spawn} = require('child_process');

const repo = path.resolve(__dirname, '..', '..');
const chrome = process.env.CHROME_BIN || '/usr/bin/google-chrome';

const argv = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] != null ? argv[i + 1] : fallback;
};
const SECONDS = Number(arg('--seconds', 65));
const OUT = arg('--out', null);

/* Headless Chrome refuses to open a window narrower than ~500 CSS px, so
 * --window-size=390 silently yields a 500 px viewport and the narrow breakpoint
 * never runs. Anything below that minimum is therefore hosted inside an iframe
 * sized to EXACTLY the requested CSS pixels: an iframe is its own viewport, so
 * the page's media queries evaluate at the real width. The probe runs inside
 * the frame and reports window.innerWidth, which the verdict checks. */
const MIN_WINDOW_CSS_PX = 500;
const VIEWPORTS = [
  {label: 'desktop', w: 1440, h: 900},
  {label: 'tablet', w: 1024, h: 768},
  {label: 'mobile', w: 390, h: 844},
];

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.md': 'text/plain; charset=utf-8',
};

const received = [];
const requestLog = [];

/* An interrupted run must not leave a headless Chrome behind: an orphan keeps
 * rendering and quietly steals the CPU that the NEXT run measures its frame
 * times with. Track every child and reap them on any exit path. */
const children = new Set();
let reaping = false;
function reapChildren() {
  if (reaping) return;
  reaping = true;
  for (const child of children) {
    try { child.kill('SIGKILL'); } catch (e) { /* already gone */ }
  }
  children.clear();
}
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(signal, function () { reapChildren(); process.exit(130); });
}
process.on('exit', reapChildren);
process.on('uncaughtException', function (error) {
  reapChildren();
  console.error(error);
  process.exit(1);
});

const server = http.createServer(function (request, response) {
  if (request.method === 'POST' && request.url === '/__probe') {
    let body = '';
    request.on('data', (c) => { body += c; });
    request.on('end', function () {
      try { received.push(JSON.parse(body)); }
      catch (e) { received.push({status: 'BAD_JSON', error: String(e), raw: body.slice(0, 400)}); }
      response.writeHead(204); response.end();
    });
    return;
  }
  const url = request.url.split('?');
  const query = new URLSearchParams(url[1] || '');
  if (url[0] === '/__frame') {
    const w = Number(query.get('w')), h = Number(query.get('h'));
    const inner = '/docs/status/index.html?probe=1&seconds='
      + encodeURIComponent(query.get('seconds') || '65')
      + '&label=' + encodeURIComponent(query.get('label') || 'framed');
    response.writeHead(200, {'content-type': MIME['.html']});
    response.end('<!doctype html><meta charset="utf-8">'
      + '<title>viewport host</title>'
      + '<style>html,body{margin:0;padding:0;background:#222}'
      + `iframe{display:block;width:${w}px;height:${h}px;border:0}</style>`
      + `<iframe src="${inner}" title="viewport host"></iframe>`);
    return;
  }
  const relative = decodeURIComponent(url[0]).replace(/^\/+/, '');
  const file = path.resolve(repo, relative || 'docs/status/index.html');
  if (!(file === repo || file.startsWith(repo + path.sep)) || !fs.existsSync(file)
      || fs.statSync(file).isDirectory()) {
    requestLog.push({url: request.url, status: 404});
    response.writeHead(404); response.end('not found'); return;
  }
  requestLog.push({url: url[0], status: 200});
  const type = MIME[path.extname(file)] || 'application/octet-stream';
  if (query.get('probe') === '1' && path.extname(file) === '.html') {
    let html = fs.readFileSync(file, 'utf8');
    const tag = '<script src="/tools/browser_validation/probe.js"></script>';
    html = html.includes('</body>') ? html.replace('</body>', tag + '\n</body>') : html + tag;
    response.writeHead(200, {'content-type': type});
    response.end(html);
    return;
  }
  response.writeHead(200, {'content-type': type});
  fs.createReadStream(file).pipe(response);
});

function runViewport(port, viewport) {
  return new Promise(function (resolve) {
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'motar-bv-'));
    const framed = viewport.w < MIN_WINDOW_CSS_PX;
    viewport.hosted_in_iframe = framed;
    const url = framed
      ? `http://127.0.0.1:${port}/__frame?w=${viewport.w}&h=${viewport.h}`
        + `&seconds=${SECONDS}&label=${viewport.label}`
      : `http://127.0.0.1:${port}/docs/status/index.html`
        + `?probe=1&seconds=${SECONDS}&label=${viewport.label}`;
    const args = [
      '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
      '--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl',
      '--ignore-gpu-blocklist', '--enable-precise-memory-info',
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      `--window-size=${framed ? Math.max(viewport.w + 80, MIN_WINDOW_CSS_PX + 80) : viewport.w},`
        + `${framed ? viewport.h + 80 : viewport.h}`,
      `--force-device-scale-factor=1`,
      `--user-data-dir=${profile}`,
      url,
    ];
    const started = Date.now();
    const child = spawn(chrome, args, {stdio: ['ignore', 'pipe', 'pipe']});
    children.add(child);
    let stderr = '';
    child.stderr.on('data', (c) => { stderr += c; });
    const before = received.length;
    const deadline = setTimeout(function () {
      child.kill('SIGKILL');
    }, (SECONDS + 35) * 1000);
    const poll = setInterval(function () {
      if (received.length > before) {
        clearInterval(poll); clearTimeout(deadline);
        setTimeout(function () { child.kill('SIGKILL'); }, 500);
      }
    }, 500);
    child.on('close', function () {
      children.delete(child);
      clearInterval(poll); clearTimeout(deadline);
      try { removeTree(profile); } catch (e) { /* best effort */ }
      const got = received.slice(before);
      resolve({
        viewport: viewport,
        wall_s: (Date.now() - started) / 1000,
        report: got.length ? got[got.length - 1] : null,
        chrome_stderr_tail: got.length ? null : stderr.slice(-1500),
      });
    });
  });
}

function removeTree(directory) {
  if (!fs.existsSync(directory)) return;
  for (const entry of fs.readdirSync(directory)) {
    const target = path.join(directory, entry);
    const stat = fs.lstatSync(target);
    if (stat.isDirectory() && !stat.isSymbolicLink()) removeTree(target);
    else fs.unlinkSync(target);
  }
  fs.rmdirSync(directory);
}

function judge(entry) {
  const r = entry.report;
  const problems = [];
  if (!r) { problems.push('no probe report'); return {pass: false, problems: problems}; }
  if (r.status !== 'OK') problems.push('probe status ' + r.status);
  if (!r.boot) problems.push('arena did not boot');
  if (r.errors && r.errors.length) problems.push('page errors: ' + r.errors.join(' | '));
  if (r.teleports) problems.push('teleports=' + r.teleports);
  // The scripted timeline changes bar count and target speed, each of which
  // rebuilds the scene on purpose. Those must be SEEN, so the run proves the
  // controls work, and must not be counted as motion discontinuities.
  if (!(r.scene_rebuilds >= 4)) {
    problems.push('scripted control changes did not rebuild the scene: ' + r.scene_rebuilds);
  }
  if (r.horizontal_overflow) problems.push('horizontal page overflow');
  // Prove the requested breakpoint actually ran. Without this the mobile row
  // reports PASS at whatever width the browser happened to grant.
  if (Math.abs(r.viewport.w - entry.viewport.w) > 2) {
    problems.push('viewport width not honoured: requested ' + entry.viewport.w
      + ', measured ' + r.viewport.w);
  }
  if (!r.gt_badge || !/NOT PPO/.test(r.gt_badge)) problems.push('GT evidence badge missing');
  if (!r.samples || r.samples < SECONDS * 5) problems.push('sampler starved: ' + r.samples);
  const timelineErrors = (r.timeline || []).filter(function (s) { return s.error; });
  if (timelineErrors.length) {
    problems.push('interaction errors: '
      + timelineErrors.map(function (s) { return s.step + ': ' + s.error; }).join(' | '));
  }
  const expected = ['cycle camera view', 'pause', 'resume', 'tap ground goal', 'auto roam'];
  for (const step of expected) {
    if (!(r.timeline || []).some(function (s) { return s.step.indexOf(step) === 0; })) {
      problems.push('interaction never ran: ' + step);
    }
  }
  if (r.distance === 'NOT_MEASURED') problems.push('tracking distance NOT_MEASURED');
  // The scene must actually advance: a paused or off-screen arena would show a
  // flat distance trace and a clean frame time while proving nothing.
  const expectedSteps = Math.floor((r.elapsed_s || 0) * 10 * 0.6);
  if (!(r.sim_time_advanced > expectedSteps)) {
    problems.push('simulation did not advance: ' + r.sim_time_advanced
      + ' steps over ' + (r.elapsed_s || 0).toFixed(1) + 's');
  }
  if (r.distance && r.distance.max != null && r.distance.max - r.distance.min < 0.5) {
    problems.push('tracking distance never changed (frozen scene?)');
  }
  return {pass: problems.length === 0, problems: problems};
}

server.listen(0, '127.0.0.1', async function () {
  const port = server.address().port;
  if (!fs.existsSync(chrome)) {
    console.error('headless Chrome missing: ' + chrome);
    process.exitCode = 1; server.close(); return;
  }
  const results = [];
  for (const viewport of VIEWPORTS) {
    process.stdout.write(`running ${viewport.label} ${viewport.w}x${viewport.h} ...\n`);
    const entry = await runViewport(port, viewport);
    entry.verdict = judge(entry);
    results.push(entry);
    const r = entry.report;
    if (r) {
      const ft = r.frame_time_ms;
      process.stdout.write(
        `  ${viewport.label}: ${entry.verdict.pass ? 'PASS' : 'FAIL'} `
        + `${r.viewport.w}x${r.viewport.h}${viewport.hosted_in_iframe ? ' (iframe)' : ''} `
        + `elapsed=${(r.elapsed_s || 0).toFixed(1)}s fps=${(r.fps_mean || 0).toFixed(1)} `
        + `frame_ms median=${ft && ft.median != null ? ft.median.toFixed(1) : 'NA'} `
        + `p95=${ft && ft.p95 != null ? ft.p95.toFixed(1) : 'NA'} `
        + `d: ${r.distance && r.distance.first != null ? r.distance.first.toFixed(1) : 'NA'}`
        + `->${r.distance && r.distance.last != null ? r.distance.last.toFixed(1) : 'NA'} `
        + `med=${r.distance && r.distance.median != null ? r.distance.median.toFixed(2) : 'NA'} `
        + `teleports=${r.teleports} rebuilds=${r.scene_rebuilds} `
        + `heapΔ=${r.heap_delta_bytes != null ? (r.heap_delta_bytes / 1048576).toFixed(1) + 'MB' : 'NOT_MEASURED'}\n`
      );
      if (!entry.verdict.pass) {
        for (const problem of entry.verdict.problems) process.stdout.write('    - ' + problem + '\n');
      }
    } else {
      process.stdout.write(`  ${viewport.label}: FAIL (no report)\n`);
      process.stdout.write('    chrome stderr: ' + (entry.chrome_stderr_tail || '') + '\n');
    }
  }
  server.close();
  const allPass = results.every(function (r) { return r.verdict.pass; });
  const out = {
    kind: 'browser_gt_tracking_browser_validation',
    boundary: ['BROWSER GT TRACKING PREVIEW', 'SIMULATION ONLY', 'NOT PPO', 'NOT PHYSX',
      'NOT RESEARCH PERFORMANCE EVIDENCE'],
    generated_utc: new Date().toISOString(),
    chrome: chrome,
    seconds_per_viewport: SECONDS,
    results: results,
    verdict: allPass ? 'PASS' : 'FAIL',
  };
  if (OUT) { fs.writeFileSync(OUT, JSON.stringify(out, null, 1) + '\n'); process.stdout.write(`wrote ${OUT}\n`); }
  process.stdout.write(allPass ? '\nBROWSER VALIDATION: PASS\n' : '\nBROWSER VALIDATION: FAIL\n');
  process.exitCode = allPass ? 0 : 1;
});
