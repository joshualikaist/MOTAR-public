'use strict';

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const {spawn} = require('child_process');

const repo = path.resolve(__dirname, '..');
const chrome = process.env.CHROME_BIN || '/usr/bin/google-chrome';
assert(fs.existsSync(chrome), `headless Chrome missing: ${chrome}`);

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

const mime = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml',
  '.md': 'text/plain; charset=utf-8',
};
const server = http.createServer(function (request, response) {
  const relative = decodeURIComponent(request.url.split('?')[0]).replace(/^\/+/, '');
  const file = path.resolve(repo, relative || 'docs/status/index.html');
  if (!(file === repo || file.startsWith(repo + path.sep)) || !fs.existsSync(file)) {
    response.writeHead(404); response.end('not found'); return;
  }
  response.writeHead(200, {'content-type': mime[path.extname(file)] || 'application/octet-stream'});
  fs.createReadStream(file).pipe(response);
});

server.listen(0, '127.0.0.1', function () {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'motar-webgl-'));
  const url = `http://127.0.0.1:${server.address().port}/docs/status/index.html`;
  const child = spawn(chrome, [
    '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
    '--use-gl=angle', '--use-angle=swiftshader',
    '--enable-webgl', '--ignore-gpu-blocklist', '--virtual-time-budget=3500', '--dump-dom',
    `--user-data-dir=${temporary}`, url,
  ], {stdio: ['ignore', 'pipe', 'pipe']});
  let output = '', errors = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { errors += chunk; });
  const timeout = setTimeout(function () { child.kill('SIGKILL'); }, 20000);
  child.on('close', function (code) {
    clearTimeout(timeout); server.close(); removeTree(temporary);
    assert.strictEqual(code, 0, errors.slice(-2000));
    assert(output.includes('<canvas'), 'Three.js did not create a WebGL canvas\n'
      + errors.slice(-2000) + '\nDOM:\n' + output.slice(-2000));
    assert(!output.includes('3D viewer could not start:'), 'viewer entered its fail screen');
    assert(output.includes('ROUTE OK') || output.includes('NO ROUTE · ZERO COMMAND'),
      'routed-preview HUD never reached a terminal planning state');
    console.log('MOTAR headless WebGL routed-preview: PASS');
  });
});
