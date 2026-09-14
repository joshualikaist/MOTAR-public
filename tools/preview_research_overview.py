#!/usr/bin/env python3
"""Read-only local browser validation; save dated screenshots and a check receipt."""
import argparse
import base64
import hashlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    import websocket
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(('127.0.0.1',0), partial(QuietHandler,directory=str(ROOT)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    checks = []
    try:
        with tempfile.TemporaryDirectory(prefix='motar-overview-chrome-', ignore_cleanup_errors=True) as tmp:
            chrome = subprocess.Popen([shutil.which('google-chrome') or 'chromium',
                '--headless=new','--no-sandbox','--disable-dev-shm-usage',
                '--use-gl=angle','--use-angle=swiftshader','--enable-webgl',
                '--ignore-gpu-blocklist','--remote-debugging-port=0',
                '--user-data-dir='+tmp, 'about:blank'],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            sock = None
            try:
                port_file = Path(tmp)/'DevToolsActivePort'
                for _ in range(100):
                    if port_file.exists():
                        break
                    if chrome.poll() is not None:
                        raise RuntimeError('Browser exited before readiness')
                    time.sleep(.1)
                port = port_file.read_text().splitlines()[0]
                with urllib.request.urlopen('http://127.0.0.1:'+port+'/json',timeout=10) as response:
                    target = next(x for x in json.load(response) if x['type']=='page')
                sock = websocket.create_connection(target['webSocketDebuggerUrl'],suppress_origin=True,timeout=30)
                seq = 0

                def call(method,params=None):
                    nonlocal seq
                    seq += 1
                    sock.send(json.dumps(dict(id=seq,method=method,params=params or {})))
                    while True:
                        data = json.loads(sock.recv())
                        if data.get('id') == seq:
                            if 'error' in data:
                                raise RuntimeError(str(data['error']))
                            return data['result']

                def js(expression):
                    data = call('Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True))
                    if data.get('exceptionDetails'):
                        raise RuntimeError(str(data))
                    return data['result'].get('value')

                call('Page.enable')
                for label,width,height in [('desktop',1360,1000),('tablet',800,1000),('mobile',390,844)]:
                    call('Emulation.clearDeviceMetricsOverride')
                    call('Emulation.setDeviceMetricsOverride',dict(
                        width=width, height=height, screenWidth=width, screenHeight=height,
                        positionX=0, positionY=0, deviceScaleFactor=1,
                        mobile=label=='mobile', dontSetVisibleSize=False))
                    call('Page.navigate',dict(url='http://127.0.0.1:%s/docs/status/index.html'%server.server_port))
                    for _ in range(100):
                        ready = js("document.readyState==='complete' && ![...document.querySelectorAll('[data-status-id]')].some(x=>x.textContent.includes('Loading')) && !!document.querySelector('#stage canvas')")
                        if ready:
                            break
                        time.sleep(.1)
                    if not ready:
                        raise RuntimeError('Page/manifest/WebGL readiness timeout')
                    js("document.querySelector('#btn-play').click(); document.documentElement.style.scrollBehavior='auto'")
                    image_ok = js("Promise.all([...document.images].map(i=>{i.loading='eager'; return i.decode().then(()=>true,()=>false)})).then(v=>v.every(Boolean))")
                    result = js("({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,statusRows:document.querySelectorAll('[data-status-id]').length,visibleNavLinks:[...document.querySelectorAll('header nav a')].filter(a=>a.getClientRects().length).length,canvas:!!document.querySelector('#stage canvas'),route:document.querySelector('#hud-route-state').textContent})")
                    result.update(viewport=label,images_decoded=image_ok)
                    if not image_ok or result['scrollWidth'] > width or result['statusRows'] != 4 or result['visibleNavLinks'] != 6:
                        raise RuntimeError('Layout/content check failed: '+str(result))
                    # Hide scripted movement before capturing; no underlying viewer source edits.
                    for section in ('top','arena','perception','evidence'):
                        if section == 'top':
                            js("scrollTo({top:0,left:0,behavior:'instant'})")
                        else:
                            js("document.getElementById(%s).scrollIntoView({block:'start',behavior:'instant'})"%json.dumps(section))
                        shot = call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
                        (args.output/(label+'-'+section+'.png')).write_bytes(base64.b64decode(shot['data']))
                    checks.append(result)
                # A failed manifest request must not leave a success-looking state.
                call('Network.enable')
                call('Network.setBlockedURLs', {'urls':['*status_manifest.json']})
                call('Page.reload', {'ignoreCache':True})
                fallback = False
                for _ in range(100):
                    fallback = js("!!document.getElementById('public-status-manifest') && document.getElementById('public-status-manifest').textContent.includes('no PASS inferred')")
                    if fallback:
                        break
                    time.sleep(.1)
                if not fallback:
                    raise RuntimeError('Missing-manifest fallback did not fail closed')
                call('Browser.close')
            finally:
                if sock:
                    try:
                        sock.send(json.dumps({'id':seq+1,'method':'Browser.close'}))
                    except (OSError, websocket.WebSocketException):
                        pass
                    sock.close()
                try:
                    chrome.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    chrome.kill(); chrome.wait(timeout=5)
    finally:
        server.shutdown(); server.server_close()
    source_paths = [ROOT/'docs/status/index.html',ROOT/'docs/status/overview.css',
                    ROOT/'docs/status/style.css',ROOT/'docs/status/status_manifest.js',
                    ROOT/'docs/status_manifest.json',Path(__file__).resolve()]
    source_paths += list((ROOT/'docs/assets/paper/overview-2026-09-13').glob('*.svg'))
    source_paths += [ROOT/'docs/status'/name for name in ('arena.js','arena_motion.js',
                     'arena_route.js','viewer.js','vendor/three.min.js','vendor/OrbitControls.js')]
    receipt = dict(status='PASS',checks=checks,missing_manifest_fails_closed=fallback,
                   source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                   scope='Local presentation checks; no simulation/perception experiment',
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()))
    (args.output/'preview.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
