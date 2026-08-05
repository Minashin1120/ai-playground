#!/usr/bin/env python3
"""Drive headless Chrome via CDP to measure the live landing page layout and
capture console/exceptions. Uses venv python (has websockets + requests)."""
import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.request

CHROME = "/home/ai-chat-minashin1120/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"
URL = "https://ai.minashin1120.com/"
DEBUG_PORT = 9333

MEASURE_JS = r"""
(function () {
    var out = {};
    out.viewport = { w: document.documentElement.clientWidth, h: window.innerHeight };
    out.doc = { scrollW: document.documentElement.scrollWidth, scrollH: document.documentElement.scrollHeight,
                bodyScrollW: document.body.scrollWidth, hasHOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1 };
    function rect(sel) {
        var el = document.querySelector(sel);
        if (!el) return null;
        var r = el.getBoundingClientRect();
        return { top: Math.round(r.top), bottom: Math.round(r.bottom), h: Math.round(r.height), w: Math.round(r.width) };
    }
    out.sections = {};
    var secs = document.querySelectorAll('section, header, footer, main');
    var i = 0;
    out.sections.hero = rect('.hero-bg');
    out.sections.demoStage = rect('.demo-stage');
    out.sections.chatDemo = rect('.chat-demo');
    out.sections.chatDemoBody = rect('.chat-demo-body');
    out.sections.hubFrame = rect('.hub-frame');
    out.sections.featuresGrid = rect('section:nth-of-type(4) .grid');
    out.sections.marquee = rect('.marquee-wrap');
    out.sections.faq = rect('.ld-faq');
    out.sections.cta = rect('section.hero-bg.py-20');
    out.header = rect('header');
    out.footer = rect('footer');
    /* negative margins anywhere that could cause awkward gaps */
    var negs = [];
    document.querySelectorAll('*').forEach(function (el) {
        var cs = window.getComputedStyle(el);
        if (cs && (parseFloat(cs.marginTop) < -1 || parseFloat(cs.marginBottom) < -1 || parseFloat(cs.marginLeft) < -1 || parseFloat(cs.marginRight) < -1)) {
            if (el.className && typeof el.className === 'string') negs.push(el.tagName + '.' + el.className.split(' ').slice(0,2).join('.') + ' mt=' + cs.marginTop + ' mb=' + cs.marginBottom);
        }
    });
    out.negativeMargins = negs.slice(0, 20);
    out.demoState = {
        built: !!document.querySelector('.chat-demo-body'),
        rows: document.querySelectorAll('.chat-demo-body .ld-row').length,
        hubCards: document.querySelectorAll('.hub-node-card').length
    };
    return JSON.stringify(out);
})();
"""

async def main():
    proc = subprocess.Popen([
        CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--remote-debugging-port=%d" % DEBUG_PORT,
        "--user-data-dir=/tmp/opencode_cdp_profile", "about:blank"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # wait for debugging endpoint
        for _ in range(40):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/json/version" % DEBUG_PORT, timeout=2)
                break
            except Exception:
                time.sleep(0.25)

        # open the page
        req = urllib.request.Request(
            "http://127.0.0.1:%d/json/new?%s" % (DEBUG_PORT, urllib.request.quote(URL, safe="")),
            method="PUT")
        target = json.loads(urllib.request.urlopen(req, timeout=5).read())
        ws_url = target["webSocketDebuggerUrl"]

        import websockets
        async with websockets.connect(ws_url, max_size=2**24) as ws:
            cmd_id = 0
            async def cmd(method, params=None):
                nonlocal cmd_id
                cmd_id += 1
                await ws.send(json.dumps({"id": cmd_id, "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == cmd_id:
                        return msg.get("result", {}), msg.get("error")
            async def listen(timeout):
                try:
                    return json.loads(await asyncio.wait_for(ws.recv(), timeout))
                except asyncio.TimeoutError:
                    return None

            await cmd("Page.enable")
            await cmd("Runtime.enable")
            await cmd("Log.enable")
            # optional viewport (desktop check)
            import os
            vp = os.environ.get("CDP_VIEWPORT", "")
            if vp:
                w, h = vp.split("x")
                await cmd("Emulation.setDeviceMetricsOverride", {"width": int(w), "height": int(h), "deviceScaleFactor": 1, "mobile": False})
            await cmd("Page.navigate", {"url": URL})
            # wait for load
            for _ in range(60):
                m = await listen(3)
                if m and m.get("method") in ("Page.loadEventFired", "Page.frameStoppedLoading"):
                    break
            await asyncio.sleep(6)  # let Rocket Loader + demo run

            # collect console/exceptions
            events = []
            async def collect(timeout):
                for _ in range(int(timeout / 0.1)):
                    m = await listen(0.1)
                    if m and m.get("method") in ("Runtime.exceptionThrown", "Log.entryAdded", "Runtime.consoleAPICalled"):
                        events.append(m)
            await collect(3)

            # evaluate measurement
            res, err = await cmd("Runtime.evaluate", {"expression": MEASURE_JS, "returnByValue": True})
            val = None
            if res and res.get("result") and res["result"].get("value"):
                val = res["result"]["value"]
            print("=== LAYOUT ===")
            print(json.dumps(json.loads(val), ensure_ascii=False, indent=1) if val else "NO VALUE")
            print("=== JS EVENTS ===")
            for e in events:
                if e["method"] == "Runtime.exceptionThrown":
                    d = e["params"]["exceptionDetails"]
                    print("EXCEPTION:", d.get("text"), d.get("exception", {}).get("description", ""))
                elif e["method"] == "Log.entryAdded":
                    print("LOG:", e["params"]["entry"].get("level"), "-", e["params"]["entry"].get("text"))
                elif e["method"] == "Runtime.consoleAPICalled":
                    args = e["params"]["args"]
                    print("CONSOLE:", " ".join(a.get("value", a.get("description", "")) for a in args)[:300])
            if not events:
                print("(none)")
    finally:
        proc.terminate()

asyncio.run(main())
