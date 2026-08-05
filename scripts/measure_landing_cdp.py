#!/usr/bin/env python3
"""Drive headless Chrome via CDP to measure the live landing page layout and
capture console/exceptions. Uses venv python (has websockets + requests).

Usage:
  CDP_VIEWPORT=375x800 venv/bin/python scripts/measure_landing_cdp.py
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

CHROME = "/home/ai-chat-minashin1120/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"
URL = "https://ai.minashin1120.com/"
DEBUG_PORT = 9500 + (int(time.time()) % 200)
PROFILE = "/tmp/opencode_cdp_profile_%d" % DEBUG_PORT

MEASURE_JS = r"""
(function () {
    var out = {};
    var vw = document.documentElement.clientWidth;
    out.viewport = { w: vw, h: window.innerHeight };
    out.doc = { scrollW: document.documentElement.scrollWidth, scrollH: document.documentElement.scrollHeight,
                hasHOverflow: document.documentElement.scrollWidth > vw + 1 };
    var offenders = [];
    var all = document.querySelectorAll('body *');
    for (var i = 0; i < all.length; i++) {
        var el = all[i];
        var r = el.getBoundingClientRect();
        if (r.width <= 0) continue;
        if (r.right > vw + 1 || r.left < -1) {
            var cs = window.getComputedStyle(el);
            var cls = (typeof el.className === 'string' && el.className) ? '.' + el.className.split(' ').slice(0, 2).join('.') : '';
            offenders.push({ tag: el.tagName.toLowerCase(), cls: cls, left: Math.round(r.left), right: Math.round(r.right), w: Math.round(r.width), pos: cs.position, overflowX: cs.overflowX });
        }
    }
    out.offenderCount = offenders.length;
    out.offenders = offenders.slice(0, 25);
    function rect(sel) {
        var el = document.querySelector(sel);
        if (!el) return null;
        var r = el.getBoundingClientRect();
        return { top: Math.round(r.top), bottom: Math.round(r.bottom), h: Math.round(r.height), w: Math.round(r.width), left: Math.round(r.left), right: Math.round(r.right) };
    }
    out.sections = {
        header: rect('header'),
        hero: rect('.hero-bg'),
        chatDemo: rect('.chat-demo'),
        chatCaption: rect('.hero-bg .text-xs.mt-4'),
        marquee: rect('.marquee-wrap'),
        hubFrame: rect('.hub-frame'),
        featuresGrid: rect('section:nth-of-type(4) .grid'),
        stepsGrid: rect('section:nth-of-type(5) .grid'),
        faq: rect('.ld-faq'),
        cta: rect('section.hero-bg.py-20'),
        footer: rect('footer')
    };
    var items = ['header', '.hero-bg', '.marquee-wrap', '.hub-frame', 'section:nth-of-type(4) .grid', 'section:nth-of-type(5) .grid', '.ld-stat', '.ld-faq', 'section.hero-bg.py-20', 'footer'];
    var rects = [];
    items.forEach(function (sel) {
        var el = document.querySelector(sel);
        if (el) { var r = el.getBoundingClientRect(); rects.push({ sel: sel, top: r.top, bottom: r.bottom }); }
    });
    var gaps = [];
    for (var k = 0; k < rects.length - 1; k++) gaps.push({ a: rects[k].sel, b: rects[k + 1].sel, gap: Math.round(rects[k + 1].top - rects[k].bottom) });
    out.sectionGaps = gaps;
    out.heroText = {
        badge: rect('.hero-bg .inline-flex'),
        h1: rect('.hero-bg h1'),
        sub: rect('.hero-bg .max-w-xl'),
        stats: rect('.hero-bg .mt-10')
    };
    out.demoState = {
        built: !!document.querySelector('.chat-demo-body'),
        rows: document.querySelectorAll('.chat-demo-body .ld-row').length,
        hubCards: document.querySelectorAll('.hub-node-card').length
    };
    return JSON.stringify(out);
})();
"""


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self._id = 0

    async def cmd(self, method, params=None):
        self._id += 1
        req_id = self._id
        await self.ws.send(json.dumps({"id": req_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == req_id:
                return msg.get("result", {}), msg.get("error")

    async def next_event(self, timeout):
        try:
            return json.loads(await asyncio.wait_for(self.ws.recv(), timeout))
        except asyncio.TimeoutError:
            return None


async def main():
    print("step: launching chrome", flush=True)
    proc = subprocess.Popen([
        CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--remote-debugging-port=%d" % DEBUG_PORT,
        "--user-data-dir=%s" % PROFILE, "about:blank"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/json/version" % DEBUG_PORT, timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        print("step: debug endpoint up", flush=True)

        from urllib.parse import quote as urlquote
        req = urllib.request.Request(
            "http://127.0.0.1:%d/json/new?%s" % (DEBUG_PORT, urlquote(URL, safe="")), method="PUT")
        target = json.loads(urllib.request.urlopen(req, timeout=5).read())
        print("step: tab opened", flush=True)

        import websockets
        async with websockets.connect(target["webSocketDebuggerUrl"], max_size=2**24) as ws:
            cdp = CDP(ws)
            await cdp.cmd("Runtime.enable")
            await cdp.cmd("Log.enable")
            vp = os.environ.get("CDP_VIEWPORT", "")
            if vp:
                w, h = vp.split("x")
                await cdp.cmd("Emulation.setDeviceMetricsOverride", {"width": int(w), "height": int(h), "deviceScaleFactor": 1, "mobile": False})
            print("step: navigating", flush=True)
            await cdp.cmd("Page.navigate", {"url": URL})
            # wait until the demo and hub are built (Rocket Loader defers scripts)
            ready = False
            i = -1
            for i in range(80):
                res, _ = await cdp.cmd("Runtime.evaluate", {"expression": "!!document.querySelector('.chat-demo-body') && !!document.querySelector('.hub-node-card')", "returnByValue": True})
                try:
                    ready = bool(res["result"]["value"])
                except Exception:
                    ready = False
                if ready:
                    break
                await asyncio.sleep(0.25)
            print("step: ready=%s after %ds" % (ready, i if ready else 20), flush=True)
            await asyncio.sleep(4)

            # capture console / exceptions (non-blocking pump)
            events = []
            for _ in range(30):
                m = await cdp.next_event(0.1)
                if m and m.get("method") in ("Runtime.exceptionThrown", "Log.entryAdded"):
                    events.append(m)

            res, err = await cdp.cmd("Runtime.evaluate", {"expression": MEASURE_JS, "returnByValue": True})
            val = None
            if res and res.get("result") and res["result"].get("value"):
                val = res["result"]["value"]
            print("=== LAYOUT ===")
            if val:
                print(json.dumps(json.loads(val), ensure_ascii=False, indent=1))
            elif res and res.get("exceptionDetails"):
                print("EVAL EXCEPTION:", res["exceptionDetails"].get("exception", {}).get("description", ""))
            else:
                print("NO VALUE", "err=", err)
            print("=== JS EVENTS ===")
            for e in events:
                if e["method"] == "Runtime.exceptionThrown":
                    d = e["params"]["exceptionDetails"]
                    print("EXCEPTION:", d.get("text"), d.get("exception", {}).get("description", ""))
                elif e["method"] == "Log.entryAdded":
                    en = e["params"]["entry"]
                    print("LOG:", en.get("level"), "-", en.get("text"))
            if not events:
                print("(none)")
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
