#!/usr/bin/env node
/**
 * Verify the landing SVG model-hub geometry before it is used on the page.
 *
 * The geometry in `static/js/landing_demo.js` is generated entirely by code
 * (no hand-written coordinates). This script re-computes it via the same pure
 * functions and asserts that every value is finite, inside the viewBox, and
 * that no model cards overlap. It also validates every sampled point of the
 * bezier routes used by <animateMotion>.
 *
 * Usage:  node scripts/verify_landing_geometry.js
 * Exit 0 on PASS, 1 on FAIL (with reasons).
 */
const path = require('path');

const {
    MODEL_HUB_CONFIG,
    computeModelHubGeometry,
    validateModelHubGeometry,
    sampleCubicBezier
} = require('../static/js/landing_demo.js');

function main() {
    const g = computeModelHubGeometry(MODEL_HUB_CONFIG);
    const errors = validateModelHubGeometry(g);

    if (errors.length) {
        console.error('FAIL');
        errors.forEach((e) => console.error('  - ' + e));
        process.exit(1);
    }

    console.log('PASS');
    console.log(`viewBox: 0 0 ${g.width} ${g.height}`);
    console.log(`hub: (${g.hub.x}, ${g.hub.y}) r=${g.hub.r}`);
    console.log(`models: ${g.nodes.length}`);
    for (const n of g.nodes) {
        const r = n.rect;
        const samples = sampleCubicBezier(n.start, n.c1, n.c2, n.end, 24);
        const dStart = Math.hypot(n.start.x - g.hub.x, n.start.y - g.hub.y).toFixed(1);
        const dEnd = Math.hypot(n.end.x - n.cx, n.end.y - n.cy).toFixed(1);
        console.log(
            `  ${n.key.padEnd(9)} center=(${n.cx.toFixed(1)}, ${n.cy.toFixed(1)})` +
            ` rect=(${r.x.toFixed(1)}, ${r.y.toFixed(1)}, ${r.w}, ${r.h})` +
            ` routeStartDist=${dStart} routeEndDist=${dEnd}` +
            ` bezierSamples=${samples.length}`
        );
    }
    console.log('All node rects inside viewBox, no overlaps, all bezier samples finite & in-bounds.');
}

main();
