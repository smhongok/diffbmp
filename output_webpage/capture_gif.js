/**
 * Capture the GSAP animation in src/index.html as a GIF.
 *
 * Usage:
 *   node capture_gif.js [output.gif] [fps] [duration_sec] [gif_scale]
 *
 * Defaults: output=animation.gif, fps=30, duration=3, gif_scale=1.0
 *
 * gif_scale: e.g. 0.5 halves both dimensions (4× smaller file).
 *
 * How it works:
 *   - Injects CSS margin so the SVG is not flush to the page edge
 *   - Uses getBoundingClientRect() for accurate screen→clip conversion
 *   - Seeks GSAP frame-by-frame (timeline paused; no live RAF)
 *   - ffmpeg 2-pass palettegen+paletteuse for high-quality GIF
 */

const puppeteer = require('puppeteer');
const path      = require('path');
const fs        = require('fs');
const { execSync } = require('child_process');

const outputGif   = process.argv[2] || 'animation.gif';
const fps         = parseInt(process.argv[3])    || 30;
const durationSec = parseFloat(process.argv[4])  || 3;
const gifScale    = parseFloat(process.argv[5])  || 1.0;   // e.g. 0.5 = half size

// Pixels of CSS margin injected around the SVG so content near the edge isn't cut off
const SVG_MARGIN = 200;  // must exceed _captureBounds SVG pad + EXTRA_PAD (pixels)

const htmlPath  = path.resolve(__dirname, 'src', 'index.html');
const framesDir = path.resolve(__dirname, '_frames_tmp');

(async () => {
  if (fs.existsSync(framesDir)) fs.rmSync(framesDir, { recursive: true });
  fs.mkdirSync(framesDir);

  const browser = await puppeteer.launch({
    headless: 'new',
    protocolTimeout: 300000,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--allow-file-access-from-files',
      '--disable-web-security',
    ],
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(300000);
  page.setDefaultNavigationTimeout(60000);

  // Tell demo_html.js to pause GSAP before any tick
  await page.evaluateOnNewDocument(() => { window.__captureMode = true; });

  // Initial viewport — will be resized after we know the clip dimensions
  await page.setViewport({ width: 800, height: 800 });
  await page.goto(`file://${htmlPath}`, { waitUntil: 'load', timeout: 60000 });

  // Wait for animation init to complete
  await page.evaluate(() => !!window._captureBounds);

  // Inject CSS margin so the SVG has space on all sides.
  // SVG is rendered at its natural pixel size; overflow:visible means content
  // outside the viewBox (e.g. the initial character row below) is still painted.
  await page.addStyleTag({ content:
    `body { margin: 0; background: white; }
     #svgsplat1 { display: block; margin: ${SVG_MARGIN}px; }`
  });

  // Compute clip in screen pixels using getBoundingClientRect, which now accounts
  // for the injected margin. Add extra screen-space padding around the bounds.
  const EXTRA_PAD = 40; // additional screen pixels beyond _captureBounds
  const clipInfo = await page.evaluate((_margin, extraPad) => {
    const bounds = window._captureBounds;
    if (!bounds) return null;
    const svg = document.getElementById(bounds.svgId);
    const rect = svg.getBoundingClientRect();
    const vb   = svg.viewBox.baseVal;
    const sx   = rect.width  / vb.width;
    const sy   = rect.height / vb.height;

    // Convert SVG-space bounds → screen pixels (rect already includes CSS margin)
    const l = Math.floor(rect.left + (bounds.x - vb.x) * sx) - extraPad;
    const t = Math.floor(rect.top  + (bounds.y - vb.y) * sy) - extraPad;
    const r = Math.ceil( rect.left + (bounds.x + bounds.width  - vb.x) * sx) + extraPad;
    const b = Math.ceil( rect.top  + (bounds.y + bounds.height - vb.y) * sy) + extraPad;

    return {
      x: Math.max(0, l), y: Math.max(0, t),
      width:  r - Math.max(0, l),
      height: b - Math.max(0, t),
      svgW: parseFloat(svg.getAttribute('width'))  || vb.width,
      svgH: parseFloat(svg.getAttribute('height')) || vb.height,
    };
  }, SVG_MARGIN, EXTRA_PAD);

  if (!clipInfo) {
    console.error('_captureBounds not found in page.');
    await browser.close();
    process.exit(1);
  }

  const clip = { x: clipInfo.x, y: clipInfo.y, width: clipInfo.width, height: clipInfo.height };

  // Resize viewport to cover the full clip area (including initial char row below SVG)
  const vpW = clip.x + clip.width  + 4;
  const vpH = clip.y + clip.height + 4;
  await page.setViewport({ width: vpW, height: vpH });

  console.log(`SVG: ${clipInfo.svgW}×${clipInfo.svgH}  margin: ${SVG_MARGIN}px`);
  console.log(`Clip: x=${clip.x} y=${clip.y} ${clip.width}×${clip.height}`);
  if (gifScale < 1)
    console.log(`GIF output: ${Math.round(clip.width * gifScale)}×${Math.round(clip.height * gifScale)} (scale=${gifScale})`);
  console.log(`Capturing ${Math.ceil(durationSec * fps)} frames @ ${fps}fps (${durationSec}s) → ${outputGif}`);

  const totalFrames = Math.ceil(durationSec * fps);
  for (let i = 0; i < totalFrames; i++) {
    // GSAP timeline has .then() — return null to avoid Puppeteer treating it as a Promise
    await page.evaluate(t => { gsap.globalTimeline.time(t); return null; }, i / fps);
    const framePath = path.join(framesDir, `frame_${String(i).padStart(5, '0')}.png`);
    await page.screenshot({ path: framePath, clip });
    if (i % 10 === 0) process.stdout.write(`  frame ${i}/${totalFrames}\r`);
  }
  console.log('\nAll frames captured.');
  await browser.close();

  // ffmpeg 2-pass: palette generation → GIF encoding
  // Optional downscale for smaller file size (gif_scale < 1.0)
  const paletteFile = path.join(framesDir, 'palette.png');

  const outW = Math.round(clip.width  * gifScale);
  const outH = Math.round(clip.height * gifScale);
  // Make dimensions even (required by some encoders)
  const evenW = outW % 2 === 0 ? outW : outW - 1;
  const evenH = outH % 2 === 0 ? outH : outH - 1;

  const scaleVF   = gifScale < 1.0 ? `scale=${evenW}:${evenH}:flags=lanczos,` : '';
  const paletteLA = gifScale < 1.0
    ? `[0:v]scale=${evenW}:${evenH}:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=5`
    : `paletteuse=dither=bayer:bayer_scale=5`;

  console.log('Generating palette...');
  execSync(
    `ffmpeg -y -framerate ${fps} -i "${framesDir}/frame_%05d.png" ` +
    `-vf "${scaleVF}palettegen=max_colors=256:stats_mode=full" "${paletteFile}"`,
    { stdio: 'inherit' }
  );

  console.log('Encoding GIF...');
  execSync(
    `ffmpeg -y -framerate ${fps} -i "${framesDir}/frame_%05d.png" ` +
    `-i "${paletteFile}" ` +
    `-lavfi "${paletteLA}" ` +
    `"${outputGif}"`,
    { stdio: 'inherit' }
  );

  fs.rmSync(framesDir, { recursive: true });
  console.log(`Done → ${outputGif}  (${fs.statSync(outputGif).size >> 10} KB)`);
})();
