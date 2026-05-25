function getMeta(name) {
  return document.querySelector(`meta[name='${name}']`)?.content;
}

const numClass = parseInt(getMeta('numClass'));

const TARGET_FILL  = 0.70;
const SPACING_RATIO = 0.20;

function animateSplatGroup(svgId, numClass, _minY, _maxY, _yBase, _scaleParams, offsetAxis) {
  const layers = gsap.utils.toArray(`#${svgId} > g > g`);
  if (layers.length === 0) return;

  // 1. Store final states BEFORE any GSAP manipulation
  const finalStates = layers.map(g => ({
    x: gsap.getProperty(g, "x"),
    y: gsap.getProperty(g, "y"),
    scale: gsap.getProperty(g, "scale"),
    rotation: gsap.getProperty(g, "rotation"),
    fill: gsap.getProperty(g, "fill"),
    fillOpacity: gsap.getProperty(g, "fill-opacity")
  }));

  // 2. Content area from SVG viewBox
  const svg = document.getElementById(svgId);
  const vb  = svg.viewBox.baseVal;
  const contentBBox = { x: vb.x, y: vb.y, width: vb.width, height: vb.height };

  // 3. One representative per class
  const classReps = Array.from({ length: numClass }, (_, c) =>
    layers.find((_, i) => i % numClass === c)
  );

  // 4a. Extract font baseline from the inner path's preserved transform.
  //     translate(0, baseline_y) scale(sx, -sy)  →  baseline_y = font ascent in SVG units.
  //     Same for all glyphs from the same font at the same size.
  function extractInnerBaselineY(gElem) {
    const innerPath = gElem?.querySelector('path');
    if (!innerPath) return null;
    const t = innerPath.getAttribute('transform') || '';
    const m = t.match(/translate\(\s*[\d.e+-]+[\s,]+\s*([\d.e+-]+)/i);
    return m ? parseFloat(m[1]) : null;
  }
  const baselineY = extractInnerBaselineY(classReps[0]) ?? 65.18;

  // 4b. Measure tight content bboxes at scale=1 (needed for scale computation).
  classReps.forEach(g => {
    if (g) gsap.set(g, { x: 0, y: 0, scale: 1, rotation: 0 });
  });
  const localBBoxes = classReps.map(g =>
    g ? g.getBBox() : { x: 0, y: 0, width: 0, height: 0 }
  );

  // 4c. Compute display scale from TARGET_FILL and character widths.
  const sumNatW  = localBBoxes.reduce((s, b) => s + b.width, 0);
  const avgNatW  = sumNatW / Math.max(1, numClass);
  const natTotal = sumNatW + avgNatW * SPACING_RATIO * (numClass - 1);
  const scale    = natTotal > 0
    ? Math.min(10, contentBBox.width * TARGET_FILL / natTotal)
    : 1;

  // 5. Adaptive initial Y — computed AFTER scale is known so the gap is large
  //    enough that the tallest character's cap-top clears the content area.
  //    Characters extend (baselineY * scale) above their baseline anchor,
  //    so the gap must exceed that height to prevent overlap with content.
  const charHeightAboveBaseline = scale * baselineY; // max vertical extent above baseline
  const gap      = charHeightAboveBaseline + Math.max(20, contentBBox.height * 0.04);
  const initialY = contentBBox.y + contentBBox.height + gap;

  const letterSpacing = scale * avgNatW * SPACING_RATIO;

  // Horizontal layout centered under content
  const contentCenterX     = contentBBox.x + contentBBox.width / 2;
  const visualWidths        = localBBoxes.map(b => b.width * scale);
  const totalWidth          = visualWidths.reduce((s, w) => s + w, 0) + letterSpacing * (numClass - 1);
  const classTargetCenterX  = [];
  let curX = contentCenterX - totalWidth / 2;
  for (let c = 0; c < numClass; c++) {
    classTargetCenterX.push(curX + visualWidths[c] / 2);
    curX += visualWidths[c] + letterSpacing;
  }

  // X: place content bbox centre at target centre (formula: tx = target - scale*localCx)
  const classGsapX = localBBoxes.map((b, c) =>
    classTargetCenterX[c] - scale * (b.x + b.width / 2)
  );

  // Y: per-class GSAP-y that produces baseline alignment given GSAP's actual
  //    transform origin (empirically: yOrigin = localBBoxes[c].y = bbox top).
  //    With yOrigin=oy: visual_y = gsapY + oy + scale*(py - oy)
  //    Setting visual_baseline = initialY:
  //      initialY = gsapY + oy + scale*(baselineY - oy)
  //      gsapY    = initialY + oy*(scale-1) - scale*baselineY
  const classGsapY = localBBoxes.map(b =>
    initialY + b.y * (scale - 1) - scale * baselineY
  );

  // 8. Set all elements to aligned initial positions (black fill).
  layers.forEach((g, i) => {
    const c = i % numClass;
    gsap.set(g, {
      x:        offsetAxis === 'x' ? classGsapX[c] : classGsapY[c],
      y:        offsetAxis === 'y' ? classGsapX[c] : classGsapY[c],
      scale,
      rotation: 0,
      fill:     '#000000',
      opacity:  1,
    });
  });

  // 9. Compute initial row visual bbox for capture bounds.
  //    With yOrigin=b.y: visual_top = gsapY + b.y  (scale-independent)
  //                      visual_bottom = gsapY + b.y + scale*b.height
  const classVisualBBoxes = localBBoxes.map((b, c) => ({
    left:   classGsapX[c] + scale * b.x,
    top:    classGsapY[c] + b.y,
    right:  classGsapX[c] + scale * (b.x + b.width),
    bottom: classGsapY[c] + b.y + scale * b.height,
  }));
  const rowBBox = {
    x:      Math.min(...classVisualBBoxes.map(b => b.left)),
    y:      Math.min(...classVisualBBoxes.map(b => b.top)),
    right:  Math.max(...classVisualBBoxes.map(b => b.right)),
    bottom: Math.max(...classVisualBBoxes.map(b => b.bottom)),
  };

  // 10. Expose capture bounds
  const pad = Math.max(60, contentBBox.width * 0.06);
  const ux  = Math.min(contentBBox.x, rowBBox.x) - pad;
  const uy  = Math.min(contentBBox.y, rowBBox.y) - pad;
  const ux2 = Math.max(contentBBox.x + contentBBox.width,  rowBBox.right)  + pad;
  const uy2 = Math.max(contentBBox.y + contentBBox.height, rowBBox.bottom) + pad;
  window._captureBounds = { svgId, x: ux, y: uy, width: ux2 - ux, height: uy2 - uy };

  // 11. Register animations (explicit FROM/TO to avoid lazy DOM reads in time())
  layers.forEach((g, i) => {
    const c  = i % numClass;
    const fs = finalStates[i];
    gsap.fromTo(
      g,
      {
        x:              offsetAxis === 'x' ? classGsapX[c] : classGsapY[c],
        y:              offsetAxis === 'y' ? classGsapX[c] : classGsapY[c],
        scale,
        rotation:       0,
        fill:           '#000000',
        immediateRender: false,
      },
      {
        x: fs.x, y: fs.y, scale: fs.scale, rotation: fs.rotation,
        fill: fs.fill, "fill-opacity": fs.fillOpacity,
        duration: 1,
        delay: 0.5 + i * 0.005,
        ease: "back.out(1.4)",
      }
    );
  });

  // 12. Pause in capture mode
  if (window.__captureMode) {
    gsap.globalTimeline.pause(0);
  }
}

animateSplatGroup("svgsplat1", numClass, -180, 140, 240, {}, 'x');
