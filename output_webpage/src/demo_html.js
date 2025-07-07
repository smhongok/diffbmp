function getMeta(name) {
  return document.querySelector(`meta[name='${name}']`)?.content;
}

const numClass = parseInt(getMeta('numClass')); // 예: 5

const minY = -180; // 원하는 영역 시작 y
const maxY = 140;  // 원하는 영역 끝 y

const maxScale = 0.8;
const minScale = 0.3;  // 너무 작아지지 않게 하한선

/**
 * SVG 그룹에 gsap 애니메이션을 적용하는 함수
 * @param {string} svgId           - SVG element의 id (ex: 'svgsplat1')
 * @param {number} numClass        - 그룹 개수
 * @param {number} minY            - 오프셋 구간 시작값
 * @param {number} maxY            - 오프셋 구간 끝값
 * @param {number} yBase           - y축 기본값 (ex: 240)
 * @param {object} scaleParams     - {max, min} scale 범위 ({max: 0.8, min: 0.3} 등)
 * @param {('x'|'y')} offsetAxis   - 'x'면 x축에, 'y'면 y축에 yOffset 적용
 */
function animateSplatGroup(svgId, numClass, minY, maxY, yBase, scaleParams, offsetAxis) {
  const maxScale = scaleParams?.max || 0.8;
  const minScale = scaleParams?.min || 0.3;
  // numClass가 커질수록 scale 줄이기, 하한/상한 보장
  const scale = Math.min(
    maxScale,
    Math.max(minScale, maxScale * Math.sqrt(5 / numClass))
  );

  // 균등 분할 offset
  const yOffsetArr = Array.from({ length: numClass }, (_, i) =>
    minY + ((maxY - minY) * i) / (numClass - 1)
  );

  const layers = gsap.utils.toArray(`#${svgId} > g > g`);
  gsap.set(layers, { opacity: 1 });

  layers.forEach((g, i) => {
    const group = i % numClass;
    const yOffset = yOffsetArr[group];
    let animProps = {
      duration: 1,
      rotation: 0,
      scale: scale,
      autoAlpha: 1,
      delay: 0.5 + i * 0.005,
      fill: "#0F0F70",
      ease: "back.out(1.4)"
    };
    if (offsetAxis === 'x') {
      animProps.x = yOffset;
      animProps.y = yBase;
    } else {
      animProps.x = yBase;
      animProps.y = yOffset;
    }
    gsap.from(g, animProps);
  });

}

// ---------------------------
// 예시: 여러 SVG 그룹 동시 적용
// ---------------------------
animateSplatGroup(
  "svgsplat1",   // svgId
  numClass,             // numClass
  minY,          // minY
  maxY,           // maxY
  240,           // yBase (기본 y값)
  { max: maxScale, min: minScale }, // scaleParams
  'x'            // offsetAxis: x축에 offset 배분
);
