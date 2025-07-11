function getMeta(name) {
  return document.querySelector(`meta[name='${name}']`)?.content;
}

const numClass = parseInt(getMeta('numClass')); // 예: 5

const minY = -180; // 원하는 영역 시작 y
const maxY = 140;  // 원하는 영역 끝 y

const maxScale = 0.8;
const minScale = 0.3;  // 너무 작아지지 않게 하한선

function setupSplatGroupAnimation(svgId, numClass, minY, maxY, yBase, scaleParams, offsetAxis) {
  const maxScale = scaleParams?.max || 0.8;
  const minScale = scaleParams?.min || 0.3;
  const scale = Math.min(
    maxScale,
    Math.max(minScale, maxScale * Math.sqrt(5 / numClass))
  );

  const yOffsetArr = Array.from({ length: 20 }, (_, i) =>
    minY + ((maxY - minY) * i) / (20 - 1)
  );

  const layers = gsap.utils.toArray(`#${svgId} > g > g`);

  const fadeDur = 0.2;  // 페이드인 지속 시간
  const moveProps = {
    duration: 1,
    rotation: 0,
    scale: scale,
    fill: "#000000",
    ease: "back.out(1.4)",
    "fill-opacity": 1
  };

  // 각 <g> 요소의 최종 상태 저장 (x, y, scale, rotation, fill, fill-opacity)
  const finalStates = layers.map(g => ({
    x: gsap.getProperty(g, "x"),
    y: gsap.getProperty(g, "y"),
    scale: gsap.getProperty(g, "scale"),
    rotation: gsap.getProperty(g, "rotation"),
    fill: gsap.getProperty(g, "fill"),
    fillOpacity: gsap.getProperty(g, "fill-opacity")
  }));

  gsap.set(layers, { opacity: 0 });

  let currentIndex = 0;

  function triggerNextAnimation() {
    if (currentIndex < layers.length) {
      const g = layers[currentIndex];
      const group = currentIndex % 20;
      const yOffset = yOffsetArr[group];
      const finalState = finalStates[currentIndex];

      // 출발점 및 moveProps 속성 설정
      gsap.set(g, {
        x: offsetAxis === 'x' ? yOffset : finalState.x,
        y: offsetAxis === 'y' ? yOffset : yBase,
        scale: scale,
        rotation: 0,
        fill: "#000000",
        "fill-opacity": 1,
        opacity: 0
      });

      // 페이드인 애니메이션
      gsap.to(g, {
        opacity: 1,
        duration: fadeDur,
        ease: "none",
        onComplete: () => {
          // 페이드인 완료 후 HTML의 원래 상태로 이동
          gsap.to(g, {
            duration: 1,
            x: finalState.x,
            y: finalState.y,
            scale: finalState.scale,
            rotation: finalState.rotation,
            fill: finalState.fill,
            "fill-opacity": finalState.fillOpacity,
            ease: "back.out(1.4)"
          });
        }
      });

      currentIndex++;
    }
  }

  document.addEventListener('keydown', (event) => {
    triggerNextAnimation();
  });
}

setupSplatGroupAnimation(
  "svgsplat1",
  numClass,
  minY,
  maxY,
  240,
  { max: maxScale, min: minScale },
  'x'
);