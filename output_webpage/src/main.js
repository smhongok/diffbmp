function getMeta(name) {
  return document.querySelector(`meta[name='${name}']`)?.content;
}

const numClass = parseInt(getMeta('numClass')); // 예: 5

const minY = -180; // 원하는 영역 시작 y
const maxY = 140;  // 원하는 영역 끝 y

// numClass에 맞게 yOffset 자동 분배
const yOffsetArr = Array.from({length: numClass}, (_, i) =>
  minY + ((maxY - minY) * i) / (numClass - 1)
);

const layers1 = gsap.utils.toArray("#svgsplat1 > g > g");
gsap.set(layers1, { opacity: 1 });

layers1.forEach((g, i) => {
  const group = i % numClass;
  const yOffset = yOffsetArr[group];

  gsap.from(g, {
    duration: 1,
    x: 240,
    y: yOffset,
    rotation: 0,
    scale: 0.8,
    autoAlpha: 1,
    delay: 0.5 + i * 0.005,
    ease: "back.out(1.4)"
  });
});
