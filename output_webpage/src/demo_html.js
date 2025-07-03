// // 1) 레이어 선택
// const layers = gsap.utils.toArray("#svgsplat > g");

// // 2) 공통된 시작값으로 from 애니메이션 (stagger만 사용)
// gsap.from(layers, {
//   duration: 1,           // 애니메이션 길이
//   x: 200,                  // 모든 레이어 동일한 시작 x
//   y: 0,               // 예: 화면 위쪽에서 내려오기
//   rotation: 0,           // 회전 없이
//   scale: 1,            // 작게 시작
//   autoAlpha: 1,          // 투명도 0 (visibility:hidden + opacity:0)
//   stagger: 0.005,          // 0.005초 간격으로 순차 등장
//   ease: "back.out(1.4)"
// });

const layers1 = gsap.utils.toArray("#svgsplat1 > g > g");
gsap.set(layers1, { opacity: 1 }); // 시작은 보이게

layers1.forEach((g, i) => {
  const group = i % 5;

  // y 오프셋 결정
  const yOffset = [-100, -20, 60, 140, 220][group];

  gsap.from(g, {
    duration: 1,
    x: 240,
    y: yOffset-80,
    rotation: 0,
    scale: 0.8,
    autoAlpha: 1,
    delay: 0.5 + i * 0.005, // stagger 효과
    ease: "back.out(1.4)"
  });
});
