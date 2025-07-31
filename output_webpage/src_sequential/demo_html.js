function getMeta(name) {
  return document.querySelector(`meta[name='${name}']`)?.content;
}

const numClass = parseInt(getMeta('numClass')); // 예: 5
const frameCount = parseInt(getMeta('frameCount')) || 1; // 프레임 수
const fps = parseInt(getMeta('fps')) || 24; // FPS

// 프레임 시퀀스 데이터 (HTML에서 주입될 예정)
let frameSequence = [];

/**
 * 프레임 시퀀스를 재생하는 함수
 * @param {string} svgId - SVG element의 id (ex: 'svgsplat1')
 * @param {Array} sequence - 프레임별 transform 배열들
 * @param {number} targetFps - 재생 FPS
 * @param {boolean} loop - 반복 재생 여부
 */
function playSequence(svgId, sequence, targetFps = 24, loop = true) {
  if (!sequence || sequence.length === 0) {
    console.error('No frame sequence data found');
    return;
  }

  const layers = gsap.utils.toArray(`#${svgId} > g > g`);
  let currentFrame = 0;
  let isPlaying = true;
  
  console.log(`Starting sequence playback: ${sequence.length} frames at ${targetFps} FPS`);
  
  function updateFrame() {
    if (!isPlaying) return;
    
    const transforms = sequence[currentFrame];
    if (transforms && transforms.length > 0) {
      layers.forEach((g, i) => {
        if (transforms[i]) {
          g.setAttribute('transform', transforms[i]);
        }
      });
    }
    
    currentFrame++;
    if (currentFrame >= sequence.length) {
      if (loop) {
        currentFrame = 0;
      } else {
        isPlaying = false;
        console.log('Sequence playback finished');
        return;
      }
    }
    
    setTimeout(updateFrame, 1000 / targetFps);
  }
  
  // 첫 프레임 설정 후 재생 시작
  updateFrame();
  
  // 재생 제어 함수들을 전역으로 노출
  window.sequenceControls = {
    play: () => { isPlaying = true; updateFrame(); },
    pause: () => { isPlaying = false; },
    stop: () => { isPlaying = false; currentFrame = 0; updateFrame(); },
    setFrame: (frame) => { 
      currentFrame = Math.max(0, Math.min(frame, sequence.length - 1)); 
      updateFrame(); 
    }
  };
}

/**
 * 단일 프레임 애니메이션 (기존 방식과 호환)
 */
function animateSplatGroup(svgId, numClass, minY, maxY, yBase, scaleParams, offsetAxis) {
  const maxScale = scaleParams?.max || 0.8;
  const minScale = scaleParams?.min || 0.3;
  const scale = Math.min(
    maxScale,
    Math.max(minScale, maxScale * Math.sqrt(5 / numClass))
  );

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

// 페이지 로드 후 자동 실행
document.addEventListener('DOMContentLoaded', function() {
  // frameSequence 데이터가 있으면 시퀀스 재생, 없으면 단일 프레임 애니메이션
  if (window.frameSequenceData && window.frameSequenceData.length > 0) {
    console.log('Sequential mode detected, starting sequence playback');
    playSequence("svgsplat1", window.frameSequenceData, fps, true);
  } else {
    console.log('Single frame mode detected, using traditional animation');
    const minY = -180;
    const maxY = 140;
    const maxScale = 0.8;
    const minScale = 0.3;
    
    animateSplatGroup(
      "svgsplat1",
      numClass,
      minY,
      maxY,
      240,
      { max: maxScale, min: minScale },
      'x'
    );
  }
});
