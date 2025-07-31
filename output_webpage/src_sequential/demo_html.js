function getMeta(name) {
  return document.querySelector(`meta[name='${name}']`)?.content;
}

const numClass = parseInt(getMeta('numClass')); // 예: 5
const frameCount = parseInt(getMeta('frameCount')) || 1; // 프레임 수
const fps = parseInt(getMeta('fps')) || 24; // FPS

// 프레임 시퀀스 데이터 (HTML에서 주입될 예정)
let frameSequence = [];

/**
 * 프레임 시퀀스를 재생하는 함수 (동기화된 버전)
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
  let startTime = Date.now();
  
  console.log(`Starting sequence playback: ${sequence.length} frames at ${targetFps} FPS`);
  
  function updateFrame() {
    if (!isPlaying) return;
    
    // 글로벌 시간 기반으로 현재 프레임 계산 (동기화)
    const elapsed = Date.now() - startTime;
    const globalFrame = Math.floor((elapsed / 1000) * targetFps) % sequence.length;
    
    const transforms = sequence[globalFrame];
    if (transforms && transforms.length > 0) {
      layers.forEach((g, i) => {
        if (transforms[i]) {
          g.setAttribute('transform', transforms[i]);
        }
      });
    }
    
    if (!loop && globalFrame >= sequence.length - 1) {
      isPlaying = false;
      console.log('Sequence playback finished');
      return;
    }
    
    requestAnimationFrame(updateFrame);
  }
  
  // 재생 시작
  updateFrame();
  
  // 재생 제어 함수들을 전역으로 노출
  window.sequenceControls = {
    play: () => { 
      if (!isPlaying) {
        startTime = Date.now(); // 재시작 시 시간 리셋
        isPlaying = true; 
        updateFrame(); 
      }
    },
    pause: () => { isPlaying = false; },
    stop: () => { 
      isPlaying = false; 
      startTime = Date.now(); 
      // 첫 번째 프레임으로 리셋
      const transforms = sequence[0];
      if (transforms && transforms.length > 0) {
        layers.forEach((g, i) => {
          if (transforms[i]) {
            g.setAttribute('transform', transforms[i]);
          }
        });
      }
    },
    setFrame: (frame) => { 
      const frameIndex = Math.max(0, Math.min(frame, sequence.length - 1));
      startTime = Date.now() - (frameIndex / targetFps * 1000);
      const transforms = sequence[frameIndex];
      if (transforms && transforms.length > 0) {
        layers.forEach((g, i) => {
          if (transforms[i]) {
            g.setAttribute('transform', transforms[i]);
          }
        });
      }
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

/**
 * 각 요소가 개별적으로 시퀀스에 합류하는 애니메이션 함수
 */
function animateSplatGroupWithStaggeredSequence(svgId, numClass, minY, maxY, yBase, scaleParams, offsetAxis, sequence, targetFps) {
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

  // 글로벌 시퀀스 시작 시간 설정
  const globalSequenceStartTime = Date.now();
  
  // 각 요소의 시퀀스 참여 상태 추적
  const elementStates = new Array(layers.length).fill(false); // false: group-in 중, true: sequence 중
  
  // 시퀀스 업데이트 함수
  function updateSequenceFrame() {
    const elapsed = Date.now() - globalSequenceStartTime;
    const globalFrame = Math.floor((elapsed / 1000) * targetFps) % sequence.length;
    
    layers.forEach((g, i) => {
      if (elementStates[i]) { // 이 요소가 시퀀스에 참여 중이면
        const transforms = sequence[globalFrame];
        if (transforms && transforms[i]) {
          g.setAttribute('transform', transforms[i]);
        }
      }
    });
    
    requestAnimationFrame(updateSequenceFrame);
  }
  
  // 시퀀스 업데이트 시작
  updateSequenceFrame();

  layers.forEach((g, i) => {
    const group = i % numClass;
    const yOffset = yOffsetArr[group];
    const delay = 0.5 + i * 0.005;
    
    let animProps = {
      duration: 1,
      rotation: 0,
      scale: scale,
      autoAlpha: 1,
      delay: delay,
      fill: "#0F0F70",
      ease: "back.out(1.4)",
      // 각 요소의 group-in 애니메이션 완료 시 시퀀스에 합류
      onComplete: function() {
        console.log(`Element ${i} joined sequence`);
        elementStates[i] = true; // 이 요소를 시퀀스 모드로 전환
      }
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
  
  // 전역 제어 함수 노출
  window.sequenceControls = {
    getElementStates: () => elementStates,
    resetAll: () => {
      elementStates.fill(false);
      // 모든 요소를 첫 번째 프레임으로 리셋
      const transforms = sequence[0];
      if (transforms && transforms.length > 0) {
        layers.forEach((g, i) => {
          if (transforms[i]) {
            g.setAttribute('transform', transforms[i]);
          }
        });
      }
    }
  };
}

// 페이지 로드 후 자동 실행
document.addEventListener('DOMContentLoaded', function() {
  if (window.frameSequenceData && window.frameSequenceData.length > 0) {
    console.log('Sequential mode detected: starting staggered group-in with immediate sequence join');
    
    const minY = -180;
    const maxY = 140;
    const maxScale = 0.8;
    const minScale = 0.3;
    
    // 각 요소가 개별적으로 시퀀스에 합류하는 애니메이션 시작
    animateSplatGroupWithStaggeredSequence(
      "svgsplat1",
      numClass,
      minY,
      maxY,
      240,
      { max: maxScale, min: minScale },
      'x',
      window.frameSequenceData,
      fps
    );
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
