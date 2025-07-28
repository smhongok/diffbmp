function getMeta(name) {
  return document.querySelector(`meta[name='${name}']`)?.content;
}

const numClass = parseInt(getMeta('numClass')); // 예: 5
const charCounts = JSON.parse(getMeta('char_counts') || '[]'); // 예: [10, 16, 25, 14, 15]
const wordLengthsPerLine = JSON.parse(getMeta('word_lengths_per_line') || '[]'); // 예: [[1, 2, 2, 1, 2, 2], [3, 2, 2, 2, 2, 2, 3]]

const minY = -180; // 원하는 영역 시작 y
const maxY = 140;  // 원하는 영역 끝 y
const centerY = (minY + maxY) / 2; // 중심 y 좌표
const letterSpacing = 20; // 글자 간 간격 (픽셀)
const wordSpacing = 10; // 단어 간 간격 (픽셀)

const maxScale = 0.8;
const minScale = 0.3;  // 너무 작아지지 않게 하한선

function setupSplatGroupAnimation(svgId, numClass, minY, maxY, yBase, scaleParams, offsetAxis) {
  const maxScale = scaleParams?.max || 0.8;
  const minScale = scaleParams?.min || 0.3;
  const scale = Math.min(
    maxScale,
    Math.max(minScale, maxScale * Math.sqrt(5 / numClass))
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

  // 문장 단위로 <g> 요소 그룹화
  const sentences = [];
  let startIndex = 0;
  charCounts.forEach(count => {
    sentences.push(layers.slice(startIndex, startIndex + count));
    startIndex += count;
  });

  gsap.set(layers, { opacity: 0 });

  let currentSentenceIndex = 0;
  let isFadedIn = false; // 현재 문장이 fade-in 완료되었는지 추적

  function triggerFadeIn() {
    if (currentSentenceIndex < sentences.length && !isFadedIn) {
      const sentence = sentences[currentSentenceIndex];
      const wordLengths = wordLengthsPerLine[currentSentenceIndex] || [];
      const charCount = sentence.length;

      // 단어와 글자 수를 기반으로 x 좌표 계산 (가운데 정렬)
      let totalWidth = 0;
      wordLengths.forEach((wordLength, i) => {
        totalWidth += (wordLength - 1) * letterSpacing;
        if (i < wordLengths.length - 1) totalWidth += wordSpacing;
      });

      const startX = centerY - totalWidth / 2; // 문장의 중심을 centerY에 맞춤

      // 단어의 시작 위치 계산
      const wordOffsets = [];
      let currentOffset = 0;
      wordLengths.forEach((wordLength) => {
        wordOffsets.push(currentOffset);
        currentOffset += wordLength * letterSpacing + wordSpacing;
      });

      // 문장의 모든 글자에 대해 출발점 설정
      sentence.forEach((g, i) => {
        const finalState = finalStates[layers.indexOf(g)];
        // 해당 글자가 속한 단어와 단어 내 위치 찾기
        let charPos = 0;
        let wordIndex = 0;
        for (let j = 0; j < wordLengths.length; j++) {
          if (i < charPos + wordLengths[j]) {
            wordIndex = j;
            break;
          }
          charPos += wordLengths[j];
        }
        const charInWord = i - charPos;
        const xOffset = startX + wordOffsets[wordIndex] + charInWord * letterSpacing;
        g.removeAttribute("data-svg-origin");
        delete g.dataset.svgOrigin;
        if (g._gsap) {
          g._gsap.xOrigin = 0;
          g._gsap.yOrigin = 0;
          g._gsap.origin = "0 0";
        }
        gsap.set(g, {
          x: offsetAxis === 'x' ? xOffset : finalState.x,
          y: offsetAxis === 'y' ? xOffset : yBase,
          scale: scale,
          rotation: 0,
          fill: "#000000",
          "fill-opacity": 1,
          opacity: 0
        });
      });

      // 문장의 모든 글자에 대해 페이드인 애니메이션
      gsap.to(sentence, {
        opacity: 1,
        duration: fadeDur,
        ease: "none",
        stagger: 0.05, // 왼쪽 글자부터 순차적으로 페이드인
        onComplete: () => {
          isFadedIn = true; // 페이드인 완료
        }
      });
    }
  }

  function triggerMove() {
    if (currentSentenceIndex < sentences.length && isFadedIn) {
      const sentence = sentences[currentSentenceIndex];
      // 페이드인 완료 후 각 글자를 HTML의 원래 상태로 이동
      sentence.forEach((g, i) => {
        const finalState = finalStates[layers.indexOf(g)];
        gsap.to(g, {
          duration: 1,
          x: finalState.x,
          y: finalState.y,
          scale: finalState.scale,
          rotation: finalState.rotation,
          fill: finalState.fill,
          "fill-opacity": finalState.fillOpacity,
          ease: "back.out(1.4)",
          delay: i * 0.05 // 이동 애니메이션도 순차적으로
        });
      });
      currentSentenceIndex++; // 다음 문장으로 이동
      isFadedIn = false; // 다음 문장을 위해 초기화
    }
  }

  // 키보드 이벤트 리스너
  document.addEventListener('keydown', (event) => {
    if (event.key === 'f') {
      triggerFadeIn(); // 'f' 키로 페이드인
    } else if (event.key === 'm') {
      triggerMove(); // 'm' 키로 이동
    }
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