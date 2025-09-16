# PSD to MP4 Animation Converter

PSD 파일의 레이어들을 MP4 애니메이션으로 변환하는 도구입니다. 각 레이어가 누적적으로 합성되어 최종 이미지가 완성되는 과정을 애니메이션으로 보여줍니다.

## 설치 요구사항

```bash
pip install psd-tools moviepy pillow numpy
```

## 기본 사용법

```bash
# 기본 변환 (모든 레이어, 24fps, 각 레이어당 1프레임)
python psd_to_mp4.py input/design.psd

# 출력 파일명 지정
python psd_to_mp4.py input/design.psd -o output/animation.mp4

# FPS와 프레임 수 설정
python psd_to_mp4.py input/design.psd --fps 30 --frames 2

# 특정 레이어 범위만 사용
python psd_to_mp4.py input/design.psd --layer-start 2 --layer-end 8

# 상세 정보 출력
python psd_to_mp4.py input/design.psd -v
```

## 옵션 설명

- `--fps`: 프레임레이트 (기본값: 24)
- `--frames`: 각 레이어가 보여지는 프레임 수 (기본값: 1)
- `--layer-start`: 시작 레이어 인덱스 (0부터 시작)
- `--layer-end`: 끝 레이어 인덱스 (포함)
- `-v, --verbose`: 상세 정보 출력

## 애니메이션 방식

이 도구는 **누적 합성(Cumulative Compositing)** 방식을 사용합니다:

1. 흰색 배경으로 시작
2. 첫 번째 레이어를 배경에 합성 → 첫 번째 프레임
3. 두 번째 레이어를 이전 결과에 합성 → 두 번째 프레임
4. 이 과정을 반복하여 최종 이미지 완성

각 레이어의 위치, 크기, 투명도 정보가 정확히 보존됩니다.

## 예시

```bash
# 레이어 3-7번만 사용하여 30fps로 애니메이션 생성
python psd_to_mp4.py input/artwork.psd --layer-start 3 --layer-end 7 --fps 30

# 각 레이어를 2프레임씩 보여주는 애니메이션 (24fps에서 약 0.083초씩)
python psd_to_mp4.py input/design.psd --frames 2 --fps 24

# 빠른 애니메이션 (60fps, 각 레이어당 1프레임)
python psd_to_mp4.py input/layers.psd --fps 60 --frames 1
```

## 디렉토리 구조

```
animation/
├── input/          # PSD 파일들을 여기에 배치
├── output/         # 생성된 MP4 파일들이 저장됨
├── psd_to_mp4.py   # 메인 스크립트
└── README.md       # 이 파일
```

## 출력

- 입력: `input/design.psd` → 출력: `output/design.mp4` (기본값)
- 코덱: H.264 (libx264)
- 오디오: 없음
- 배경: 흰색
- 레이어 블렌딩: Alpha over 합성
