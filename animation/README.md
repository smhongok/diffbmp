# Animation Tools

PSD 파일을 활용한 애니메이션 도구 모음입니다.

## PSD to MP4 Animation Converter

PSD 파일의 레이어들을 MP4 애니메이션으로 변환하는 도구입니다. 각 레이어가 누적적으로 합성되어 최종 이미지가 완성되는 과정을 애니메이션으로 보여줍니다.

## PSD Layer Dropout Tool

PSD 파일의 레이어들을 위치 기반으로 확률적 dropout을 적용하여 sparse한 효과를 만드는 도구입니다.

## 설치 요구사항

```bash
pip install psd-tools moviepy pillow numpy
```

## 기본 사용법

### PSD to MP4 변환

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

### PSD Layer Dropout

```bash
# 기본 linear dropout (x축, 왼쪽→오른쪽)
python psd_layer_dropout.py input.psd -o output.psd --strength 0.9

# 방향 설정 (positive: 왼쪽 안전, negative: 오른쪽 안전, both: 중앙 안전)
python psd_layer_dropout.py input.psd -o output.psd --axis x --direction positive --strength 0.8

# Y축 dropout (위→아래)
python psd_layer_dropout.py input.psd -o output.psd --axis y --strength 0.7

# 방사형 dropout (중심→가장자리)
python psd_layer_dropout.py input.psd -o output.psd --mode radial --strength 0.6

# 타원형 dropout (가로가 세로보다 1.5배 넓은 타원)
python psd_layer_dropout.py input.psd -o output.psd --mode radial --ellipse-ratio 1.5 --strength 0.5

# 재현 가능한 결과를 위한 시드 설정
python psd_layer_dropout.py input.psd -o output.psd --seed 42 --strength 0.9 -v
```

## 옵션 설명

### PSD to MP4 옵션

- `--fps`: 프레임레이트 (기본값: 24)
- `--frames`: 각 레이어가 보여지는 프레임 수 (기본값: 1)
- `--layer-start`: 시작 레이어 인덱스 (0부터 시작)
- `--layer-end`: 끝 레이어 인덱스 (포함)
- `-v, --verbose`: 상세 정보 출력

### PSD Layer Dropout 옵션

- `--mode`: dropout 모드 (`linear` 또는 `radial`, 기본값: linear)
- `--axis`: linear 모드의 축 (`x`, `y`, `both`, 기본값: x)
- `--direction`: dropout 방향 (`positive`, `negative`, `both`, 기본값: positive)
- `--strength`: 최대 dropout 확률 (0.0-1.0, 기본값: 0.5)
- `--ellipse-ratio`: 방사형 모드의 가로/세로 비율 (기본값: 1.0)
- `--seed`: 재현 가능한 결과를 위한 랜덤 시드
- `-v, --verbose`: 상세 정보 출력

### Dropout 패턴 설명

**Linear 모드:**
- `axis=x, direction=positive`: 왼쪽 절반 안전, 오른쪽으로 갈수록 dropout 증가
- `axis=x, direction=negative`: 오른쪽 절반 안전, 왼쪽으로 갈수록 dropout 증가
- `axis=x, direction=both`: 중앙 안전, 양쪽 끝으로 갈수록 dropout 증가
- `axis=y`: 위아래 방향으로 동일한 패턴 적용
- `axis=both`: 대각선 방향으로 패턴 적용

**Radial 모드:**
- 중심에서 가장자리로 갈수록 dropout 확률 증가
- `ellipse-ratio > 1.0`: 가로로 넓은 타원형 패턴
- `ellipse-ratio < 1.0`: 세로로 긴 타원형 패턴

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
├── input/                  # PSD 파일들을 여기에 배치
├── output/                 # 생성된 MP4 파일들과 dropout PSD 파일들이 저장됨
├── psd_to_mp4.py          # PSD to MP4 변환 스크립트
├── psd_layer_dropout.py   # PSD 레이어 dropout 도구
└── README.md              # 이 파일
```

## 출력

- 입력: `input/design.psd` → 출력: `output/design.mp4` (기본값)
- 코덱: H.264 (libx264)
- 오디오: 없음
- 배경: 흰색
- 레이어 블렌딩: Alpha over 합성
