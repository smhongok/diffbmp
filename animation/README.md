# Animation Tools

PSD 파일을 활용한 2D 및 3D 애니메이션 도구 모음입니다.

## 도구 목록

### 2D 애니메이션
- **psd_to_mp4.py**: 기본 누적 합성 애니메이션
- **psd_flickering_mp4.py**: 깜빡임 효과 애니메이션
- **psd_parallax_mp4.py**: 시차 효과 애니메이션
- **psd_falling_mp4.py**: 낙하 효과 애니메이션

### 3D 애니메이션 (PyVista 기반)
- **psd_3d_pyvista_mp4.py**: 3D 카메라 움직임 (줌아웃 + 궤도 회전)
- **psd_3d_flickering_mp4.py**: 3D 카메라 + 레이어 순차 등장

### 유틸리티
- **psd_layer_dropout.py**: 위치 기반 확률적 dropout
- **run_all.sh**: 모든 애니메이션 일괄 실행

## 설치 요구사항

### 기본 패키지 (2D 애니메이션)
```bash
pip install psd-tools moviepy pillow numpy
```

### 3D 애니메이션 추가 패키지
```bash
pip install pyvista
```

## 기본 사용법

### 2D 애니메이션

#### 기본 PSD to MP4 변환
```bash
# 기본 변환 (모든 레이어, 24fps, 각 레이어당 1프레임)
python psd_to_mp4.py input/design.psd

# 출력 파일명 지정
python psd_to_mp4.py input/design.psd -o output/animation.mp4

# FPS와 프레임 수 설정
python psd_to_mp4.py input/design.psd --fps 30 --frames 2

# 특정 레이어 범위만 사용
python psd_to_mp4.py input/design.psd --layer-start 2 --layer-end 8
```

#### 깜빡임 애니메이션
```bash
# 기본 깜빡임 효과
python psd_flickering_mp4.py input.psd -o output_flicker.mp4

# 깜빡임 강도 조절
python psd_flickering_mp4.py input.psd --intensity 0.5 --duration 10 --fps 24
```

#### 시차 효과 애니메이션
```bash
# 원형 움직임
python psd_parallax_mp4.py input.psd -o output_parallax.mp4 --movement circular

# 심장형 움직임
python psd_parallax_mp4.py input.psd --movement cardioid --duration 8
```

#### 낙하 효과 애니메이션
```bash
# 위에서 아래로 낙하
python psd_falling_mp4.py input.psd -o output_falling.mp4 --pattern top_to_bottom

# 랜덤 낙하
python psd_falling_mp4.py input.psd --pattern random --duration 12
```

### 3D 애니메이션 (PyVista)

#### 3D 카메라 움직임
```bash
# 기본 3D 카메라 (줌아웃 + 궤도 회전)
python psd_3d_pyvista_mp4.py input.psd -o output_3d.mp4

# 카메라 거리 및 줌 비율 조절
python psd_3d_pyvista_mp4.py input.psd --distance 1500 --zoom-ratio 0.3 --duration 10

# 깊이 범위 설정
python psd_3d_pyvista_mp4.py input.psd --depth-range 100,1000 --fps 24
```

#### 3D 카메라 + 레이어 순차 등장
```bash
# 3D 카메라 + 랜덤 순서로 레이어 등장
python psd_3d_flickering_mp4.py input.psd -o output_3d_appear.mp4

# 레이어 범위 및 속도 조절
python psd_3d_flickering_mp4.py input.psd --layer-start 0 --layer-end 999 \
  --duration 20 --fps 12 --distance 1500

# 줌아웃 비율 조절 (30% 줌아웃, 70% 궤도 회전)
python psd_3d_flickering_mp4.py input.psd --zoom-ratio 0.3 --duration 15
```

### 유틸리티

#### PSD Layer Dropout
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

#### 모든 애니메이션 일괄 실행
```bash
# 기본 실행 (모든 애니메이션 생성)
./run_all.sh input.psd --out-dir ./output --fps 24

# 특정 애니메이션 옵션 지정
./run_all.sh input.psd --out-dir ./output \
  --flicker-opts "--intensity 0.3" \
  --parallax-opts "--movement cardioid --duration 8" \
  --falling-opts "--pattern top_to_bottom" \
  --3d-opts "--duration 10 --distance 1500" \
  --3d-flicker-opts "--duration 15 --fps 12"
```

## 옵션 설명

### 공통 옵션
- `--fps`: 프레임레이트 (기본값: 24)
- `--duration`: 애니메이션 길이 (초)
- `--layer-start`: 시작 레이어 인덱스 (0부터 시작)
- `--layer-end`: 끝 레이어 인덱스 (포함)
- `-v, --verbose`: 상세 정보 출력

### 2D 애니메이션 옵션
- `--frames`: 각 레이어가 보여지는 프레임 수 (psd_to_mp4.py)
- `--intensity`: 깜빡임 강도 (psd_flickering_mp4.py)
- `--movement`: 움직임 패턴 (circular, cardioid 등, psd_parallax_mp4.py)
- `--pattern`: 낙하 패턴 (top_to_bottom, random 등, psd_falling_mp4.py)

### 3D 애니메이션 옵션
- `--distance`: 초기 카메라 거리 (기본값: 1500)
- `--zoom-ratio`: 줌아웃 단계 비율 (기본값: 0.3, 30% 줌아웃 + 70% 궤도)
- `--depth-range`: 레이어 깊이 범위 (기본값: 100,1000)
- `--flicker-intensity`: 깜빡임 강도 (psd_3d_flickering_mp4.py, 사용 안 함)

### PSD Layer Dropout 옵션
- `--mode`: dropout 모드 (`linear` 또는 `radial`, 기본값: linear)
- `--axis`: linear 모드의 축 (`x`, `y`, `both`, 기본값: x)
- `--direction`: dropout 방향 (`positive`, `negative`, `both`, 기본값: positive)
- `--strength`: 최대 dropout 확률 (0.0-1.0, 기본값: 0.5)
- `--ellipse-ratio`: 방사형 모드의 가로/세로 비율 (기본값: 1.0)
- `--seed`: 재현 가능한 결과를 위한 랜덤 시드

### Dropout 패턴
**Linear 모드:**
- `axis=x, direction=positive`: 왼쪽 안전, 오른쪽으로 dropout 증가
- `axis=x, direction=negative`: 오른쪽 안전, 왼쪽으로 dropout 증가
- `axis=x, direction=both`: 중앙 안전, 양쪽 끝으로 dropout 증가

**Radial 모드:**
- 중심에서 가장자리로 갈수록 dropout 증가
- `ellipse-ratio > 1.0`: 가로로 넓은 타원형

## 애니메이션 방식

### 2D 애니메이션
**누적 합성(Cumulative Compositing)** 방식:
1. 흰색 배경으로 시작
2. 첫 번째 레이어를 배경에 합성 → 첫 번째 프레임
3. 두 번째 레이어를 이전 결과에 합성 → 두 번째 프레임
4. 이 과정을 반복하여 최종 이미지 완성

### 3D 애니메이션 (PyVista)
**3D 공간 배치 + 카메라 움직임**:
1. 각 레이어를 3D 공간에 깊이별로 배치 (Z축)
2. 카메라가 줌아웃하며 전체 구조 드러냄
3. 카메라가 레이어 중심을 기준으로 궤도 회전
4. 레이어들이 랜덤 순서로 나타남 (psd_3d_flickering_mp4.py)

**주요 특징**:
- 레이어 Z 좌표 중앙값을 기준으로 궤도 회전
- 궤도 반지름이 Z 중심 이동에 따라 자동 조정
- 시작점은 동일하게 유지하면서 더 큰 궤도로 회전

## 예시

### 2D 애니메이션
```bash
# 레이어 3-7번만 사용하여 30fps로 애니메이션 생성
python psd_to_mp4.py input/artwork.psd --layer-start 3 --layer-end 7 --fps 30

# 각 레이어를 2프레임씩 보여주는 애니메이션
python psd_to_mp4.py input/design.psd --frames 2 --fps 24

# 깜빡임 효과로 10초 애니메이션
python psd_flickering_mp4.py input.psd --duration 10 --intensity 0.4
```

### 3D 애니메이션
```bash
# 1000개 레이어로 20초 3D 애니메이션 (12fps)
python psd_3d_flickering_mp4.py input.psd --layer-start 0 --layer-end 999 \
  --duration 20 --fps 12 --distance 1500 -v

# 빠른 테스트 (10초, 2fps)
python psd_3d_flickering_mp4.py input.psd --layer-start 0 --layer-end 999 \
  --duration 10 --fps 2 -v

# 카메라 거리 및 줌 비율 조절
python psd_3d_pyvista_mp4.py input.psd --distance 2000 --zoom-ratio 0.4 \
  --duration 15 --fps 24
```

## 디렉토리 구조

```
animation/
├── input/                      # PSD 파일들을 여기에 배치
├── output/                     # 생성된 MP4 파일들이 저장됨
├── psd_to_mp4.py              # 기본 누적 합성 애니메이션
├── psd_flickering_mp4.py      # 깜빡임 효과 애니메이션
├── psd_parallax_mp4.py        # 시차 효과 애니메이션
├── psd_falling_mp4.py         # 낙하 효과 애니메이션
├── psd_3d_pyvista_mp4.py      # 3D 카메라 애니메이션
├── psd_3d_flickering_mp4.py   # 3D 카메라 + 레이어 등장
├── psd_layer_dropout.py       # 레이어 dropout 도구
├── run_all.sh                 # 모든 애니메이션 일괄 실행
└── README.md                  # 이 파일
```

## 출력

### 2D 애니메이션
- 입력: `input/design.psd` → 출력: `output/design.mp4` (기본값)
- 코덱: H.264 (libx264)
- 오디오: 없음
- 배경: 흰색
- 레이어 블렌딩: Alpha over 합성

### 3D 애니메이션
- 입력: `input/design.psd` → 출력: `output/design_3d.mp4` (기본값)
- 렌더러: PyVista (VTK 기반)
- 배경: 흰색
- 3D 효과: 깊이별 레이어 배치 + 카메라 움직임
- 카메라 움직임: 줌아웃 → 궤도 회전 (레이어 Z 중앙값 기준)

## 성능 팁

### 3D 애니메이션
- **렌더링 속도**: PyVista는 CPU 기반이므로 많은 레이어와 높은 FPS는 느릴 수 있습니다
- **권장 설정**: 1000 레이어 기준 fps=2-12 정도가 적당
- **테스트**: 먼저 낮은 fps(2-3)로 테스트 후 원하는 fps로 증가
- **레이어 수**: `--layer-start`와 `--layer-end`로 일부만 사용하여 테스트
