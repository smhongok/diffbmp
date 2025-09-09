# PSD Export CUDA 커널 구현 계획서

## 📋 개요

이 문서는 circle_art 프로젝트에서 효율적인 PSD 레이어 내보내기를 위한 새로운 FP16 CUDA 커널 구현 계획을 다룹니다. 목표는 현재의 비효율적인 PyTorch 기반 접근법을 고도로 최적화된 primitive-parallel CUDA 구현으로 대체하는 것입니다.

## 🎯 동기 및 목표

### 현재 문제점
- **성능 병목**: 현재 PSD 내보내기는 Python 루프에서 PyTorch `F.grid_sample()`을 사용하여 심각한 성능 저하 발생
- **메모리 비효율성**: 개별 primitive 렌더링으로 불필요한 중간 텐서 생성
- **타입 불일치 문제**: CUDA 커널과 PyTorch 연산 간 Float16/Float32 호환성 문제
- **순차 처리**: GPU 병렬성을 활용하지 못하고 primitive를 하나씩 처리

### 목표
1. **성능**: PSD 내보내기 작업에서 10-50배 속도 향상
2. **메모리 효율성**: FP16 정밀도 사용으로 50% 메모리 절약
3. **확장성**: 고해상도 캔버스(2048x2048+)에서 2000+ primitive 지원
4. **정확성**: 기존 렌더러 출력과 픽셀 단위 일치 유지
5. **통합성**: 기존 코드베이스와 원활한 통합, 호환성 유지

## 🏗️ 아키텍처 설계

### 고수준 아키텍처
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PSD 내보내기   │    │  새로운 CUDA     │    │  최종 PNG       │
│   (새 커널)     │    │  커널            │    │  (기존 렌더러)   │
│                 │    │  (Primitive-     │    │                 │
│  개별 레이어     │    │   Parallel)      │    │  Porter-Duff    │
│                 │    │                  │    │  합성           │
│                 │    │  합성 없음       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 커널 설계 철학
- **Primitive-Parallel**: 각 primitive를 독립적인 스레드 블록으로 처리
- **2D 스레드 그리드**: 각 primitive의 bounding box 내에서 픽셀 수준 병렬성
- **Forward-Only**: 역전파 없음 (내보내기 전용 작업)
- **직접 출력**: [H,W,4] RGBA 텐서 직접 생성

## 🔧 기술적 구현

### 1. 커널 아키텍처

#### 2단계 커널 구조 (메모리 효율적 크롭 출력)
```cpp
// 1단계: Bounding box 계산 커널
__global__ void compute_bounding_boxes_kernel(
    const float* means2D,            // [N, 2] - primitive 위치
    const float* radii,              // [N] - primitive 크기  
    const float* rotations,          // [N] - primitive 회전
    const float* primitive_templates, // [P, H_t, W_t] - 블러되지 않은 템플릿
    const int* global_bmp_sel,       // [N] - 템플릿 선택 인덱스
    int* output_bounding_boxes,      // [N, 4] - 계산된 bounding box
    int* cropped_sizes,              // [N, 2] - 각 primitive의 크롭된 크기 (width, height)
    PSDExportConfig config
);

// 2단계: 크롭된 레이어 생성 커널
__global__ void generate_cropped_layers_kernel(
    const float* means2D, const float* radii, const float* rotations,
    const float* colors, const float* visibility,
    const float* primitive_templates, const int* global_bmp_sel,
    const int* bounding_boxes,       // [N, 4] - 1단계에서 계산된 bbox
    const int* layer_offsets,        // [N] - 각 레이어의 출력 버퍼 오프셋
    uint8_t* cropped_output_buffer,  // 연속된 크롭 데이터 버퍼
    PSDExportConfig config
);
```

#### 2단계 실행 구성 (메모리 효율적)
```cpp
// 1단계: Bounding box 계산
dim3 bbox_grid((N_primitives + 255) / 256);
dim3 bbox_block(256);
compute_bounding_boxes_kernel<<<bbox_grid, bbox_block>>>(
    means2D, radii, rotations, primitive_templates, global_bmp_sel,
    bounding_boxes, cropped_sizes, config
);
cudaDeviceSynchronize();

// CPU에서 총 메모리 요구량 계산 및 오프셋 설정
int* h_cropped_sizes = new int[N * 2];
cudaMemcpy(h_cropped_sizes, cropped_sizes, N * 2 * sizeof(int), cudaMemcpyDeviceToHost);

size_t total_pixels = 0;
int* h_layer_offsets = new int[N];
for (int i = 0; i < N; i++) {
    h_layer_offsets[i] = total_pixels;
    total_pixels += h_cropped_sizes[i * 2] * h_cropped_sizes[i * 2 + 1] * 4; // RGBA
}

// 크롭된 출력 버퍼 할당 (전체 캔버스 대신 필요한 만큼만)
uint8_t* cropped_output_buffer;
cudaMalloc(&cropped_output_buffer, total_pixels * sizeof(uint8_t));

int* d_layer_offsets;
cudaMalloc(&d_layer_offsets, N * sizeof(int));
cudaMemcpy(d_layer_offsets, h_layer_offsets, N * sizeof(int), cudaMemcpyHostToDevice);

// 2단계: 크롭된 레이어 생성
dim3 layer_grid(N_primitives, 16, 16);
dim3 layer_block(16, 16);
generate_cropped_layers_kernel<<<layer_grid, layer_block>>>(
    means2D, radii, rotations, colors, visibility,
    primitive_templates, global_bmp_sel, bounding_boxes,
    d_layer_offsets, cropped_output_buffer, config
);
```

### 2. 핵심 구성 요소

#### A. 1단계: Bounding Box 계산 커널
```cpp
__global__ void compute_bounding_boxes_kernel(
    const float* means2D, const float* radii, const float* rotations,
    const float* primitive_templates, const int* global_bmp_sel,
    int* output_bounding_boxes, int* cropped_sizes,
    PSDExportConfig config
) {
    int prim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (prim_id >= config.N) return;
    
    // 이 primitive의 파라미터
    float x = means2D[prim_id * 2 + 0] * config.scale_factor;
    float y = means2D[prim_id * 2 + 1] * config.scale_factor;
    float r = radii[prim_id] * config.scale_factor;
    
    // 보수적 bounding box 계산 (회전 고려)
    int min_x = max(0, (int)floorf(x - r));
    int min_y = max(0, (int)floorf(y - r));
    int max_x = min(config.W, (int)ceilf(x + r));
    int max_y = min(config.H, (int)ceilf(y + r));
    
    // 유효한 영역이 있는지 확인
    if (max_x <= min_x || max_y <= min_y) {
        // 빈 bounding box
        output_bounding_boxes[prim_id * 4 + 0] = 0;
        output_bounding_boxes[prim_id * 4 + 1] = 0;
        output_bounding_boxes[prim_id * 4 + 2] = 1;
        output_bounding_boxes[prim_id * 4 + 3] = 1;
        cropped_sizes[prim_id * 2 + 0] = 1;
        cropped_sizes[prim_id * 2 + 1] = 1;
    } else {
        output_bounding_boxes[prim_id * 4 + 0] = min_x;
        output_bounding_boxes[prim_id * 4 + 1] = min_y;
        output_bounding_boxes[prim_id * 4 + 2] = max_x;
        output_bounding_boxes[prim_id * 4 + 3] = max_y;
        cropped_sizes[prim_id * 2 + 0] = max_x - min_x;
        cropped_sizes[prim_id * 2 + 1] = max_y - min_y;
    }
}
```

#### B. 2단계: 크롭된 레이어 생성 커널
```cpp
__global__ void generate_cropped_layers_kernel(
    const float* means2D, const float* radii, const float* rotations,
    const float* colors, const float* visibility,
    const float* primitive_templates, const int* global_bmp_sel,
    const int* bounding_boxes, const int* layer_offsets,
    uint8_t* cropped_output_buffer, PSDExportConfig config
) {
    int prim_id = blockIdx.x;
    int local_y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (prim_id >= config.N) return;
    
    // 이 primitive의 bounding box
    int bbox_left = bounding_boxes[prim_id * 4 + 0];
    int bbox_top = bounding_boxes[prim_id * 4 + 1];
    int bbox_right = bounding_boxes[prim_id * 4 + 2];
    int bbox_bottom = bounding_boxes[prim_id * 4 + 3];
    int bbox_w = bbox_right - bbox_left;
    int bbox_h = bbox_bottom - bbox_top;
    
    if (local_x >= bbox_w || local_y >= bbox_h) return;
    
    // 실제 캔버스 좌표
    int px = bbox_left + local_x;
    int py = bbox_top + local_y;
    
    // primitive 파라미터
    float x = means2D[prim_id * 2 + 0] * config.scale_factor;
    float y = means2D[prim_id * 2 + 1] * config.scale_factor;
    float r = radii[prim_id] * config.scale_factor;
    float theta = rotations[prim_id];
    
    float3 rgb = make_float3(colors[prim_id * 3 + 0], 
                            colors[prim_id * 3 + 1], 
                            colors[prim_id * 3 + 2]);
    float vis = visibility[prim_id];
    
    // 그리드 샘플링
    float template_val = cuda_grid_sample_bilinear(
        primitive_templates, global_bmp_sel[prim_id],
        px, py, x, y, r, theta, config
    );
    
    // uint8 RGBA 계산
    uint8_t final_r = (uint8_t)(template_val * rgb.x * 255.0f);
    uint8_t final_g = (uint8_t)(template_val * rgb.y * 255.0f);
    uint8_t final_b = (uint8_t)(template_val * rgb.z * 255.0f);
    uint8_t final_a = (uint8_t)(template_val * vis * 255.0f);
    
    // 크롭된 버퍼에 저장
    int buffer_offset = layer_offsets[prim_id];
    int pixel_offset = (local_y * bbox_w + local_x) * 4;
    cropped_output_buffer[buffer_offset + pixel_offset + 0] = final_r;
    cropped_output_buffer[buffer_offset + pixel_offset + 1] = final_g;
    cropped_output_buffer[buffer_offset + pixel_offset + 2] = final_b;
    cropped_output_buffer[buffer_offset + pixel_offset + 3] = final_a;
}
```

#### C. 최적화된 그리드 샘플링 (float32)
```cpp
__device__ float cuda_grid_sample_bilinear(
    const float* template_data, int template_idx,
    int px, int py, float x, float y, float r, float theta,
    PSDExportConfig config
) {
    // 좌표를 [-1, 1]로 정규화 (기존 PyTorch 방식과 동일)
    float u_norm = ((float)px - x) / r;
    float v_norm = ((float)py - y) / r;
    
    // 역회전 적용
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float u_rot = cos_t * u_norm + sin_t * v_norm;
    float v_rot = -sin_t * u_norm + cos_t * v_norm;
    
    // 템플릿 좌표로 변환 [0, W_t-1] x [0, H_t-1]
    int H_t = config.template_height;
    int W_t = config.template_width;
    float u_template = (u_rot + 1.0f) * 0.5f * (W_t - 1);
    float v_template = (v_rot + 1.0f) * 0.5f * (H_t - 1);
    
    // 경계 검사
    if (u_template < 0.0f || u_template >= W_t - 1 ||
        v_template < 0.0f || v_template >= H_t - 1) {
        return 0.0f;
    }
    
    // 이중선형 보간
    int u0 = (int)floorf(u_template);
    int v0 = (int)floorf(v_template);
    int u1 = u0 + 1;
    int v1 = v0 + 1;
    
    float wu = u_template - u0;
    float wv = v_template - v0;
    float wu_inv = 1.0f - wu;
    float wv_inv = 1.0f - wv;
    
    // 템플릿 데이터 인덱싱
    int base_idx = template_idx * H_t * W_t;
    float val00 = template_data[base_idx + v0 * W_t + u0];
    float val01 = template_data[base_idx + v0 * W_t + u1];
    float val10 = template_data[base_idx + v1 * W_t + u0];
    float val11 = template_data[base_idx + v1 * W_t + u1];
    
    // 이중선형 보간 결과
    return wu_inv * wv_inv * val00 + wu * wv_inv * val01 +
           wu_inv * wv * val10 + wu * wv * val11;
}
```

### 3. 파일 구조

```
cuda_tile_rasterizer/
├── cuda_psd_export/
│   ├── psd_export_uint8.h          # 헤더 정의
│   ├── psd_export_uint8.cu         # 통합 커널 구현
│   └── psd_common.h                # 공통 구조체 및 유틸리티
├── ext_psd_export.cpp              # Python 바인딩
├── setup_psd_export.py             # 빌드 구성
└── __init__.py                     # Python 인터페이스
```

### 4. Python 통합

#### Python 래퍼 (크롭된 출력)
```python
# cuda_tile_rasterizer/psd_export_uint8.py
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from . import _C_psd_export

def export_psd_layers_cuda_uint8(
    renderer_S: torch.Tensor,        # [P, H_t, W_t] primitive 템플릿
    x: torch.Tensor,                 # [N] x 위치
    y: torch.Tensor,                 # [N] y 위치
    r: torch.Tensor,                 # [N] 크기
    theta: torch.Tensor,             # [N] 회전
    v: torch.Tensor,                 # [N] 가시성 (sigmoid 적용됨)
    c: torch.Tensor,                 # [N, 3] 색상 (sigmoid 적용됨)
    global_bmp_sel: torch.Tensor,    # [N] 템플릿 선택 인덱스
    H: int, W: int,                  # 캔버스 크기
    scale_factor: float = 1.0        # 내보내기 스케일링
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    2단계 CUDA 커널을 사용하여 크롭된 primitive 레이어를 PIL 이미지로 직접 내보내기.
    메모리 효율적: 전체 캔버스 대신 필요한 크롭 영역만 할당.
    
    반환값:
        (PIL 이미지 리스트, bounding box 리스트)
    """
    N = len(x)
    device = x.device
    
    # 입력을 float32로 변환
    x_f32 = x.float()
    y_f32 = y.float()
    r_f32 = r.float()
    theta_f32 = theta.float()
    v_f32 = torch.sigmoid(v).float()
    c_f32 = torch.sigmoid(c).float()
    renderer_S_f32 = renderer_S.float()
    
    # 1단계: Bounding box 계산
    bounding_boxes = torch.zeros(N, 4, dtype=torch.int32, device=device)
    cropped_sizes = torch.zeros(N, 2, dtype=torch.int32, device=device)
    
    _C_psd_export.compute_bounding_boxes(
        x_f32, y_f32, r_f32, theta_f32, renderer_S_f32, global_bmp_sel.int(),
        bounding_boxes, cropped_sizes, H, W, scale_factor
    )
    
    # CPU에서 총 메모리 요구량 계산
    h_cropped_sizes = cropped_sizes.cpu().numpy()
    total_pixels = 0
    layer_offsets = []
    
    for i in range(N):
        layer_offsets.append(total_pixels)
        w, h = h_cropped_sizes[i]
        total_pixels += w * h * 4  # RGBA
    
    # 크롭된 출력 버퍼 할당 (전체 캔버스 대신 필요한 만큼만)
    cropped_buffer = torch.zeros(total_pixels, dtype=torch.uint8, device=device)
    layer_offsets_tensor = torch.tensor(layer_offsets, dtype=torch.int32, device=device)
    
    # 2단계: 크롭된 레이어 생성
    _C_psd_export.generate_cropped_layers(
        x_f32, y_f32, r_f32, theta_f32, v_f32, c_f32,
        renderer_S_f32, global_bmp_sel.int(), bounding_boxes,
        layer_offsets_tensor, cropped_buffer, H, W, scale_factor
    )
    
    # PIL 이미지 리스트로 변환
    pil_images = []
    bounds_list = []
    h_bounding_boxes = bounding_boxes.cpu().numpy()
    h_cropped_buffer = cropped_buffer.cpu().numpy()
    
    for i in range(N):
        left, top, right, bottom = h_bounding_boxes[i]
        w, h = h_cropped_sizes[i]
        
        if w <= 1 or h <= 1:  # 빈 레이어
            pil_images.append(Image.fromarray(np.zeros((1, 1, 4), dtype=np.uint8), 'RGBA'))
            bounds_list.append((0, 0, 1, 1))
        else:
            # 크롭된 데이터 추출
            offset = layer_offsets[i]
            layer_data = h_cropped_buffer[offset:offset + w * h * 4]
            rgba_array = layer_data.reshape(h, w, 4)
            
            pil_images.append(Image.fromarray(rgba_array, 'RGBA'))
            bounds_list.append((left, top, right, bottom))
    
    return pil_images, bounds_list
```

#### PSDExporter와의 통합
```python
# util/psd_exporter.py
def add_layers_batch_optimized(self, primitive_templates, x_tensor, y_tensor, 
                              r_tensor, theta_tensor, v_tensor, c_tensor, names=None):
    """현재 구현을 통합 CUDA 커널 호출로 완전 대체."""
    
    try:
        # 새로운 uint8 CUDA 커널 사용 - 한 번의 호출로 모든 작업 완료
        from cuda_tile_rasterizer.psd_export_uint8 import export_psd_layers_cuda_uint8
        
        # global_bmp_sel 생성 (기존 로직과 동일)
        N = len(x_tensor)
        if primitive_templates.dim() == 2:
            global_bmp_sel = torch.zeros(N, dtype=torch.int32, device=x_tensor.device)
        else:
            p = primitive_templates.shape[0]
            global_bmp_sel = torch.arange(N, device=x_tensor.device) % p
        
        # 단일 CUDA 커널 호출로 PIL 이미지와 bounding box 직접 생성
        pil_images, bounds_list = export_psd_layers_cuda_uint8(
            primitive_templates, x_tensor, y_tensor, r_tensor, theta_tensor,
            v_tensor, c_tensor, global_bmp_sel,
            self.export_height, self.export_width, self.scale_factor
        )
        
        # PSD 레이어로 직접 추가 (추가 변환 없음)
        for i in range(N):
            layer_name = names[i] if names else f"primitive_{i}"
            left, top, right, bottom = bounds_list[i]
            layer = PixelLayer.frompil(pil_images[i], self.psd, layer_name, top=top, left=left)
            self.psd.append(layer)
            
    except (ImportError, RuntimeError) as e:
        # CUDA 커널 실패 시 기존 PyTorch 구현으로 폴백
        print(f"CUDA 커널 실패, PyTorch 폴백 사용: {e}")
        self._add_layers_batch_pytorch_fallback(
            primitive_templates, x_tensor, y_tensor, r_tensor, 
            theta_tensor, v_tensor, c_tensor, names
        )
```

## 📊 성능 분석

### 예상 성능 향상

#### 메모리 사용량 비교
```
현재 (PyTorch 분리된 처리):
- _apply_transformations_batch: O(N × H × W × 4 bytes) = 4N×H×W bytes (float32)
- _batch_templates_to_pil_with_bounds: 추가 numpy 변환 및 PIL 생성
- 중간 텐서: 2-3배 추가 오버헤드
- 총 메모리: ~12-16N×H×W bytes

새로운 방식 (크롭된 uint8 CUDA):
- 크롭된 영역만 출력: O(Σ(bbox_w × bbox_h) × 4 bytes)
- 전체 캔버스 대신 실제 사용 영역만 할당
- 일반적으로 95-99% 메모리 절약 (primitive 크기에 따라)
- 총 메모리: ~0.01-0.05N×H×W bytes (95-99% 절약)
```

#### 속도 비교
```
테스트 케이스: 2000 primitive, 2048×2048 캔버스

현재 구현:
- _apply_transformations_batch: 배치 F.grid_sample (개선됨)
- _batch_templates_to_pil_with_bounds: CPU numpy 변환 + PIL 생성
- 두 단계 분리 처리: GPU→CPU→PIL 변환
- 예상 시간: 10-20초

새로운 구현:  
- 단일 통합 CUDA 커널: 모든 작업 GPU에서 완료
- 직접 uint8 출력: 추가 변환 없음
- 원자적 bounding box 계산: 별도 단계 불필요
- 예상 시간: 0.5-2초 (10-40배 속도 향상)
```

### 확장성 분석
```
메모리 요구사항 (크롭된 출력, 실제 사용량):
- 1000 primitive, 평균 64x64 크롭: ~16 MB (99.6% 절약)
- 2000 primitive, 평균 128x128 크롭: ~131 MB (99.6% 절약)
- 4000 primitive, 평균 256x256 크롭: ~1 GB (98.5% 절약)

GPU 메모리 한계 (극대 메모리 절약):
- RTX 3090: 24 GB → 2048²에서 ~50,000+ primitive 처리 가능
- RTX 4090: 24 GB → 2048²에서 ~50,000+ primitive 처리 가능
- A100: 80 GB → 2048²에서 ~150,000+ primitive 처리 가능

주의: 실제 메모리 사용량은 primitive의 크기와 밀도에 따라 결정됨
```

## 🛠️ 구현 일정

### 1단계: 핵심 커널 개발 (1-2주차)
- [ ] 기본 FP16 커널 구조 구현
- [ ] Bounding box 계산 커널 생성
- [ ] FP16 이중선형 보간 구현
- [ ] 기본 그리드 샘플링 기능
- [ ] 개별 구성 요소 단위 테스트

### 2단계: 통합 및 최적화 (3주차)
- [ ] Python 바인딩 구현
- [ ] 기존 PSDExporter와 통합
- [ ] 성능 벤치마킹
- [ ] 메모리 사용량 최적화
- [ ] 오류 처리 및 예외 상황

### 3단계: 테스트 및 검증 (4주차)
- [ ] 픽셀 단위 정확도 검증
- [ ] 대규모 성능 테스트
- [ ] 메모리 누수 감지
- [ ] 크로스 플랫폼 호환성
- [ ] 문서화 및 코드 리뷰

### 4단계: 배포 (5주차)
- [ ] 최종 통합 테스트
- [ ] 성능 회귀 테스트
- [ ] 사용자 승인 테스트
- [ ] 프로덕션 배포
- [ ] 모니터링 및 최적화

## 🧪 테스트 전략

### 단위 테스트
1. **Bounding Box 정확도**: 다양한 primitive 구성에 대한 올바른 bbox 계산 검증
2. **그리드 샘플링 정밀도**: FP16 결과를 FP32 참조 구현과 비교
3. **메모리 관리**: 메모리 누수 및 적절한 정리 테스트
4. **예외 상황**: 경계 밖 좌표, 0 반지름, 극단적 회전 처리

### 통합 테스트  
1. **PSD 호환성**: 내보낸 PSD 파일이 유효하고 로드 가능한지 확인
2. **시각적 정확도**: 현재 구현과 픽셀 단위 비교
3. **성능 벤치마크**: 다양한 구성에서 속도 향상 측정
4. **메모리 사용량**: 50% 메모리 절약 주장 검증

### 스트레스 테스트
1. **대규모**: 최대 해상도에서 4000+ primitive 테스트
2. **메모리 한계**: GPU 메모리 한계 근처에서 동작 테스트
3. **동시 접근**: 여러 동시 내보내기 작업
4. **오류 복구**: CUDA 오류 및 폴백의 우아한 처리

## 🔄 하위 호환성

### 폴백 전략
```python
def add_layers_batch_optimized(self, ...):
    try:
        # 새로운 FP16 CUDA 커널 시도
        return self._cuda_fp16_export(...)
    except (ImportError, RuntimeError) as e:
        # 현재 PyTorch 구현으로 폴백
        logger.warning(f"CUDA FP16 내보내기 실패: {e}, 폴백 사용")
        return self._pytorch_export(...)
```

### 구성 옵션
```python
# config.json
{
    "postprocessing": {
        "export_psd": true,
        "psd_export_method": "cuda_fp16",  # "cuda_fp16" | "pytorch" | "auto"
        "psd_scale_factor": 1.0,
        "psd_memory_limit_gb": 20.0
    }
}
```

## 📈 성공 지표

### 성능 목표
- [ ] **속도**: 일반적인 워크로드(1000 primitive, 1024²)에서 최소 10배 속도 향상
- [ ] **메모리**: 현재 구현 대비 50% 메모리 절약
- [ ] **확장성**: 24GB GPU에서 2048² 캔버스에 2000+ primitive 지원
- [ ] **정확도**: 현재 렌더러 출력과 픽셀 단위 일치

### 품질 목표
- [ ] **신뢰성**: 메모리 누수 없음, 적절한 오류 처리
- [ ] **호환성**: CUDA 11.0+ 및 주요 GPU 아키텍처에서 작동
- [ ] **유지보수성**: 프로젝트 규약을 따르는 깔끔하고 문서화된 코드
- [ ] **사용성**: 원활한 통합, 기존 API에 대한 호환성 유지

## 🚀 향후 개선사항

### 잠재적 최적화
1. **회전 인식 Bounding Box**: 회전을 고려한 더 타이트한 bounding box
2. **템플릿 캐싱**: 반복되는 primitive에 대한 GPU 측 템플릿 캐싱
3. **멀티 GPU 지원**: 여러 GPU에 primitive 분산
4. **스트리밍**: 메모리 제한 시나리오를 위한 배치 단위 primitive 처리
5. **압축**: 대용량 레이어 출력을 위한 실시간 압축

### 확장 기능
1. **실시간 미리보기**: 최적화 중 라이브 PSD 레이어 업데이트
2. **선택적 내보내기**: 수정된 레이어만 내보내기로 증분 업데이트
3. **포맷 지원**: 추가 내보내기 포맷 (TIFF, OpenEXR 등)
4. **품질 모드**: 다양한 사용 사례를 위한 속도 vs 품질 트레이드오프

---

## 📝 참고사항

이 구현 계획은 기존 코드베이스와의 원활한 통합을 보장하면서 성능, 메모리 효율성, 유지보수성을 우선시합니다. FP16 접근법은 기존 인프라를 활용하며 대규모 primitive 처리에 상당한 이점을 제공합니다.

모듈식 설계는 점진적 개발과 테스트를 가능하게 하며, 전환 기간 동안 시스템 안정성을 보장하는 명확한 폴백 메커니즘을 제공합니다.
