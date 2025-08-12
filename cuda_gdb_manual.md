# CUDA-GDB 수동 디버깅 가이드

## 🎯 성공적인 브레이크포인트 설정 확인됨!

CUDA-GDB가 성공적으로 `tile_rasterize_forward_kernel` 함수를 찾고 브레이크포인트를 설정했습니다.

## 🚀 수동 디버깅 단계

### 1. 대화형 CUDA-GDB 시작
```bash
source /home/sonic/anaconda3/bin/activate svgsplat
cd /home/sonic/ICL_SMH/Research_compass_aftersubmit/circle_art
cuda-gdb python test_cuda_forward.py
```

### 2. GDB 명령어 순서
```gdb
# 브레이크포인트 설정
break tile_rasterize_forward_kernel

아직 run 하기 전이니까 못찾음. future 에 찾게됨.

# 프로그램 실행
run

# 현재 브레이크포인트에서 중단된 상태에서:
(cuda-gdb) step
(cuda-gdb) step

# 몇 단계 진행 후 다시 확인
info cuda kernels
info cuda blocks  
info cuda threads

# 첫 번째 스레드로 전환
cuda thread (0,0,0)

# 변수 검사
print global_x
print global_y
print blockIdx.x
print blockIdx.y
print threadIdx.x
print threadIdx.y

# 몇 단계 실행
step 5

# 중요 변수들 확인
print num_primitives
print pixel_color[0]
print pixel_color[1] 
print pixel_color[2]
print pixel_alpha

# 첫 번째 primitive 처리 확인
step 10
print prim_id
print mean_x
print mean_y
print radius
print alpha

# 좌표 변환 확인
step 5
print dx
print dy
print norm_dx
print norm_dy
print u
print v

# Template 샘플링 확인
step 5
print template_idx
print mask_value

# 색상 처리 확인
step 5
print color_r
print color_g
print color_b

# 계속 실행
continue
```

### 3. 주요 디버깅 포인트

1. **좌표 변환**: `dx`, `dy`, `norm_dx`, `norm_dy`, `u`, `v` 값들이 PyTorch와 일치하는가?
2. **Template 샘플링**: `template_idx`, `mask_value`가 올바른가?
3. **Alpha compositing**: `alpha`, `pixel_alpha` 계산이 정확한가?
4. **색상 처리**: `color_r`, `color_g`, `color_b` 값들이 예상과 일치하는가?

### 4. 예상 결과

브레이크포인트에서 중단되면 다음과 같은 정보를 얻을 수 있습니다:
- 각 스레드의 정확한 픽셀 좌표
- Primitive 처리 순서와 값들
- 중간 계산 결과들
- 최종 픽셀 색상 값들

## 🔍 문제 진단 가능성

CUDA-GDB를 통해 다음을 확인할 수 있습니다:
1. **좌표계 문제**: 픽셀 좌표가 예상과 다른가?
2. **Template 문제**: 잘못된 template이 선택되는가?
3. **Alpha compositing 문제**: Porter-Duff 계산이 틀린가?
4. **색상 문제**: Sigmoid 변환이나 색상 처리가 틀린가?

## ⚡ 빠른 실행

브레이크포인트가 이미 설정되어 있으므로:
```bash
cuda-gdb python test_cuda_forward.py
(gdb) break tile_rasterize_forward_kernel
(gdb) run
# 브레이크포인트에서 중단됨!
```

이제 실제 커널 내부 값들을 직접 확인할 수 있습니다!
