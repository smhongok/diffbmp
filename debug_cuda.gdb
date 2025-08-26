# CUDA 커널 디버깅 스크립트 (대화형 사용)
set pagination off
set confirm off

# 프로그램 시작
run

# 프로그램이 종료된 후 CUDA 정보 확인
info cuda kernels
info cuda devices
info functions tile_rasterize

# 커널 함수에 브레이크포인트 설정
break tile_rasterize_forward_kernel

printf "\n=== 브레이크포인트 설정 완료 ===\n"
printf "이제 'run' 명령으로 다시 실행하세요.\n"
printf "커널이 실행되면 브레이크포인트에서 중단됩니다.\n"