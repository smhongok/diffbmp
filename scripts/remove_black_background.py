#!/usr/bin/env python3
"""
검정색 배경을 투명하게 만드는 스크립트
지구 이미지처럼 검정 배경이 있는 이미지를 투명 배경 PNG로 변환
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def remove_black_background(input_path, output_path, threshold=30):
    """
    검정색 배경을 투명하게 제거
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 PNG 경로
        threshold: 검정색 판단 임계값 (0-255, 낮을수록 엄격)
    """
    # 이미지 읽기
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    
    # Alpha channel 생성
    # 모든 RGB 값이 threshold 이하면 투명하게
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # 검정색 판단: 모든 채널이 threshold 이하
    is_black = (r <= threshold) & (g <= threshold) & (b <= threshold)
    
    # Alpha channel: 검정색이면 0 (투명), 아니면 255 (불투명)
    alpha = np.where(is_black, 0, 255).astype(np.uint8)
    
    # RGBA 이미지 생성
    rgba_array = np.dstack((img_array, alpha))
    rgba_img = Image.fromarray(rgba_array, 'RGBA')
    
    # PNG로 저장
    rgba_img.save(output_path, 'PNG')
    
    # 통계 출력
    total_pixels = alpha.size
    transparent_pixels = np.sum(is_black)
    transparent_percent = (transparent_pixels / total_pixels) * 100
    
    print(f"  투명 픽셀: {transparent_pixels:,} / {total_pixels:,} ({transparent_percent:.1f}%)")
    
    return True


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python remove_black_background.py <input_image> [output_image] [threshold]")
        print("예제: python remove_black_background.py earth.jpg earth_no_bg.png 30")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # 출력 경로 결정
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # 입력 파일명에 _no_bg 추가
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_no_bg.png"
    
    # 임계값 설정
    threshold = int(sys.argv[3]) if len(sys.argv) >= 4 else 30
    
    if not os.path.exists(input_path):
        print(f"Error: 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    print(f"입력 파일: {input_path}")
    print(f"출력 파일: {output_path}")
    print(f"검정색 임계값: {threshold}")
    print()
    
    try:
        remove_black_background(input_path, output_path, threshold)
        print(f"\n✓ 완료: {output_path}")
    except Exception as e:
        print(f"\n✗ 오류: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
