from PIL import Image
import os
import sys

def count_gif_frames(file_path):
    """
    GIF 파일의 프레임 수를 세는 함수
    
    Args:
        file_path (str): GIF 파일 경로
    
    Returns:
        int: 프레임 수
    """
    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 파일이 GIF인지 확인
        if not file_path.lower().endswith('.gif'):
            raise ValueError("GIF 파일이 아닙니다.")
        
        # PIL로 이미지 열기
        with Image.open(file_path) as img:
            # GIF 파일인지 다시 한번 확인
            if img.format != 'GIF':
                raise ValueError("유효한 GIF 파일이 아닙니다.")
            
            # 프레임 수 세기
            frame_count = 0
            try:
                while True:
                    img.seek(frame_count)
                    frame_count += 1
            except EOFError:
                # 모든 프레임을 다 읽었을 때 발생하는 예외
                pass
            
            return frame_count
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return 0

def main():
    """
    메인 함수 - 명령줄 인수로 파일 경로를 받거나 사용자 입력을 받음
    """
    if len(sys.argv) > 1:
        # 명령줄 인수로 파일 경로가 제공된 경우
        file_path = sys.argv[1]
    else:
        # 사용자로부터 파일 경로 입력받기
        file_path = input("GIF 파일 경로를 입력하세요: ").strip()
    
    if not file_path:
        print("파일 경로가 입력되지 않았습니다.")
        return
    
    frame_count = count_gif_frames(file_path)
    
    if frame_count > 0:
        print(f"파일: {file_path}")
        print(f"프레임 수: {frame_count}")
    else:
        print("프레임 수를 세는데 실패했습니다.")

if __name__ == "__main__":
    main()