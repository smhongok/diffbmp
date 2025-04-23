import svgwrite
from fontTools.ttLib import TTFont
from fontTools.subset import Subsetter, Options
from pathlib import Path
import os
import unicodedata
import base64
import io
base_folder = Path(__file__).resolve().parent.parent
font_folder = os.path.join(base_folder, "assets", "font") # .ttf 또는 .otf 경로
svg_folder = os.path.join(base_folder, "assets", "svg") # svg 경로

class FontParser:
    def __init__(self, font_name):
        self.font_path = font_name
        self.font_path = os.path.join(font_folder, font_name)
        self.output_svg_path = os.path.join(svg_folder, os.path.splitext(font_name)[0])
        self.font = TTFont(self.font_path)
            
    def estimate_text_width(self, text, font_size):
        """
        Text width estimation, giving different factor to the character type to estimate the text width
        """
        total_width = 0.0
        for ch in text:
            if ch == ' ':
                factor = 0.35
            elif unicodedata.category(ch).startswith('P'):  # 특수문자
                factor = 0.5
            elif '0' <= ch <= '9':
                factor = 0.55
            elif '\uAC00' <= ch <= '\uD7AF':  # 한글
                factor = 0.6
            elif ch.isalpha():
                factor = 0.6
            else:
                factor = 0.6  # 기본값
            total_width += font_size * factor
        return int(total_width)

    def subset_font_data(self, text):
        opts = Options()
        opts.flavor = None       # 유지할 포맷(w/o woff 등)
        opts.with_zopfli = False # zopfli 압축은 느리므로 건너뛰기
        subsetter = Subsetter(options=opts)
        subsetter.populate(text=text)
        subsetter.subset(self.font)

        buf = io.BytesIO()
        self.font.save(buf)
        return buf.getvalue()

    def text_to_svg(self, text, font_size=72, position=(10, 100), margin=10):
        # 2) 폰트 파일 읽어서 Base64 인코딩
        subset_data = self.subset_font_data(text)
        b64 = base64.b64encode(subset_data).decode()
        mime, fmt = 'font/truetype','truetype'  # ttf 기준. otf면 변경
            
        # Extract font name for SVG usage
        font_name = self.font['name'].getName(1, 3, 1).toStr()
        
        # Estimate SVG width and height based on text
        text_width = self.estimate_text_width(text, font_size)
        svg_width = text_width + margin*2
        svg_height = int(font_size * 1.5) + margin # 여유를 둔 세로 높이

        # Create SVG drawing
        self.output_svg_path = self.output_svg_path + f"_{text}.svg"
        dwg = svgwrite.Drawing(
            self.output_svg_path, 
            size=(f"{svg_width}px", f"{svg_height}px"),
            viewBox=f"0 0 {svg_width} {svg_height}"
        )
        
        # Embed font via @font-face
        font_face = f"""
        @font-face {{
            font-family: '{font_name}';
            src: url('data:{mime};base64,{b64}') format('{fmt}');
        }}
        """
        dwg.defs.add(dwg.style(font_face))
        
        # Add text element
        dwg.add(dwg.text(text,
                        insert=position,
                        font_size=font_size,
                        font_family=font_name,
                        fill="black"))

        # Save SVG
        dwg.save()
        size_kb = Path(self.output_svg_path).stat().st_size/1024
        print("SVG Saved to:", self.output_svg_path, "({:.1f} KB)".format(size_kb))
        return self.output_svg_path

# 예시 사용법
if __name__ == "__main__":
    text = "HELLO"
    font_name = "MaruBuri-Bold.otf"
    font_parser = FontParser(font_name)
    font_parser.text_to_svg(text)
