import svgwrite
from fontTools.ttLib import TTFont
from pathlib import Path
import os
import unicodedata
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

    def text_to_svg(self, text, font_size=72, position=(10, 100), margin=10):
        # Load font using fontTools (for validation, not rendering)
        font = TTFont(self.font_path)

        # Extract font name for SVG usage
        font_name = font['name'].getName(1, 3, 1).toStr()
        
        # Estimate SVG width and height based on text
        text_width = self.estimate_text_width(text, font_size)
        svg_width = text_width + margin
        svg_height = int(font_size * 1.5) + margin # 여유를 둔 세로 높이

        # Create SVG drawing
        self.output_svg_path = os.path.join(self.output_svg_path, f"_{text}.svg")    
        dwg = svgwrite.Drawing(self.output_svg_path, size=(f"{svg_width}px", f"{svg_height}px"))
        
        # Embed font via @font-face
        font_face = f"""
        @font-face {{
            font-family: '{font_name}';
            src: url('{Path(self.font_path).name}');
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

        print(f"SVG saved to: {self.output_svg_path}")
        return self.output_svg_path

# 예시 사용법
if __name__ == "__main__":
    text = "MarryMe"
    font_name = "MaruBuri-Bold.otf"
    font_parser = FontParser(font_name)
    font_parser.text_to_svg(text)
