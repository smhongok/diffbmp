import svgwrite
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.subset import Subsetter, Options
from pathlib import Path
from scour import scour
import torch
import os
import unicodedata
import base64
import io
import math,random
from PIL import Image, ImageDraw, ImageFont

base_folder = Path(__file__).resolve().parent.parent
font_folder = os.path.join(base_folder, "assets", "font") # .ttf 또는 .otf 경로
svg_folder = os.path.join(base_folder, "assets", "svg") # svg 경로

class FontParser:
    def __init__(self, font_name):
        self.font_path = Path(font_folder) / font_name
        self.svg_folder = Path(svg_folder)
        self.svg_folder.mkdir(parents=True, exist_ok=True)
        
        # Load full font for metrics and outlines
        self.font = TTFont(self.font_path)
        self.glyph_set = self.font.getGlyphSet()
        self.upem = self.font['head'].unitsPerEm
            
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

    def subset_font_data(self, text: str, flavor: str = None, zopfli: bool = False) -> bytes:
        opts = Options()
        opts.flavor = flavor       # 유지할 포맷(w/o woff 등)
        opts.with_zopfli = zopfli # zopfli 압축은 느리므로 건너뛰기
        subsetter = Subsetter(options=opts)
        subsetter.populate(text=text)
        subsetter.subset(self.font)

        buf = io.BytesIO()
        self.font.save(buf)
        return buf.getvalue()
    
    def embed_font_face(self, dwg: svgwrite.Drawing, data: bytes, mime: str, fmt: str, family: str):
        b64 = base64.b64encode(data).decode('ascii')
        css = f"""
        @font-face {{
          font-family: '{family}';
          src: url("data:{mime};base64,{b64}") format('{fmt}');
        }}
        """
        dwg.defs.add(dwg.style(css))
    
    def get_output_path(self, text: str) -> Path:
        safe = ''.join(c for c in text if c.isalnum()) or 'text'
        filename = f"{self.font_path.stem}_{safe}.svg"
        return self.svg_folder / filename

    def prepare_drawing(self, width: int, height: int, output_path: Path) -> svgwrite.Drawing:
        return svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
    def minify_svg(self, svg_path: Path):
        # SVG minification with path optimization
        opts = scour.sanitizeOptions()
        opts.remove_metadata = True
        opts.strip_comments = True
        opts.enable_viewboxing = True
        opts.shorten_ids = True
        opts.style_to_attributes = True
        opts.group_collapse = True
        opts.remove_descriptive_elements = True
        opts.convert_path_data = True
        # enable path data optimizations (absolute→relative, command merging)
        #opts.simple_colors = True  # allows path simplification
        minified = scour.scourString(svg_path.read_text(encoding='utf-8'), opts)
        svg_path.write_text(minified, encoding='utf-8')

    def text_to_svg(self, text, mode='opt-path', font_size=72, position=(0, 100), margin=15):
        """
        mode 'opt-path': optimized path (<path> elements per glyph, scour minification, SVGZ compression)
        """
        if mode == 'opt-path':
            return self.generate_path_optimized(text, font_size, position, margin)
        elif mode == 'path':
            return self.generate_path(text, font_size, position, margin)
        elif mode == 'text':
            return self.generate_text(text, font_size, position, margin)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def generate_path_optimized(self, text: str, font_size: float, position: tuple, margin: int) -> Path:
        pen = SVGPathPen(self.glyph_set)
        x, y = position
        scale = font_size / self.upem
        hhea = self.font['hhea']
        ascent = hhea.ascent * scale
        descent = hhea.descent * scale
        # Prepare SVG canvas
        width = self.estimate_text_width(text, font_size) + margin * 2 + 15*len(text)
        height = int(ascent - descent) + margin
        baseline = ascent
        out_path = self.get_output_path(text)
        dwg = self.prepare_drawing(width, height, out_path)

        # Add each glyph as its own <path> with transform
        cmap = self.font.getBestCmap()
        for ch in text:
            glyph_name = cmap.get(ord(ch))
            if not glyph_name:
                x += font_size * 0.6
                continue
            glyph = self.glyph_set[glyph_name]
            pen.path = ''
            glyph.draw(pen)
            d = pen.getCommands()
            dwg.add(
                dwg.path(
                    d=d,
                    fill='black',
                    transform=f"translate({x},{baseline}) scale({scale}, {-scale})"
                )
            )
            x += glyph.width * scale

        # Save and then minify
        dwg.save()
        self.minify_svg(out_path)

        size_kb = out_path.stat().st_size / 1024
        print(f"Saved optimized SVG: {out_path} ({size_kb:.1f} KB)")
        return out_path
    
    def generate_path(self, text: str, font_size: float, position: tuple, margin: int) -> Path:
        pen = SVGPathPen(self.glyph_set)
        x, y = position
        group = svgwrite.container.Group(fill='black')
        scale = font_size / self.upem

        cmap = self.font.getBestCmap()
        for ch in text:
            glyph_name = cmap.get(ord(ch))
            if not glyph_name:
                x += font_size * 0.6
                continue
            glyph = self.glyph_set[glyph_name]
            pen.path = ''
            glyph.draw(pen)
            d = pen.getCommands()
            transform = f"translate({x},{y}) scale({scale})"
            group.add(svgwrite.path.Path(d=d, transform=transform))
            x += glyph.width * scale

        width = self.estimate_text_width(text, font_size) + margin*2
        height = int(font_size * 1.5) + margin
        out_path = self.get_output_path(text)
        dwg = self.prepare_drawing(width, height, out_path)
        dwg.add(group)
        dwg.save()
        print(f"Saved SVG (path mode): {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
        return out_path
    
    def generate_text(self, text: str, font_size: float, position: tuple, margin: int) -> Path:
        # Subset to original font type (TTF/OTF) for broad language support
        data = self.subset_font_data(text, flavor=None, zopfli=False)
        # Determine MIME type and format from the font extension
        ext = self.font_path.suffix.lower()
        if ext == '.ttf':
            mime, fmt = 'font/truetype', 'truetype'
        elif ext == '.otf':
            mime, fmt = 'font/opentype', 'opentype'
        else:
            mime, fmt = 'font/opentype', 'opentype'
        family = 'SubsetFont'

        width = self.estimate_text_width(text, font_size) + margin*2
        height = int(font_size * 1.5) + margin
        out_path = self.get_output_path(text)
        dwg = self.prepare_drawing(width, height, out_path)

        # Embed the subsetted TTF/OTF so that all glyphs (한글·CJK 포함)가 살아납니다
        self.embed_font_face(dwg, data, mime, fmt, family)
        dwg.add(dwg.text(text,
                         insert=position,
                         font_size=font_size,
                         font_family=family,
                         fill='black'))
        dwg.save()
        print(f"Saved SVG (text mode with full subset): {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
        return out_path
    
    def generate_calligraphy_svg(self,
        text: str,
        font_path: str,
        output_path: str,
        image_size: tuple = (1024, 512),
        font_size: int = 300,
        char_spacing: int = 20,
        line_spacing: float = 1.2,
        max_angle: float = 5.0,
        max_offset: float = 10.0,
        bg_color: str = 'white',
        fg_color: str = 'black',
        mode: str = 'wrap',
        circle_params: dict = None
    ) -> Path:
        """
        주어진 텍스트로 캘리그라피 스타일 SVG를 생성합니다.

        Parameters:
            text: 변환할 문자열 (\n로 강제 개행)
            font_path: OTF/TTF 폰트 파일 경로
            output_path: 저장할 SVG 경로
            mode: 'wrap' (일반 줄바꿈) 또는 'circle' (원 경로에 fitting)
            circle_params: mode='circle'일 때 사용, {'center':(x,y), 'radius':r, 'start_angle':rad}
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        dwg = svgwrite.Drawing(
            filename=str(out),
            size=(f"{image_size[0]}px", f"{image_size[1]}px"),
            viewBox=f"0 0 {image_size[0]} {image_size[1]}"
        )
        # 배경
        dwg.add(dwg.rect(insert=(0,0), size=(image_size[0], image_size[1]), fill=bg_color))

        if mode == 'circle':
            self.add_text_on_circle(dwg, text, font_path, image_size, font_size,
                                char_spacing, fg_color, circle_params)
        else:
            self.add_text_wrap(dwg, text, font_path, image_size, font_size,
                        char_spacing, line_spacing, max_angle,
                        max_offset, fg_color)

        dwg.save()
        print(f"Calligraphy SVG saved to: {out}")
        return out


    def add_text_wrap(self, dwg, text, font_path, image_size, font_size,
                    char_spacing, line_spacing, max_angle,
                    max_offset, fg_color):
        # 줄바꿈 계산
        # 임시 렌더용 시트
        temp = svgwrite.Drawing()
        def text_width(s):
            # 단순 추정: 문자 수 * font_size * 0.6
            return len(s) * font_size * 0.6
        max_width = image_size[0] - 2 * char_spacing
        # 줄 분리
        words = text.split(' ')
        lines = []
        curr = ''
        for word in words:
            if '\n' in word:
                parts = word.split('\n')
                for i, part in enumerate(parts):
                    test = f"{curr} {part}".strip()
                    if text_width(test) <= max_width:
                        curr = test
                    else:
                        lines.append(curr)
                        curr = part
                    if i < len(parts)-1:
                        lines.append(curr)
                        curr = ''
            else:
                test = f"{curr} {word}".strip()
                if text_width(test) <= max_width:
                    curr = test
                else:
                    lines.append(curr)
                    curr = word
        if curr:
            lines.append(curr)

        # position
        y = char_spacing + font_size
        line_h = font_size * line_spacing
        for line in lines:
            x = char_spacing
            for ch in line:
                angle = random.uniform(-max_angle, max_angle)
                ox = random.uniform(-max_offset, max_offset)
                oy = random.uniform(-max_offset, max_offset)
                # 텍스트 요소
                txt = dwg.text(
                    ch,
                    insert=(x+ox, y+oy),
                    font_size=font_size,
                    font_family=Path(font_path).stem,
                    fill=fg_color
                )
                txt.rotate(angle, center=(x+ox, y+oy))
                dwg.add(txt)
                # 다음 x
                x += text_width(ch) + char_spacing
            y += line_h
            if y > image_size[1] - font_size:
                break


    def add_text_on_circle(self, dwg, text, font_path, image_size, font_size,
                            char_spacing, fg_color, params):
        cx, cy = params.get('center', (image_size[0]/2, image_size[1]/2))
        r = params.get('radius', min(image_size)/2 - font_size)
        start_angle = params.get('start_angle', -math.pi/2)
        # 호 전체 길이
        circ = 2 * math.pi * r
        # 문자별 호 길이
        lens = []
        for ch in text:
            # 추정 너비
            w = font_size * 0.6
            lens.append(w + char_spacing)
        # 각 문자 각도 계산
        angles = []
        angle = start_angle
        for l in lens:
            delta = (l/circ) * 2*math.pi
            angle += delta/2
            angles.append(angle)
            angle += delta/2
        # 문자 배치
        for ch, theta in zip(text, angles):
            w = font_size * 0.6
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            deg = math.degrees(theta) + 90
            txt = dwg.text(
                ch,
                insert=(x, y),
                font_size=font_size,
                font_family=Path(font_path).stem,
                fill=fg_color
            )
            txt.rotate(deg, center=(x, y))
            dwg.add(txt)

# 예시 사용법
if __name__ == "__main__":
    from svg_loader import SVGLoader
    text = "고"
    font_name = "MaruBuri-Bold.otf"
    font_parser = FontParser(font_name)
    svg_path = font_parser.text_to_svg(text, mode='opt-path')
    svg_loader = SVGLoader(
        svg_path=svg_path,
        output_width=128,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    classify_svg = svg_loader.classify_svg()
    print(f"SVG is classified as: {classify_svg}")
