import svgwrite
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.subset import Subsetter, Options
from pathlib import Path
from scour import scour
import os
import unicodedata
import base64
import io
import gzip

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

    def text_to_svg(self, text, mode='text', font_size=72, position=(10, 100), margin=10):
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
        
        # Prepare SVG canvas
        width = self.estimate_text_width(text, font_size) + margin * 2
        height = int(font_size * 1.5) + margin
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
                    transform=f"translate({x},{y}) scale({scale})"
                )
            )
            x += glyph.width * scale

        # Save, minify, and compress to SVGZ
        dwg.save()
        self.minify_svg(out_path)

        svgz_path = out_path.with_suffix('.svgz')
        with gzip.open(svgz_path, 'wb') as gz:
            gz.write(out_path.read_bytes())

        size_kb = svgz_path.stat().st_size / 1024
        print(f"Saved optimized SVGZ: {svgz_path} ({size_kb:.1f} KB)")
        return svgz_path
    
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
        # Subset to WOFF for smaller size
        data = self.subset_font_data(text, flavor='woff', zopfli=True)
        mime, fmt = 'font/woff', 'woff'
        family = 'SubsetFont'

        width = self.estimate_text_width(text, font_size) + margin*2
        height = int(font_size * 1.5) + margin
        out_path = self.get_output_path(text)
        dwg = self.prepare_drawing(width, height, out_path)

        self.embed_font_face(dwg, data, mime, fmt, family)
        dwg.add(dwg.text(text,
                         insert=position,
                         font_size=font_size,
                         font_family=family,
                         fill='black'))
        dwg.save()
        print(f"Saved SVG (text mode): {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
        return out_path

# 예시 사용법
if __name__ == "__main__":
    text = "가나다"
    font_name = "MaruBuri-Bold.otf"
    font_parser = FontParser(font_name)
    font_parser.text_to_svg(text, mode='opt-path')
