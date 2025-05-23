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
import numpy as np
import cv2

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

class ImageToSVG:
    def __init__(self, output_folder=None):
        self.svg_folder = Path(svg_folder) if output_folder is None else Path(output_folder)
        self.svg_folder.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, img_path, suffix=""):
        img_path = Path(img_path)
        filename = f"{img_path.stem}{suffix}.svg"
        return self.svg_folder / filename
    
    def minify_svg(self, svg_path: Path):
        """SVG 최적화 및 경로 간소화"""
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
        minified = scour.scourString(svg_path.read_text(encoding='utf-8'), opts)
        svg_path.write_text(minified, encoding='utf-8')
    
    def embed_image(self, img_path, output_path=None, compression_quality=90):
        """
        이미지를 SVG 내부에 base64로 인코딩하여 포함합니다.
        주의: 이 방법은 실제 경로 기반 SVG가 아닙니다.
        """
        img = Image.open(img_path)
        width, height = img.size
        
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(img_path, "_embedded")
        
        # 이미지를 base64로 인코딩
        buffer = io.BytesIO()
        img.save(buffer, format="PNG" if img_path.lower().endswith(".png") else "JPEG", 
                quality=compression_quality)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # SVG 생성
        img_format = "png" if img_path.lower().endswith(".png") else "jpeg"
        img_data = f"data:image/{img_format};base64,{img_base64}"
        
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        dwg.add(dwg.image(href=img_data, insert=(0, 0), size=(width, height)))
        dwg.save()
        
        size_kb = output_path.stat().st_size / 1024
        print(f"Saved embedded image SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path
    
    def vectorize_image(self, img_path, output_path=None, threshold=128, 
                       simplify=True, width=None, height=None):
        """
        이미지를 벡터화하여 경로 기반 SVG로 변환합니다.
        실제 path 요소를 사용하여 벡터화된 SVG를 생성합니다.
        모든 윤곽선(외부 + 내부)을 처리합니다.
        """
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(img_path, "_vector")
        
        # 이미지 불러오기
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # 크기 조정 (옵션)
        if width is not None and height is not None:
            img = cv2.resize(img, (width, height))
        
        orig_height, orig_width = img.shape[:2]
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거와 디테일 보존을 위한 전처리
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 이진화 - 적응형 임계값으로 더 나은 디테일 감지
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 엣지 감지 - 낮은 임계값과 높은 임계값 조정으로 더 많은 엣지 감지
        edges = cv2.Canny(binary, 30, 100)
        
        # 모폴로지 연산으로 엣지 강화
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 윤곽선 찾기 - RETR_TREE로 모든 계층 구조 감지
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{orig_width}px", f"{orig_height}px"),
            viewBox=f"0 0 {orig_width} {orig_height}"
        )
        
        # 윤곽선을 SVG 경로로 변환
        path_group = dwg.g(fill="none", stroke="black", stroke_width=1)
        
        for i, contour in enumerate(contours):
            if simplify:
                # 윤곽선 단순화 (더글라스-포이커 알고리즘)
                epsilon = 0.005 * cv2.arcLength(contour, True)  # 더 정밀한 단순화
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # SVG 경로 문자열 생성
            path_data = "M"
            for i, point in enumerate(contour):
                x, y = point[0]
                if i == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            path_data += " Z"  # 경로 닫기
            
            # 경로 추가
            path_group.add(dwg.path(d=path_data))
        
        dwg.add(path_group)
        dwg.save()
        
        # SVG 최적화
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized path-based SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path
    
    def trace_image(self, img_path, output_path=None, 
                   color_mode="color", quantize=True, num_colors=8,
                   simplify=True, despeckle=True):
        """
        컬러 이미지를 계층화된 경로 기반 SVG로 변환합니다.
        색상별 레이어를 path 요소로 생성합니다.
        모든 내부/외부 윤곽선을 포함합니다.
        """
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(img_path, "_traced")
        
        # 이미지 불러오기
        original_img = Image.open(img_path)
        width, height = original_img.size
        
        # OpenCV로 변환
        img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 배경 추가
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))
        
        if color_mode == "color" and quantize:
            # 이미지 양자화 (색상 수 감소)
            img_array = np.float32(img).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(img_array, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            
            # 각 색상에 대한 그룹 생성
            for i in range(num_colors):
                # 해당 색상 마스크 생성
                mask = cv2.inRange(labels.reshape(height, width), i, i)
                
                # 노이즈 제거 (옵션)
                if despeckle:
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # 윤곽선 찾기 - RETR_TREE로 모든 계층 구조 감지
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # 색상값 가져오기
                color = centers[i]
                color_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
                
                # 같은 색상의 경로를 그룹화
                color_group = dwg.g(fill=color_hex, stroke="none")
                
                # 계층 구조에 따라 처리
                if hierarchy is not None:
                    for j, contour in enumerate(contours):
                        if cv2.contourArea(contour) < 10:  # 너무 작은 윤곽선 무시
                            continue
                        
                        if simplify:
                            # 윤곽선 단순화
                            epsilon = 0.001 * cv2.arcLength(contour, True)  # 더 정밀한 단순화
                            contour = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # SVG 경로 생성
                        path_data = "M"
                        for k, point in enumerate(contour):
                            x, y = point[0]
                            if k == 0:
                                path_data += f" {x},{y}"
                            else:
                                path_data += f" L {x},{y}"
                        path_data += " Z"
                        
                        # 경로 추가 (내부 윤곽선은 구멍으로 처리)
                        is_hole = hierarchy[0][j][3] >= 0  # 부모가 있으면 구멍
                        if is_hole:
                            # 구멍은 별도 경로로 추가
                            path = dwg.path(d=path_data, fill="white")
                            color_group.add(path)
                        else:
                            # 일반 경로
                            path = dwg.path(d=path_data)
                            color_group.add(path)
                
                # 색상 그룹 추가
                if len(color_group.elements) > 0:
                    dwg.add(color_group)
        else:
            # 그레이스케일 처리
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 적응형 이진화로 더 많은 디테일 감지
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            
            # 모든 계층 구조 감지
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # 단일 색상 그룹
            path_group = dwg.g(fill="black", stroke="none")
            
            if hierarchy is not None:
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) < 10:  # 너무 작은 윤곽선 무시
                        continue
                    
                    if simplify:
                        # 윤곽선 단순화
                        epsilon = 0.001 * cv2.arcLength(contour, True)
                        contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # SVG 경로 생성
                    path_data = "M"
                    for j, point in enumerate(contour):
                        x, y = point[0]
                        if j == 0:
                            path_data += f" {x},{y}"
                        else:
                            path_data += f" L {x},{y}"
                    path_data += " Z"
                    
                    # 내부 윤곽선은 구멍으로 처리
                    is_hole = hierarchy[0][i][3] >= 0  # 부모가 있으면 구멍
                    if is_hole:
                        path = dwg.path(d=path_data, fill="white")
                    else:
                        path = dwg.path(d=path_data)
                    path_group.add(path)
            
            dwg.add(path_group)
        
        dwg.save()
        
        # SVG 최적화
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized path-based SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path
            
    def _add_holes_recursively(self, dwg, group, contours, hierarchy, parent_idx, opttolerance):
        """
        윤곽선의 계층 구조를 재귀적으로 처리하여 구멍(holes)을 추가합니다.
        """
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == parent_idx:  # 현재 검사 중인 윤곽선이 직접적인 자식인 경우
                if cv2.contourArea(contour) < 10:  # 너무 작은 윤곽선 무시
                    continue
                    
                # 곡선 간소화
                epsilon = opttolerance * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # SVG 경로 생성
                path_data = "M"
                for j, point in enumerate(approx):
                    x, y = point[0]
                    if j == 0:
                        path_data += f" {x},{y}"
                    else:
                        path_data += f" L {x},{y}"
                path_data += " Z"
                
                # 구멍 추가 (흰색으로 채움)
                path_element = dwg.path(d=path_data, fill="white")
                group.add(path_element)
                
                # 이 구멍 안에 다른 윤곽선이 있는지 확인하고 재귀적으로 처리
                self._add_holes_recursively(dwg, group, contours, hierarchy, i, opttolerance)
                
    def extract_shape_from_image(self, img_path, output_path=None, bg_color=(255, 255, 255),
                                threshold=25, simplify=True, edge_detail=True, smooth_curves=True):
        """
        일반적인 도장이나 로고와 같은 형태의 이미지에서 모양을 추출하고 SVG로 변환합니다.
        배경색과 객체 간의 명확한 구분이 있는 이미지에 적합합니다.
        내부/외부 경로를 일관되게 처리하며, 베지어 곡선으로 매끄럽게 표현합니다.
        """
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(img_path, "_shape")
        
        # 이미지 불러오기
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        
        # 배경색 확인 및 마스크 생성
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 배경색과의 차이에 기반한 마스크 생성
        for y in range(height):
            for x in range(width):
                b, g, r = img[y, x]
                # 배경색과의 색상 차이 계산
                diff = abs(int(b) - bg_color[0]) + abs(int(g) - bg_color[1]) + abs(int(r) - bg_color[2])
                # 임계값보다 차이가 크면 객체로 판단
                if diff > threshold * 3:
                    mask[y, x] = 255
        
        # 노이즈 제거 및 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 가우시안 블러로 부드럽게
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        if edge_detail:
            # 엣지 감지를 통해 디테일 보존
            edges = cv2.Canny(mask, 30, 100)
            mask = cv2.bitwise_or(mask, edges)
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 모든 경로를 하나의 복합 경로로 처리
        # 단일 경로로 처리하여 내부/외부 일관성 유지
        path_data = ""
        
        # 윤곽선 찾기 - NONE으로 모든 점 보존
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if hierarchy is not None and len(contours) > 0:
            # 먼저 원형인 윤곽선 식별
            circle_contours = []
            normal_contours = []
            
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] < 0 and cv2.contourArea(contour) >= 10:  # 외부 윤곽선만
                    if self._is_circle(contour):
                        # 원 정보 계산
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        circle_contours.append((i, contour, True, (x, y, radius)))
                    else:
                        normal_contours.append((i, contour))
            
            # 원형 윤곽선 처리
            for i, contour, (x, y, radius) in circle_contours:
                if len(path_data) > 0:
                    path_data += " "
                
                # 원 경로 추가
                path_data += self._create_svg_circle_path(x, y, radius)
                
                # 이 원 내부의 구멍 처리
                for j, hole_contour in enumerate(contours):
                    if hierarchy[0][j][3] == i:  # 직접적인 자식
                        if cv2.contourArea(hole_contour) < 10:
                            continue
                            
                        # 구멍이 원형인지 확인
                        if self._is_circle(hole_contour):
                            (hx, hy), hradius = cv2.minEnclosingCircle(hole_contour)
                            path_data += " " + self._create_svg_circle_path(hx, hy, hradius)
                        else:
                            # 일반 구멍은 부드러운 곡선 처리
                            if simplify:
                                epsilon = 0.0005 * cv2.arcLength(hole_contour, True)
                                hole_contour = cv2.approxPolyDP(hole_contour, epsilon, True)
                            
                            # 내부 경로는 반시계 방향
                            if cv2.isContourConvex(hole_contour):
                                hole_contour = np.flip(hole_contour, axis=0)
                                
                            path_data += " " + self._create_smooth_path(hole_contour, smooth_curves)
            
            # 일반 윤곽선 처리
            for i, contour in normal_contours:
                if len(path_data) > 0:
                    path_data += " "
                
                # 외부 윤곽선 처리
                if simplify:
                    epsilon = 0.0005 * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # 외부 경로는 시계 방향
                if not cv2.isContourConvex(contour):
                    contour = np.flip(contour, axis=0)
                
                path_data += self._create_smooth_path(contour, smooth_curves)
                
                # 이 외부 윤곽선 내부의 구멍 처리
                for j, hole_contour in enumerate(contours):
                    if hierarchy[0][j][3] == i:  # 직접적인 자식
                        if cv2.contourArea(hole_contour) < 10:
                            continue
                        
                        # 구멍이 원형인지 확인
                        if self._is_circle(hole_contour):
                            (hx, hy), hradius = cv2.minEnclosingCircle(hole_contour)
                            path_data += " " + self._create_svg_circle_path(hx, hy, hradius)
                        else:
                            # 일반 구멍은 부드러운 곡선 처리
                            if simplify:
                                epsilon = 0.0005 * cv2.arcLength(hole_contour, True)
                                hole_contour = cv2.approxPolyDP(hole_contour, epsilon, True)
                            
                            # 내부 경로는 반시계 방향
                            if cv2.isContourConvex(hole_contour):
                                hole_contour = np.flip(hole_contour, axis=0)
                                
                            path_data += " " + self._create_smooth_path(hole_contour, smooth_curves)
        
        # 단일 경로로 추가 (evenodd 규칙 사용)
        if path_data:
            # 원하는 색상으로 채우기
            dwg.add(dwg.path(d=path_data, fill="red", fill_rule="evenodd"))
        
        dwg.save()
        
        # SVG 최적화
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized shape SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path

    def create_logo_svg(self, img_path, output_path=None, bg_color=(0, 0, 0),
                       fill_color="red", stroke_color="white", stroke_width=1,
                       threshold=25, simplify=True, smooth_curves=True):
        """
        로고나 아이콘을 SVG로 변환합니다.
        내부/외부 경로를 일관되게 처리하여 깔끔한 벡터 로고를 생성합니다.
        곡선 부분을 매끄럽게 표현하기 위해 베지어 곡선을 사용합니다.
        """
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(img_path, "_logo")
        
        # 이미지 불러오기
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 이진화 (배경이 어두운 경우를 고려)
        if np.mean(bg_color) < 128:  # 어두운 배경
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:  # 밝은 배경
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 노이즈 제거와 모서리 부드럽게 하기
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 가우시안 블러로 부드럽게
        binary = cv2.GaussianBlur(binary, (5, 5), 0)
        
        # 엣지 추출 대신 윤곽선 직접 찾기
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 모든 윤곽선을 단일 복합 경로로 처리
        path_data = ""
        
        if hierarchy is not None and len(contours) > 0:
            # 외부 윤곽선 먼저 처리 (시계 방향)
            ext_contours = []
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] < 0:  # 부모가 없는 윤곽선
                    if cv2.contourArea(contour) < 10:
                        continue
                    
                    # 원형 감지 및 특별 처리
                    if self._is_circle(contour):
                        # 원 정보 계산
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        ext_contours.append((i, contour, True, (x, y, radius)))
                    else:
                        # 일반 윤곽선은 단순화 후 처리
                        if simplify:
                            epsilon = 0.0005 * cv2.arcLength(contour, True)  # 더 정밀한 단순화
                            contour = cv2.approxPolyDP(contour, epsilon, True)
                        ext_contours.append((i, contour, False, None))
            
            # 면적 기준 정렬 (큰 것부터)
            ext_contours.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
            
            # 외부 윤곽선 추가
            for idx, contour, is_circle, circle_info in ext_contours:
                # 외부 윤곽선은 시계 방향 (SVG 규칙)
                if not cv2.isContourConvex(contour):
                    contour = np.flip(contour, axis=0)
                
                if len(path_data) > 0:
                    path_data += " "
                
                if is_circle and circle_info:
                    # 원으로 처리
                    x, y, radius = circle_info
                    path_data += self._create_svg_circle_path(x, y, radius)
                else:
                    # 일반 경로로 처리 (곡선 보간 사용)
                    path_data += self._create_smooth_path(contour, smooth_curves)
                
                # 이 외부 윤곽선에 속한 모든 구멍 추가
                for i, hole_contour in enumerate(contours):
                    if hierarchy[0][i][3] == idx:  # 직접적인 자식
                        if cv2.contourArea(hole_contour) < 10:
                            continue
                        
                        # 구멍도 원형인지 확인
                        is_hole_circle = self._is_circle(hole_contour)
                        
                        if is_hole_circle:
                            (hx, hy), hradius = cv2.minEnclosingCircle(hole_contour)
                            path_data += " " + self._create_svg_circle_path(hx, hy, hradius)
                        else:
                            # 구멍도 단순화
                            if simplify:
                                epsilon = 0.0005 * cv2.arcLength(hole_contour, True)
                                hole_contour = cv2.approxPolyDP(hole_contour, epsilon, True)
                            
                            # 내부 윤곽선은 반시계 방향 (SVG 규칙)
                            if cv2.isContourConvex(hole_contour):
                                hole_contour = np.flip(hole_contour, axis=0)
                                
                            path_data += " " + self._create_smooth_path(hole_contour, smooth_curves)
        
        # 단일 경로로 추가
        if path_data:
            dwg.add(dwg.path(
                d=path_data, 
                fill=fill_color, 
                stroke=stroke_color if stroke_width > 0 else "none",
                stroke_width=stroke_width,
                fill_rule="evenodd"
            ))
        
        dwg.save()
        
        # SVG 최적화
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized logo SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path

    def _is_circle(self, contour, tolerance=0.01):
        """
        윤곽선이 원형인지 확인합니다.
        """
        # 면적과 둘레를 이용한 원형 측정
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False
            
        # 원형 측정값 (1에 가까울수록 원형)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 최소 경계 원과 면적 비교
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        area_ratio = area / circle_area if circle_area > 0 else 0
        
        # 두 측정값 모두 기준 충족시 원형으로 판단
        return circularity > 0.9 and area_ratio > 0.9
    
    def _create_svg_circle_path(self, x, y, radius):
        """
        SVG 원 경로를 생성합니다.
        """
        # SVG 경로 명령어로 원 그리기
        # M cx,cy 
        # m -r,0 
        # a r,r 0 1,0 (r*2),0 
        # a r,r 0 1,0 -(r*2),0
        cx, cy = x, y
        r = radius
        return f"M {cx},{cy-r} a {r},{r} 0 1,0 {r*2},0 a {r},{r} 0 1,0 -{r*2},0 Z"
    
    def _create_smooth_path(self, contour, use_curves=True):
        """
        매끄러운 SVG 경로를 생성합니다. 베지어 곡선 사용 옵션 지원.
        """
        if len(contour) < 3:
            # 점이 너무 적으면 직선 경로 사용
            path_data = "M"
            for j, point in enumerate(contour):
                x, y = point[0]
                if j == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            return path_data + " Z"
        
        if not use_curves:
            # 곡선 사용하지 않는 경우
            path_data = "M"
            for j, point in enumerate(contour):
                x, y = point[0]
                if j == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            return path_data + " Z"
        
        # 베지어 곡선으로 부드러운 경로 생성
        points = np.array([point[0] for point in contour])
        n_points = len(points)
        
        # 시작점
        path_data = f"M {points[0][0]},{points[0][1]}"
        
        # 점이 충분히 많은 경우만 곡선 적용
        if n_points > 3:
            # 각 점 사이에 베지어 곡선 생성
            for i in range(1, n_points + 1):
                # 현재 점, 이전 점, 다음 점
                curr = points[i % n_points]
                prev = points[(i - 1) % n_points]
                next_pt = points[(i + 1) % n_points]
                
                # 제어점 계산 (이전-현재-다음 점 사이 중간 위치)
                cp1x = prev[0] + (curr[0] - prev[0]) * 0.5
                cp1y = prev[1] + (curr[1] - prev[1]) * 0.5
                cp2x = curr[0] + (next_pt[0] - curr[0]) * 0.5
                cp2y = curr[1] + (next_pt[1] - curr[1]) * 0.5
                
                # 베지어 곡선 추가
                path_data += f" C {cp1x},{cp1y} {cp2x},{cp2y} {curr[0]},{curr[1]}"
        else:
            # 점이 적은 경우는 직선 사용
            for i in range(1, n_points):
                path_data += f" L {points[i][0]},{points[i][1]}"
        
        # 경로 닫기
        return path_data + " Z"

    def svg_from_json(self, json_path, output_path=None, stroke_color="black", 
                     stroke_width=1, fill="none"):
        """
        JSON 형식의 경로 데이터에서 SVG를 생성합니다.
        JSON 형식: {"paths": [{"points": [[x1, y1], [x2, y2], ...], "closed": true/false}, ...]}
        """
        import json
        
        # JSON 데이터 로드
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(Path(json_path), "_svg")
        
        # 캔버스 크기 결정 (모든 점의 최대/최소 좌표 활용)
        all_points = []
        for path in data.get("paths", []):
            all_points.extend(path.get("points", []))
        
        if not all_points:
            raise ValueError("No path data found in JSON")
        
        # 좌표 범위 계산
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 여백 추가
        padding = max(10, int(max(max_x - min_x, max_y - min_y) * 0.05))
        width = max_x - min_x + padding * 2
        height = max_y - min_y + padding * 2
        
        # 좌표 오프셋 (모든 점을 양수 영역으로)
        offset_x = -min_x + padding
        offset_y = -min_y + padding
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 모든 경로를 하나의 복합 경로로 통합
        path_data = ""
        
        # 경로 처리
        for path in data.get("paths", []):
            points = path.get("points", [])
            closed = path.get("closed", True)
            
            if not points:
                continue
            
            if len(path_data) > 0:
                path_data += " "
            
            # 시작점
            path_data += f"M {points[0][0] + offset_x},{points[0][1] + offset_y}"
            
            # 나머지 점
            for i in range(1, len(points)):
                path_data += f" L {points[i][0] + offset_x},{points[i][1] + offset_y}"
            
            # 경로 닫기 (옵션)
            if closed:
                path_data += " Z"
        
        # 경로 추가
        if path_data:
            dwg.add(dwg.path(
                d=path_data,
                fill=fill,
                stroke=stroke_color,
                stroke_width=stroke_width
            ))
        
        dwg.save()
        
        # SVG 최적화
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Created SVG from JSON: {output_path} ({size_kb:.1f} KB)")
        return output_path
        
    def extract_all_outlines(self, img_path, output_path=None, stroke_color="black", 
                           stroke_width=3, fill="none", bg_color=(0, 0, 0),
                           threshold=128, min_area_ratio=0.0001):
        """
        이미지에서 모든 중요한 윤곽선을 추출하여 SVG로 변환합니다.
        내부/외부 윤곽선을 모두 포함하며 필요한 최소한의 필터링만 적용합니다.
        
        Parameters:
            img_path: 입력 이미지 경로
            output_path: 출력 SVG 경로
            stroke_color: 선 색상
            stroke_width: 선 두께
            fill: 채우기 색상 (보통 "none")
            bg_color: 배경색 (이진화용)
            threshold: 이진화 임계값
            min_area_ratio: 무시할 최소 윤곽선 크기 비율
        """
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(img_path, "_outlines")
        
        # 이미지 불러오기
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        img_area = height * width
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 배경 색상에 따른 이진화
        if np.mean(bg_color) < 128:  # 어두운 배경
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:  # 밝은 배경
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 노이즈 제거 (매우 약한 처리)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 찾기 - 내부와 외부 윤곽선 모두 가져옴
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 그룹 요소 생성 - 선 스타일 속성 추가
        outline_group = dwg.g(
            fill=fill, 
            stroke=stroke_color, 
            stroke_width=stroke_width,
            stroke_linejoin="round",  # 모서리를 둥글게 (miter, round, bevel 중 선택)
            stroke_linecap="round"    # 선 끝을 둥글게 (butt, round, square 중 선택)
        )
        
        # 모든 윤곽선 처리 (너무 작은 것만 제외)
        min_area = img_area * min_area_ratio
        contour_count = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 너무 작은 윤곽선은 무시
            if area < min_area:
                continue
            
            # 점이 너무 적은 경우도 무시 (노이즈 제거)
            if len(contour) < 3:
                continue
                
            # 윤곽선 단순화 (중요한 점만 유지)
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # SVG 경로 생성
            path_data = "M"
            for j, point in enumerate(approx):
                x, y = point[0]
                if j == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            path_data += " Z"
            
            # 경로 추가
            outline_group.add(dwg.path(d=path_data))
            contour_count += 1
        
        # 그룹 추가
        if contour_count > 0:
            dwg.add(outline_group)
        
        # SVG 저장
        dwg.save()
        
        # 최적화
        self.minify_svg(Path(output_path))
        
        print(f"Created outline SVG with {contour_count} contours: {output_path}")
        return output_path

    def extract_filled_outlines(self, img_path, output_path=None, stroke_color="black",
                                stroke_width=1, fill_color="#000000", bg_color=(0, 0, 0),
                                threshold=100, min_area_ratio=0.00001, color_extraction=False):
        """
        이미지에서 모든 중요한 윤곽선을 추출하여 내부가 채워진 SVG로 변환합니다.
        투명 영역(alpha=0)은 컨투어 계산에서 제외하고, SVG 배경은 투명으로 유지합니다.
        """
        if output_path is None:
            output_path = self.get_output_path(img_path, "_filled")

        # --- 1) UNCHANGED 플래그로 읽어서 alpha 채널 확보 ---
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # 채널 분리
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[..., :3]
            alpha = img[..., 3]
        else:
            bgr = img
            alpha = None

        height, width = bgr.shape[:2]
        img_area = height * width

        # 그레이스케일 변환 (투명 픽셀도 BGR에서 변환하되, 이후에 마스킹)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 배경 색상에 따른 이진화
        if np.mean(bg_color) < 128:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # --- 2) alpha 채널이 있을 때, 투명 픽셀을 모두 배경(0)으로 ---
        if alpha is not None:
            # alpha == 0 이면 투명 => 컨투어 검사에서 제외
            binary[alpha == 0] = 0

        # 노이즈 제거
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 윤곽선 찾기
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )

        # 이하 기존 로직 유지...
        min_area = img_area * min_area_ratio
        contour_count = 0
        external_contours, internal_contours = [], []

        if hierarchy is not None:
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                epsilon = 0.0005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if hierarchy[0][i][3] < 0:
                    external_contours.append((approx, area, i))
                else:
                    parent_idx = hierarchy[0][i][3]
                    internal_contours.append((approx, area, i, parent_idx))

        external_contours.sort(key=lambda x: x[1], reverse=True)
        for contour, _, idx in external_contours:
            # 색상 추출 옵션
            current_fill = fill_color
            if color_extraction:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_color = cv2.mean(bgr, mask=mask)
                b, g, r = mean_color[:3]
                current_fill = f"#{int(r):02x}{int(g):02x}{int(b):02x}"

            # 경로 문자열 생성
            path_data = "M " + " L ".join(f"{pt[0][0]},{pt[0][1]}" for pt in contour) + " Z"
            contour_path = dwg.path(
                d=path_data,
                fill=current_fill,
                stroke=stroke_color,
                stroke_width=stroke_width,
                stroke_linejoin="round",
                stroke_linecap="round",
                fill_rule="evenodd"
            )
            dwg.add(contour_path)
            contour_count += 1

            # 구멍(hole) 추가
            for hole_contour, _, _, parent in internal_contours:
                if parent == idx:
                    hole_data = "M " + " L ".join(f"{pt[0][0]},{pt[0][1]}" for pt in hole_contour) + " Z"
                    hole_path = dwg.path(
                        d=hole_data,
                        fill="white",
                        stroke=stroke_color,
                        stroke_width=stroke_width,
                        stroke_linejoin="round",
                        stroke_linecap="round"
                    )
                    dwg.add(hole_path)

        dwg.save()
        self.minify_svg(Path(output_path))
        print(f"Created filled SVG with {contour_count} contours: {output_path}")
        return output_path

    def extract_smooth_outlines(self, img_path, output_path=None, stroke_color="black", 
                             stroke_width=3, fill="none", bg_color=(0, 0, 0),
                             threshold=100, min_area_ratio=0.000001):
        """
        이미지에서 윤곽선을 추출하여 매우 매끄러운 SVG로 변환합니다.
        원형 모양은 <circle> 요소로 변환하고, 다른 모양은 부드러운 곡선으로 처리합니다.
        
        Parameters:
            img_path: 입력 이미지 경로
            output_path: 출력 SVG 경로
            stroke_color: 선 색상
            stroke_width: 선 두께
            fill: 채우기 색상 (보통 "none")
            bg_color: 배경색 (이진화용)
            threshold: 이진화 임계값
            min_area_ratio: 무시할 최소 윤곽선 크기 비율
        """
        # 저장 경로 준비
        if output_path is None:
            output_path = self.get_output_path(img_path, "_smooth")
        
        # 이미지 불러오기
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        img_area = height * width
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 필터 적용하여 노이즈 제거 및 부드럽게
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 배경 색상에 따른 이진화
        if np.mean(bg_color) < 128:  # 어두운 배경
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        else:  # 밝은 배경
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 노이즈 제거 (매우 약한 처리)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 윤곽선 찾기 - 모든 점을 보존하는 모드로 (CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 그룹 요소 생성 - 선 스타일 속성 추가
        outline_group = dwg.g(
            fill=fill, 
            stroke=stroke_color, 
            stroke_width=stroke_width,
            stroke_linejoin="round",  # 모서리를 둥글게
            stroke_linecap="round"    # 선 끝을 둥글게
        )
        
        # 모든 윤곽선 처리 (너무 작은 것만 제외)
        min_area = img_area * min_area_ratio
        contour_count = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 너무 작은 윤곽선은 무시
            if area < min_area:
                continue
            
            # 점이 너무 적은 경우도 무시 (노이즈 제거)
            if len(contour) < 3:
                continue
                
            # 원형인지 확인
            is_circle = self._is_circle(contour, tolerance=0.02)
            
            if is_circle:
                # 원 파라미터 계산
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                
                # SVG 원 요소 추가
                outline_group.add(dwg.circle(
                    center=(center_x, center_y),
                    r=radius,
                    stroke_width=stroke_width
                ))
                contour_count += 1
            else:
                # 곡선 보간을 위해 점 간격 기반으로 보간
                # 많은 점으로 더 부드럽게 처리
                approx_points = []
                for j in range(len(contour)):
                    p1 = contour[j][0]
                    p2 = contour[(j + 1) % len(contour)][0]
                    
                    # 거리 계산
                    dist = np.linalg.norm(p2 - p1)
                    
                    # 현재 점 추가
                    approx_points.append(p1)
                    
                    # 거리가 5보다 크면, 중간 점 추가
                    if dist > 5:
                        num_points = max(1, int(dist / 5))
                        for k in range(1, num_points):
                            t = k / num_points
                            mid = p1 * (1 - t) + p2 * t
                            approx_points.append(mid)
                
                # 부드러운 곡선 경로 생성
                path_data = ""
                
                # 베지어 곡선 또는 부드러운 곡선을 사용하여 경로 생성
                if len(approx_points) >= 3:
                    # 시작점
                    path_data = f"M {approx_points[0][0]},{approx_points[0][1]}"
                    
                    # 베지어 곡선 명령어로 경로 연결
                    for j in range(1, len(approx_points), 3):
                        # 충분한 점이 있는지 확인
                        if j + 2 < len(approx_points):
                            # 입방 베지어 곡선
                            x1, y1 = approx_points[j]
                            x2, y2 = approx_points[j + 1]
                            x3, y3 = approx_points[j + 2]
                            path_data += f" C {x1},{y1} {x2},{y2} {x3},{y3}"
                        elif j + 1 < len(approx_points):
                            # 이차 베지어 곡선
                            x1, y1 = approx_points[j]
                            x2, y2 = approx_points[j + 1]
                            path_data += f" Q {x1},{y1} {x2},{y2}"
                        else:
                            # 마지막 점은 선으로 연결
                            x, y = approx_points[j]
                            path_data += f" L {x},{y}"
                    
                    # 경로 닫기
                    path_data += " Z"
                else:
                    # 점이 부족하면 기본 경로 생성
                    path_data = "M"
                    for point in approx_points:
                        x, y = point
                        if path_data == "M":
                            path_data += f" {x},{y}"
                        else:
                            path_data += f" L {x},{y}"
                    path_data += " Z"
                
                # 경로 추가
                outline_group.add(dwg.path(d=path_data))
                contour_count += 1
        
        # 그룹 추가
        if contour_count > 0:
            dwg.add(outline_group)
        
        # SVG 저장
        dwg.save()
        
        # 최적화
        self.minify_svg(Path(output_path))
        
        print(f"Created smooth outline SVG with {contour_count} contours: {output_path}")
        return output_path
        
    def extract_logo_paths(self, img_path, output_path=None, bg_color=(0, 0, 0)):
        """
        보다 확실한 방법으로 로고나 단순한 그래픽의 경로를 추출합니다.
        여러 방법을 시도하고 최상의 결과를 선택합니다.
        
        이 함수는 특히 apple.svg나 tesla_logo.svg와 같은 단순한 로고 이미지에 적합합니다.
        """
        if output_path is None:
            output_path = self.get_output_path(img_path, "_logo_paths")
        
        # 이미지 불러오기
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        img_area = height * width
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 여러 가지 전처리 방법 시도
        methods = []
        
        # 방법 1: Otsu 이진화
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        methods.append(("otsu", binary1))
        
        # 방법 2: 적응형 이진화
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        methods.append(("adaptive", binary2))
        
        # 방법 3: 고정 임계값 이진화
        if np.mean(bg_color) < 128:  # 어두운 배경
            _, binary3 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        else:  # 밝은 배경
            _, binary3 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        methods.append(("fixed", binary3))
        
        # 최적의 방법 찾기
        best_contours = []
        best_method = ""
        
        for method_name, binary in methods:
            # 윤곽선 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 충분히 큰 윤곽선만 유지
            significant_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > img_area * 0.001:  # 이미지 면적의 0.1% 이상
                    perimeter = cv2.arcLength(contour, True)
                    # 복잡성 측정 (높을수록 복잡)
                    complexity = perimeter * perimeter / (4 * np.pi * area)
                    significant_contours.append((contour, area, complexity))
            
            # 윤곽선 결과가 더 좋은지 평가
            if len(significant_contours) > 0:
                if len(best_contours) == 0 or len(significant_contours) > len(best_contours):
                    best_contours = significant_contours
                    best_method = method_name
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 모든 윤곽선 추가
        for contour, _, _ in best_contours:
            # 단순화
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 경로 데이터 생성
            path_data = "M"
            for i, point in enumerate(approx):
                x, y = point[0]
                if i == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            path_data += " Z"
            
            # 경로 추가
            dwg.add(dwg.path(d=path_data, fill="none", stroke="black", stroke_width="1"))
        
        # SVG 저장
        dwg.save()
        
        # 최적화
        self.minify_svg(Path(output_path))
        
        print(f"Created logo SVG using {best_method} method with {len(best_contours)} paths: {output_path}")
        return output_path
    
    def create_clean_svg(self, img_path, output_path=None, stroke_color="black", 
                        stroke_width=1, fill="none", fix_empty=True):
        """
        이미지에서 깔끔한 윤곽선을 추출하여 SVG로 변환합니다.
        여러 방법을 시도하고 빈 결과가 나오면 자동으로 다른 방법을 시도합니다.
        
        Parameters:
            img_path: 입력 이미지 경로
            output_path: 출력 SVG 경로
            stroke_color: 선 색상
            stroke_width: 선 두께
            fill: 채우기 색상
            fix_empty: 결과가 비어있으면 다른 방법을 시도할지 여부
        """
        if output_path is None:
            output_path = self.get_output_path(img_path, "_clean")
            
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        height, width = img.shape[:2]
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 노이즈 제거 (약한 처리)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny 엣지 감지
        edges = cv2.Canny(blurred, 50, 150)
        
        # 모폴로지 연산으로 엣지 연결
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # SVG 생성
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # 중요한 윤곽선만 선택
        significant_contours = []
        min_area = width * height * 0.0001  # 전체 이미지 면적의 0.01%
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # 단순화
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 너무 단순한 형태 제외 (점이 3개 미만)
            if len(approx) < 3:
                continue
                
            significant_contours.append(approx)
        
        # 각 윤곽선을 개별 경로로 추가
        for contour in significant_contours:
            # 경로 데이터 생성
            path_data = "M"
            for i, point in enumerate(contour):
                x, y = point[0]
                if i == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            path_data += " Z"
            
            # 경로 추가
            dwg.add(dwg.path(d=path_data, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
        
        # SVG 저장
        dwg.save()
        
        # 빈 결과를 확인하고 해결 시도
        if fix_empty and len(significant_contours) == 0:
            print("No contours found. Trying alternative method...")
            
            # 다른 방법 시도: 이진화 기반
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 가장 큰 윤곽선 찾기
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                
                # 단순화
                epsilon = 0.001 * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True)
                
                # 새 SVG 생성
                dwg = svgwrite.Drawing(
                    filename=str(output_path),
                    size=(f"{width}px", f"{height}px"),
                    viewBox=f"0 0 {width} {height}"
                )
                
                # 경로 생성
                path_data = "M"
                for i, point in enumerate(approx):
                    x, y = point[0]
                    if i == 0:
                        path_data += f" {x},{y}"
                    else:
                        path_data += f" L {x},{y}"
                path_data += " Z"
                
                # 경로 추가
                dwg.add(dwg.path(d=path_data, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
                
                # 저장
                dwg.save()
                
                print(f"Created SVG with alternative method: {output_path}")
            else:
                print("Failed to find contours with alternative method.")
        
        # 최적화
        self.minify_svg(Path(output_path))
        
        print(f"Created clean SVG with {len(significant_contours)} contours: {output_path}")
        return output_path

# 예시 사용법
if __name__ == "__main__":
    from svg_loader import SVGLoader
    text = "WAVE"
    font_name = "MaruBuri-Bold.otf"
    font_parser = FontParser(font_name)
    svg_path = font_parser.text_to_svg(text, mode='opt-path')
    svg_loader = SVGLoader(
        svg_path=svg_path,
        output_width=128,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # classify_svg = svg_loader.classify_svg()
    # print(f"SVG is classified as: {classify_svg}")
    
    # img_converter = ImageToSVG()
    # img_path = "/data/jameskim/repos/circle_art/assets/image/elec_stamp.png"
    # #traced_svg = img_converter.trace_image(img_path, color_mode="color", quantize=True, simplify=False, num_colors=8)
    
    # #outline_svg = img_converter.extract_all_outlines(img_path, threshold=100, min_area_ratio=0.000001)
    
    # filled_outline_svg = img_converter.extract_filled_outlines(img_path, threshold=100, min_area_ratio=0.000001)
    # out_svg = img_converter.create_logo_svg(img_path, simplify=False)
    # img_converter.extract_logo_paths(img_path)
    # img_converter.vectorize_image(img_path)
    
    #smooth_svg = img_converter.extract_smooth_outlines(img_path, threshold=100, min_area_ratio=0.000001)
    
    