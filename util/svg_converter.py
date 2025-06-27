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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2

base_folder = Path(__file__).resolve().parent.parent
font_folder = os.path.join(base_folder, "assets", "font") # .ttf or .otf path
svg_folder = os.path.join(base_folder, "assets", "svg") # svg path

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
            elif unicodedata.category(ch).startswith('P'):  # Special characters
                factor = 0.5
            elif '0' <= ch <= '9':
                factor = 0.55
            elif '\uAC00' <= ch <= '\uD7AF':  # Korean characters
                factor = 0.6
            elif ch.isalpha():
                factor = 0.6
            else:
                factor = 0.6  # Default value
            total_width += font_size * factor
        return int(total_width)

    def subset_font_data(self, text: str, flavor: str = None, zopfli: bool = False) -> bytes:
        opts = Options()
        opts.flavor = flavor       # Format to preserve (w/o woff etc.)
        opts.with_zopfli = zopfli  # Skip zopfli compression as it's slow
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
        """SVG optimization and path simplification"""
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
        # pen = SVGPathPen(self.glyph_set)
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
            # pen.path = ''
            # glyph.draw(pen)
            # d = pen.getCommands()
            # ───────────────────────────────────────────────
            # 루프마다 새로운 pen 생성 → 이전 경로 누적 방지
            pen = SVGPathPen(self.glyph_set)
            glyph.draw(pen)
            d = pen.getCommands()
            # ───────────────────────────────────────────────
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

        # Embed the subsetted TTF/OTF so that all glyphs (including Korean and CJK) are preserved
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
        Generate calligraphy style SVG from the given text.

        Parameters:
            text: Text string to convert (\n for forced line breaks)
            font_path: Path to OTF/TTF font file
            output_path: Path to save the SVG
            mode: 'wrap' (normal line breaks) or 'circle' (fit on a circular path)
            circle_params: Used when mode='circle', {'center':(x,y), 'radius':r, 'start_angle':rad}
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Create SVG
        dwg = svgwrite.Drawing(
            filename=output_path,
            size=(f"{image_size[0]}px", f"{image_size[1]}px"),
            viewBox=f"0 0 {image_size[0]} {image_size[1]}"
        )
        
        # Background
        dwg.add(dwg.rect(insert=(0, 0), size=(image_size[0], image_size[1]), fill=bg_color))
        
        # Add text based on mode
        if mode == 'wrap':
            self.add_text_wrap(dwg, text, font_path, image_size, font_size,
                          char_spacing, line_spacing, max_angle,
                          max_offset, fg_color)
        elif mode == 'circle':
            if circle_params is None:
                circle_params = {}
            self.add_text_on_circle(dwg, text, font_path, image_size, font_size,
                               char_spacing, fg_color, circle_params)
        
        # Save SVG
        dwg.save()
        self.minify_svg(Path(output_path))
        print(f"Generated calligraphy SVG: {output_path}")
        return Path(output_path)
    
    def add_text_wrap(self, dwg, text, font_path, image_size, font_size,
                    char_spacing, line_spacing, max_angle,
                    max_offset, fg_color):
        # Line break calculation
        # Temporary render sheet
        width, height = image_size
        def text_width(s):
            # Simple estimation: number of characters * font_size * 0.6
            return len(s) * font_size * 0.6 + char_spacing * (len(s) - 1)
        
        # Line separation
        words = text.split(' ')
        lines = []
        curr = ''
        
        for word in words:
            if '\n' in word:
                parts = word.split('\n')
                for i, part in enumerate(parts):
                    test = f"{curr} {part}".strip()
                    if text_width(test) <= width:
                        curr = test
                    else:
                        lines.append(curr)
                        curr = part
                    if i < len(parts) - 1:
                        lines.append(curr)
                        curr = ''
            else:
                test = f"{curr} {word}".strip()
                if text_width(test) <= width:
                    curr = test
                else:
                    lines.append(curr)
                    curr = word
        
        if curr:
            lines.append(curr)
        
        # Positioning
        font_family = Path(font_path).stem
        start_x = 20
        start_y = font_size + 20
        
        # Text elements
        font = font_path
        
        for i, line in enumerate(lines):
            y = start_y + i * font_size * line_spacing
            x = char_x = start_x
            group = dwg.g(fill=fg_color, font_family=font_family, font_size=font_size)
            dwg.add(group)
            
            for j, char in enumerate(line):
                # Next x
                char_width = font_size * 0.6 if char != ' ' else font_size * 0.35
                if char != ' ':
                    angle = random.uniform(-max_angle, max_angle)
                    ox = random.uniform(-max_offset, max_offset)
                    oy = random.uniform(-max_offset, max_offset)
                    # Text element
                    txt = dwg.text(
                        ch,
                        insert=(x+ox, y+oy),
                        font_size=font_size,
                        font_family=font_family,
                        fill=fg_color
                    )
                    txt.rotate(angle, center=(x+ox, y+oy))
                    dwg.add(txt)
                    # Next x position
                    x += char_width + char_spacing
        
        # Next y position
        y += font_size * line_spacing
        if y > image_size[1] - font_size:
            return

    def add_text_on_circle(self, dwg, text, font_path, image_size, font_size,
                            char_spacing, fg_color, params):
        cx, cy = params.get('center', (image_size[0]/2, image_size[1]/2))
        r = params.get('radius', min(image_size)/2 - font_size)
        start_angle = params.get('start_angle', -math.pi/2)
        # Total arc length
        circ = 2 * math.pi * r
        # Length per character
        lens = []
        for ch in text:
            # Estimated width
            w = font_size * 0.6
            lens.append(w + char_spacing)
        # Calculate angle for each character
        angles = []
        angle = start_angle
        for l in lens:
            delta = (l/circ) * 2*math.pi
            angle += delta/2
            angles.append(angle)
            angle += delta/2
        # Character placement
        for ch, theta in zip(text, angles):
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            group = dwg.g(fill=fg_color, font_family=font_path, font_size=font_size)
            dwg.add(group)
            
            # Next x
            char_width = font_size * 0.6 if ch != ' ' else font_size * 0.35
            if ch != ' ':
                angle = (random.random() * 2 - 1) * max_angle if max_angle > 0 else 0
                offset_x = (random.random() * 2 - 1) * max_offset if max_offset > 0 else 0
                offset_y = (random.random() * 2 - 1) * max_offset if max_offset > 0 else 0
                
                transform = f"translate({x + offset_x}, {y + offset_y}) rotate({angle}, {font_size/2}, 0)"
                group.add(dwg.text(ch, insert=(0, 0), transform=transform))
            
            # Next x
            x += char_width + char_spacing

class ImageToSVG:
    def __init__(self, output_folder=None):
        self.svg_folder = Path(svg_folder) if output_folder is None else Path(output_folder)
        self.svg_folder.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, img_path, suffix=""):
        img_path = Path(img_path)
        filename = f"{img_path.stem}{suffix}.svg"
        return self.svg_folder / filename
    
    def minify_svg(self, svg_path: Path):
        """SVG optimization and path simplification"""
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
        Embeds an image as base64 data URI in an SVG file.
        
        Args:
            img_path: Path to input image
            output_path: Path for output SVG (default: auto-generated)
            compression_quality: JPEG compression quality (default: 90)
        
        Returns:
            Path to the generated SVG file
        """
        # Prepare output path
        if output_path is None:
            output_path = self.get_output_path(img_path, "_embedded")
        
        # Encode image to base64
        with Image.open(img_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=compression_quality)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            width, height = img.size
        
        # Generate SVG
        img_format = "png" if img_path.lower().endswith(".png") else "jpeg"
        img_data = f"data:image/{img_format};base64,{img_b64}"
        
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
        Vectorizes a bitmap image to SVG outlines.
        
        Args:
            img_path: Path to input image
            output_path: Path for output SVG (default: auto-generated)
            threshold: Threshold for image binarization (0-255)
            simplify: Whether to simplify paths for smaller file size
            width: Output width (default: same as input)
            height: Output height (default: same as input)
            
        Returns:
            Path to the generated SVG file
        """
        # Prepare output path
        if output_path is None:
            output_path = self.get_output_path(img_path, "_vector")
        
        # Load image
        img = Image.open(img_path)
        
        # Resize if needed (optional)
        if width is not None or height is not None:
            width = width or int(img.width * height / img.height)
            height = height or int(img.height * width / img.width)
            img = img.resize((width, height), Image.LANCZOS)
        
        # Convert to grayscale
        img_gray = img.convert('L')
        
        # Noise removal and detail preservation preprocessing
        img_blur = img_gray.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Binarization - adaptive threshold for better detail detection
        np_img = np.array(img_blur)
        _, bin_img = cv2.threshold(np_img, threshold, 255, cv2.THRESH_BINARY)
        
        # Edge detection - adjust low and high threshold for more edges
        edges = cv2.Canny(bin_img, 30, 100)
        
        # Morphological operations to enhance edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours - RETR_TREE to capture all hierarchy structure
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Convert contours to SVG paths
        path_group = dwg.g(fill="none", stroke="black", stroke_width=1)
        
        for i, contour in enumerate(contours):
            if simplify:
                # Simplify contour (Douglas-Peucker algorithm)
                epsilon = 0.005 * cv2.arcLength(contour, True)  # More precise simplification
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Generate SVG path string
            path_data = "M"
            for i, point in enumerate(contour):
                x, y = point[0]
                if i == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            path_data += " Z"  # Close path
            
            # Add path
            path_group.add(dwg.path(d=path_data))
        
        dwg.add(path_group)
        dwg.save()
        
        # SVG optimization
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized path-based SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path
    
    def trace_image(self, img_path, output_path=None, 
                   color_mode="color", quantize=True, num_colors=8,
                   simplify=True, despeckle=True):
        """
        Convert a color image to a layered path-based SVG.
        Create path elements for each color layer.
        Include all internal/external contours.
        """
        # Prepare output path
        if output_path is None:
            output_path = self.get_output_path(img_path, "_traced")
        
        # Load image
        original_img = Image.open(img_path)
        width, height = original_img.size
        
        # Convert to OpenCV
        img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))
        
        if color_mode == "color" and quantize:
            # Image quantization (reduce number of colors)
            img_array = np.float32(img).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(img_array, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            
            # Create path elements for each color layer
            for i in range(num_colors):
                # Create mask for current color
                mask = cv2.inRange(labels.reshape(height, width), i, i)
                
                # Remove noise (optional)
                if despeckle:
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours - RETR_TREE to capture all hierarchy structure
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create SVG path string for current color layer
                path_data = "M"
                for i, contour in enumerate(contours):
                    if simplify:
                        # Simplify contour (Douglas-Peucker algorithm)
                        epsilon = 0.001 * cv2.arcLength(contour, True)  # More precise simplification
                        contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Generate SVG path string
                    path_data += "M"
                    for i, point in enumerate(contour):
                        x, y = point[0]
                        if i == 0:
                            path_data += f" {x},{y}"
                        else:
                            path_data += f" L {x},{y}"
                    path_data += " Z"  # Close path
                    
                    # Add path
                    path_group = dwg.g(fill=f"#{centers[i][2]:02x}{centers[i][1]:02x}{centers[i][0]:02x}", stroke="none")
                    dwg.add(path_group)
                    path_group.add(dwg.path(d=path_data))
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))
        
        # Add text
        dwg.add(dwg.text(
            text,
            insert=(10, height - 10),
            font_size=20,
            fill="black"
        ))
        
        # Save SVG
        dwg.save()
        self.minify_svg(Path(output_path))
        print(f"Created traced SVG: {output_path}")
        return output_path
    
    def extract_shape_from_image(self, img_path, output_path=None, bg_color=(255, 255, 255),
                                threshold=25, simplify=True, edge_detail=True, smooth_curves=True):
        """
        Extract shapes from images such as stamps or logos and convert them to SVG.
        Suitable for images with clear distinction between background and objects.
        Consistently processes internal/external paths and renders them smoothly with Bezier curves.
        """
        # Prepare output path
        if output_path is None:
            output_path = self.get_output_path(img_path, "_shape")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        
        # Create mask based on background color
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create mask based on difference from background color
        for y in range(height):
            for x in range(width):
                b, g, r = img[y, x]
                # Calculate color difference from background color
                diff = abs(int(b) - bg_color[0]) + abs(int(g) - bg_color[1]) + abs(int(r) - bg_color[2])
                # Consider as object if difference is greater than threshold
                if diff > threshold * 3:
                    mask[y, x] = 255
        
        # Remove noise and apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur for smoothing
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        if edge_detail:
            # Preserve details through edge detection
            edges = cv2.Canny(mask, 30, 100)
            mask = cv2.bitwise_or(mask, edges)
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Process all paths as a single compound path
        # Use single path to maintain internal/external consistency
        path_data = ""
        
        # Find contours - NONE mode to preserve all points
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if hierarchy is not None and len(contours) > 0:
            # First identify circular contours
            circle_contours = []
            normal_contours = []
            
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] < 0 and cv2.contourArea(contour) >= 10:  # External contours only
                    if self._is_circle(contour):
                        # Calculate circle parameters
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        circle_contours.append((i, contour, True, (x, y, radius)))
                    else:
                        normal_contours.append((i, contour))
            
            # Process circular contours
            for i, contour, (x, y, radius) in circle_contours:
                if len(path_data) > 0:
                    path_data += " "
                
                # Add circle path
                path_data += self._create_svg_circle_path(x, y, radius)
                
                # Process holes inside this circle
                for j, hole_contour in enumerate(contours):
                    if hierarchy[0][j][3] == i:  # Direct child
                        if cv2.contourArea(hole_contour) < 10:
                            continue
                            
                        # Check if hole is circular
                        if self._is_circle(hole_contour):
                            (hx, hy), hradius = cv2.minEnclosingCircle(hole_contour)
                            path_data += " " + self._create_svg_circle_path(hx, hy, hradius)
                        else:
                            # Process regular holes with smooth curves
                            if simplify:
                                epsilon = 0.0005 * cv2.arcLength(hole_contour, True)
                                hole_contour = cv2.approxPolyDP(hole_contour, epsilon, True)
                            
                            # Internal paths are counter-clockwise
                            if cv2.isContourConvex(hole_contour):
                                hole_contour = np.flip(hole_contour, axis=0)
                                
                            path_data += " " + self._create_smooth_path(hole_contour, smooth_curves)
            
            # Process regular contours
            for i, contour in normal_contours:
                if len(path_data) > 0:
                    path_data += " "
                
                # Process external contour
                if simplify:
                    epsilon = 0.0005 * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # External paths are clockwise
                if not cv2.isContourConvex(contour):
                    contour = np.flip(contour, axis=0)
                
                path_data += self._create_smooth_path(contour, smooth_curves)
                
                # Process holes inside this external contour
                for j, hole_contour in enumerate(contours):
                    if hierarchy[0][j][3] == i:  # Direct child
                        if cv2.contourArea(hole_contour) < 10:
                            continue
                        
                        # Check if hole is circular
                        if self._is_circle(hole_contour):
                            (hx, hy), hradius = cv2.minEnclosingCircle(hole_contour)
                            path_data += " " + self._create_svg_circle_path(hx, hy, hradius)
                        else:
                            # Process regular holes with smooth curves
                            if simplify:
                                epsilon = 0.0005 * cv2.arcLength(hole_contour, True)
                                hole_contour = cv2.approxPolyDP(hole_contour, epsilon, True)
                            
                            # Internal paths are counter-clockwise
                            if cv2.isContourConvex(hole_contour):
                                hole_contour = np.flip(hole_contour, axis=0)
                                
                            path_data += " " + self._create_smooth_path(hole_contour, smooth_curves)
        
        # Add single path (use evenodd rule)
        if path_data:
            # Fill with desired color
            dwg.add(dwg.path(d=path_data, fill="red", fill_rule="evenodd"))
        
        dwg.save()
        
        # Optimize SVG
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized shape SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path

    def create_logo_svg(self, img_path, output_path=None, bg_color=(0, 0, 0),
                       fill_color="red", stroke_color="white", stroke_width=1,
                       threshold=25, simplify=True, smooth_curves=True):
        """
        Convert a logo or icon to SVG.
        Consistently processes internal/external paths to create a clean vector logo.
        Uses Bezier curves for smooth representation of curved parts.
        """
        # Prepare output path
        if output_path is None:
            output_path = self.get_output_path(img_path, "_logo")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarization (considering dark background)
        if np.mean(bg_color) < 128:  # Dark background
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:  # Light background
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Remove noise and smooth edges
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Smooth with Gaussian blur
        binary = cv2.GaussianBlur(binary, (5, 5), 0)
        
        # Find contours directly instead of using edge extraction
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Process all contours as a single compound path
        path_data = ""
        
        # Process external contours first (clockwise)
        ext_contours = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] < 0:  # No parent contour
                if cv2.contourArea(contour) < 10:
                    continue
                
                # Circle detection and special handling
                if self._is_circle(contour):
                    # Calculate circle parameters
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    ext_contours.append((i, contour, True, (x, y, radius)))
                else:
                    # Regular contours are simplified then processed
                    if simplify:
                        epsilon = 0.0005 * cv2.arcLength(contour, True)  # More precise simplification
                        contour = cv2.approxPolyDP(contour, epsilon, True)
                    ext_contours.append((i, contour, False, None))
        
        # Sort by area (largest first)
        ext_contours.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
        
        # Add external contours
        for idx, contour, is_circle, circle_info in ext_contours:
            # External contours are clockwise (SVG rule)
            if not cv2.isContourConvex(contour):
                contour = np.flip(contour, axis=0)
            
            if len(path_data) > 0:
                path_data += " "
            
            if is_circle and circle_info:
                # Process as circle
                x, y, radius = circle_info
                path_data += self._create_svg_circle_path(x, y, radius)
            else:
                # Process as regular path (using curve interpolation)
                path_data += self._create_smooth_path(contour, smooth_curves)
            
            # Process all holes belonging to this external contour
            for i, hole_contour in enumerate(contours):
                if hierarchy[0][i][3] == idx:  # Direct child
                    if cv2.contourArea(hole_contour) < 10:
                        continue
                    
                    # Check if hole is also circular
                    is_hole_circle = self._is_circle(hole_contour)
                    
                    if is_hole_circle:
                        (hx, hy), hradius = cv2.minEnclosingCircle(hole_contour)
                        path_data += " " + self._create_svg_circle_path(hx, hy, hradius)
                    else:
                        # Simplify hole too
                        if simplify:
                            epsilon = 0.0005 * cv2.arcLength(hole_contour, True)
                            hole_contour = cv2.approxPolyDP(hole_contour, epsilon, True)
                        
                        # Internal contours are counter-clockwise (SVG rule)
                        if cv2.isContourConvex(hole_contour):
                            hole_contour = np.flip(hole_contour, axis=0)
                            
                        path_data += " " + self._create_smooth_path(hole_contour, smooth_curves)
        
        # Add single path (use evenodd rule)
        if path_data:
            dwg.add(dwg.path(
                d=path_data, 
                fill=fill_color, 
                stroke=stroke_color if stroke_width > 0 else "none",
                stroke_width=stroke_width,
                fill_rule="evenodd"
            ))
        
        dwg.save()
        
        # Optimize SVG
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized logo SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path

    def _is_circle(self, contour, tolerance=0.01):
        """
        Check if a contour is circular.
        """
        # Use area and perimeter to measure circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False
            
        # Circularity measure (closer to 1 means more circular)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Compare area with minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        area_ratio = area / circle_area if circle_area > 0 else 0
        
        # Consider circular if both measures meet criteria
        return circularity > 0.9 and area_ratio > 0.9
    
    def _create_svg_circle_path(self, x, y, radius):
        """
        Create an SVG circle path.
        """
        # Draw a circle with SVG path commands
        # M cx,cy 
        # m -r,0 
        # a r,r 0 1,0 (r*2),0 
        # a r,r 0 1,0 -(r*2),0
        cx, cy = x, y
        r = radius
        return f"M {cx},{cy-r} a {r},{r} 0 1,0 {r*2},0 a {r},{r} 0 1,0 -{r*2},0 Z"
    
    def _create_smooth_path(self, contour, use_curves=True):
        """
        Create a smooth SVG path. Supports Bezier curve option.
        """
        if len(contour) < 3:
            # Use straight-line path if too few points
            path_data = "M"
            for j, point in enumerate(contour):
                x, y = point[0]
                if j == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            return path_data + " Z"
        
        if not use_curves:
            # Don't use curves if not requested
            path_data = "M"
            for j, point in enumerate(contour):
                x, y = point[0]
                if j == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            return path_data + " Z"
        
        # Create smooth path with Bezier curves
        points = np.array([point[0] for point in contour])
        n_points = len(points)
        
        # Starting point
        path_data = f"M {points[0][0]},{points[0][1]}"
        
        # Apply curves only if enough points
        if n_points > 3:
            # Create smooth curve path
            path_data = ""
            
            # Generate path using Bezier curves or smooth curves
            if len(points) >= 3:
                # Starting point
                path_data = f"M {points[0][0]},{points[0][1]}"
                
                # Connect path with Bezier curve commands
                for j in range(1, len(points), 3):
                    # Check if enough points remain
                    if j + 2 < len(points):
                        # Cubic Bezier curve
                        x1, y1 = points[j]
                        x2, y2 = points[j + 1]
                        x3, y3 = points[j + 2]
                        path_data += f" C {x1},{y1} {x2},{y2} {x3},{y3}"
                    elif j + 1 < len(points):
                        # Quadratic Bezier curve
                        x1, y1 = points[j]
                        x2, y2 = points[j + 1]
                        path_data += f" Q {x1},{y1} {x2},{y2}"
                    else:
                        # Connect last point with a line
                        x, y = points[j]
                        path_data += f" L {x},{y}"
            
                # Close path
                path_data += " Z"
        else:
            # Use straight lines if too few points
            for i in range(1, n_points):
                path_data += f" L {points[i][0]},{points[i][1]}"
        
        # Close path
        return path_data + " Z"

    def svg_from_json(self, json_path, output_path=None, stroke_color="black", 
                     stroke_width=1, fill="none"):
        """
        Create SVG from JSON format path data.
        JSON format: {"paths": [{"points": [[x1, y1], [x2, y2], ...], "closed": true/false}, ...]}
        """
        import json
        
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Prepare output path
        if output_path is None:
            output_path = self.get_output_path(Path(json_path), "_svg")
        
        # Determine canvas size (using min/max coordinates from all points)
        all_points = []
        for path in data.get("paths", []):
            all_points.extend(path.get("points", []))
        
        if not all_points:
            raise ValueError("No path data found in JSON")
        
        # Calculate coordinate range
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add padding
        padding = max(10, int(max(max_x - min_x, max_y - min_y) * 0.05))
        width = max_x - min_x + padding * 2
        height = max_y - min_y + padding * 2
        
        # Coordinate offset (move all points to positive area)
        offset_x = -min_x + padding
        offset_y = -min_y + padding
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Combine all paths into a single compound path
        path_data = ""
        
        # Process paths
        for path in data.get("paths", []):
            points = path.get("points", [])
            closed = path.get("closed", True)
            
            if not points:
                continue
            
            if len(path_data) > 0:
                path_data += " "
            
            # Starting point
            path_data += f"M {points[0][0] + offset_x},{points[0][1] + offset_y}"
            
            # Remaining points
            for i in range(1, len(points)):
                path_data += f" L {points[i][0] + offset_x},{points[i][1] + offset_y}"
            
            # Close path (optional)
            if closed:
                path_data += " Z"
        
        # Add path
        if path_data:
            dwg.add(dwg.path(
                d=path_data,
                fill=fill,
                stroke=stroke_color,
                stroke_width=stroke_width
            ))
        
        dwg.save()
        
        # SVG optimization
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Created SVG from JSON: {output_path} ({size_kb:.1f} KB)")
        return output_path
        
    def extract_all_outlines(self, img_path, output_path=None, stroke_color="black", 
                           stroke_width=3, fill="none", bg_color=(0, 0, 0),
                           threshold=128, min_area_ratio=0.0001):
        """
        Extract all important contours from an image and convert them to SVG.
        Includes both internal and external contours with minimal filtering.
        
        Parameters:
            img_path: Input image path
            output_path: Output SVG path
            stroke_color: Line color
            stroke_width: Line thickness
            fill: Fill color (usually "none")
            bg_color: Background color (for binarization)
            threshold: Binarization threshold
            min_area_ratio: Minimum contour size ratio to keep
        """
        # Prepare output path
        if output_path is None:
            output_path = self.get_output_path(img_path, "_outlines")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        img_area = height * width
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarization based on background color
        if np.mean(bg_color) < 128:  # Dark background
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:  # Light background
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Remove noise (very light processing)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours - get both internal and external contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Create group element - add line style attributes
        outline_group = dwg.g(
            fill=fill, 
            stroke=stroke_color, 
            stroke_width=stroke_width,
            stroke_linejoin="round",  # Round corners (options: miter, round, bevel)
            stroke_linecap="round"    # Round line ends (options: butt, round, square)
        )
        
        # Process all contours (except very small ones)
        min_area = img_area * min_area_ratio
        contour_count = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Ignore too small contours
            if area < min_area:
                continue
            
            # Ignore cases with too few points (noise removal)
            if len(contour) < 3:
                continue
                
            # Contour simplification (keep only important points)
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Generate SVG path
            path_data = "M"
            for j, point in enumerate(approx):
                x, y = point[0]
                if j == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            path_data += " Z"
            
            # Add path
            outline_group.add(dwg.path(d=path_data))
            contour_count += 1
        
        # Add group
        if contour_count > 0:
            dwg.add(outline_group)
        
        # Save SVG
        dwg.save()
        
        # Optimize
        self.minify_svg(Path(output_path))
        
        print(f"Created outline SVG with {contour_count} contours: {output_path}")
        return output_path

    def extract_filled_outlines(self, img_path, output_path=None, stroke_color="black",
                                stroke_width=1, fill_color="#000000", bg_color=(0, 0, 0),
                                threshold=100, min_area_ratio=0.00001, color_extraction=False):
        """
        Extract all significant contours from the image and convert to SVG with filled interiors.
        Transparent areas (alpha=0) are excluded from contour calculations, and SVG background remains transparent.
        """
        if output_path is None:
            output_path = self.get_output_path(img_path, "_filled")

        # --- 1) Read with UNCHANGED flag to preserve alpha channel ---
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Channel separation
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[..., :3]
            alpha = img[..., 3]
        else:
            bgr = img
            alpha = None

        height, width = bgr.shape[:2]
        img_area = height * width

        # Convert to grayscale (convert transparent pixels from BGR too, but mask later)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Binarization based on background color
        if np.mean(bg_color) < 128:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # --- 2) If alpha channel exists, set all transparent pixels to background (0) ---
        if alpha is not None:
            # alpha == 0 means transparent => exclude from contour detection
            binary[alpha == 0] = 0

        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )

        # Keep existing logic...
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
            # Color extraction option
            current_fill = fill_color
            if color_extraction:
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                mean_color = cv2.mean(bgr, mask=mask)
                b, g, r = mean_color[:3]
                current_fill = f"#{int(r):02x}{int(g):02x}{int(b):02x}"

            # Generate path string
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

            # Add holes
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
        Extract contours from image and convert to very smooth SVG.
        Circular shapes are converted to <circle> elements, and other shapes are processed with smooth curves.
        
        Parameters:
            img_path: Input image path
            output_path: Output SVG path
            stroke_color: Line color
            stroke_width: Line thickness
            fill: Fill color (usually "none")
            bg_color: Background color (for binarization)
            threshold: Binarization threshold
            min_area_ratio: Minimum contour size ratio to keep
        """
        if output_path is None:
            output_path = self.get_output_path(img_path, "_smooth")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        img_area = height * width
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply filter for noise removal and smoothing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binarization based on background color
        if np.mean(bg_color) < 128:  # Dark background
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        else:  # Light background
            _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Remove noise (very light processing)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours - use mode to preserve all points (CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Create group element - add line style attributes
        outline_group = dwg.g(
            fill=fill, 
            stroke=stroke_color, 
            stroke_width=stroke_width,
            stroke_linejoin="round",  # Round corners
            stroke_linecap="round"    # Round line ends
        )
        
        # Process all contours (except very small ones)
        min_area = img_area * min_area_ratio
        contour_count = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Ignore too small contours
            if area < min_area:
                continue
            
            # Ignore cases with too few points (noise removal)
            if len(contour) < 3:
                continue
                
            # Check if contour is circular
            is_circle = self._is_circle(contour, tolerance=0.02)
            
            if is_circle:
                # Calculate circle parameters
                (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                
                # Add SVG circle element
                outline_group.add(dwg.circle(
                    center=(center_x, center_y),
                    r=radius,
                    stroke_width=stroke_width
                ))
                contour_count += 1
            else:
                # Curve interpolation based on point spacing
                # Use more points for smoother results
                approx_points = []
                for j in range(len(contour)):
                    p1 = contour[j][0]
                    p2 = contour[(j + 1) % len(contour)][0]
                    
                    # Calculate distance
                    dist = np.linalg.norm(p2 - p1)
                    
                    # Add current point
                    approx_points.append(p1)
                    
                    # If distance > 5, add intermediate points
                    if dist > 5:
                        num_points = max(1, int(dist / 5))
                        for k in range(1, num_points):
                            t = k / num_points
                            mid = p1 * (1 - t) + p2 * t
                            approx_points.append(mid)
                
                # Create smooth curve path
                path_data = ""
                
                # Generate path using Bezier curves or smooth curves
                if len(approx_points) >= 3:
                    # Starting point
                    path_data = f"M {approx_points[0][0]},{approx_points[0][1]}"
                    
                    # Connect path with Bezier curve commands
                    for j in range(1, len(approx_points), 3):
                        # Check if enough points remain
                        if j + 2 < len(approx_points):
                            # Cubic Bezier curve
                            x1, y1 = approx_points[j]
                            x2, y2 = approx_points[j + 1]
                            x3, y3 = approx_points[j + 2]
                            path_data += f" C {x1},{y1} {x2},{y2} {x3},{y3}"
                        elif j + 1 < len(approx_points):
                            # Quadratic Bezier curve
                            x1, y1 = approx_points[j]
                            x2, y2 = approx_points[j + 1]
                            path_data += f" Q {x1},{y1} {x2},{y2}"
                        else:
                            # Connect last point with a line
                            x, y = approx_points[j]
                            path_data += f" L {x},{y}"
            
                # Close path
                path_data += " Z"
            
            # Add path
            dwg.add(dwg.path(d=path_data, fill=fill, stroke=stroke_color, stroke_width=stroke_width))
        
        # Save SVG
        dwg.save()
        
        # Optimize SVG
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Created smooth outline SVG with {len(contours)} contours: {output_path} ({size_kb:.1f} KB)")
        return output_path
        
    def extract_logo_paths(self, img_path, output_path=None, bg_color=(0, 0, 0)):
        """
        Extract paths from logos or simple graphics using a more reliable method.
        Tries multiple approaches and selects the best result.
        
        This function is particularly suitable for simple logo images like apple.svg or tesla_logo.svg.
        """
        if output_path is None:
            output_path = self.get_output_path(img_path, "_logo_paths")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        height, width = img.shape[:2]
        img_area = height * width
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing methods
        methods = []
        
        # Method 1: Otsu thresholding
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        methods.append(("otsu", binary1))
        
        # Method 2: Adaptive thresholding
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        methods.append(("adaptive", binary2))
        
        # Method 3: Fixed threshold
        if np.mean(bg_color) < 128:  # Dark background
            _, binary3 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        else:  # Light background
            _, binary3 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        methods.append(("fixed", binary3))
        
        # Find optimal method
        best_contours = []
        best_method = ""
        
        for method_name, binary in methods:
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Keep only sufficiently large contours
            significant_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > img_area * 0.001:  # Greater than 0.1% of image area
                    perimeter = cv2.arcLength(contour, True)
                    # Complexity measurement (higher means more complex)
                    complexity = perimeter * perimeter / (4 * np.pi * area)
                    significant_contours.append((contour, area, complexity))
            
            # Evaluate if contour results are better
            if len(significant_contours) > 0:
                if len(best_contours) == 0 or len(significant_contours) > len(best_contours):
                    best_contours = significant_contours
                    best_method = method_name
        
        # Generate SVG
        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}"
        )
        
        # Add all contours
        for contour, _, _ in best_contours:
            # Simplify
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Generate path data
            path_data = "M"
            for i, point in enumerate(approx):
                x, y = point[0]
                if i == 0:
                    path_data += f" {x},{y}"
                else:
                    path_data += f" L {x},{y}"
            path_data += " Z"
            
            # Add path
            dwg.add(dwg.path(d=path_data, fill="none", stroke="black", stroke_width="1"))
        
        # Save SVG
        dwg.save()
        
        # Optimize
        self.minify_svg(Path(output_path))
        
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"Saved optimized logo SVG: {output_path} ({size_kb:.1f} KB)")
        return output_path
    
    def _add_holes_recursively(self, dwg, group, contours, hierarchy, parent_idx, opttolerance):
        """
        Recursively process contour hierarchy to add holes.
        """
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == parent_idx:  # Current contour is a direct child
                if cv2.contourArea(contour) < 10:  # Ignore too small contours
                    continue
                    
                # Curve simplification
                epsilon = opttolerance * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Generate SVG path
                path_data = "M"
                for j, point in enumerate(approx):
                    x, y = point[0]
                    if j == 0:
                        path_data += f" {x},{y}"
                    else:
                        path_data += f" L {x},{y}"
                path_data += " Z"
                
                # Add hole (filled with white)
                path_element = dwg.path(d=path_data, fill="white")
                group.add(path_element)
                
                # Check for and recursively process contours inside this hole
                self._add_holes_recursively(dwg, group, contours, hierarchy, i, opttolerance)

# Example usage
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
    
    