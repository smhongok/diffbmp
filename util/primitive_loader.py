from pathlib import Path
from cairosvg import svg2png
from PIL import Image
import numpy as np
import tempfile
import torch
from svgpathtools import svg2paths2
import cv2
import re
import xml.etree.ElementTree as ET
from typing import Union, List, Tuple
import base64
import io

class PrimitiveLoader:
    """
    Loads SVG, PNG, or JPG files as primitives and converts them to alpha bitmap tensors.
    Supports mixed primitive types in a single loader.
    Backward compatible with SVGLoader interface.
    """
    
    def __init__(self, primitive_paths: Union[str, List[str]], output_width: int = 128, device=None, bg_threshold: int = 250):
        # Accept single path or list of paths
        if isinstance(primitive_paths, (list, tuple)):
            self.primitive_paths = primitive_paths
        else:
            self.primitive_paths = [primitive_paths]
        
        self.output_width = output_width
        self.device = device or torch.device('cpu')
        self.bg_threshold = bg_threshold  # Threshold for background removal (0-255)
        
        # Analyze primitive types and set dimensions
        self.primitive_types = []
        self.primitive_data = []
        
        for path in self.primitive_paths:
            ptype, data = self._analyze_primitive(path)
            self.primitive_types.append(ptype)
            self.primitive_data.append(data)
        
        # Use first primitive to set canvas size (backward compatibility)
        first_data = self.primitive_data[0]
        if self.primitive_types[0] == 'svg':
            self.svg_width, self.svg_height = first_data['width'], first_data['height']
        else:
            # For raster primitives, use the actual image dimensions
            self.svg_width = self.svg_height = self.output_width

    def _analyze_primitive(self, path: str) -> Tuple[str, dict]:
        """Analyze primitive type and extract metadata"""
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        
        if ext == '.svg':
            # SVG analysis (existing logic)
            paths, attributes, svg_attrs = svg2paths2(path)
            viewbox = svg_attrs.get('viewBox')
            if viewbox is None:
                raise ValueError(f"SVG missing viewBox: {path}")
            parts = viewbox.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid viewBox: {viewbox}")
            min_x, min_y, width, height = map(float, parts)
            if min_x != 0 or min_y != 0:
                raise ValueError(f"viewBox must start at 0,0 but got {min_x, min_y}")
            
            return 'svg', {
                'path': path,
                'width': width,
                'height': height,
                'viewbox': viewbox
            }
            
        elif ext in ['.png', '.jpg', '.jpeg']:
            # Raster analysis
            img = Image.open(path).convert('RGBA')
            width, height = img.size
            
            return 'raster', {
                'path': path,
                'width': width,
                'height': height,
                'format': ext[1:],  # Remove the dot
                'image': img
            }
        else:
            raise ValueError(f"Unsupported primitive format: {ext}")

    def get_primitive_size(self, index: int = 0):
        """Return the (width, height) of the specified primitive"""
        data = self.primitive_data[index]
        return data['width'], data['height']

    def load_alpha_bitmap(self):
        """
        Render each primitive to an alpha bitmap and return:
        - If one primitive: tensor of shape [H, W]
        - If multiple primitives: tensor of shape [p, H, W]
        """
        bitmaps = []
        
        for i, (ptype, data) in enumerate(zip(self.primitive_types, self.primitive_data)):
            if ptype == 'svg':
                bitmap = self._load_svg_bitmap(data)
            else:  # raster
                bitmap = self._load_raster_bitmap(data)
            
            bitmaps.append(bitmap)
        
        # Stack or return single
        if len(bitmaps) == 1:
            return bitmaps[0]
        else:
            return torch.stack(bitmaps, dim=0)

    def _load_svg_bitmap(self, data: dict) -> torch.Tensor:
        """Load SVG as bitmap (existing SVGLoader logic)"""
        path = data['path']
        svg_bytes = Path(path).read_bytes()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
            svg2png(bytestring=svg_bytes, write_to=tmp.name, 
                   output_width=self.output_width, output_height=self.output_width)
            img = Image.open(tmp.name).convert('RGBA')
            arr = np.array(img)

        # Pad to square
        h, w, _ = arr.shape
        new_size = max(h, w)
        pad_h = (new_size - h) // 2
        pad_w = (new_size - w) // 2
        padded = np.pad(
            arr,
            ((pad_h, new_size - h - pad_h), (pad_w, new_size - w - pad_w), (0, 0)),
            mode='constant', constant_values=0
        )
        alpha = padded[:, :, 3]
        binary = (alpha > 0).astype(np.float32)
        return torch.tensor(binary, device=self.device)

    def _load_raster_bitmap(self, data: dict) -> torch.Tensor:
        """Load raster image as bitmap with background removal"""
        img = data['image']
        
        # Resize to target size while maintaining aspect ratio
        img.thumbnail((self.output_width, self.output_width), Image.Resampling.LANCZOS)
        
        # Convert to RGB first, then create alpha channel based on background removal
        if img.mode == 'RGBA':
            # If already has alpha, preserve it but also apply background removal
            rgb_img = img.convert('RGB')
            original_alpha = np.array(img)[:, :, 3]
        else:
            rgb_img = img.convert('RGB')
            original_alpha = None
        
        # Convert to grayscale to detect background
        gray_img = rgb_img.convert('L')
        rgb_arr = np.array(rgb_img)
        gray_arr = np.array(gray_img)
        
        # Create alpha channel: transparent for background pixels
        # Background pixels are those with grayscale value > bg_threshold
        alpha_from_bg = (gray_arr <= self.bg_threshold).astype(np.uint8) * 255
        
        # If original image had alpha, combine with background removal
        if original_alpha is not None:
            # Keep pixels that are both non-background AND originally non-transparent
            alpha_channel = np.minimum(alpha_from_bg, original_alpha)
        else:
            alpha_channel = alpha_from_bg
        
        # Create RGBA array
        h, w = rgb_arr.shape[:2]
        rgba_arr = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_arr[:, :, :3] = rgb_arr  # RGB channels
        rgba_arr[:, :, 3] = alpha_channel  # Alpha channel
        
        # Pad to square
        new_size = max(h, w, self.output_width)
        pad_h = (new_size - h) // 2
        pad_w = (new_size - w) // 2
        padded = np.pad(
            rgba_arr,
            ((pad_h, new_size - h - pad_h), (pad_w, new_size - w - pad_w), (0, 0)),
            mode='constant', constant_values=0
        )
        
        # Use alpha channel for mask
        alpha = padded[:, :, 3]
        binary = (alpha > 0).astype(np.float32)
        return torch.tensor(binary, device=self.device)

    def get_html_embedding_data(self, index: int) -> dict:
        """
        Get data needed for HTML embedding of this primitive.
        Returns different data based on primitive type.
        """
        ptype = self.primitive_types[index]
        data = self.primitive_data[index]
        
        if ptype == 'svg':
            # Return SVG content for inline embedding
            with open(data['path'], 'r') as f:
                svg_content = f.read()
            return {
                'type': 'svg',
                'content': svg_content,
                'width': data['width'],
                'height': data['height']
            }
        else:
            # Return base64-encoded image for data URL embedding
            img = data['image']
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                'type': 'raster',
                'data_url': f"data:image/png;base64,{img_str}",
                'width': data['width'],
                'height': data['height']
            }

    def has_raster_primitives(self) -> bool:
        """Check if any loaded primitives are raster (PNG/JPG)"""
        return any(ptype == 'raster' for ptype in self.primitive_types)

    # Backward compatibility methods
    def get_svg_size(self):
        """Backward compatibility with SVGLoader"""
        return self.get_primitive_size(0)
    
    @property
    def svg_path(self):
        """Backward compatibility with SVGLoader"""
        return self.primitive_paths

    def svg_to_image(self) -> np.ndarray:
        """Backward compatibility with SVGLoader"""
        if self.primitive_types[0] == 'svg':
            png_bytes = svg2png(url=self.primitive_paths[0], output_width=self.output_width)
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            return img
        else:
            # For raster primitives, convert to grayscale
            img = self.primitive_data[0]['image'].convert('L')
            img.thumbnail((self.output_width, self.output_width), Image.Resampling.LANCZOS)
            return np.array(img)

    def extract_angles(self, img: np.ndarray) -> np.ndarray:
        """Backward compatibility with SVGLoader"""
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        angles = (np.arctan2(gy, gx) + np.pi) % np.pi
        return angles

    def compute_metrics(self, angles: np.ndarray, mask: np.ndarray):
        """Backward compatibility with SVGLoader"""
        hist, bins = np.histogram(angles, bins=36, range=(0, np.pi))
        hist = hist.astype(float) / hist.sum()
        centers = 0.5 * (bins[:-1] + bins[1:])
        idx0 = np.argmin(np.abs(centers - 0))
        idx90 = np.argmin(np.abs(centers - np.pi / 2))
        straight_mass = hist[idx0] + hist[idx90]
        R = np.sqrt((hist * np.cos(centers)).sum() ** 2 + (hist * np.sin(centers)).sum() ** 2)
        cir_var = 1 - R
        p = hist + 1e-8
        ent = -np.sum(p * np.log(p))
        edges = (angles > -1)
        lap = cv2.Laplacian(mask.astype(np.float32), cv2.CV_32F, ksize=3)
        curv = np.mean(np.abs(lap))
        return straight_mass, cir_var, ent, curv

    def classify_svg(self, curve_thresh: float = 0.5, straight_thresh: float = 0.5) -> str:
        """Backward compatibility with SVGLoader"""
        if self.primitive_types[0] != 'svg':
            return 'raster'  # New classification for raster primitives
            
        tree = ET.parse(self.primitive_paths[0])
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        curve_count = straight_count = 0
        for path in root.findall('.//svg:path', ns):
            d = path.get('d')
            if not d:
                continue
            cmds = re.findall(r'[A-Za-z]', d)
            for c in cmds:
                if c in 'QqCcSsTtAa': curve_count += 1
                elif c in 'LlHhVv': straight_count += 1
        total = curve_count + straight_count + 1e-8
        curve_ratio = curve_count / total
        straight_ratio = straight_count / total
        if curve_ratio > curve_thresh: return 'curve'
        if straight_ratio > straight_thresh: return 'straight'
        return 'mixed'
