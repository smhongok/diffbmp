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

class SVGLoader:
    """
    Loads one or multiple SVG files (or generated text-based SVG paths), validates viewBox,
    and converts to padded binary alpha bitmap tensor or tensor stack.
    """
    def __init__(self, svg_path, output_width: int = 128, device=None):
        # Accept single path or list of paths
        if isinstance(svg_path, (list, tuple)):
            self.svg_path = svg_path
        else:
            self.svg_path = [svg_path]
        self.output_width = output_width
        self.device = device or torch.device('cpu')

        # Validate viewBox for the first SVG to set size
        first_path = self.svg_path[0]
        paths, attributes, svg_attrs = svg2paths2(first_path)
        viewbox = svg_attrs.get('viewBox')
        if viewbox is None:
            raise ValueError(f"SVG missing viewBox: {first_path}")
        parts = viewbox.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid viewBox: {viewbox}")
        min_x, min_y, width, height = map(float, parts)
        if min_x != 0 or min_y != 0:
            raise ValueError(f"viewBox must start at 0,0 but got {min_x, min_y}")
        self.svg_width, self.svg_height = width, height

    def get_svg_size(self):
        """
        Return the (width, height) of the loaded SVG(s).
        """
        return self.svg_width, self.svg_height

    def load_alpha_bitmap(self):
        """
        Render each SVG to an alpha bitmap, pad to square, and return:
        - If one SVG: tensor of shape [H, W]
        - If multiple SVGs: tensor of shape [p, H, W]
        """
        bitmaps = []
        for path in self.svg_path:
            # Render SVG to PNG via cairosvg
            svg_bytes = Path(path).read_bytes()
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                svg2png(bytestring=svg_bytes, write_to=tmp.name, output_width=self.output_width)
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
            tensor = torch.tensor(binary, device=self.device)
            bitmaps.append(tensor)

        # Stack or return single
        if len(bitmaps) == 1:
            return bitmaps[0]
        else:
            return torch.stack(bitmaps, dim=0)

    def svg_to_image(self) -> np.ndarray:
        png_bytes = svg2png(url=self.svg_path[0], output_width=self.output_width)
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return img

    def extract_angles(self, img: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        angles = (np.arctan2(gy, gx) + np.pi) % np.pi
        return angles

    def compute_metrics(self, angles: np.ndarray, mask: np.ndarray):
        # (unchanged)
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
        tree = ET.parse(self.svg_path[0])
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