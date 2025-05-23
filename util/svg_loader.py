# svg_loader.py
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
    Loads an SVG file, validates viewBox,
    converts to padded binary alpha bitmap tensor.
    """
    def __init__(self, svg_path: str, output_width: int = 128, device=None):
        self.svg_path = svg_path
        self.output_width = output_width
        self.device = device or torch.device('cpu')
        paths, attributes, svg_attrs = svg2paths2(svg_path)
        viewbox = svg_attrs.get('viewBox')
        if viewbox is None:
            raise ValueError("SVG missing viewBox")
        parts = viewbox.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid viewBox: {viewbox}")
        min_x, min_y, width, height = map(float, parts)
        if min_x != 0 or min_y != 0:
            raise ValueError(f"viewBox must start at 0,0 but got {min_x, min_y}")
        self.svg_width, self.svg_height = width, height

    def get_svg_size(self):
        # Return the size of the SVG
        return self.svg_width, self.svg_height

    def load_alpha_bitmap(self):
        # Read the entire SVG into bytes
        svg_bytes = Path(self.svg_path).read_bytes()
        # render SVG to PNG
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
            svg2png(bytestring=svg_bytes, write_to=tmp.name, output_width=self.output_width)
            img = Image.open(tmp.name).convert('RGBA')
            arr = np.array(img)
        # pad to square
        h, w, _ = arr.shape
        new_size = max(h, w)
        pad_h = (new_size - h) // 2
        pad_w = (new_size - w) // 2
        padded = np.pad(arr,
                        ((pad_h, new_size-h-pad_h), (pad_w, new_size-w-pad_w), (0,0)),
                        mode='constant', constant_values=0)
        alpha = padded[:, :, 3]
        binary = (alpha > 0).astype(np.float32)
        tensor = torch.tensor(binary, device=self.device)
        return tensor
    
    def svg_to_image(self) -> np.ndarray:
        png_bytes = svg2png(url=self.svg_path, output_width=self.output_width)
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)  # as grayscale
        return img

    def extract_angles(self, img: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        angles = (np.arctan2(gy, gx) + np.pi) % np.pi  # [0, π)
        return angles
    
    def compute_metrics(self, angles: np.ndarray, mask: np.ndarray):
        # 1) Orientation histogram
        hist, bins = np.histogram(angles, bins=36, range=(0, np.pi))
        hist = hist.astype(float) / hist.sum()
        centers = 0.5*(bins[:-1] + bins[1:])

        # 2) straight_mass: 0° & 90° only (exclude diagonals) 
        idx0  = np.argmin(np.abs(centers -   0    ))
        idx90 = np.argmin(np.abs(centers - np.pi/2))
        straight_mass = hist[idx0] + hist[idx90]

        # 3) circular variance V = 1 - R  :contentReference[oaicite:5]{index=5}
        R = np.sqrt((hist*np.cos(centers)).sum()**2 +
                    (hist*np.sin(centers)).sum()**2)
        cir_var = 1 - R

        # 4) entropy H = -Σ p log p  :contentReference[oaicite:6]{index=6}
        p = hist + 1e-8
        ent = -np.sum(p * np.log(p))

        # 5) curvature: Laplacian on binary edge mask  :contentReference[oaicite:7]{index=7}
        edges = (angles > -1)  # dummy to get same shape
        lap = cv2.Laplacian(mask.astype(np.float32), cv2.CV_32F, ksize=3)
        curv = np.mean(np.abs(lap))

        return straight_mass, cir_var, ent, curv
    
    def classify_svg(self, curve_thresh: float = 0.5, straight_thresh: float = 0.5) -> str:
        """
        Calculate command ratios from the d string of <path> elements to classify as 'straight'/'curve'/'mixed'.
        - Curve commands: Q, q, C, c, S, s, T, t, A, a
        - Straight commands: L, l, H, h, V, v
        """
        # XML parsing
        tree = ET.parse(self.svg_path)
        root = tree.getroot()
        # Namespace handling
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        curve_count = straight_count = 0
        # Iterate through all <path> elements
        for path in root.findall('.//svg:path', ns):
            d = path.get('d')
            if not d:
                continue
            # Extract only command characters
            cmds = re.findall(r'[A-Za-z]', d)
            for c in cmds:
                if c in 'QqCcSsTtAa':
                    curve_count += 1
                elif c in 'LlHhVv':
                    straight_count += 1

        total = curve_count + straight_count + 1e-8
        curve_ratio = curve_count / total
        straight_ratio = straight_count / total

        # Classification criteria
        if curve_ratio  > curve_thresh:
            return 'curve'
        if straight_ratio > straight_thresh:
            return 'straight'
        return 'mixed'