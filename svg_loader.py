# svg_loader.py
from cairosvg import svg2png
from PIL import Image
import numpy as np
import tempfile
import torch
from svgpathtools import svg2paths2

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
        # render SVG to PNG
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
            svg2png(url=self.svg_path, write_to=tmp.name, output_width=self.output_width)
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
