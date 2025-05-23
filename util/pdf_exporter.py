# pdf_exporter.py
import tempfile
from copy import deepcopy
import xml.etree.ElementTree as ET
from cairosvg import svg2pdf
import numpy as np
import os
import torch

class PDFExporter:
    """
    Exports optimized vector elements back into an SVG and renders PDF.
    """
    def __init__(self, svg_path: str, canvas_size: tuple, viewbox_size: tuple, 
                 alpha_upper_bound: float = 1.0, stroke_width: float = 3.0):
        self.svg_path = svg_path
        self.canvas_w, self.canvas_h = canvas_size
        self.view_w, self.view_h = viewbox_size
        self.norm_scale = 2 / max(self.view_w, self.view_h)
        self.alpha_upper_bound = alpha_upper_bound
        self.stroke_width = stroke_width
        # config["postprocessing"].get("linewidth", 3.0)

    def _remove_styles(self, root):
        for elem in root.iter():
            for attr in ['style', 'stroke', 'fill', 'stroke-opacity', 'fill-opacity']:
                if attr in elem.attrib:
                    del elem.attrib[attr]
        to_remove = [e for e in root.iter() if e.tag.endswith('style')]
        for child in to_remove:
            parent = self._find_parent(root, child)
            if parent is not None:
                parent.remove(child)

    def _find_parent(self, root, child):
        for elem in root.iter():
            for sub in elem:
                if sub is child:
                    return elem
        return None

    def export(self, x, y, r, theta, v, c, output_path: str, svg_hollow=False):
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Convert torch.Tensor to numpy array
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()  # radians
        v_np = v.detach().cpu().numpy()
        c_np = torch.sigmoid(c).detach().cpu().numpy()  # shape (N, 3)
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))
        
        tree = ET.parse(self.svg_path)
        root = tree.getroot()
        self._remove_styles(root)
        # set canvas
        root.attrib['width'] = str(self.canvas_w)
        root.attrib['height'] = str(self.canvas_h)
        root.attrib['viewBox'] = f"{-self.canvas_w/2} {-self.canvas_h/2} {self.canvas_w} {self.canvas_h}"
        children = list(root)
        for ch in children:
            root.remove(ch)
        N = len(x_np)
        for i in reversed(range(N)):
            theta_deg = np.degrees(theta_np[i])
            transform = (
                f"translate({x_np[i]-self.canvas_w/2},{y_np[i]-self.canvas_h/2}) "
                f"rotate({theta_deg}) "
                f"scale({r_np[i]}) "
                f"scale({self.norm_scale}) " 
                f"translate({-self.view_w/2},{-self.view_h/2})"
            )
            g = ET.Element('g', {'transform': transform})
            #g.attrib["transform"] = transform_str
            # Set color (stroke), convert RGB from [0,1] to integers 0-255
            r_color, g_color, b_color = c_np[i]
            r_int = int(np.clip(r_color * 255, 0, 255))
            g_int = int(np.clip(g_color * 255, 0, 255))
            b_int = int(np.clip(b_color * 255, 0, 255))
            if svg_hollow:
                g.attrib["stroke"] = f"rgb({r_int},{g_int},{b_int})"
                g.attrib["stroke-opacity"] = str(alpha_vals[i])
                g.attrib["stroke-width"] = str(self.stroke_width)
                g.attrib["fill"] = f"rgb({r_int},{g_int},{b_int})"
                g.attrib["fill-opacity"] = str(0.0)
            else:
                g.attrib["fill"] = f"rgb({r_int},{g_int},{b_int})"
                g.attrib["fill-opacity"] = str(alpha_vals[i])
            
            for elem in children:
                g.append(deepcopy(elem))
            root.append(g)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
        tree.write(tmp.name)
        svg2pdf(url=tmp.name, write_to=output_path)
        tmp.close()
        os.remove(tmp.name)

    # For figure 1
    def export_dropout_right_third(self,
                                   x, y, r, theta, v, c,
                                   output_path: str,
                                   svg_hollow=False):
        """
        • Left ⅔ area splats         → Keep all
        • Right ⅓ area (x > 2W/3) → Drop with linear probability proportional to x
        Then save PDF with self.export(...)
        """
        W = self.canvas_w

        with torch.no_grad():
            keep_left = x < ( W / 2)
            prob      = torch.where(keep_left,
                                    torch.ones_like(x),
                                    (W - x) / (W / 2))  # decreases as x increases
            keep_idx  = keep_left | (torch.rand_like(prob) < prob)

            # Filtered parameters
            x_d, y_d, r_d     = [t[keep_idx] for t in (x, y, r)]
            theta_d, v_d, c_d = [t[keep_idx] for t in (theta, v, c)]

        # Export to PDF
        self.export(x_d, y_d, r_d, theta_d, v_d, c_d,
                    output_path=output_path,
                    svg_hollow=svg_hollow)
        print(f"drop-out PDF saved to {output_path}")