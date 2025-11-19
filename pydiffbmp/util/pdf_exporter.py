import tempfile
import xml.etree.ElementTree as ET
from cairosvg import svg2pdf, svg2png
import numpy as np
import os
import torch
import glob
import cv2
from copy import deepcopy
import shutil

# Register default namespace so it exports without prefixes
ET.register_namespace('', 'http://www.w3.org/2000/svg')
SVG_NS = 'http://www.w3.org/2000/svg'

class PDFExporter:
    """
    Exports optimized vector elements into a combined SVG and renders to PDF.
    Supports cycling through multiple SVG shape templates (e.g., per-letter primitives).
    """
    def __init__(self,
                 svg_paths,
                 canvas_size: tuple,
                 viewbox_size: tuple,
                 alpha_upper_bound: float = 1.0,
                 stroke_width: float = 3.0):
        # Accept single string or list/tuple of strings
        if isinstance(svg_paths, (list, tuple)):
            self.svg_paths = list(svg_paths)
        else:
            self.svg_paths = [svg_paths]

        self.canvas_w, self.canvas_h = canvas_size
        self.view_w, self.view_h = viewbox_size
        self.norm_scale = 2 / max(self.view_w, self.view_h)
        self.alpha_upper_bound = alpha_upper_bound
        self.stroke_width = stroke_width

    def _remove_styles(self, root):
        # Remove style-related attributes and <style> tags
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

    def export_svg(self,
               x: torch.Tensor,
               y: torch.Tensor,
               r: torch.Tensor,
               theta: torch.Tensor,
               v: torch.Tensor,
               c: torch.Tensor,
               output_path: str,
               svg_hollow: bool = False):
        # Convert tensors to numpy arrays
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        c_np = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))

        # Create new SVG root element
        root = ET.Element(f'{{{SVG_NS}}}svg', {
            'width': str(self.canvas_w),
            'height': str(self.canvas_h),
            'viewBox': f"{-self.canvas_w/2} {-self.canvas_h/2} {self.canvas_w} {self.canvas_h}"
        })

        N = len(x_np)
        p = len(self.svg_paths)

        for i in reversed(range(N)):
            idx = i % p
            # Re-parse template to get fresh children
            tree = ET.parse(self.svg_paths[idx])
            template_root = tree.getroot()
            self._remove_styles(template_root)
            children = list(template_root)

            # Compute transform string
            theta_deg = np.degrees(theta_np[i])
            transform = (
                f"translate({x_np[i]-self.canvas_w/2:.3f},{y_np[i]-self.canvas_h/2:.3f}) "
                f"rotate({theta_deg:.3f}) "
                f"scale({r_np[i]:.3f}) "
                f"scale({self.norm_scale:.4f}) "
                f"translate({-self.view_w/2},{-self.view_h/2})"
            )
            g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})

            # Style attributes
            r_color, g_color, b_color = c_np[i]
            r_int = int(np.clip(r_color * 255, 0, 255))
            g_int = int(np.clip(g_color * 255, 0, 255))
            b_int = int(np.clip(b_color * 255, 0, 255))
            if svg_hollow:
                g.attrib.update({
                    'stroke': f'rgb({r_int},{g_int},{b_int})',
                    'stroke-opacity': f"{alpha_vals[i]:.4f}",
                    'stroke-width': str(self.stroke_width),
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': '0'
                })
            else:
                g.attrib.update({
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': f"{alpha_vals[i]:.4f}"
                })

            # Append fresh children
            for child in children:
                g.append(deepcopy(child))
            root.append(g)

        # Write combined SVG to temp and convert to PDF
        svg_path = output_path.replace(".pdf", ".svg")
        tree = ET.ElementTree(root)
        tree.write(svg_path, encoding='utf-8', xml_declaration=True)
        svg2pdf(url=svg_path, write_to=output_path)
        # tmp.close()
        # os.remove(tmp.name)

    def export(self,
            x: torch.Tensor,
            y: torch.Tensor,
            r: torch.Tensor,
            theta: torch.Tensor,
            v: torch.Tensor,
            c: torch.Tensor,
            output_path: str,
            svg_hollow: bool = False,
            html_extra_path = "output_webpage/src/index.html",
            export_pdf: bool = False,
            html_extra_meta: dict = {}):

        SVG_NS = "http://www.w3.org/2000/svg"

        output_svg_path = output_path.replace(".pdf", ".svg")
        output_html_path = output_path.replace(".pdf", ".html")

        # 1. Tensor → numpy
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        c_np = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))

        N = len(x_np)
        p = len(self.svg_paths)

        # --- For SVG file (no transform) ---
        # 2. Create SVG root
        root_svg = ET.Element(f'{{{SVG_NS}}}svg', {
            'id': 'svgsplat1',
            'style': 'overflow: visible;',
            'width': str(int(self.canvas_w)),
            'height': str(int(self.canvas_h)),
            'viewBox': f"{int(-self.canvas_w/2)} {int(-self.canvas_h/2)} {int(self.canvas_w)} {int(self.canvas_h)}"
        })
        wrapper_g_svg = ET.Element('g', {'id': 'wrapper'})

        # Create elements
        for i in reversed(range(N)):
            idx = (N-i-1) % p
            tree = ET.parse(self.svg_paths[idx])
            template_root = tree.getroot()
            self._remove_styles(template_root)
            children = list(template_root)

            theta_deg = np.degrees(theta_np[i])
            transform = (
                f"translate({x_np[i]-self.canvas_w/2:.3f},{y_np[i]-self.canvas_h/2:.3f}) "
                f"rotate({theta_deg:.3f}) "
                f"scale({r_np[i]:.3f}) "
                f"scale({self.norm_scale:.4f}) "
                f"translate({-self.view_w/2},{-self.view_h/2})"
            )
            g = ET.Element('g', {'transform': transform})

            r_color, g_color, b_color = c_np[i]
            r_int = int(np.clip(r_color * 255, 0, 255))
            g_int = int(np.clip(g_color * 255, 0, 255))
            b_int = int(np.clip(b_color * 255, 0, 255))
            if svg_hollow:
                g.attrib.update({
                    'stroke': f'rgb({r_int},{g_int},{b_int})',
                    'stroke-opacity': f"{alpha_vals[i]:.4f}",
                    'stroke-width': str(self.stroke_width),
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': '0'
                })
            else:
                g.attrib.update({
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': f"{alpha_vals[i]:.4f}"
                })

            for child in children:
                g.append(deepcopy(child))
            wrapper_g_svg.append(g)

        root_svg.append(wrapper_g_svg)

        # 3. Save as SVG file
        tree_svg = ET.ElementTree(root_svg)
        tree_svg.write(output_svg_path, encoding='utf-8', xml_declaration=True)

        # --- For HTML embed (with transform) ---
        root_html = ET.Element(f'{{{SVG_NS}}}svg', {
            'id': 'svgsplat1',
            'style': 'overflow: visible;',
            'width': str(self.canvas_w),
            'height': str(self.canvas_h),
            'viewBox': f"{-self.canvas_w/2} {-self.canvas_h/2} {self.canvas_w} {self.canvas_h}"
        })
        wrapper_g_html = ET.Element('g', {'id': 'wrapper', 'transform': 'translate(0,0)'})
        # Copy and add existing g
        for g in list(wrapper_g_svg):
            wrapper_g_html.append(deepcopy(g))
        root_html.append(wrapper_g_html)

        # Save as temporary SVG file (for HTML)
        tmp_html_svg = output_svg_path.replace(".svg", ".html_temp.svg")
        tree_html = ET.ElementTree(root_html)
        tree_html.write(tmp_html_svg, encoding='utf-8', xml_declaration=True)
        with open(tmp_html_svg, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        os.remove(tmp_html_svg)

        # 6. Define HTML header/footer
        meta_tags = []
        for k, v in html_extra_meta.items():
            meta_tags.append(f'<meta name="{k}" content="{v}">')
        html_head = f"""<!DOCTYPE html>
                    <html lang="en">
                    <head>
                    <meta charset="UTF-8">
                    <meta name="numClass" content="{len(self.svg_paths)}">
                    {chr(10).join(meta_tags)}
                    <title>HTML Demo</title>
                    <link rel="stylesheet" href="demo_html.css">
                    </head>
                    <body id="demo_html">
                    """
        html_tail = """
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/Draggable.min.js"></script>
                    <script src="demo_html.js"></script>
                    </body>
                    </html>
                    """

        # 7. Save as HTML
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_head)
            f.write(svg_content)
            f.write(html_tail)

        # 8. Copy to desired location (overwrite)
        shutil.copyfile(output_html_path, html_extra_path)

        # 9. If exporting PDF as well
        if export_pdf:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
            tree_svg.write(tmp.name)
            svg2pdf(url=tmp.name, write_to=output_path)
            tmp.close()
            os.remove(tmp.name)

    def export_dropout_right_third(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   r: torch.Tensor,
                                   theta: torch.Tensor,
                                   v: torch.Tensor,
                                   c: torch.Tensor,
                                   output_path: str,
                                   svg_hollow: bool = False):
        W = self.canvas_w
        with torch.no_grad():
            keep_left = x < (W / 2)
            prob = torch.where(keep_left,
                                torch.ones_like(x),
                                (W - x) / (W / 2))
            keep_idx = keep_left | (torch.rand_like(prob) < prob)

            x_d, y_d, r_d = [t[keep_idx] for t in (x, y, r)]
            theta_d, v_d, c_d = [t[keep_idx] for t in (theta, v, c)]

        self.export(x_d, y_d, r_d, theta_d, v_d, c_d,
                    output_path=output_path,
                    svg_hollow=svg_hollow)
        print(f"drop-out PDF saved to {output_path}")

    def export_with_pngs(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         r: torch.Tensor,
                         theta: torch.Tensor,
                         v: torch.Tensor,
                         c: torch.Tensor,
                         output_folder: str,
                         svg_hollow: bool = False,
                         border_color: str = 'red',
                         border_width: float = 6.0):
        """
        For each primitive added (bottom to top), export a PNG frame
        highlighting that primitive with a colored border.
        """
        # Prepare arrays
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        c_np = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))

        N = len(x_np)
        p = len(self.svg_paths)
        rev_indices = list(reversed(range(N)))

        os.makedirs(output_folder, exist_ok=True)

        # Step through adding one primitive at a time
        for frame_idx, idx in enumerate(rev_indices):
            # Build SVG root
            root = ET.Element(f'{{{SVG_NS}}}svg', {
                'width': str(int(self.canvas_w)),
                'height': str(int(self.canvas_h)),
                'viewBox': f"{int(-self.canvas_w/2)} {int(-self.canvas_h/2)} {int(self.canvas_w)} {int(self.canvas_h)}"
            })
            # Draw all primitives up to this step
            for j in rev_indices[:frame_idx+1]:
                i_mod = j % p
                tree = ET.parse(self.svg_paths[i_mod])
                tpl_root = tree.getroot()
                self._remove_styles(tpl_root)
                children = list(tpl_root)

                theta_deg = np.degrees(theta_np[j])
                transform = (
                    f"translate({x_np[j]-self.canvas_w/2:.3f},{y_np[j]-self.canvas_h/2:.3f}) "
                    f"rotate({theta_deg:.3f}) "
                    f"scale({r_np[j]:.3f}) "
                    f"scale({self.norm_scale:.4f}) "
                    f"translate({-self.view_w/2},{-self.view_h/2})"
                )

                # Main group
                g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
                # Style fill or hollow
                r_col, g_col, b_col = c_np[j]
                style = ({
                    'fill': f'rgb({int(r_col*255)},{int(g_col*255)},{int(b_col*255)})',
                    'fill-opacity': f"{alpha_vals[j]:.4f}"
                } if not svg_hollow else {
                    'stroke': f'rgb({int(r_col*255)},{int(g_col*255)},{int(b_col*255)})',
                    'stroke-opacity': f"{alpha_vals[j]:.4f}",
                    'stroke-width': str(border_width),
                    'fill': f'rgb({int(r_col*255)},{int(g_col*255)},{int(b_col*255)})',
                    'fill-opacity': '0'
                })
                g.attrib.update(style)
                for ch in children:
                    g.append(deepcopy(ch))
                root.append(g)

                # If this is the newly added primitive, draw colored border
                if j == idx:
                    border_g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
                    border_g.attrib.update({
                        'fill': 'none',
                        'stroke': border_color,
                        'stroke-width': str(border_width)
                    })
                    for ch in children:
                        border_g.append(deepcopy(ch))
                    root.append(border_g)

            # Write temp SVG and export PNG with white background
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.svg')
            ET.ElementTree(root).write(tmp.name, encoding='utf-8', xml_declaration=True)
            frame_path = os.path.join(output_folder, f'frame_{frame_idx:04d}.png')
            svg2png(
                url=tmp.name,
                write_to=frame_path,
                output_width=self.canvas_w,
                background_color='white'
            )
            tmp.close()
            os.remove(tmp.name)
            print(f"Wrote frame {frame_idx} -> {frame_path}")

        self.make_mp4(frames_folder=output_folder,
                        video_path=output_folder[:-1]+".mp4",
                        )


    def make_mp4(self,
                frames_folder: str,
                video_path: str,
                fps: int = 60):
        """
        Assemble all PNG frames in `frames_folder` named 'frame_*.png' into an MP4.
        """
        frame_paths = sorted(glob.glob(os.path.join(frames_folder, 'frame_*.png')))
        if not frame_paths:
            raise ValueError(f"No frames found in {frames_folder}")
        # Read first to get size
        first = cv2.imread(frame_paths[0])
        if first is None:
            raise RuntimeError(f"Failed to read {frame_paths[0]}")
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            writer.write(img)
        writer.release()
        print(f"MP4 video saved to {video_path}")


    def export_to_mp4_incremental(self,
                                x: torch.Tensor,
                                y: torch.Tensor,
                                r: torch.Tensor,
                                theta: torch.Tensor,
                                v: torch.Tensor,
                                c: torch.Tensor,
                                video_path: str,
                                fps: int = 60,
                                svg_hollow: bool = False,
                                border_color: str = 'red',
                                border_width: float = 6.0):
        """
        - Single loop accumulating primitives one by one
        - Convert to PNG frame directly in the loop → record to VideoWriter
        """
        import io
        import numpy as np
        import cv2
        import xml.etree.ElementTree as ET
        from cairosvg import svg2png

        # 1) Tensor → NumPy
        x_np     = x.detach().cpu().numpy()
        y_np     = y.detach().cpu().numpy()
        r_np     = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np     = v.detach().cpu().numpy()
        c_np     = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))

        N = len(x_np)
        p = len(self.svg_paths)
        rev_indices = list(reversed(range(N)))

        # 2) Create one empty SVG root, then accumulate primitives and highlight borders
        root = ET.Element(f'{{{SVG_NS}}}svg', {
            'width': str(int(self.canvas_w)),
            'height': str(int(self.canvas_h)),
            'viewBox': f"{int(-self.canvas_w/2)} {int(-self.canvas_h/2)} {int(self.canvas_w)} {int(self.canvas_h)}"
        })

        # 3) Determine video size — generate first frame
        # (Here, create temporary root with only first primitive added)
        first_idx = rev_indices[0]
        self._append_primitive_to_root(root, first_idx,
                                    x_np, y_np, r_np, theta_np, c_np, alpha_vals,
                                    svg_hollow, border_color, border_width)
        svg_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        png_bytes = svg2png(bytestring=svg_bytes,
                            output_width=self.canvas_w,
                            background_color='white')
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        first = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        h, w = first.shape[:2]

        # 4) Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        import time
        # 5) Single loop: accumulate primitives one by one → record one frame per cycle
        #    First primitive already added to root, so enumerate starts at 0
        for frame_idx, idx in enumerate(rev_indices):
            t_start = time.time()

            if frame_idx > 0:
                t_primitive_start = time.time()
                self._append_primitive_to_root(
                    root, idx,
                    x_np, y_np, r_np, theta_np, c_np, alpha_vals,
                    svg_hollow, border_color, border_width
                )
                t_primitive_end = time.time()
                print(f"[{frame_idx}] Primitive add time: {t_primitive_end - t_primitive_start:.4f}s")
            else:
                print(f"[{frame_idx}] First frame: Primitive add skipped")

            t_svg_start = time.time()
            svg_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            t_svg_end = time.time()

            t_png_start = time.time()
            png_bytes = svg2png(
                bytestring=svg_bytes,
                output_width=self.canvas_w,
                background_color='white'
            )
            t_png_end = time.time()

            t_decode_start = time.time()
            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            t_decode_end = time.time()

            t_write_start = time.time()
            writer.write(frame)
            t_write_end = time.time()

            print(f"[{frame_idx}] SVG to string: {t_svg_end - t_svg_start:.4f}s")
            print(f"[{frame_idx}] SVG to PNG:   {t_png_end - t_png_start:.4f}s")
            print(f"[{frame_idx}] PNG decode:   {t_decode_end - t_decode_start:.4f}s")
            print(f"[{frame_idx}] Frame write:  {t_write_end - t_write_start:.4f}s")
            print(f"[{frame_idx}] Total loop time: {t_write_end - t_start:.4f}s\n")
            print("asdf")

        writer.release()
        print(f"Saved MP4 video to {video_path}")

    # -----------------------------------------------------
    def _append_primitive_to_root(self, root, j,
                                x_np, y_np, r_np, theta_np, c_np, alpha_vals,
                                svg_hollow, border_color, border_width):
        """
        Helper to append to root(<svg> Element):
        - j-th primitive and
        - highlight border (for this primitive only)
        """
        tree     = ET.parse(self.svg_paths[j % len(self.svg_paths)])
        tpl_root = tree.getroot()
        self._remove_styles(tpl_root)
        children = list(tpl_root)

        theta_deg = np.degrees(theta_np[j])
        transform = (
            f"translate({x_np[j]-self.canvas_w/2:.3f},{y_np[j]-self.canvas_h/2:.3f}) "
            f"rotate({theta_deg:.3f}) "
            f"scale({r_np[j]:.3f}) "
            f"scale({self.norm_scale:.4f}) "
            f"translate({-self.view_w/2},{-self.view_h/2})"
        )

        # 1) Main shape
        g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
        if not svg_hollow:
            g.attrib.update({
                'fill':   f'rgb({int(c_np[j][0]*255)},{int(c_np[j][1]*255)},{int(c_np[j][2]*255)})',
                'fill-opacity': f"{alpha_vals[j]:.4f}"
            })
        else:
            g.attrib.update({
                'stroke':       f'rgb({int(c_np[j][0]*255)},{int(c_np[j][1]*255)},{int(c_np[j][2]*255)})',
                'stroke-opacity': f"{alpha_vals[j]:.4f}",
                'stroke-width': str(border_width),
                'fill':         f'rgb({int(c_np[j][0]*255)},{int(c_np[j][1]*255)},{int(c_np[j][2]*255)})',
                'fill-opacity': '0'
            })
        for ch in children:
            g.append(deepcopy(ch))
        root.append(g)

        # 2) Highlight border
        border_g = ET.Element(f'{{{SVG_NS}}}g', {'transform': transform})
        border_g.attrib.update({
            'fill': 'none',
            'stroke': border_color,
            'stroke-width': str(border_width)
        })
        for ch in children:
            border_g.append(deepcopy(ch))
        root.append(border_g)

    def export_sequence(self,
                       frame_results: list,
                       output_html_path: str,
                       svg_hollow: bool = False,
                       fps: int = 24,
                       html_extra_meta: dict = {}):
        """
        Export multiple frames as a sequence for HTML animation.
        
        Args:
            frame_results: List of frame result dictionaries containing x,y,r,theta,v,c tensors
            output_html_path: Path for HTML output file
            svg_hollow: Whether to use hollow SVG style
            fps: Frames per second for animation
        """
        import json
        
        SVG_NS = "http://www.w3.org/2000/svg"
        
        if not frame_results:
            raise ValueError("No frame results provided")
        
        print(f"Exporting sequence of {len(frame_results)} frames to HTML...")
        
        # Extract frame data and generate transform sequences
        frame_sequence = []
        
        for frame_idx, frame_result in enumerate(frame_results):
            print(f"Processing frame {frame_idx + 1}/{len(frame_results)}...")
            
            # Extract parameters from frame result
            x_np = frame_result['x'].detach().cpu().numpy()
            y_np = frame_result['y'].detach().cpu().numpy()
            r_np = frame_result['r'].detach().cpu().numpy()
            theta_np = frame_result['theta'].detach().cpu().numpy()
            v_np = frame_result['v'].detach().cpu().numpy()
            c_np = torch.sigmoid(frame_result['c']).detach().cpu().numpy()
            alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))
            
            N = len(x_np)
            p = len(self.svg_paths)
            
            # Generate transform strings for this frame
            frame_transforms = []
            
            for i in reversed(range(N)):
                idx = (N-i-1) % p
                
                theta_deg = np.degrees(theta_np[i])
                transform = (
                    f"translate({x_np[i]-self.canvas_w/2:.3f},{y_np[i]-self.canvas_h/2:.3f}) "
                    f"rotate({theta_deg:.3f}) "
                    f"scale({r_np[i]:.3f}) "
                    f"scale({self.norm_scale:.4f}) "
                    f"translate({-self.view_w/2},{-self.view_h/2})"
                )
                frame_transforms.append(transform)
            
            frame_sequence.append(frame_transforms)
        
        # Create SVG for the first frame (for initial display)
        first_frame = frame_results[0]
        x_np = first_frame['x'].detach().cpu().numpy()
        y_np = first_frame['y'].detach().cpu().numpy()
        r_np = first_frame['r'].detach().cpu().numpy()
        theta_np = first_frame['theta'].detach().cpu().numpy()
        v_np = first_frame['v'].detach().cpu().numpy()
        c_np = torch.sigmoid(first_frame['c']).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))
        
        N = len(x_np)
        p = len(self.svg_paths)
        
        # Create SVG root for HTML
        root_html = ET.Element(f'{{{SVG_NS}}}svg', {
            'id': 'svgsplat1',
            'style': 'overflow: visible;',
            'width': str(self.canvas_w),
            'height': str(self.canvas_h),
            'viewBox': f"{-self.canvas_w/2} {-self.canvas_h/2} {self.canvas_w} {self.canvas_h}"
        })
        wrapper_g_html = ET.Element('g', {'id': 'wrapper', 'transform': 'translate(0,0)'})
        
        # Add primitives for first frame
        for i in reversed(range(N)):
            idx = (N-i-1) % p
            tree = ET.parse(self.svg_paths[idx])
            template_root = tree.getroot()
            self._remove_styles(template_root)
            children = list(template_root)
            
            theta_deg = np.degrees(theta_np[i])
            transform = (
                f"translate({x_np[i]-self.canvas_w/2:.3f},{y_np[i]-self.canvas_h/2:.3f}) "
                f"rotate({theta_deg:.3f}) "
                f"scale({r_np[i]:.3f}) "
                f"scale({self.norm_scale:.4f}) "
                f"translate({-self.view_w/2},{-self.view_h/2})"
            )
            g = ET.Element('g', {'transform': transform})
            
            r_color, g_color, b_color = c_np[i]
            r_int = int(np.clip(r_color * 255, 0, 255))
            g_int = int(np.clip(g_color * 255, 0, 255))
            b_int = int(np.clip(b_color * 255, 0, 255))
            
            if svg_hollow:
                g.attrib.update({
                    'stroke': f'rgb({r_int},{g_int},{b_int})',
                    'stroke-opacity': f"{alpha_vals[i]:.4f}",
                    'stroke-width': str(self.stroke_width),
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': '0'
                })
            else:
                g.attrib.update({
                    'fill': f'rgb({r_int},{g_int},{b_int})',
                    'fill-opacity': f"{alpha_vals[i]:.4f}"
                })
            
            for child in children:
                g.append(deepcopy(child))
            wrapper_g_html.append(g)
        
        root_html.append(wrapper_g_html)
        
        # Convert SVG to string
        svg_content = ET.tostring(root_html, encoding='unicode')
        
        # Prepare metadata
        meta_tags = []
        meta_tags.append(f'<meta name="frameCount" content="{len(frame_results)}">')
        meta_tags.append(f'<meta name="fps" content="{fps}">')
        for k, v in html_extra_meta.items():
            meta_tags.append(f'<meta name="{k}" content="{v}">')
        
        # Create HTML content with proper indentation
        html_head = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="numClass" content="{len(self.svg_paths)}">
            {chr(10).join('            ' + tag for tag in meta_tags)}
            <title>Sequential Splatting Animation</title>
            <link rel="stylesheet" href="demo_html.css">
        </head>
        <body id="demo_html">
        """
        
        # Embed frame sequence data as JavaScript
        frame_sequence_js = f"""
            <script>
            // Frame sequence data for animation
            window.frameSequenceData = {json.dumps(frame_sequence)};
            console.log('Loaded frame sequence:', window.frameSequenceData.length, 'frames');
            </script>
        """
        
        html_tail = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/Draggable.min.js"></script>
            <script src="demo_html.js"></script>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_head)
            f.write(svg_content)
            f.write(frame_sequence_js)
            f.write(html_tail)
        
        print(f"Sequential HTML export completed: {output_html_path}")
        print(f"Frames: {len(frame_results)}, FPS: {fps}")
        
        return output_html_path
